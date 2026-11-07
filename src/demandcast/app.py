"""FastAPI service: forecasts, tomorrow's prep list, supplier orders, staffing, backtest results, and the UI.

Artifacts come from `demandcast train` (data/artifacts.json). If the file is missing the service builds them on
startup with the quick config, so `demandcast serve` always works.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse

from . import briefing, pipeline
from .logs import get_logger, log_event
from .metrics import Metrics

STATIC = Path(__file__).parent / "static"


def data_dir() -> Path:
    return Path(os.environ.get("DEMANDCAST_DATA_DIR", "data"))


def load_or_build(path: Path | None = None) -> dict:
    path = path or data_dir() / "artifacts.json"
    if path.exists():
        return pipeline.load(path)
    art = pipeline.build(pipeline.PipelineConfig.quick(), log=lambda m: None)
    return json_roundtrip(art)


def json_roundtrip(art: dict) -> dict:
    import json

    return json.loads(json.dumps(art, default=pipeline._default))


def create_app(artifacts: dict | None = None, provider: briefing.BriefingProvider | None = None) -> FastAPI:
    log = get_logger()
    t0 = time.perf_counter()
    art = artifacts if artifacts is not None else load_or_build()
    provider = provider or briefing.make_provider()
    metrics = Metrics()
    bt = art["backtest"]
    log_event(
        log,
        "artifacts_ready",
        seconds=round(time.perf_counter() - t0, 2),
        tomorrow=art["tomorrow"],
        model=bt["model"],
        gate_passed=bt["gate"]["passed"],
    )
    best = [r for r in bt["main"] if r["model"] == bt["model"] and r["level"] == "bottom"]
    for r in best:
        metrics.set_gauge(
            "demandcast_backtest_wape", "Backtest WAPE of the chosen model, item level.", r["wape"], horizon=str(r["H"])
        )
        if r["coverage"] is not None:
            metrics.set_gauge(
                "demandcast_interval_coverage",
                "Backtest coverage of the 90% interval, item level.",
                r["coverage"],
                horizon=str(r["H"]),
            )
    for lid, loc in art["locations"].items():
        metrics.set_gauge(
            "demandcast_forecast_tomorrow_units",
            "Forecast item units for tomorrow.",
            loc["tomorrow"]["mean"],
            location=lid,
        )
    for p in art["simulation"]["policies"]:
        metrics.set_gauge(
            "demandcast_sim_total_cost_usd",
            "Simulated month: waste plus lost margin, by policy.",
            p["total_cost"],
            policy=p["policy"],
        )

    app = FastAPI(title="Demand forecast and prep planning", version="1.0.0")

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t = time.perf_counter()
        response = await call_next(request)
        dt = time.perf_counter() - t
        route = getattr(request.scope.get("route"), "path", "unmatched")
        metrics.observe_request(route, response.status_code, dt)
        response.headers["x-request-id"] = rid
        if route not in ("/healthz", "/metrics"):
            log_event(
                log,
                "request",
                request_id=rid,
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                ms=round(dt * 1000, 1),
            )
        return response

    def loc_or_404(location: str) -> dict:
        if location not in art["locations"]:
            raise HTTPException(404, f"unknown location {location}")
        return art["locations"][location]

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "tomorrow": art["tomorrow"], "model": bt["model"], "gate_passed": bt["gate"]["passed"]}

    @app.get("/metrics")
    def prom():
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/locations")
    def locations():
        return {
            "today": art["today"],
            "tomorrow": art["tomorrow"],
            "weather_tomorrow": art["weather_tomorrow"],
            "events_ahead": art["events_ahead"],
            "locations": [
                {
                    "id": k,
                    "name": v["name"],
                    "mobile": v["mobile"],
                    "tomorrow_units": v["tomorrow"]["mean"],
                    "open_tomorrow": v["tomorrow"]["open"],
                    "site_tomorrow": v["tomorrow"]["site"],
                }
                for k, v in art["locations"].items()
            ],
            "generated": art["generated"],
            "cold_start": art["cold_start"],
        }

    @app.get("/api/forecast")
    def forecast(location: str = "lodo", days: int = 7, items: bool = False):
        if not 1 <= days <= len(art["total"]):
            raise HTTPException(400, f"days must be between 1 and {len(art['total'])}")
        loc = loc_or_404(location)
        out = {"location": location, "daily": loc["daily"][:days]}
        if items:
            out["items"] = {k: {s: v[s][:days] for s in ("mean", "lo", "hi")} for k, v in loc["items"].items()}
        return out

    @app.get("/api/forecast/total")
    def forecast_total(days: int = 7):
        return {"daily": art["total"][:days]}

    @app.get("/api/forecast/hourly")
    def hourly(location: str = "lodo"):
        loc = loc_or_404(location)
        t = loc["tomorrow"]
        return {
            "location": location,
            "date": art["tomorrow"],
            "open": t["open"],
            "site": t["site"],
            "daily": {k: t[k] for k in ("mean", "lo", "hi")},
            "hours": t["hours"],
        }

    @app.get("/api/prep")
    def prep_list(location: str = "lodo"):
        loc = loc_or_404(location)
        return {"location": location, "date": art["tomorrow"], **loc["prep"]}

    @app.get("/api/orders")
    def orders(location: str = "lodo", supplier: str | None = None):
        loc = loc_or_404(location)
        lines = [x for x in loc["orders"]["lines"] if supplier is None or x["supplier"] == supplier]
        nxt = [x for x in loc["orders"]["next"] if supplier is None or x["supplier"] == supplier]
        return {"location": location, "today": art["today"], "next": nxt, "lines": lines}

    @app.get("/api/labor")
    def labor(location: str = "lodo"):
        loc = loc_or_404(location)
        return {"location": location, "date": art["tomorrow"], **loc["labor"]}

    @app.get("/api/backtest")
    def backtest():
        return {k: v for k, v in bt.items() if k != "scores"}

    @app.get("/api/simulation")
    def simulation():
        return art["simulation"]

    @app.get("/api/briefing")
    def morning(location: str = "lodo"):
        loc_or_404(location)
        f = briefing.facts(art, location)
        try:
            out = provider.write(f)
        except Exception as e:  # provider outage: fall back, never fail the page
            log_event(log, "briefing_provider_error", provider=provider.name, error=str(e)[:200])
            out = {**briefing.OfflineProvider().write(f), "fallback": True}
        metrics.summary(out["provider"], out["fallback"])
        return {**out, "facts": f}

    return app
