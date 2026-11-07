import json

import pytest
from fastapi.testclient import TestClient

from demandcast.app import create_app
from demandcast.cli import backtest_markdown


class Broken:
    name = "claude"

    def write(self, f):
        raise RuntimeError("provider down")


@pytest.fixture(scope="module")
def client(art):
    return TestClient(create_app(art))


def test_healthz(client):
    r = client.get("/healthz").json()
    assert r["status"] == "ok" and r["gate_passed"] is True


def test_metrics_exposition(client):
    client.get("/api/prep?location=lodo")
    t = client.get("/metrics").text
    assert "demandcast_backtest_wape" in t and "demandcast_http_requests_total" in t
    assert 'demandcast_sim_total_cost_usd{policy="newsvendor"}' in t


def test_ui_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "prep planner" in r.text


def test_locations(client):
    r = client.get("/api/locations").json()
    assert [x["id"] for x in r["locations"]] == ["lodo", "cap-hill", "highlands", "truck"]
    assert r["tomorrow"] > r["today"]


def test_forecast_and_items(client):
    r = client.get("/api/forecast?location=cap-hill&days=7&items=true").json()
    assert len(r["daily"]) == 7
    d = r["daily"][0]
    assert d["lo"] <= d["mean"] <= d["hi"]
    assert all(len(v["mean"]) == 7 for v in r["items"].values())


def test_forecast_errors(client):
    assert client.get("/api/forecast?location=nowhere").status_code == 404
    assert client.get("/api/forecast?days=0").status_code == 400
    assert client.get("/api/prep?location=nowhere").status_code == 404


def test_hourly_band(client):
    r = client.get("/api/forecast/hourly?location=lodo").json()
    assert len(r["hours"]) == 12
    assert all(h["lo"] <= h["mean"] <= h["hi"] for h in r["hours"])
    assert abs(sum(h["mean"] for h in r["hours"]) - r["daily"]["mean"]) < 1.5


def test_prep_list(client):
    r = client.get("/api/prep?location=lodo").json()
    assert r["items"] and r["ingredients"]
    assert all(x["prep"] >= x["forecast"] for x in r["items"])


def test_orders_filter(client):
    r = client.get("/api/orders?location=lodo&supplier=meats").json()
    assert {x["supplier"] for x in r["lines"]} == {"meats"}
    assert len(r["next"]) == 1


def test_labor(client):
    r = client.get("/api/labor?location=lodo").json()
    assert r["labor_hours"] > 0 and r["shifts"]


def test_backtest_hides_scores(client):
    r = client.get("/api/backtest").json()
    assert "scores" not in r and r["main"]


def test_simulation(client):
    r = client.get("/api/simulation").json()
    assert [p["policy"] for p in r["policies"]] == ["gut_feel", "forecast_mean", "newsvendor"]


def test_briefing(client):
    r = client.get("/api/briefing?location=highlands").json()
    assert r["provider"] == "offline" and r["text"] and r["facts"]["location"] == "Highlands"


def test_briefing_falls_back_when_provider_throws(art):
    c = TestClient(create_app(art, provider=Broken()))
    r = c.get("/api/briefing").json()
    assert r["fallback"] is True and r["provider"] == "offline"


def test_request_id_echo(client):
    r = client.get("/api/locations", headers={"x-request-id": "abc123"})
    assert r.headers["x-request-id"] == "abc123"


def test_artifacts_are_json_and_markdown_renders(art):
    json.dumps(art)
    md = backtest_markdown(art["backtest"])
    assert "PASS" in md and "| H | model |" in md
