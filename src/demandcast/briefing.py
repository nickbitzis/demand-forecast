"""Morning briefing for a kitchen manager, written only from computed numbers.

`facts(artifacts, location)` turns tomorrow's forecast, prep list, orders, and staffing into pre-formatted strings.
The offline writer (default) fills a template from them. The Claude provider gets the same facts and must copy
numbers exactly; any number in its text that is not in the facts makes the service fall back to the offline text.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

DEFAULT_MODEL = "claude-opus-5-5"
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def facts(art: dict, location: str) -> dict:
    loc = art["locations"][location]
    t = loc["tomorrow"]
    items = sorted(loc["prep"]["items"], key=lambda r: -r["prep"])
    peak = max(t["hours"], key=lambda h: h["mean"]) if t["hours"] else None
    nxt = [o for o in loc["orders"]["next"] if o["order_date"] == art["today"] and o["lines"]]
    risks = [x for x in loc["orders"]["lines"] if x["flag"].startswith("projected")]
    w = art["weather_tomorrow"]
    f = {
        "location": loc["name"],
        "date": art["tomorrow"],
        "interval": "90%",
        "open": "yes" if t["open"] else "no",
        "forecast_units": f"{t['mean']:,.0f}",
        "forecast_lo": f"{t['lo']:,.0f}",
        "forecast_hi": f"{t['hi']:,.0f}",
        "high_temp_c": f"{w['temp_max']:.0f}",
        "precip_mm": f"{w['precip_mm']:.0f}",
        "snow_cm": f"{w['snow_cm']:.0f}",
        "site": t["site"],
        "peak_hour": peak["hour"] if peak else "",
        "peak_hour_units": f"{peak['mean']:,.0f}" if peak else "",
        "top_prep": [f"{r['name']} {r['prep']} portions" for r in items[:4]],
        "prep_expected_cost": f"${loc['prep']['expected_cost']:,.0f}",
        "gut_expected_cost": f"${loc['prep']['gut_expected_cost']:,.0f}",
        "orders_today": [f"{o['supplier_name']} ${o['total']:,.0f}" for o in nxt],
        "run_out_risks": sorted({x["name"] for x in risks}),
        "labor_hours": f"{loc['labor']['labor_hours']:.0f}",
        "peak_staff": str(loc["labor"]["peak_staff"]),
    }
    ev = [e for e in art["events_ahead"] if e["date"] == art["tomorrow"]]
    if ev:
        e = max(ev, key=lambda e: e["attendance"])
        f["event"] = f"{e['venue_name']} ({e['kind']}), {e['attendance']:,} expected"
    return f


def numbers(text: str) -> set[str]:
    return {m.group(0).replace(",", "").rstrip(".") for m in NUM.finditer(text)}


def unsupported_numbers(text: str, f: dict) -> list[str]:
    return sorted(numbers(text) - numbers(json.dumps(f)))


class BriefingProvider(Protocol):
    name: str

    def write(self, f: dict) -> dict: ...


class OfflineProvider:
    name = "offline"

    def write(self, f: dict) -> dict:
        if f["open"] == "no":
            text = f"{f['location']} is closed on {f['date']}. No prep and no orders are needed for it."
        else:
            parts = [
                f"{f['location']}, {f['date']}: forecast {f['forecast_units']} items ({f['interval']} range "
                f"{f['forecast_lo']} to {f['forecast_hi']}), busiest at {f['peak_hour']} with about "
                f"{f['peak_hour_units']}.",
                f"Weather: high of {f['high_temp_c']} C, {f['precip_mm']} mm of rain, {f['snow_cm']} cm of snow.",
            ]
            if "event" in f:
                parts.append(f"Event nearby: {f['event']}.")
            parts.append("Prep the most: " + ", ".join(f["top_prep"]) + ".")
            parts.append(
                f"Expected waste plus lost margin with this prep list is {f['prep_expected_cost']}, against "
                f"{f['gut_expected_cost']} for prepping last week's sales."
            )
            if f["orders_today"]:
                parts.append("Orders to place tonight: " + ", ".join(f["orders_today"]) + ".")
            if f["run_out_risks"]:
                parts.append("May run short before the next delivery: " + ", ".join(f["run_out_risks"]) + ".")
            parts.append(f"Staffing: {f['labor_hours']} labor hours, {f['peak_staff']} people at the peak.")
            text = " ".join(parts)
        return {"provider": self.name, "model": None, "text": text, "unsupported_numbers": [], "fallback": False}


SYSTEM = (
    "You write a short morning briefing for a restaurant kitchen manager. Use only the facts provided. Every number "
    "you write must appear exactly as formatted in the facts. Do not compute, round, or add numbers. Plain English, "
    "complete sentences, no headings, no bullet points, 80 to 140 words."
)


class ClaudeProvider:
    name = "claude"

    def __init__(self, client=None, model: str | None = None, max_tokens: int = 500):
        self.model = model or os.environ.get("DEMANDCAST_CLAUDE_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import anthropic  # lazy: the app works without the extra installed

            self._client = anthropic.Anthropic()
        return self._client

    def write(self, f: dict) -> dict:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM,
            messages=[{"role": "user", "content": "Facts:\n" + json.dumps(f, indent=1)}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        bad = unsupported_numbers(text, f)
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        if bad or not text:
            off = OfflineProvider().write(f)
            return {**off, "fallback": True, "rejected_text": text, "unsupported_numbers": bad, "usage": usage}
        return {
            "provider": self.name,
            "model": self.model,
            "text": text,
            "unsupported_numbers": [],
            "fallback": False,
            "usage": usage,
        }


def make_provider(name: str | None = None) -> BriefingProvider:
    name = name or os.environ.get("DEMANDCAST_BRIEFING_PROVIDER", "offline")
    if name == "claude":
        return ClaudeProvider()
    return OfflineProvider()
