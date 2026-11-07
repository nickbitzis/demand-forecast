"""Command line: train (everything to data/artifacts.json), backtest (with the CI gate), serve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import backtest, pipeline
from .generator import GenConfig, generate


def _table(rows: list[dict], cols: list[str]) -> str:
    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def backtest_markdown(bt: dict) -> str:
    rows = [r for r in bt["main"] if r["level"] == "bottom"]
    s = [
        f"Origins: {bt['origins'][0]} to {bt['origins'][-1]} ({len(bt['origins'])}). "
        f"Chosen reconciliation: {bt['chosen_reconciliation']}.",
        "",
        _table(rows, ["H", "model", "wape", "mase", "bias", "coverage", "improvement_vs_snaive"]),
        "",
        _table(bt["reconciliation"], ["H", "method", "bottom", "location", "total", "mean"]),
        "",
        _table(bt["by_event"], ["day_type", "H", "snaive", "gbm", "coverage"]),
        "",
        _table(bt["cold_start"], ["segment", "H", "snaive", "gbm_no_cold_start", "gbm_cold_start", "n"]),
    ]
    g = bt["gate"]
    s += [
        "",
        f"Gate (item level, {g['horizon']}-day horizon): model WAPE {g['model_wape']:.4f} vs seasonal naive "
        f"{g['snaive_wape']:.4f}: {'PASS' if g['passed'] else 'FAIL'}",
    ]
    return "\n".join(s)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="demandcast")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="generate, backtest, forecast, plan, simulate; write artifacts.json")
    t.add_argument("--quick", action="store_true", help="small history and fewer origins (about 20 seconds)")
    t.add_argument("--seed", type=int, default=42)
    b = sub.add_parser("backtest", help="rolling-origin backtest only")
    b.add_argument("--quick", action="store_true")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--gate", action="store_true", help="exit 1 unless the model beats seasonal naive at 7 days")
    b.add_argument("--markdown", type=Path, help="append the tables as markdown (for $GITHUB_STEP_SUMMARY)")
    b.add_argument("--json", type=Path, help="write the backtest summary as JSON")
    s = sub.add_parser("serve", help="API and UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8824)
    a = ap.parse_args(argv)

    def log(m):
        print(m, file=sys.stderr)

    if a.cmd == "train":
        cfg = pipeline.PipelineConfig.quick(a.seed) if a.quick else pipeline.PipelineConfig(gen=GenConfig(seed=a.seed))
        art = pipeline.build(cfg, log=log)
        out = a.data_dir / "artifacts.json"
        pipeline.save(art, out)
        print(backtest_markdown(art["backtest"]))
        print(f"wrote {out} in {art['seconds']}s")
    elif a.cmd == "backtest":
        cfg = pipeline.PipelineConfig.quick(a.seed) if a.quick else pipeline.PipelineConfig(gen=GenConfig(seed=a.seed))
        bt = backtest.run(generate(cfg.gen), cfg.backtest, log=log)
        bt.pop("_frame")
        md = backtest_markdown(bt)
        print(md)
        if a.markdown:
            with a.markdown.open("a") as f:
                f.write("## Backtest\n\n" + md + "\n")
        if a.json:
            a.json.parent.mkdir(parents=True, exist_ok=True)
            a.json.write_text(json.dumps({k: v for k, v in bt.items() if k != "scores"}, indent=1))
        if a.gate and not bt["gate"]["passed"]:
            print("backtest gate failed: model is not better than seasonal naive", file=sys.stderr)
            sys.exit(1)
    elif a.cmd == "serve":
        import os

        import uvicorn

        os.environ.setdefault("DEMANDCAST_DATA_DIR", str(a.data_dir))
        from .app import create_app

        uvicorn.run(
            create_app(),
            host=os.environ.get("HOST", a.host),
            port=int(os.environ.get("PORT", a.port)),
            log_level="warning",
        )
