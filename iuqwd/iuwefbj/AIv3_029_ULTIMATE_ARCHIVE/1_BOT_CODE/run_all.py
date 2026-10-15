"""
Single-command end-to-end pipeline runner.

Usage:
    python3 scripts/run_all.py                 # full run, cached where possible
    python3 scripts/run_all.py --force-download  # re-download Binance data
    python3 scripts/run_all.py --from-scratch    # delete all caches and re-run

Stages:
    1. fetch_data.py       — download & cache BTC/ETH 15m + 1h from Binance
    2. generate_strategies.py — emit 2,500 strategy configs
    3. evaluate.py         — Pass 1 → Pass 2 → Pass 3 funnel + results.csv
    4. robustness.py       — Phase 5 robustness + trades_best.csv + equity_curve.png
    5. strategy_search_report.md already written; this script just confirms
       the file is present.

Each stage caches its intermediate outputs to data/ so a re-run picks up where
the previous run left off. Use --from-scratch to wipe caches.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data"
DOWNLOAD = ROOT / "download"


def run(cmd: list[str], cwd: Path = SCRIPTS) -> int:
    print(f"\n>>> {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(cwd))
    print(f"    (exit code {rc}, {time.time()-t0:.1f}s)")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-download", action="store_true",
                    help="Re-download Binance data even if cached Parquet exists")
    ap.add_argument("--from-scratch", action="store_true",
                    help="Delete all caches (data/, download/) and re-run from zero")
    args = ap.parse_args()

    if args.from_scratch:
        print("Wiping caches for full from-scratch run...")
        for sub in [DATA, DOWNLOAD]:
            if sub.exists():
                shutil.rmtree(sub)
            sub.mkdir(parents=True)

    # Stage 1: data
    fetch_args = ["python3", "fetch_data.py"]
    if args.force_download:
        fetch_args.append("--force")
    rc = run(fetch_args)
    if rc != 0:
        print("FATAL: fetch_data failed")
        sys.exit(rc)

    # Stage 2: strategy generation
    rc = run(["python3", "generate_strategies.py"])
    if rc != 0:
        print("FATAL: generate_strategies failed")
        sys.exit(rc)

    # Stage 3: funnel (Pass 1 → 2 → 3) + results.csv
    rc = run(["python3", "evaluate.py"])
    if rc != 0:
        print("FATAL: evaluate failed")
        sys.exit(rc)

    # Stage 4: robustness + trade log + equity curve
    rc = run(["python3", "robustness.py"])
    if rc != 0:
        print("FATAL: robustness failed")
        sys.exit(rc)

    # Stage 5: confirm report exists (the report is hand-authored with
    # empirical numbers from the actual run; if you re-run with different
    # data or costs, you should re-verify the numbers in the report).
    report = DOWNLOAD / "strategy_search_report.md"
    if report.exists():
        print(f"\n✓ Report present: {report}")
    else:
        print(f"\n⚠ Report MISSING: {report}")

    # Summary
    print("\n=== Pipeline complete ===")
    print(f"Deliverables in {DOWNLOAD}/:")
    for p in sorted(DOWNLOAD.iterdir()):
        if p.is_file():
            print(f"  {p.name}  ({p.stat().st_size:>8} bytes)")


if __name__ == "__main__":
    main()
