# Project Worklog — Autonomous Strategy Research Agent

---
Task ID: 1
Agent: main (Super Z)
Task: Find, backtest, and rank BTC/ETH trading strategies meeting 7 hard gates (70% win rate, 1:2 R:R, +net profit, ≤25% DD, ≥200 OOS trades, OOS confirmation, no lookahead) — or honestly report that none pass.

Work Log:
- Phase 0: Environment check. Python 3.12.15, pandas 2.2.3, numpy 2.1.3, requests 2.32.5, matplotlib 3.9.2, scipy 1.14.1. Installed pyarrow 26.0.0 for Parquet support. No vectorbt/backtesting.py — wrote hand-rolled event-driven backtester per spec allowance.
- Phase 1: `scripts/fetch_data.py` — Downloaded 7.8 years of Binance klines (2019-01-01 → 2026-10-14) for BTCUSDT+ETHUSDT at 15m+1h. Cached as Parquet. ~272k 15m candles and ~68k 1h candles per symbol; ~21 small gaps from Binance maintenance (<2h each). Metadata logged to JSON.
- Phase 2: `scripts/generate_strategies.py` — Sampled 625 distinct rule+side combinations from a 283,920-element cartesian grid, replicated across 2 symbols × 2 timeframes = 2,500 distinct strategy configs. Good distribution across all required component families (trend filters, entry triggers, confirmations, stops, targets, exits, sides).
- Phase 3: `scripts/backtester.py` — Event-driven backtester with strict correctness: indicators use only closed-bar data, entries fill at next-bar open with slippage+commission, conservative stop-first-when-both-hit rule, 1% equity risk per trade, full trade CSV logging. Also includes `pass1_fast_filter` for vectorized Pass 1.
- Phase 3b + 4: `scripts/evaluate.py` — Three-stage funnel. Pass 1 (vectorized, 6-mo IS): 363/2500 survived (14.5%). Pass 2 (full 70% IS, ≥200 trades + net positive): 29/363 survived (8.0%). Pass 3 (untouched 30% OOS): all 29 measured. Wrote `download/results.csv` with all metrics + per-gate pass/fail flags.
- Phase 5: `scripts/robustness.py` — Ran cost stress (2× commission+slippage), parameter sensitivity (±10%, ±25%), walk-forward (5 windows), and regime split (ADX>25 vs ADX<20) on top 10 candidates (top 5 by OOS net profit + top 5 with ≥200 OOS trades). All 10 failed cost stress. Only 1/10 passed walk-forward (and that one had −20% OOS net). 7/10 regime-dependent. Exported `download/trades_best.csv` (238 trades) and `download/equity_curve.png` for best candidate.
- Phase 6: `download/strategy_search_report.md` — Final report with executive summary, top-10 table, robustness table, best-candidate failure analysis (smoking gun: gross-profitable +8% but net-negative −4% after 12% cost drag), structural analysis of why no strategy passed, reproducibility section (Python version, package versions, data URLs, single shell command).
- `scripts/run_all.py` + `requirements.txt` — Single-command end-to-end runner with `--force-download` and `--from-scratch` flags.

Stage Summary:
- **Honest finding: NO strategy passed all 7 hard gates.** The closest candidate had 37.3% OOS win rate (vs 70% gate) and 99.9% drawdown (vs 25% gate). Among configs that satisfied the 200-trade OOS gate (10/29), best OOS net profit was −9.2%.
- Even in-sample, no configuration reached 50% win rate at 1:2 R:R. The 70%/1:2 gate is mathematically unreachable with simple technical rules on liquid BTC/ETH.
- The "best candidate" (ETHUSDT_1h_long_d7f1c1d5ec, 238 OOS trades) is gross-profitable (+8.0%) but net-negative (−4.1%) after 12.1% cost drag — the textbook failure mode of retail crypto strategies.
- All 10 robustness candidates failed cost stress (doubling costs flipped all negative). Only 1/10 passed walk-forward (with −20% OOS net anyway). 7/10 were regime-dependent.
- All deliverables in `/home/z/my-project/download/`: results.csv, robustness_results.csv, trades_best.csv, equity_curve.png, strategy_search_report.md. All scripts in `/home/z/my-project/scripts/`. Rerun with `python3 scripts/run_all.py`.
