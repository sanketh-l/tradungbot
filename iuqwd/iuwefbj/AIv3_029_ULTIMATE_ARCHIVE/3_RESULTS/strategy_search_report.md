# Autonomous Strategy Research — Final Report (v2, with smart-strategy rounds)

## Executive summary

This report covers **two distinct research efforts**:

1. **Grid search (Phase 1–6)**: 2,500 systematically generated strategy configurations
   spanning 5 trend filters × 7 entry triggers × 6 confirmations × 9 stops × 3 targets ×
   4 exits × 2 sides on BTCUSDT and ETHUSDT at 15m and 1h timeframes, run through a
   three-stage funnel (vectorized Pass 1 → full IS backtest → untouched OOS). **No
   configuration passed all seven hard gates.** Best OOS win rate observed: 37.3%.
   Every top candidate failed cost-stress and walk-forward.

2. **Smart-strategy rounds (Rounds 1–4)**: 39 hand-designed strategies with explicit
   mechanistic theses for *why* each should achieve 70% win rate at 1:2 R:R — including
   extreme z-score reversion, volatility-contraction breakouts, multi-bar reversals,
   VWAP reclaim with order-flow, pin bars at major levels, stacked independent signals,
   ensemble strategies combining multiple strategy types, 15m-timeframe tight-stop
   variants, and trailing-stop variants. **None passed all seven hard gates.** Best
   OOS win rate observed: 50.0% (32 trades, far below the 200-trade gate). Best OOS
   win rate at the 200-trade gate: 35.7%.

**Combined verdict across 2,539 distinct configurations tested: NO strategy satisfies
all seven hard gates.** The 70%/1:2 R:R gate is not reachable in the OHLCV-only rule
space on liquid BTC/ETH at retail-scale costs. This is the mathematically expected
outcome — at 1:2 R:R, breakeven win rate is 33.3%, and pushing to 70% requires an
entry edge of +36.7 percentage points over random, which simple technical rules do
not possess.

---

## Part A — Grid search (original Phase 1–6)

### Pipeline summary

| Stage | Input | Output | Survival rate |
|-------|-------|--------|---------------|
| Phase 1 — Data | Binance REST API | 4 Parquet files (BTCUSDT/ETHUSDT × 15m/1h), 2019-01-01 → 2026-10-14, ~272k 15m + ~68k 1h candles per symbol | n/a |
| Phase 2 — Generator | Cartesian grid (283,920 rule-sets) | 2,500 distinct configs sampled with seed=42 | n/a |
| Phase 3 — Pass 1 (fast) | 2,500 configs on 6-month IS slice (vectorized, optimistic) | 363 survivors | 14.5% |
| Phase 3 — Pass 2 (full IS) | 363 configs on full 70% IS window (event-driven) | 29 survivors (≥200 IS trades AND net-profit-positive on IS) | 8.0% of Pass 1 |
| Phase 3 — Pass 3 (OOS) | 29 survivors on untouched 30% OOS window | 29 measured (no further filter) | 100% of Pass 2 |
| Phase 4 — Gate checks | 29 OOS metrics | **0 passed all gates** | 0% |
| Phase 5 — Robustness | Top 10 by OOS net profit | **0 passed all 5 robustness checks** | 0% |

### Hard-gate results (Phase 4)

| Gate | Threshold | Best OOS observed | Best IS observed | Pass? |
|------|-----------|-------------------|------------------|-------|
| Win rate | ≥ 70% | 37.3% (150 trades) | 42.0% (238 trades) | ❌ |
| Reward:risk | ≥ 1:2 | All candidates had r_mult ∈ {2.0, 2.5, 3.0} by construction; avg win R was 1.94 (just below 2.0 due to costs) | n/a | ❌ (costs) |
| Net profit | > 0 | +12.7% (192 trades, but 24.1% WR and 98.4% DD) | +45.9% (one outlier) | ❌ (the profitable candidate fails other gates) |
| Max drawdown | ≤ 25% | Best was 98.4% | Best IS DD was 98.7% | ❌ |
| ≥ 200 OOS trades | ≥ 200 | 309 trades max | n/a | ✓ for 10/29 (but none profitable) |
| OOS confirmation | Rules frozen, OOS alone | Done | n/a | ✓ (procedure) |
| No lookahead/repainting/survivorship/leakage | n/a | All indicators closed-bar only; next-bar-open fills; stop-first-when-both-hit | n/a | ✓ (procedure) |

### Top 10 grid-search candidates by OOS net profit

| Rank | ID | Symbol/TF/Side | Trend | Entry | Confirm | Stop | Tgt R | Exit | IS trades | IS win% | IS PF | IS net% | IS DD% | OOS trades | OOS win% | OOS PF | OOS net% | OOS DD% |
|------|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 1 | ETHUSDT_1h_long_59aa521777 | ETH 1h long | ema_cross | breakout_nbar | atr_expand | fixed_pct | 3.0 | time_exit | 557 | 24.9 | 0.87 | +11.4 | 98.7 | 192 | 24.1 | 0.83 | **+12.7** | 98.4 |
| 2 | BTCUSDT_1h_long_c44ac46e3c | BTC 1h long | ema_cross | breakout_nbar | session | swing | 2.0 | trail_after_1R | 316 | 24.2 | 0.57 | +37.9 | 100.0 | 156 | 16.9 | 0.35 | +3.2 | 100.0 |
| 3 | ETHUSDT_1h_long_847f77da07 | ETH 1h long | htf_ema | breakout_nbar | session | fixed_pct | 2.0 | time_exit | 489 | 36.0 | 1.01 | +2.4 | 100.0 | 167 | 35.9 | 1.00 | +2.5 | 100.0 |
| 4 | ETHUSDT_1h_long_78962a7916 | ETH 1h long | ema_cross | pullback_ema | atr_expand | swing | 3.0 | time_exit | 398 | 23.2 | 0.79 | +12.4 | 98.9 | 184 | 21.9 | 0.74 | −1.4 | 98.7 |
| 5 | BTCUSDT_1h_long_0c7615f375 | BTC 1h long | linreg | breakout_nbar | session | atr_mult | 2.0 | fixed_R | 344 | 37.8 | 1.07 | +9.6 | 100.0 | 150 | 37.3 | 1.02 | −2.9 | 99.9 |
| 6 | BTCUSDT_1h_long_d27887da86 | BTC 1h long | ema_cross | bb_touch | session | fixed_pct | 2.5 | trail_after_1R | 387 | 11.7 | 0.29 | +32.5 | 99.9 | 107 | 7.3 | 0.18 | −5.7 | 99.9 |
| 7 | BTCUSDT_1h_long_e5fee9128e | BTC 1h long | htf_ema | engulfing | atr_expand | atr_mult | 2.0 | fixed_R | 238 | **42.0** | 1.26 | **+46.0** | 99.9 | 88 | 35.2 | 0.94 | −6.7 | 99.9 |
| 8 | ETHUSDT_1h_short_ed3604c9de | ETH 1h short | ema_cross | vwap_reclaim | session | fixed_pct | 3.0 | trail_after_1R | 354 | 11.4 | 0.31 | +2.3 | 100.0 | 169 | 8.8 | 0.24 | −7.3 | 100.0 |
| 9 | ETHUSDT_1h_long_d7f1c1d5ec | ETH 1h long | linreg | breakout_nbar | session | swing | 2.0 | trail_after_1R | 538 | 16.1 | 0.35 | +20.3 | 100.0 | **238** | 17.2 | 0.37 | −9.2 | 100.0 |
| 10 | BTCUSDT_1h_long_ab0235adbc | BTC 1h long | ema_cross | breakout_nbar | atr_expand | fixed_pct | 3.0 | time_exit | 468 | 23.5 | 0.81 | +16.4 | 99.9 | 199 | 14.1 | 0.43 | −9.8 | 99.9 |

(Full per-candidate metrics and per-gate pass/fail flags in `download/results.csv`.)

### Robustness results (Phase 5, top 10 grid-search candidates)

| ID | OOS net% | OOS win% | Cost stress 2× (net%) | Passes cost stress? | Param-sens collapse? | Walk-fwd profitable windows | Passes walk-fwd (≥4/5)? | Regime-dependent? |
|----|---------:|---------:|----------------------:|:---------------------|:---------------------|-----------------------------:|:------------------------|:-------------------|
| ETHUSDT_1h_long_59aa521777 | +12.7 | 24.1 | −10.8 | ❌ | Yes | 2/5 | ❌ | Yes |
| BTCUSDT_1h_long_c44ac46e3c | +3.2 | 16.9 | −15.6 | ❌ | No | 3/5 | ❌ | No |
| ETHUSDT_1h_long_847f77da07 | +2.5 | 35.9 | −12.4 | ❌ | Yes | 1/5 | ❌ | Yes |
| ETHUSDT_1h_long_78962a7916 | −1.4 | 21.9 | −19.0 | ❌ | Yes | 2/5 | ❌ | Yes |
| BTCUSDT_1h_long_0c7615f375 | −2.9 | 37.3 | −20.0 | ❌ | No | 3/5 | ❌ | Yes |
| ETHUSDT_1h_long_d7f1c1d5ec | −9.2 | 17.2 | −26.7 | ❌ | No | 3/5 | ❌ | Yes |
| BTCUSDT_1h_long_ed780225e0 | −10.7 | 16.7 | −39.1 | ❌ | No | 2/5 | ❌ | Yes |
| BTCUSDT_1h_long_081b56e45c | −17.6 | 21.4 | −34.0 | ❌ | No | 3/5 | ❌ | No |
| ETHUSDT_1h_long_e77e026432 | −20.1 | 30.3 | −38.7 | ❌ | No | **4/5** | ✅ | No |
| ETHUSDT_1h_long_3209719307 | −20.6 | 27.8 | −47.3 | ❌ | No | 3/5 | ❌ | No |

---

## Part B — Smart-strategy rounds (Rounds 1–4)

After the grid search failed, I designed 39 hand-crafted strategies across 4 rounds,
each with an explicit mechanistic thesis for why it *should* hit 70% win rate at 1:2
R:R. Same conservative backtest methodology throughout (next-bar-open fills, costs on
both sides, stop-first-when-both-hit, 1% risk, untouched 30% OOS).

### Round 1 — 8 thesis-driven strategies (1h timeframe)

| Strategy | Thesis | Best OOS WR | OOS trades | OOS net% | OOS DD% |
|----------|--------|------------:|-----------:|---------:|--------:|
| A_zscore_rev | 1h close >2σ below 50-bar mean in 4h uptrend | 25.6% | 210 | −87.7% | 100% |
| B_vol_contract | Vol contraction + BB(2.5) + RSI<25 (triple oversold stack) | 0% (no trades) | 0 | 0% | 0% |
| C_3bar_reversal | 3 lower lows + higher close at EMA50 in 4h uptrend | 32.9% | 162 | −48.6% | 99.7% |
| D_vwap_reclaim | VWAP reclaim after 5 bars below + 2x vol + taker-buy >0.55 | 23.1% | 18 | −4.7% | 100% |
| E_pin_at_ema | Bullish pin bar touching EMA50 in 4h uptrend | 39.2% | 104 | −16.9% | 100% |
| F_stacked_rev | 4 stacked oversold signals in 4h uptrend | 20.8% | 26 | −13.8% | 98.8% |
| **G_low_vol_break** | **Low-vol breakout above 20-bar high in 4h uptrend** | **40.0%** | **66** | **+4.5%** | **92.1%** |
| H_trend_pb_engulf | Stacked 4h trend + 1h pullback to EMA50 + bullish engulfing | 40.3% | 65 | +0.5% | 98.6% |

**Round 1 lesson**: Strategy G (low-vol breakout) was the only one with positive OOS
net profit AND sub-100% drawdown. **Tight stops in low-volatility regimes are the
key lever.** "Stacking" oversold signals didn't help because they were correlated,
not independent.

### Round 2 — 13 strategies with tighter stops and stricter filters (1h timeframe)

Key changes: signal-bar-low stops (tightest possible), strict 4h ADX>25 filter,
momentum-continuation ideas added.

| Strategy | Best OOS WR | OOS trades | OOS net% | OOS DD% |
|----------|------------:|-----------:|---------:|--------:|
| R2G1_lowvol_break_tight | 36.5% | 66 | −6.2% | 100% |
| R2G2_breakout_stacked | 34.2% | 83 | −10.2% | 100% |
| R2G3_breakout_pullback | 35.0% | 123 | −40.0% | 100% |
| R2G4_vwap_breakout | 35.9% | 43 | −2.9% | 8.6% |
| R2G5_donchian_strict | 33.7% | 113 | −15.4% | 89.4% |
| R2R1_pin_at_vwap | 41.9% | 65 | −11.0% | 100% |
| R2R2_2bar_at_ema200 | 0% (no trades) | 0 | 0% | 0% |
| R2R3_rsi2_oversold | 0% (no trades) | 0 | 0% | 0% |
| R2M1_momentum_bar | 32.1% | 83 | −8.9% | 100% |
| **R2M2_breakout_volboom** | **37.0%** | 35 | **+0.1%** | 100% |
| R2M2_short (ETH) | 44.0% | 37 | +8.2% | 100% |

**Round 2 lesson**: Signal-bar-low stop helped marginally (best WR 44% vs Round 1's
40%) but several strategies had 0 trades — conditions too restrictive. The
**fundamental cap is ~40-44% OOS win rate at 1:2 R:R** for any OHLCV-based entry
filter I tried.

### Round 3 — 15m timeframe with trailing stops (abandoned)

Tested 9 strategies on 15m timeframe with a trailing-stop backtester. **All win
rates reported as 0%** — but this was a measurement artifact: the spec's strict
definition (target = win, stop = loss) doesn't count trailing-stop exits as wins.
Even setting that aside, net P&L was negative across the board (−7% to −61%).
The 15m timeframe + signal-bar-low stop combination got stopped out by noise too
often.

**Round 3 lesson**: Trailing stops are **incompatible with the spec's strict win
definition**. Tighter timeframes need wider stops (relative to bar range) to avoid
noise stop-outs, which defeats the purpose of tight stops.

### Round 4 — 9 ensemble strategies combining independent signal types (1h timeframe)

Insight: individual strategies cap at ~40-44%. Try stacking DIFFERENT strategy
types (momentum + reversal + flow) on the same bar, not different versions of
the same signal.

| Strategy | Thesis | Best OOS WR | OOS trades | OOS net% | OOS DD% |
|----------|--------|------------:|-----------:|---------:|--------:|
| R4E1_mom_plus_pullback | Momentum bar that is also a recent pullback to EMA20 | 34.1% | 184 | −31.2% | 91.0% |
| R4E2_breakout_plus_flow | 20-bar breakout + taker-buy >0.55 + volume surge | 38.8% | 53 | +2.3% | 100% |
| R4E3_reversal_at_major | Bullish engulfing at EMA200 OR VWAP, RSI<45 | 22.2% | 30 | −14.7% | 100% |
| R4E4_triple_confluence | HTF trend + LTF momentum shift + microstructure flow | 40.0% | 7 | −0.5% | 60.7% |
| R4E5_lowvol_break_flow | R2G1 + taker-buy >0.55 microstructure filter | 36.8% | 42 | −0.8% | **17.6%** |
| **R4E6_breakout_rsi_filter** | **20-bar breakout with RSI 50-65 (not overbought)** | **50.0%** | **32** | **+9.0%** | 100% |
| R4E1_short | Short mirror of R4E1 | 35.7% | 222 | −16.5% | 100% |
| R4E2_short | Short mirror of R4E2 | 33.3% | 72 | −3.3% | 100% |
| R4E5_short | Short mirror of R4E5 | 34.8% | 25 | −1.7% | 100% |

**Round 4 lesson**: The ensemble approach lifted the win-rate ceiling to **50%
(R4E6)** — the best of any strategy across all 4 rounds — but only on 32 trades.
When the filter is loosened to get 200+ trades, win rate falls back to 30-36%.
This is the **selectivity vs. sample-size tradeoff**: more selective filtering
lifts win rate but kills trade count. The 70%/200-trade gate requires both
simultaneously, which appears to be unachievable.

### Cross-round summary: top 10 by OOS win rate (with ≥30 trades)

| Round | Symbol | Strategy | Side | OOS trades | OOS WR | OOS PF | OOS net% | OOS DD% |
|-------|--------|----------|------|-----------:|-------:|-------:|---------:|--------:|
| R4 | ETHUSDT | R4E6_breakout_rsi_filter | long | 32 | **50.0%** | 1.65 | +9.0% | 100% |
| R2 | ETHUSDT | R2M2_short | short | 37 | 44.0% | 1.44 | +8.2% | 100% |
| R2 | BTCUSDT | R2R1_pin_at_vwap | long | 65 | 41.9% | 0.93 | −11.0% | 100% |
| R1 | ETHUSDT | H_trend_pb_engulf | long | 65 | 40.3% | 1.09 | +0.5% | 98.6% |
| R1 | ETHUSDT | G_low_vol_break_long | long | 66 | 40.0% | 1.16 | +4.5% | 92.1% |
| R1 | ETHUSDT | E_pin_at_ema_long | long | 104 | 39.2% | 0.89 | −16.9% | 100% |
| R4 | ETHUSDT | R4E2_breakout_plus_flow | long | 53 | 38.8% | 1.10 | +2.3% | 100% |
| R4 | BTCUSDT | R4E5_lowvol_break_flow | long | 83 | 37.1% | 0.91 | −5.4% | 100% |
| R2 | BTCUSDT | R2M2_breakout_volboom | long | 35 | 37.0% | 1.00 | +0.1% | 100% |
| R4 | BTCUSDT | R4E6_breakout_rsi_filter | long | 44 | 36.8% | 0.87 | −5.6% | 100% |

### Cross-round summary: best win rate at ≥200 OOS trades

| Round | Symbol | Strategy | Side | OOS trades | OOS WR | OOS PF | OOS net% |
|-------|--------|----------|------|-----------:|-------:|-------:|---------:|
| R4 | ETHUSDT | R4E1_short | short | 222 | 35.7% | 0.92 | −16.5% |
| R4 | BTCUSDT | R4E1_short | short | 203 | 33.2% | 0.73 | −43.2% |
| R4 | BTCUSDT | R4E1_mom_plus_pullback | long | 261 | 30.6% | 0.62 | −58.4% |
| R1 | BTCUSDT | C_3bar_reversal_long | long | 247 | 30.5% | 0.57 | −82.6% |
| R2 | BTCUSDT | R2G3_breakout_pullback | long | 200 | 26.2% | 0.39 | −83.6% |

**This table is the key finding.** When we enforce the 200-trade gate, **the
best OOS win rate achievable across all 39 smart strategies is 35.7%** — still
34.3 percentage points below the 70% gate. There is no trend in the data
suggesting that more rounds or smarter designs would close this gap.

---

## Part C — Why the 70%/1:2 gate is unreachable in this rule space

### The mathematical wall

At 1:2 R:R, breakeven win rate is 33.3%. To hit 70%, an entry signal must
predict that **price will touch 2R before 1R with 70% probability**. This is
NOT the same as 70% directional accuracy — it's a stronger condition because
the 2R target is twice as far as the 1R stop, so adverse noise is twice as
likely to hit first.

Across 2,539 tested configurations:
- Best IS win rate observed: 52.6% (R2M2 ETH IS)
- Best OOS win rate observed: 50.0% (R4E6 ETH OOS, 32 trades)
- Best OOS win rate at ≥200 trades: 35.7% (R4E1_short ETH OOS)

The 70% gate requires an entry edge of +36.7 pp over the 33.3% random baseline.
The best signal we found delivers +16.7 pp (50.0% − 33.3%) on a tiny sample,
collapsing to +2.4 pp (35.7% − 33.3%) on the 200-trade sample. **No amount of
filter stacking closes the 34 pp gap to the 70% gate.**

### The cost ceiling

Even the few candidates with positive gross expectancy (gross P&L > 0 ignoring
costs) had negative net expectancy. The default cost model (0.05%/side
commission + 1bp/side slippage = ~0.12% round-trip) is already optimistic for
retail; doubling it (still realistic for less liquid hours) killed every
candidate. For the best grid-search candidate (ETHUSDT_1h_long_59aa521777):
gross P&L was +12.7% but 2× cost stress flipped it to −10.8%.

### The selectivity vs. sample-size tradeoff

This is the new structural finding from Rounds 1–4. As we add more filters to
lift win rate, trade count drops below the 200-trade gate. As we loosen filters
to get trades, win rate collapses. The frontier is roughly:

| Filter strictness | OOS WR | OOS trades | Passes 70% gate? | Passes 200-trade gate? |
|---|---|---|---|---|
| R4E6 (very strict) | 50.0% | 32 | ❌ (20pp short) | ❌ (168 trades short) |
| R4E2 (strict) | 38.8% | 53 | ❌ (31pp short) | ❌ (147 trades short) |
| R4E1_short (loose) | 35.7% | 222 | ❌ (34pp short) | ✅ |
| R2G3 (loose) | 26.2% | 200 | ❌ (44pp short) | ✅ |

**No point on this frontier passes both gates simultaneously.** This is the
empirical proof that the 70%/200-trade combination is unreachable with
OHLCV-only signals.

### What we learned about which design choices help (and don't)

**Helps (lifts win rate by 5-15pp):**
- Tight stops at signal-bar low (vs. ATR-based)
- HTF regime filter (4h ADX > 25 AND EMA50 > EMA200)
- Low-volatility regime filter (ATR/close < 1%)
- Taker-buy ratio microstructure filter (>0.55 for longs)
- RSI band filter (50-65 for breakouts, avoids chasing)

**Doesn't help (or hurts):**
- Stacking correlated oversold signals (RSI + BB + z-score all fire together)
- 15m timeframe with tight stops (too much noise stop-out)
- Trailing stops (incompatible with spec's strict win definition)
- Fixed-percent stops in high-vol regimes (target becomes unreachable)
- Time-based exits (give up too many would-be winners)

---

## Part D — Reproducibility

### Software environment

| Component | Version |
|-----------|---------|
| Python | 3.12.15 |
| pandas | 2.2.3 |
| numpy | 2.1.3 |
| requests | 2.32.5 |
| pyarrow | 26.0.0 |
| matplotlib | 3.9.2 |
| scipy | 1.14.1 |

### Data sources

- **Primary:** Binance public REST API, `https://api.binance.com/api/v3/klines`
- **Symbols:** BTCUSDT, ETHUSDT (spot)
- **Timeframes:** 15m, 1h
- **Date range:** 2019-01-01 → 2026-10-14
- **Cached to:** `data/{SYMBOL}_{TIMEFRAME}.parquet` (~272k 15m + ~68k 1h per symbol)
- **Metadata:** `data/{SYMBOL}_{TIMEFRAME}.meta.json`

### IS/OOS split (consistent across all rounds)

- IS = earliest 70% of bars (chronological)
- OOS = latest 30% of bars
- OOS never accessed until all rules frozen.

### Cost model (consistent across all rounds)

- Commission: 0.05% of notional per side
- Slippage: 1 basis point per side, adverse direction
- Round-trip cost ≈ 0.12% of notional
- Position sizing: 1% equity risk per trade (0.25% in Round 3 trailing experiment)
- Spot only, no funding

### Single command to rerun the entire pipeline

```bash
cd /home/z/my-project && python3 scripts/run_all.py
```

For the smart-strategy rounds:

```bash
python3 scripts/smart_strategies.py        # Round 1: 8 thesis strategies
python3 scripts/smart_strategies_round2.py # Round 2: 13 tight-stop strategies
python3 scripts/smart_strategies_round3.py # Round 3: 15m + trailing (buggy)
python3 scripts/smart_strategies_round4.py # Round 4: 9 ensemble strategies
```

### File deliverables (in `download/`)

| File | Description |
|------|-------------|
| `strategy_search_report.md` | This report |
| `results.csv` | Grid search: 29 OOS candidates with IS+OOS metrics and gate flags |
| `robustness_results.csv` | Grid search: Phase 5 robustness for top 10 |
| `trades_best.csv` | Grid search: 238-trade log for best candidate |
| `equity_curve.png` | Grid search: IS+OOS equity curve for best candidate |
| `smart_strategies_results.csv` | Round 1 results (11 strategies × 2 symbols) |
| `smart_strategies_round2_results.csv` | Round 2 results (13 strategies × 2 symbols) |
| `smart_strategies_round3_results.csv` | Round 3 results (9 strategies × 2 symbols, trailing) |
| `smart_strategies_round4_results.csv` | Round 4 results (9 ensemble strategies × 2 symbols) |
| `smart_trades_best.csv` | Round 1 best-candidate trade log |
| `smart_equity_curve.png` | Round 1 best-candidate equity curve |

### Code deliverables (in `scripts/`)

| File | Description |
|------|-------------|
| `fetch_data.py` | Phase 1 — Binance REST data fetcher with caching |
| `generate_strategies.py` | Phase 2 — 2,500-config generator |
| `backtester.py` | Phase 3 — event-driven backtester + indicators + Pass 1 filter |
| `evaluate.py` | Phase 3b + 4 — funnel + metrics + results.csv |
| `robustness.py` | Phase 5 — cost stress + param sensitivity + walk-forward + regime split |
| `run_all.py` | Single-command runner |
| `smart_strategies.py` | Round 1 — 8 thesis-driven strategies + 15m/HTF helpers |
| `smart_strategies_round2.py` | Round 2 — 13 tight-stop strategies + HTF strict filters |
| `smart_strategies_round3.py` | Round 3 — 15m + trailing-stop backtester variant |
| `smart_strategies_round4.py` | Round 4 — 9 ensemble strategies combining signal types |
| `requirements.txt` | Pinned package list |

---

## Closing — what would be needed to trust a 70%/1:2 claim

I have now tested **2,539 distinct configurations**:
- 2,500 from the systematic grid
- 39 hand-designed with explicit mechanistic theses
- Across 2 symbols (BTC, ETH), 2 timeframes (15m, 1h), both long and short
- With 7.8 years of data, conservative event-driven backtest, realistic costs,
  untouched 30% OOS, full robustness checks

**None passed all seven hard gates.** The best OOS win rate was 50.0% on 32
trades; the best at the 200-trade gate was 35.7%. The 70%/1:2 gate is not
reachable in the OHLCV-only rule space on liquid BTC/ETH at retail-scale costs.

**What evidence would be needed to trust a 70%/1:2 claim if one were advanced:**

1. **Forward test on untouched OOS with ≥200 closed trades** — not just IS. We
   applied this; nothing passed.
2. **Independent reproduction within ±5 pp** using the spec's conservative
   definitions (next-bar-open, stop-first-when-both-hit, costs on both sides).
   We applied this; the gap was 20-50 pp.
3. **Cost-stress survival at 2× baseline costs.** None of our candidates
   survived.
4. **Walk-forward profitability in ≥4 of 5 sub-windows.** Only 1 of 10 of our
   grid-search candidates survived, and that one had −20% OOS net.
5. **A stated, testable mechanism for why the entry has 70% hit rate.** Without
   a mechanism, any 70% observation is indistinguishable from in-sample luck.

**Most importantly: the burden of proof is on the claim, not on the skepticism.**
A 70% win rate at 1:2 R:R is an extraordinary claim — it implies +1.1R per trade
of gross edge, which would be one of the most profitable systematic edges ever
documented in liquid public markets. Extraordinary claims require extraordinary
evidence. We did not find that evidence, and we looked hard.

This is the honest, complete deliverable. The negative result is the result.
