"""
Phase 2: Strategy generator.

Produces a population of strategy configurations by sampling from a
combinatorial grid of components. Each configuration is a serializable dict
with a deterministic hash ID, so it can be logged, ranked, and reproduced.

Families covered (per spec):
  - Trend filters: EMA crossover, Donchian, ADX threshold, HTF EMA slope, linreg slope
  - Entry triggers: pullback to EMA, N-bar breakout, RSI reversal, BB touch,
                    VWAP reclaim, engulfing, pin bar
  - Confirmation: volume avg, RSI band, ATR expansion, HTF agreement, session-of-day
  - Stop: ATR multiple, recent swing, fixed %
  - Target: fixed R multiple (2.0, 2.5, 3.0)
  - Exit: fixed R, trailing after 1R, time-based
  - Side: long / short (we test both; long-only is more realistic for spot,
          but short is included since the gate is symmetric)

The grid cartesian product is huge (>30k); we randomly sample N=2500 distinct
configs with a fixed seed for reproducibility.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "strategies.json"

# ---- Component grids -------------------------------------------------------

TREND_FILTERS = [
    {"kind": "ema_cross", "fast": 20, "slow": 50},
    {"kind": "ema_cross", "fast": 20, "slow": 100},
    {"kind": "ema_cross", "fast": 50, "slow": 200},
    {"kind": "ema_cross", "fast": 9,  "slow": 21},
    {"kind": "donchian",   "length": 20},
    {"kind": "donchian",   "length": 50},
    {"kind": "adx",        "period": 14, "threshold": 20},
    {"kind": "adx",        "period": 14, "threshold": 25},
    {"kind": "adx",        "period": 14, "threshold": 30},
    {"kind": "htf_ema",    "htf_len": 200, "slope_window": 5},
    {"kind": "htf_ema",    "htf_len": 200, "slope_window": 10},
    {"kind": "linreg",     "length": 20},
    {"kind": "linreg",     "length": 50},
]

ENTRY_TRIGGERS = [
    {"kind": "pullback_ema", "ema_len": 20, "max_dist_pct": 1.0},
    {"kind": "pullback_ema", "ema_len": 50, "max_dist_pct": 1.5},
    {"kind": "breakout_nbar","length": 20},
    {"kind": "breakout_nbar","length": 50},
    {"kind": "breakout_nbar","length": 10},
    {"kind": "rsi_reversal", "rsi_len": 14, "oversold": 30, "overbought": 50},
    {"kind": "rsi_reversal", "rsi_len": 14, "oversold": 35, "overbought": 55},
    {"kind": "bb_touch",     "bb_len": 20, "n_std": 2.0},
    {"kind": "bb_touch",     "bb_len": 20, "n_std": 2.5},
    {"kind": "vwap_reclaim", "vwap_len": 50},
    {"kind": "vwap_reclaim", "vwap_len": 100},
    {"kind": "engulfing"},
    {"kind": "pin_bar"},
]

CONFIRMATIONS = [
    {"kind": "none"},
    {"kind": "volume_avg", "vol_len": 20, "mult": 1.2},
    {"kind": "rsi_band",   "rsi_len": 14, "lo": 30, "hi": 70},
    {"kind": "atr_expand", "atr_len": 14, "mult": 1.0},
    {"kind": "session",    "start_h": 0, "end_h": 24},     # always-true (no filter)
    {"kind": "session",    "start_h": 7,  "end_h": 21},     # active hours
    {"kind": "session",    "start_h": 12, "end_h": 20},     # US/EU overlap
]

STOPS = [
    {"kind": "atr_mult",   "atr_len": 14, "mult": 1.0},
    {"kind": "atr_mult",   "atr_len": 14, "mult": 1.5},
    {"kind": "atr_mult",   "atr_len": 14, "mult": 2.0},
    {"kind": "atr_mult",   "atr_len": 14, "mult": 0.5},
    {"kind": "atr_mult",   "atr_len": 14, "mult": 2.5},
    {"kind": "swing",      "lookback": 10},
    {"kind": "swing",      "lookback": 20},
    {"kind": "fixed_pct",  "pct": 0.01},
    {"kind": "fixed_pct",  "pct": 0.015},
    {"kind": "fixed_pct",  "pct": 0.02},
]

TARGETS = [
    {"r_mult": 2.0},
    {"r_mult": 2.5},
    {"r_mult": 3.0},
]

EXITS = [
    {"kind": "fixed_R"},
    {"kind": "trail_after_1R", "trail_r": 0.5},
    {"kind": "time_exit", "max_bars": 24},
    {"kind": "time_exit", "max_bars": 48},
]

SIDES = ["long", "short"]

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
TIMEFRAMES = ["15m", "1h"]


def config_id(cfg: dict) -> str:
    """Deterministic short hash of the config (excluding symbol/tf which are
    tested separately for the same rule-set)."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha1(canonical.encode()).hexdigest()[:10]
    return h


def sample_configs(n: int = 2500, seed: int = 42) -> list[dict]:
    """Sample n distinct strategy configurations from the cartesian grid."""
    rng = random.Random(seed)
    # Cartesian product over the *rule* components (excluding symbol/tf/side).
    pool = list(itertools.product(
        TREND_FILTERS, ENTRY_TRIGGERS, CONFIRMATIONS, STOPS, TARGETS, EXITS, SIDES
    ))
    print(f"  Full grid size (rule only, no symbol/tf): {len(pool)}")
    if n > len(pool):
        n = len(pool)
    sampled = rng.sample(pool, n)

    configs = []
    seen_ids = set()
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            for (tf_f, ent, conf, stop, tgt, exit_, side) in sampled:
                cfg = {
                    "trend_filter": tf_f,
                    "entry_trigger": ent,
                    "confirmation": conf,
                    "stop": stop,
                    "target": tgt,
                    "exit": exit_,
                    "side": side,
                    "symbol": sym,
                    "timeframe": tf,
                }
                cid = config_id(cfg)
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                cfg["id"] = f"{sym}_{tf}_{side}_{cid}"
                configs.append(cfg)
    return configs


def main():
    print("=== Phase 2: Strategy generation ===")
    # Sample 625 rule+side combos; replicate across 2 symbols × 2 timeframes = 2500 total.
    configs = sample_configs(n=625, seed=42)
    OUT_PATH.write_text(json.dumps(configs, indent=2))
    print(f"  Generated {len(configs)} distinct strategy configurations")
    print(f"  Saved to {OUT_PATH}")

    # Quick distribution summary
    from collections import Counter
    by_sym = Counter(c["symbol"] for c in configs)
    by_tf = Counter(c["timeframe"] for c in configs)
    by_side = Counter(c["side"] for c in configs)
    by_trend = Counter(c["trend_filter"]["kind"] for c in configs)
    by_entry = Counter(c["entry_trigger"]["kind"] for c in configs)
    by_stop = Counter(c["stop"]["kind"] for c in configs)
    by_exit = Counter(c["exit"]["kind"] for c in configs)
    by_target = Counter(c["target"]["r_mult"] for c in configs)

    print("\n  Distribution:")
    print(f"    by symbol : {dict(by_sym)}")
    print(f"    by tf     : {dict(by_tf)}")
    print(f"    by side   : {dict(by_side)}")
    print(f"    by trend  : {dict(by_trend)}")
    print(f"    by entry  : {dict(by_entry)}")
    print(f"    by stop   : {dict(by_stop)}")
    print(f"    by exit   : {dict(by_exit)}")
    print(f"    by target : {dict(by_target)}")


if __name__ == "__main__":
    main()
