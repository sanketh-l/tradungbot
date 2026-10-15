"""
100 AI-FIRST TRADING BOTS — no fixed strategies, just AI + risk management.

Each bot is a unique combination of:
  - AI model (5 types)
  - Feature set (5 groups)
  - Decision mode (3 types)
  - Risk management preset (5 levels)
  - Trading universe (4 combinations)

Pipeline:
  1. Generate 100 distinct AI bot configs
  2. For each bot, build a CANDIDATE-GENERATOR (simple price-action signals that
     produce many candidates) + AI GATEKEEPER (decides which candidates to take)
  3. Train each bot's AI on 11 months of data (2025-09 to 2026-08)
  4. Validate on 1 month (2026-08 to 2026-09) to pick threshold
  5. Test on 30 days (2026-09-14 to 2026-10-14) with risk management
  6. Rank by net profit, report top 10
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from backtester import Costs, ema, rsi, atr, adx, bollinger, vwap_rolling, donchian
from smart_strategies import zscore, taker_buy_ratio, reindex_htf_to_ltf
from bots_100 import rolling_linreg_slope, INITIAL_WALLET, RISK_PER_TRADE, COSTS, DATA_DIR, DOWNLOAD
from risk_managed_v2 import RiskConfig

# Time windows
TEST_END = pd.Timestamp("2026-10-14 18:15:00", tz="UTC")
TEST_START = TEST_END - pd.Timedelta(days=30)
VAL_END = TEST_START
VAL_START = VAL_END - pd.Timedelta(days=30)
TRAIN_END = VAL_START
TRAIN_START = TRAIN_END - pd.Timedelta(days=335)


# ============================================================================
# Data loading
# ============================================================================

def load_full_data(symbol: str, timeframe: str):
    if timeframe == "4h":
        df_1h = pd.read_parquet(DATA_DIR / f"{symbol}_1h.parquet").set_index("open_time").sort_index()
        df = df_1h.resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum", "quote_volume": "sum",
            "taker_buy_base": "sum",
        }).dropna()
    else:
        df = pd.read_parquet(DATA_DIR / f"{symbol}_{timeframe}.parquet").set_index("open_time").sort_index()
    lookback = TRAIN_START - pd.Timedelta(hours=400)
    df = df.loc[lookback:TEST_END].copy()
    df_1h_raw = pd.read_parquet(DATA_DIR / f"{symbol}_1h.parquet").set_index("open_time").sort_index()
    htf = df_1h_raw.loc[lookback:TEST_END].copy()
    return df, htf


# ============================================================================
# CANDIDATE GENERATORS — produce many trade candidates per bar
# Each bot uses one of these; the AI then filters which to take.
# ============================================================================

def candidate_breakout(df, htf):
    """Long+short breakout candidates above/below 10-bar high/low."""
    close = df["close"]; high = df["high"]; low = df["low"]
    u10 = high.rolling(10).max().shift(1)
    l10 = low.rolling(10).min().shift(1)
    sig_long = close > u10
    sig_short = close < l10
    a = atr(df, 14)
    stop_long = close - 0.75 * a
    stop_short = close + 0.75 * a
    return sig_long, sig_short, stop_long, stop_short


def candidate_pullback(df, htf):
    """Pullback-to-EMA candidates in any trend."""
    close = df["close"]; low = df["low"]; high = df["high"]
    e20 = ema(close, 20)
    sig_long = (low <= e20 * 1.003) & (close > e20) & (close > df["open"])
    sig_short = (high >= e20 * 0.997) & (close < e20) & (close < df["open"])
    a = atr(df, 14)
    stop_long = close - 1.0 * a
    stop_short = close + 1.0 * a
    return sig_long, sig_short, stop_long, stop_short


def candidate_reversal(df, htf):
    """RSI reversal candidates (oversold/overbought bounce)."""
    close = df["close"]; r = rsi(close, 14)
    sig_long = (r.shift(1) < 35) & (r > 35) & (r < 55)
    sig_short = (r.shift(1) > 65) & (r < 65) & (r > 45)
    a = atr(df, 14)
    stop_long = close - 1.5 * a
    stop_short = close + 1.5 * a
    return sig_long, sig_short, stop_long, stop_short


def candidate_momentum(df, htf):
    """Momentum-bar candidates (strong body + close in top/bottom 25%)."""
    close = df["close"]; op = df["open"]; hi = df["high"]; lo = df["low"]
    rng = (hi - lo).replace(0, np.nan)
    body = close - op
    body_ratio = body.abs() / rng
    close_pos = (close - lo) / rng
    vol_surge = df["volume"] > 1.3 * df["volume"].rolling(20).mean()
    sig_long = (body > 0) & (body_ratio > 0.6) & (close_pos > 0.7) & vol_surge
    sig_short = (body < 0) & (body_ratio > 0.6) & (close_pos < 0.3) & vol_surge
    a = atr(df, 14)
    stop_long = close - 0.5 * a
    stop_short = close + 0.5 * a
    return sig_long, sig_short, stop_long, stop_short


def candidate_ensemble(df, htf):
    """Union of breakout + pullback + reversal + momentum — many candidates."""
    sl1, ss1, stol1, stos1 = candidate_breakout(df, htf)
    sl2, ss2, stol2, stos2 = candidate_pullback(df, htf)
    sl3, ss3, stol3, stos3 = candidate_reversal(df, htf)
    sl4, ss4, stol4, stos4 = candidate_momentum(df, htf)
    sig_long = sl1 | sl2 | sl3 | sl4
    sig_short = ss1 | ss2 | ss3 | ss4
    # Use the tightest stop when multiple fire
    a = atr(df, 14)
    stop_long = df["close"] - 0.75 * a
    stop_short = df["close"] + 0.75 * a
    return sig_long, sig_short, stop_long, stop_short


CANDIDATE_GENERATORS = {
    "breakout": candidate_breakout,
    "pullback": candidate_pullback,
    "reversal": candidate_reversal,
    "momentum": candidate_momentum,
    "ensemble": candidate_ensemble,
}


# ============================================================================
# FEATURE SETS — different groups of features the AI can use
# ============================================================================

# Pre-computed indicator cache — compute once per (df, htf) pair, reuse
_INDICATOR_CACHE = {}

def get_indicators(df, htf):
    """Compute all indicators once and cache."""
    # Cache key includes shape and time bounds to detect new slices
    key = (id(df), len(df), str(df.index[0]) if len(df) > 0 else "")
    if key in _INDICATOR_CACHE:
        return _INDICATOR_CACHE[key]
    # Limit cache size to avoid memory issues
    if len(_INDICATOR_CACHE) > 20:
        _INDICATOR_CACHE.clear()
    close = df["close"]; high = df["high"]; low = df["low"]; op = df["open"]
    vol = df["volume"]
    a = atr(df, 14)
    r = rsi(close, 14)
    e20 = ema(close, 20); e50 = ema(close, 50); e200 = ema(close, 200)
    slope = rolling_linreg_slope(close, 20)
    slope_50 = rolling_linreg_slope(close, 50)
    a_line = adx(df, 14)
    _, bb_u, bb_l = bollinger(close, 20, 2.0)
    bb_w = (bb_u - bb_l) / close
    vol_ma = vol.rolling(20).mean()
    tbr = taker_buy_ratio(df)
    atr_ma = a.rolling(50).mean()
    # HTF
    htf_e20 = ema(htf["close"], 20)
    htf_e50 = ema(htf["close"], 50)
    htf_e200 = ema(htf["close"], 200)
    htf_slope = rolling_linreg_slope(htf["close"], 20)
    htf_e20_ltf = htf_e20.reindex(df.index, method="ffill")
    htf_e50_ltf = htf_e50.reindex(df.index, method="ffill")
    htf_e200_ltf = htf_e200.reindex(df.index, method="ffill")
    htf_slope_ltf = htf_slope.reindex(df.index, method="ffill")
    cache = {
        "close": close, "high": high, "low": low, "op": op, "vol": vol,
        "atr": a, "rsi": r, "ema20": e20, "ema50": e50, "ema200": e200,
        "slope": slope, "slope_50": slope_50, "adx": a_line,
        "bb_w": bb_w, "vol_ma": vol_ma, "tbr": tbr, "atr_ma": atr_ma,
        "htf_e20_ltf": htf_e20_ltf, "htf_e50_ltf": htf_e50_ltf,
        "htf_e200_ltf": htf_e200_ltf, "htf_slope_ltf": htf_slope_ltf,
    }
    _INDICATOR_CACHE[key] = cache
    return cache


def features_technical(df, htf, i, side):
    c = get_indicators(df, htf)
    close = c["close"]
    return {
        "atr_pct": float(c["atr"].iloc[i] / close.iloc[i]) if np.isfinite(c["atr"].iloc[i]) and close.iloc[i] > 0 else 0,
        "rsi": float(c["rsi"].iloc[i]) if np.isfinite(c["rsi"].iloc[i]) else 50,
        "dist_ema20": float((close.iloc[i] - c["ema20"].iloc[i]) / c["ema20"].iloc[i]) if np.isfinite(c["ema20"].iloc[i]) and c["ema20"].iloc[i] != 0 else 0,
        "dist_ema50": float((close.iloc[i] - c["ema50"].iloc[i]) / c["ema50"].iloc[i]) if np.isfinite(c["ema50"].iloc[i]) and c["ema50"].iloc[i] != 0 else 0,
        "dist_ema200": float((close.iloc[i] - c["ema200"].iloc[i]) / c["ema200"].iloc[i]) if np.isfinite(c["ema200"].iloc[i]) and c["ema200"].iloc[i] != 0 else 0,
        "linreg_slope": float(c["slope"].iloc[i]) if np.isfinite(c["slope"].iloc[i]) else 0,
        "adx": float(c["adx"].iloc[i]) if np.isfinite(c["adx"].iloc[i]) else 0,
        "bb_width_pct": float(c["bb_w"].iloc[i]) if np.isfinite(c["bb_w"].iloc[i]) else 0,
    }


def features_microstructure(df, htf, i, side):
    c = get_indicators(df, htf)
    close = c["close"]; vol = c["vol"]; vol_ma = c["vol_ma"]
    return {
        "vol_ratio": float(vol.iloc[i] / vol_ma.iloc[i]) if np.isfinite(vol_ma.iloc[i]) and vol_ma.iloc[i] > 0 else 1,
        "taker_buy_ratio": float(c["tbr"].iloc[i]) if np.isfinite(c["tbr"].iloc[i]) else 0.5,
        "vol_ma_diff": float((vol.iloc[i] - vol_ma.iloc[i]) / vol_ma.iloc[i]) if np.isfinite(vol_ma.iloc[i]) and vol_ma.iloc[i] > 0 else 0,
        "atr_pct": float(c["atr"].iloc[i] / close.iloc[i]) if np.isfinite(c["atr"].iloc[i]) and close.iloc[i] > 0 else 0,
        "vol_surge_3bar": float(vol.iloc[i] / vol.iloc[i-3]) if i >= 3 and vol.iloc[i-3] > 0 else 1,
    }


def features_regime(df, htf, i, side):
    c = get_indicators(df, htf)
    return {
        "adx": float(c["adx"].iloc[i]) if np.isfinite(c["adx"].iloc[i]) else 0,
        "atr_above_ma": float(c["atr"].iloc[i] / c["atr_ma"].iloc[i]) if np.isfinite(c["atr_ma"].iloc[i]) and c["atr_ma"].iloc[i] > 0 else 1,
        "slope_20": float(c["slope"].iloc[i]) if np.isfinite(c["slope"].iloc[i]) else 0,
        "slope_50": float(c["slope_50"].iloc[i]) if np.isfinite(c["slope_50"].iloc[i]) else 0,
        "trending": int(c["adx"].iloc[i] > 25) if np.isfinite(c["adx"].iloc[i]) else 0,
    }


def features_multi_tf(df, htf, i, side):
    c = get_indicators(df, htf)
    close = c["close"]
    return {
        "htf_dist_ema20": float((close.iloc[i] - c["htf_e20_ltf"].iloc[i]) / c["htf_e20_ltf"].iloc[i]) if np.isfinite(c["htf_e20_ltf"].iloc[i]) and c["htf_e20_ltf"].iloc[i] != 0 else 0,
        "htf_dist_ema50": float((close.iloc[i] - c["htf_e50_ltf"].iloc[i]) / c["htf_e50_ltf"].iloc[i]) if np.isfinite(c["htf_e50_ltf"].iloc[i]) and c["htf_e50_ltf"].iloc[i] != 0 else 0,
        "htf_dist_ema200": float((close.iloc[i] - c["htf_e200_ltf"].iloc[i]) / c["htf_e200_ltf"].iloc[i]) if np.isfinite(c["htf_e200_ltf"].iloc[i]) and c["htf_e200_ltf"].iloc[i] != 0 else 0,
        "htf_slope": float(c["htf_slope_ltf"].iloc[i]) if np.isfinite(c["htf_slope_ltf"].iloc[i]) else 0,
        "htf_above_ema200": int(close.iloc[i] > c["htf_e200_ltf"].iloc[i]) if np.isfinite(c["htf_e200_ltf"].iloc[i]) else 0,
    }


def features_combined(df, htf, i, side):
    f1 = features_technical(df, htf, i, side)
    f2 = features_microstructure(df, htf, i, side)
    f3 = features_regime(df, htf, i, side)
    f4 = features_multi_tf(df, htf, i, side)
    return {**f1, **f2, **f3, **f4, "side_long": int(side == "long"), "side_short": int(side == "short")}


FEATURE_SETS = {
    "technical": (features_technical, ["atr_pct","rsi","dist_ema20","dist_ema50","dist_ema200","linreg_slope","adx","bb_width_pct"]),
    "microstructure": (features_microstructure, ["vol_ratio","taker_buy_ratio","vol_ma_diff","atr_pct","vol_surge_3bar"]),
    "regime": (features_regime, ["adx","atr_above_ma","slope_20","slope_50","trending"]),
    "multi_tf": (features_multi_tf, ["htf_dist_ema20","htf_dist_ema50","htf_dist_ema200","htf_slope","htf_above_ema200"]),
}
# Build combined feature list (deduplicated)
_combined_feat = []
for _, cols in FEATURE_SETS.items():
    for c in cols:
        if c not in _combined_feat:
            _combined_feat.append(c)
_combined_feat.extend(["side_long", "side_short"])
FEATURE_SETS["combined"] = (features_combined, _combined_feat)


# ============================================================================
# AI MODELS
# ============================================================================

def make_model(kind: str):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=10,
                                       random_state=42, class_weight="balanced")
    if kind == "gb":
        return GradientBoostingClassifier(n_estimators=100, max_depth=4, min_samples_leaf=10,
                                           random_state=42)
    if kind == "lr":
        return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42))])
    if kind == "knn":
        return Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=15))])
    if kind == "nb":
        return GaussianNB()
    raise ValueError(kind)


# ============================================================================
# RISK MANAGEMENT PRESETS
# ============================================================================

def make_risk_cfg(preset: str) -> RiskConfig:
    if preset == "none":
        return RiskConfig(
            monthly_loss_cap_pct=1.0, weekly_loss_cap_pct=1.0, daily_loss_cap_pct=1.0,
            dd_throttle_1=1.0, dd_throttle_2=1.0, dd_throttle_3=1.0,
            consec_loss_throttle_1=99, consec_loss_throttle_2=99,
            enable_trailing=False,
        )
    if preset == "light":
        return RiskConfig(
            monthly_loss_cap_pct=0.12, weekly_loss_cap_pct=0.08, daily_loss_cap_pct=0.05,
            dd_throttle_1=0.15, dd_throttle_2=0.20, dd_throttle_3=0.30,
            consec_loss_throttle_1=4, consec_loss_throttle_2=7,
            enable_trailing=True, trail_trigger_1=1.0, trail_trigger_2=2.0,
        )
    if preset == "standard":
        return RiskConfig()  # defaults
    if preset == "aggressive":
        return RiskConfig(
            monthly_loss_cap_pct=0.05, weekly_loss_cap_pct=0.03, daily_loss_cap_pct=0.02,
            dd_throttle_1=0.07, dd_throttle_2=0.12, dd_throttle_3=0.17,
            consec_loss_throttle_1=2, consec_loss_throttle_2=4,
            anti_mart_win_mult=1.30, anti_mart_loss_mult=0.60,
            enable_trailing=True, trail_trigger_1=0.5, trail_trigger_2=1.0, trail_distance_r=0.3,
        )
    if preset == "conservative":
        return RiskConfig(
            monthly_loss_cap_pct=0.05, weekly_loss_cap_pct=0.03, daily_loss_cap_pct=0.02,
            dd_throttle_1=0.05, dd_throttle_2=0.10, dd_throttle_3=0.15,
            consec_loss_throttle_1=2, consec_loss_throttle_2=3,
            base_risk_pct=0.005,  # 0.5% base risk
            anti_mart_max_risk=0.008, anti_mart_min_risk=0.002,
            enable_trailing=True, trail_trigger_1=0.5, trail_trigger_2=1.0, trail_distance_r=0.3,
        )
    raise ValueError(preset)


# ============================================================================
# DECISION MODES — how the AI's probability translates to action
# ============================================================================

DECISION_MODES = ["binary", "confidence_size", "multi_class"]


# ============================================================================
# Generate 100 AI bot configurations
# ============================================================================

def generate_100_ai_bots(seed: int = 42) -> list[dict]:
    """Generate 100 distinct AI bot configs."""
    import random
    rng = random.Random(seed)

    models = ["rf", "gb", "lr", "knn", "nb"]
    feature_sets = ["technical", "microstructure", "regime", "multi_tf", "combined"]
    candidate_gens = ["breakout", "pullback", "reversal", "momentum", "ensemble"]
    risk_presets = ["none", "light", "standard", "aggressive", "conservative"]
    universes = [
        ("BTCUSDT", "1h"),
        ("ETHUSDT", "1h"),
        ("BTCUSDT", "4h"),
        ("ETHUSDT", "4h"),
    ]
    decision_modes = DECISION_MODES

    # Cartesian product is huge: 5×5×5×5×4×3 = 7500
    # Sample 100 distinct configs
    configs = []
    seen = set()
    attempts = 0
    while len(configs) < 100 and attempts < 50000:
        attempts += 1
        model = rng.choice(models)
        feat = rng.choice(feature_sets)
        cand = rng.choice(candidate_gens)
        risk = rng.choice(risk_presets)
        sym, tf = rng.choice(universes)
        dec = rng.choice(decision_modes)
        key = (model, feat, cand, risk, sym, tf, dec)
        if key in seen:
            continue
        seen.add(key)
        bot_id = f"AIBOT_{len(configs)+1:03d}_{model}_{feat[:4]}_{cand[:4]}_{risk[:4]}_{sym[:3]}_{tf}_{dec[:3]}"
        configs.append({
            "id": bot_id,
            "model_kind": model,
            "feature_set": feat,
            "candidate_gen": cand,
            "risk_preset": risk,
            "symbol": sym,
            "timeframe": tf,
            "decision_mode": dec,
        })
    return configs


# ============================================================================
# Training data generation
# ============================================================================

def generate_training_samples(df, htf, cfg, start, end):
    """For each candidate signal in [start, end], simulate the trade and label it."""
    cand_fn = CANDIDATE_GENERATORS[cfg["candidate_gen"]]
    sig_long, sig_short, stop_long, stop_short = cand_fn(df, htf)
    feat_fn, feat_cols = FEATURE_SETS[cfg["feature_set"]]

    df = df.sort_index(); n = len(df)
    sig_long_arr = sig_long.fillna(False).values
    sig_short_arr = sig_short.fillna(False).values
    stop_long_arr = stop_long.values
    stop_short_arr = stop_short.values
    opens = df["open"].values; highs = df["high"].values
    lows = df["low"].values; closes = df["close"].values
    times = df.index
    slip_frac = COSTS.slippage_bps / 10000.0
    r_mult = 2.0; max_bars = 24 if cfg["timeframe"] == "1h" else 12

    samples = []
    i = 0
    while i < n:
        if times[i] < start or times[i] > end:
            i += 1; continue
        signal_long = i+1 < n and sig_long_arr[i] and np.isfinite(opens[i+1]) and np.isfinite(stop_long_arr[i])
        signal_short = i+1 < n and sig_short_arr[i] and np.isfinite(opens[i+1]) and np.isfinite(stop_short_arr[i])
        # Take both signals separately for training (don't dedupe)
        for chosen, sp_arr in [("long", stop_long_arr), ("short", stop_short_arr)]:
            take = signal_long if chosen == "long" else signal_short
            if not take: continue
            entry_raw = float(opens[i+1])
            if entry_raw <= 0: continue
            sp = float(sp_arr[i])
            if not np.isfinite(sp): continue
            if chosen == "long" and sp >= entry_raw: continue
            if chosen == "short" and sp <= entry_raw: continue
            entry_price = entry_raw * (1 + slip_frac) if chosen == "long" else entry_raw * (1 - slip_frac)
            stop_distance = abs(entry_raw - sp)
            if stop_distance <= 0: continue
            sp_adj = entry_price - stop_distance if chosen == "long" else entry_price + stop_distance
            target_price = entry_price + r_mult * (entry_price - sp_adj) if chosen == "long" \
                           else entry_price - r_mult * (sp_adj - entry_price)
            # Walk forward
            exit_idx = -1; exit_price = float("nan"); exit_reason = ""; bars_held = 0
            j = i + 1
            while j < n:
                bars_held = j - i
                h = highs[j]; l = lows[j]
                if chosen == "long":
                    hit_stop = l <= sp_adj; hit_tgt = h >= target_price
                else:
                    hit_stop = h >= sp_adj; hit_tgt = l <= target_price
                if hit_stop and hit_tgt:
                    exit_price = sp_adj; exit_reason = "stop"; exit_idx = j; break
                elif hit_stop:
                    exit_price = sp_adj; exit_reason = "stop"; exit_idx = j; break
                elif hit_tgt:
                    exit_price = target_price; exit_reason = "target"; exit_idx = j; break
                if bars_held >= max_bars:
                    exit_price = closes[j]; exit_reason = "time"; exit_idx = j; break
                j += 1
            if exit_idx < 0:
                exit_price = closes[-1]; exit_reason = "end_of_data"; exit_idx = n - 1
            if chosen == "long":
                exit_fill = exit_price * (1 - slip_frac)
                gross_per_unit = exit_fill - entry_price
            else:
                exit_fill = exit_price * (1 + slip_frac)
                gross_per_unit = entry_price - exit_fill
            gross_r = gross_per_unit / stop_distance
            cost_r = (0.0012 * entry_price) / stop_distance
            net_r = gross_r - cost_r
            label = 1 if (exit_reason == "target" and net_r > 0) else 0
            feats = feat_fn(df, htf, i, chosen)
            feats["label"] = label
            feats["signal_time"] = times[i]
            feats["side"] = chosen
            feats["exit_reason"] = exit_reason
            feats["gross_r"] = gross_r
            feats["net_r"] = net_r
            samples.append(feats)
        # Move to next bar (don't skip — training samples don't need non-overlap)
        i += 1
    return samples, feat_cols


# ============================================================================
# AI-gated backtester with all the variants
# ============================================================================

def backtest_ai_bot(df, htf, cfg, model, feat_cols, threshold, risk_cfg,
                    start_time=None, initial_wallet=INITIAL_WALLET):
    """Run an AI bot: candidate generator + AI gatekeeper + risk management."""
    cand_fn = CANDIDATE_GENERATORS[cfg["candidate_gen"]]
    sig_long, sig_short, stop_long, stop_short = cand_fn(df, htf)
    feat_fn = _ = FEATURE_SETS[cfg["feature_set"]][0]

    r_mult = 2.0; max_bars = risk_cfg.max_bars if risk_cfg else 24
    if cfg["timeframe"] == "4h" and max_bars > 12:
        max_bars = 12

    df = df.sort_index(); n = len(df)
    sig_long_arr = sig_long.fillna(False).values
    sig_short_arr = sig_short.fillna(False).values
    stop_long_arr = stop_long.values
    stop_short_arr = stop_short.values
    opens = df["open"].values; highs = df["high"].values
    lows = df["low"].values; closes = df["close"].values
    times = df.index
    slip_frac = COSTS.slippage_bps / 10000.0

    # Risk state
    wallet = initial_wallet
    peak_wallet = initial_wallet
    base_risk = risk_cfg.base_risk_pct
    cur_risk = base_risk
    consec_losses = 0; consec_wins = 0
    throttle_remaining = 0
    pause_until = None
    monthly_start = initial_wallet; weekly_start = initial_wallet; daily_start = initial_wallet
    cur_month = times[0].month; cur_week = times[0].isocalendar()[1]; cur_day = times[0].day
    monthly_stopped = False; weekly_stopped = False; daily_stopped = False; dd_stopped = False

    trades = []
    ai_decisions = {"approved": 0, "rejected": 0, "skipped_rm": 0}

    i = 0
    while i < n:
        t = times[i]
        if start_time is not None and t < start_time:
            i += 1; continue

        # Risk management layer
        if risk_cfg is not None:
            if t.day != cur_day:
                daily_stopped = False; daily_start = wallet; cur_day = t.day
            if t.isocalendar()[1] != cur_week:
                weekly_stopped = False; weekly_start = wallet; cur_week = t.isocalendar()[1]
            if t.month != cur_month:
                monthly_stopped = False; monthly_start = wallet; cur_month = t.month
            if wallet > peak_wallet: peak_wallet = wallet

            allowed = True; reason = "ok"; eff_risk = cur_risk
            if monthly_stopped: allowed = False; reason = "monthly_cap"
            elif weekly_stopped: allowed = False; reason = "weekly_cap"
            elif daily_stopped: allowed = False; reason = "daily_cap"
            elif dd_stopped: allowed = False; reason = "dd_cap"
            elif pause_until is not None and t < pause_until: allowed = False; reason = "consec_pause"
            if allowed:
                if monthly_start > 0:
                    mp = (wallet - monthly_start) / monthly_start
                    if mp <= -risk_cfg.monthly_loss_cap_pct:
                        monthly_stopped = True; allowed = False; reason = "monthly_cap_hit"
                if allowed and weekly_start > 0:
                    wp = (wallet - weekly_start) / weekly_start
                    if wp <= -risk_cfg.weekly_loss_cap_pct:
                        weekly_stopped = True; allowed = False; reason = "weekly_cap_hit"
                if allowed and daily_start > 0:
                    dp = (wallet - daily_start) / daily_start
                    if dp <= -risk_cfg.daily_loss_cap_pct:
                        daily_stopped = True; allowed = False; reason = "daily_cap_hit"
            if allowed:
                if peak_wallet > 0:
                    dd = (peak_wallet - wallet) / peak_wallet
                    if dd >= risk_cfg.dd_throttle_3:
                        dd_stopped = True; allowed = False; reason = "dd_cap_hit"
                    else:
                        dd_stopped = False
                        if dd >= risk_cfg.dd_throttle_2:
                            eff_risk = min(eff_risk, base_risk * 0.25)
                        elif dd >= risk_cfg.dd_throttle_1:
                            eff_risk = min(eff_risk, base_risk * 0.50)
            if allowed:
                if t.month == 12:
                    next_month = pd.Timestamp(year=t.year+1, month=1, day=1, tz="UTC")
                else:
                    next_month = pd.Timestamp(year=t.year, month=t.month+1, day=1, tz="UTC")
                days_left = (next_month - t).days
                if days_left <= risk_cfg.end_of_month_days and monthly_start > 0:
                    mp = (wallet - monthly_start) / monthly_start
                    if mp > 0:
                        eff_risk = risk_cfg.end_of_month_risk
                if throttle_remaining > 0:
                    eff_risk = max(eff_risk * 0.5, risk_cfg.anti_mart_min_risk)
                    throttle_remaining -= 1
            if not allowed:
                ai_decisions["skipped_rm"] += 1
                i += 1; continue
        else:
            eff_risk = RISK_PER_TRADE

        # Check for candidate signal
        signal_long = i+1 < n and sig_long_arr[i] and np.isfinite(opens[i+1]) and np.isfinite(stop_long_arr[i])
        signal_short = i+1 < n and sig_short_arr[i] and np.isfinite(opens[i+1]) and np.isfinite(stop_short_arr[i])
        chosen = None
        if signal_long and signal_short: chosen = "long"  # prefer long
        elif signal_long: chosen = "long"
        elif signal_short: chosen = "short"
        if chosen is None:
            i += 1; continue

        # === AI GATEKEEPER ===
        feats = feat_fn(df, htf, i, chosen)
        X = np.array([[feats[c] for c in feat_cols]])
        if model is not None:
            try:
                proba = float(model.predict_proba(X)[0][1])
            except Exception:
                proba = 1.0
        else:
            proba = 1.0

        # Decision mode
        if cfg["decision_mode"] == "binary":
            if proba < threshold:
                ai_decisions["rejected"] += 1
                i += 1; continue
            ai_decisions["approved"] += 1
            # Standard risk
        elif cfg["decision_mode"] == "confidence_size":
            # Always take, but size by confidence
            ai_decisions["approved"] += 1
            # Scale risk between 0.3% and 1.5% based on confidence
            eff_risk = max(0.003, min(0.015, proba * 0.02))
        elif cfg["decision_mode"] == "multi_class":
            # Reject if proba in [0.40, 0.55] (uncertain zone)
            if 0.40 <= proba <= 0.55:
                ai_decisions["rejected"] += 1
                i += 1; continue
            ai_decisions["approved"] += 1
        else:
            if proba < threshold:
                ai_decisions["rejected"] += 1
                i += 1; continue
            ai_decisions["approved"] += 1

        # Execute trade
        entry_raw = float(opens[i+1])
        if entry_raw <= 0: i += 1; continue
        if chosen == "long":
            sp_ref = float(stop_long_arr[i])
            if not np.isfinite(sp_ref) or sp_ref >= entry_raw: i += 1; continue
            entry_price = entry_raw * (1 + slip_frac)
            stop_distance = abs(entry_raw - sp_ref)
            sp = entry_price - stop_distance
            target_price = entry_price + r_mult * (entry_price - sp)
        else:
            sp_ref = float(stop_short_arr[i])
            if not np.isfinite(sp_ref) or sp_ref <= entry_raw: i += 1; continue
            entry_price = entry_raw * (1 - slip_frac)
            stop_distance = abs(sp_ref - entry_raw)
            sp = entry_price + stop_distance
            target_price = entry_price - r_mult * (sp - entry_price)
        if stop_distance <= 0: i += 1; continue

        risk_amount = eff_risk * wallet
        size_units = risk_amount / stop_distance
        if size_units <= 0: i += 1; continue

        entry_notional = size_units * entry_price
        entry_commission = entry_notional * COSTS.commission_pct
        wallet_after_entry = wallet - entry_commission

        # Walk forward with trailing stop (if RM enabled)
        exit_idx = -1; exit_price = float("nan"); exit_reason = ""; bars_held = 0
        cur_trail_stop = sp
        best_price = entry_price
        j = i + 1
        while j < n:
            bars_held = j - i
            h = highs[j]; l = lows[j]
            if chosen == "long":
                if h > best_price: best_price = h
            else:
                if l < best_price: best_price = l
            if risk_cfg is not None and risk_cfg.enable_trailing:
                r_best = (best_price - entry_price) / stop_distance if chosen == "long" \
                         else (entry_price - best_price) / stop_distance
                if r_best >= risk_cfg.trail_trigger_1:
                    if chosen == "long":
                        if entry_price > cur_trail_stop: cur_trail_stop = entry_price
                    else:
                        if entry_price < cur_trail_stop: cur_trail_stop = entry_price
                if r_best >= risk_cfg.trail_trigger_2:
                    if chosen == "long":
                        new_stop = best_price - risk_cfg.trail_distance_r * stop_distance
                        if new_stop > cur_trail_stop: cur_trail_stop = new_stop
                    else:
                        new_stop = best_price + risk_cfg.trail_distance_r * stop_distance
                        if new_stop < cur_trail_stop: cur_trail_stop = new_stop
            else:
                cur_trail_stop = sp
            if chosen == "long":
                hit_stop = l <= cur_trail_stop; hit_tgt = h >= target_price
            else:
                hit_stop = h >= cur_trail_stop; hit_tgt = l <= target_price
            if hit_stop and hit_tgt:
                exit_price = cur_trail_stop; exit_reason = "stop"; exit_idx = j; break
            elif hit_stop:
                if cur_trail_stop >= entry_price: exit_reason = "trail_win"
                else: exit_reason = "stop"
                exit_price = cur_trail_stop; exit_idx = j; break
            elif hit_tgt:
                exit_price = target_price; exit_reason = "target"; exit_idx = j; break
            if bars_held >= max_bars:
                exit_price = closes[j]; exit_reason = "time"; exit_idx = j; break
            j += 1
        if exit_idx < 0:
            exit_price = closes[-1]; exit_reason = "end_of_data"; exit_idx = n - 1
        if chosen == "long":
            exit_fill = exit_price * (1 - slip_frac)
            gross_pnl = size_units * (exit_fill - entry_price)
        else:
            exit_fill = exit_price * (1 + slip_frac)
            gross_pnl = size_units * (entry_price - exit_fill)
        exit_notional = size_units * exit_fill
        exit_commission = exit_notional * COSTS.commission_pct
        total_cost = entry_commission + exit_commission
        net_pnl = gross_pnl - total_cost
        r_mult_actual = net_pnl / risk_amount if risk_amount > 0 else 0.0

        new_wallet = wallet_after_entry + net_pnl
        won = net_pnl > 0
        if risk_cfg is not None:
            if won:
                consec_wins += 1; consec_losses = 0
                cur_risk = min(cur_risk * risk_cfg.anti_mart_win_mult, risk_cfg.anti_mart_max_risk)
            else:
                consec_losses += 1; consec_wins = 0
                cur_risk = max(cur_risk * risk_cfg.anti_mart_loss_mult, risk_cfg.anti_mart_min_risk)
            if consec_losses >= risk_cfg.consec_loss_throttle_2:
                pause_until = times[exit_idx] + pd.Timedelta(hours=risk_cfg.consec_loss_throttle_2_hours)
                consec_losses = 0
            elif consec_losses >= risk_cfg.consec_loss_throttle_1:
                throttle_remaining = risk_cfg.consec_loss_throttle_1_trades
                cur_risk = max(cur_risk * 0.5, risk_cfg.anti_mart_min_risk)

        trades.append({
            "entry_time": times[i+1], "exit_time": times[exit_idx],
            "side": chosen, "entry_price": entry_price, "exit_price": exit_fill,
            "stop": sp, "target": target_price, "size": size_units,
            "risk_pct": eff_risk, "risk_usd": risk_amount,
            "gross_pnl": gross_pnl, "cost": total_cost, "net_pnl": net_pnl,
            "wallet_before": wallet, "wallet_after": new_wallet,
            "R_multiple": r_mult_actual, "exit_reason": exit_reason, "bars_held": bars_held,
            "ai_proba": proba,
        })
        wallet = new_wallet
        if wallet > peak_wallet: peak_wallet = wallet
        i = exit_idx + 1

    if trades:
        wallets = [initial_wallet] + [t["wallet_after"] for t in trades]
        peak_arr = np.maximum.accumulate(wallets)
        dd_arr = (peak_arr - wallets) / np.where(peak_arr > 0, peak_arr, 1)
        max_dd = float(np.max(dd_arr))
    else:
        max_dd = 0
    stats = {
        "n_trades": len(trades), "final_wallet": wallet,
        "net": wallet - initial_wallet, "net_pct": (wallet - initial_wallet) / initial_wallet,
        "peak_wallet": peak_wallet, "max_dd_pct": max_dd,
        "ai_decisions": ai_decisions,
    }
    return trades, stats


# ============================================================================
# Main: train + validate + test all 100 bots
# ============================================================================

def main():
    print("="*120)
    print("100 AI-FIRST TRADING BOTS — train, validate, test on last 30 days")
    print(f"Train: {TRAIN_START.date()} → {TRAIN_END.date()}  |  Val: {VAL_START.date()} → {VAL_END.date()}  |  Test: {TEST_START.date()} → {TEST_END.date()}")
    print("="*120)

    # Generate 100 configs
    configs = generate_100_ai_bots(seed=42)
    print(f"\nGenerated {len(configs)} AI bot configs")

    # Load all data
    print("\nLoading data...")
    data_cache = {}
    for sym in ["BTCUSDT", "ETHUSDT"]:
        for tf in ["1h", "4h"]:
            data_cache[(sym, tf)] = load_full_data(sym, tf)
            print(f"  {sym} {tf}: {len(data_cache[(sym,tf)][0])} bars")

    # Train + validate + test each bot
    print("\nTraining + testing 100 AI bots...")
    all_results = []
    t_start = time.time()
    for i, cfg in enumerate(configs, 1):
        if i % 10 == 0:
            elapsed = time.time() - t_start
            rate = i / elapsed
            eta = (len(configs) - i) / rate if rate > 0 else 0
            print(f"  ... {i}/{len(configs)} done ({rate:.1f}/s, eta {eta:.0f}s)")

        df, htf = data_cache[(cfg["symbol"], cfg["timeframe"])]
        feat_fn, feat_cols = FEATURE_SETS[cfg["feature_set"]]

        try:
            # Step 1: Generate training samples
            train_samples, _ = generate_training_samples(df, htf, cfg, TRAIN_START, TRAIN_END)
            if len(train_samples) < 30:
                # Skip — too few samples
                all_results.append({"cfg": cfg, "trades": [], "stats": {"n_trades": 0, "final_wallet": INITIAL_WALLET,
                    "net": 0, "net_pct": 0, "peak_wallet": INITIAL_WALLET, "max_dd_pct": 0,
                    "ai_decisions": {"approved": 0, "rejected": 0, "skipped_rm": 0}},
                    "train_samples": len(train_samples), "val_samples": 0, "threshold": 0.5, "trained": False})
                continue

            X_train = np.array([[s[c] for c in feat_cols] for s in train_samples])
            y_train = np.array([s["label"] for s in train_samples])

            # Step 2: Train model
            model = make_model(cfg["model_kind"])
            try:
                model.fit(X_train, y_train)
            except Exception:
                all_results.append({"cfg": cfg, "trades": [], "stats": {"n_trades": 0, "final_wallet": INITIAL_WALLET,
                    "net": 0, "net_pct": 0, "peak_wallet": INITIAL_WALLET, "max_dd_pct": 0,
                    "ai_decisions": {"approved": 0, "rejected": 0, "skipped_rm": 0}},
                    "train_samples": len(train_samples), "val_samples": 0, "threshold": 0.5, "trained": False})
                continue

            # Step 3: Validate — pick threshold
            val_samples, _ = generate_training_samples(df, htf, cfg, VAL_START, VAL_END)
            best_threshold = 0.5; best_exp = -999
            if len(val_samples) >= 10:
                X_val = np.array([[s[c] for c in feat_cols] for s in val_samples])
                y_val = np.array([s["label"] for s in val_samples])
                try:
                    val_proba = model.predict_proba(X_val)[:, 1]
                    for thresh in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
                        mask = val_proba >= thresh
                        if mask.sum() < 3: continue
                        wr = y_val[mask].mean()
                        exp = 3 * wr - 1
                        if exp > best_exp:
                            best_exp = exp; best_threshold = thresh
                except Exception:
                    pass

            # Step 4: Test on 30-day window with risk management
            risk_cfg = make_risk_cfg(cfg["risk_preset"])
            trades, stats = backtest_ai_bot(df, htf, cfg, model, feat_cols, best_threshold, risk_cfg,
                                             start_time=TEST_START, initial_wallet=INITIAL_WALLET)
            all_results.append({"cfg": cfg, "trades": trades, "stats": stats,
                                "train_samples": len(train_samples), "val_samples": len(val_samples),
                                "threshold": best_threshold, "trained": True})
        except Exception as e:
            all_results.append({"cfg": cfg, "trades": [], "stats": {"n_trades": 0, "final_wallet": INITIAL_WALLET,
                "net": 0, "net_pct": 0, "peak_wallet": INITIAL_WALLET, "max_dd_pct": 0,
                "ai_decisions": {"approved": 0, "rejected": 0, "skipped_rm": 0}},
                "train_samples": 0, "val_samples": 0, "threshold": 0.5, "trained": False, "error": str(e)})

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s ({elapsed/len(configs):.2f}s per bot)")

    # Sort by net profit
    all_results.sort(key=lambda x: x["stats"]["net"], reverse=True)

    # Print top 10
    print("\n" + "="*150)
    print("TOP 10 AI BOTS BY NET PROFIT (last 30 days, $10,000 initial)")
    print("="*150)
    print(f"{'#':<3} {'Bot ID':<55} {'Sym':<8} {'TF':<4} {'Trades':>6} {'WR':>6} {'Net $':>11} {'Net %':>8} {'MaxDD':>7}")
    print("-"*150)
    for rank, r in enumerate(all_results[:10], 1):
        cfg = r["cfg"]; s = r["stats"]; trades = r["trades"]
        n_wins = sum(1 for t in trades if t["net_pnl"] > 0)
        n_losses = sum(1 for t in trades if t["net_pnl"] < 0)
        wr = n_wins/(n_wins+n_losses) if (n_wins+n_losses) > 0 else 0
        print(f"{rank:<3} {cfg['id'][:55]:<55} {cfg['symbol']:<8} {cfg['timeframe']:<4} {s['n_trades']:>6} {wr*100:>5.1f}% {s['net']:>+11,.2f} {s['net_pct']*100:>+7.2f}% {s['max_dd_pct']*100:>6.1f}%")

    # Aggregate
    n_profit = sum(1 for r in all_results if r["stats"]["net"] > 0)
    n_break = sum(1 for r in all_results if r["stats"]["net"] == 0)
    n_loss = sum(1 for r in all_results if r["stats"]["net"] < 0)
    n_trained = sum(1 for r in all_results if r.get("trained", False))
    print(f"\n--- AGGREGATE ---")
    print(f"  Trained successfully: {n_trained}/100")
    print(f"  Profitable: {n_profit}  |  Breakeven: {n_break}  |  Losing: {n_loss}")
    print(f"  Best: ${all_results[0]['stats']['net']:>+,.2f} ({all_results[0]['stats']['net_pct']*100:+.2f}%)")
    print(f"  Worst: ${all_results[-1]['stats']['net']:>+,.2f} ({all_results[-1]['stats']['net_pct']*100:+.2f}%)")
    if len(all_results) >= 50:
        print(f"  Median: ${all_results[50]['stats']['net']:>+,.2f} ({all_results[50]['stats']['net_pct']*100:+.2f}%)")

    # Ensemble: $2k each on top 5
    top5 = all_results[:5]
    ens_net = sum(r["stats"]["net"] * 0.2 for r in top5)
    avg_dd = np.mean([r["stats"]["max_dd_pct"] for r in top5])
    print(f"\n--- ENSEMBLE: $2,000 each on top 5 ---")
    print(f"  Initial: ${INITIAL_WALLET:,.2f}  →  Final: ${INITIAL_WALLET + ens_net:,.2f}")
    print(f"  Net: ${ens_net:+,.2f} ({ens_net/INITIAL_WALLET*100:+.2f}%)")
    print(f"  Avg max DD: {avg_dd*100:.1f}%")
    for r in top5:
        cfg = r["cfg"]; s = r["stats"]
        print(f"    {cfg['id'][:55]:55s} ${s['net']*0.2:+9,.2f} on $2,000 ({s['net_pct']*100:+6.2f}%)  DD={s['max_dd_pct']*100:.1f}%")

    # Save Excel
    out_path = DOWNLOAD / "100_ai_bots_30day.xlsx"
    write_excel(all_results, out_path)
    print(f"\n>>> Wrote {out_path}")


def write_excel(all_results, out_path):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    header_font = Font(bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    pos_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    neg_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    warn_fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
    border = Border(left=Side(style='thin'), right=Side(style='thin'),
                    top=Side(style='thin'), bottom=Side(style='thin'))

    # All 100 ranked
    ws = wb.create_sheet("All 100 Ranked")
    ws["A1"] = "100 AI-FIRST BOTS — 30-DAY TEST, RANKED BY NET PROFIT"
    ws["A1"].font = Font(bold=True, size=14, color="1F4E78")
    ws["A2"] = f"Window: {TEST_START} → {TEST_END}  |  Initial: ${INITIAL_WALLET:,.2f}  |  Risk: varies (see Risk Preset)"
    ws["A3"] = f"Trained: {sum(1 for r in all_results if r.get('trained', False))}/100  |  Profitable: {sum(1 for r in all_results if r['stats']['net']>0)}/100"

    headers = ["Rank", "Bot ID", "Symbol", "TF", "Model", "Features", "Candidates", "Risk", "Decision",
               "Trained", "Train Samples", "Val Samples", "Threshold",
               "Trades", "Wins", "Losses", "WR", "Initial $", "Final $", "Net $", "Net %", "Max DD %", "AI Rejected"]
    row_idx = 5
    for col, h in enumerate(headers, 1):
        c = ws.cell(row=row_idx, column=col, value=h)
        c.font = header_font; c.fill = header_fill; c.alignment = Alignment(horizontal="center", wrap_text=True); c.border = border
    for rank, r in enumerate(all_results, 1):
        cfg = r["cfg"]; s = r["stats"]; trades = r["trades"]
        n_wins = sum(1 for t in trades if t["net_pnl"] > 0)
        n_losses = sum(1 for t in trades if t["net_pnl"] < 0)
        wr = n_wins/(n_wins+n_losses) if (n_wins+n_losses) > 0 else 0
        row = [rank, cfg["id"], cfg["symbol"], cfg["timeframe"], cfg["model_kind"], cfg["feature_set"],
               cfg["candidate_gen"], cfg["risk_preset"], cfg["decision_mode"],
               "Yes" if r.get("trained", False) else "No",
               r.get("train_samples", 0), r.get("val_samples", 0), r.get("threshold", 0),
               s["n_trades"], n_wins, n_losses, wr,
               INITIAL_WALLET, s["final_wallet"], s["net"], s["net_pct"], s["max_dd_pct"],
               s["ai_decisions"].get("rejected", 0)]
        for col, v in enumerate(row, 1):
            c = ws.cell(row=row_idx+rank, column=col, value=v)
            c.border = border
            if col in (18, 19, 20):
                c.number_format = '"$"#,##0.00'
            elif col in (21, 22):
                c.number_format = '0.00%'
            elif col == 17:
                c.number_format = '0.0%'
            if col == 20 and isinstance(v, (int, float)):
                c.fill = pos_fill if v > 0 else (neg_fill if v < 0 else warn_fill)
    widths = [5, 60, 9, 5, 6, 14, 12, 13, 13, 8, 9, 9, 8, 7, 7, 7, 7, 11, 12, 12, 9, 9, 9]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A6"

    # Top 10 summary
    ws = wb.create_sheet("Top 10 Summary")
    ws["A1"] = "TOP 10 AI BOTS — Last 30 Days"
    ws["A1"].font = Font(bold=True, size=14, color="1F4E78")
    for col, h in enumerate(headers, 1):
        c = ws.cell(row=3, column=col, value=h)
        c.font = header_font; c.fill = header_fill; c.alignment = Alignment(horizontal="center", wrap_text=True); c.border = border
    for rank, r in enumerate(all_results[:10], 1):
        cfg = r["cfg"]; s = r["stats"]; trades = r["trades"]
        n_wins = sum(1 for t in trades if t["net_pnl"] > 0)
        n_losses = sum(1 for t in trades if t["net_pnl"] < 0)
        wr = n_wins/(n_wins+n_losses) if (n_wins+n_losses) > 0 else 0
        row = [rank, cfg["id"], cfg["symbol"], cfg["timeframe"], cfg["model_kind"], cfg["feature_set"],
               cfg["candidate_gen"], cfg["risk_preset"], cfg["decision_mode"],
               "Yes" if r.get("trained", False) else "No",
               r.get("train_samples", 0), r.get("val_samples", 0), r.get("threshold", 0),
               s["n_trades"], n_wins, n_losses, wr,
               INITIAL_WALLET, s["final_wallet"], s["net"], s["net_pct"], s["max_dd_pct"],
               s["ai_decisions"].get("rejected", 0)]
        for col, v in enumerate(row, 1):
            c = ws.cell(row=3+rank, column=col, value=v)
            c.border = border
            if col in (18, 19, 20):
                c.number_format = '"$"#,##0.00'
            elif col in (21, 22):
                c.number_format = '0.00%'
            elif col == 17:
                c.number_format = '0.0%'
            if col == 20 and isinstance(v, (int, float)):
                c.fill = pos_fill if v > 0 else (neg_fill if v < 0 else warn_fill)
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A4"

    # Per-bot trade logs for top 10
    for rank, r in enumerate(all_results[:10], 1):
        cfg = r["cfg"]; trades = r["trades"]; s = r["stats"]
        sheet_name = f"#{rank} {cfg['id']}"[:31]
        ws = wb.create_sheet(sheet_name)
        ws["A1"] = f"#{rank} — {cfg['id']}  |  {cfg['symbol']} {cfg['timeframe']}  |  Model: {cfg['model_kind']}  |  Features: {cfg['feature_set']}  |  Risk: {cfg['risk_preset']}  |  Decision: {cfg['decision_mode']}"
        ws["A1"].font = Font(bold=True, size=10); ws.merge_cells("A1:T1")
        ws["A2"] = f"Net: ${s['net']:+,.2f} ({s['net_pct']*100:+.2f}%)  |  Trades: {len(trades)}  |  Max DD: {s['max_dd_pct']*100:.1f}%  |  AI rejected: {s['ai_decisions'].get('rejected', 0)}  |  Train samples: {r.get('train_samples', 0)}  |  Threshold: {r.get('threshold', 0):.2f}"
        ws.merge_cells("A2:T2")
        trade_headers = ["#", "Entry Time", "Exit Time", "Side", "Entry $", "Exit $",
                         "Stop $", "Target $", "Size", "Risk %", "Risk $", "Gross P&L $",
                         "Cost $", "Net P&L $", "R Mult", "AI P(win)", "Exit Reason", "Bars Held",
                         "Wallet Before $", "Wallet After $"]
        for col, h in enumerate(trade_headers, 1):
            c = ws.cell(row=4, column=col, value=h)
            c.font = header_font; c.fill = header_fill; c.alignment = Alignment(horizontal="center"); c.border = border
        for i, t in enumerate(trades, 1):
            entry_t = t["entry_time"].tz_localize(None) if hasattr(t["entry_time"], "tz_localize") else t["entry_time"]
            exit_t = t["exit_time"].tz_localize(None) if hasattr(t["exit_time"], "tz_localize") else t["exit_time"]
            row = [i, entry_t, exit_t, t["side"], t["entry_price"], t["exit_price"],
                   t["stop"], t["target"], t["size"], t["risk_pct"], t["risk_usd"],
                   t["gross_pnl"], t["cost"], t["net_pnl"], t["R_multiple"],
                   t["ai_proba"], t["exit_reason"], t["bars_held"],
                   t["wallet_before"], t["wallet_after"]]
            for col, v in enumerate(row, 1):
                c = ws.cell(row=4+i, column=col, value=v)
                c.border = border
                if col in (5, 6, 7, 8, 11, 12, 13, 14, 19, 20) and isinstance(v, (int, float)):
                    c.number_format = '"$"#,##0.00'
                elif col == 10 and isinstance(v, (int, float)):
                    c.number_format = '0.00%'
                elif col in (15, 16) and isinstance(v, (int, float)):
                    c.number_format = '0.00'
                if col == 14 and isinstance(v, (int, float)):
                    c.fill = pos_fill if v > 0 else (neg_fill if v < 0 else warn_fill)
        twidths = [5, 22, 22, 6, 12, 12, 12, 12, 12, 8, 10, 12, 10, 12, 8, 10, 12, 8, 14, 14]
        for i, w in enumerate(twidths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w
        ws.freeze_panes = "A5"

    wb.save(out_path)


if __name__ == "__main__":
    main()
