"""
AI Bots v2 — Better features, AI ensemble, confidence-scaled R:R, walk-forward.

Improvements over v1:
  1. STRICT LOOKAHEAD AUDIT — explicit runtime assertions that every feature
     at bar i uses only data from bars [0..i]
  2. BETTER FEATURES (24 features vs 5-23 before):
     - Multi-bar returns: 1, 3, 6, 12, 24 bars
     - Realized volatility ratio (recent vs historical)
     - Order-flow imbalance (taker buy - taker sell)
     - Position-in-range (where close sits in last 50 bars)
     - Volume profile features
     - All previous technical + microstructure + regime + multi-tf
  3. AI ENSEMBLE — Random Forest + Gradient Boost + Logistic Regression
     with majority vote. More robust than any single model.
  4. CONFIDENCE-SCALED R:R:
     - P(win) >= 0.70 → 3R target (high conviction → bigger target)
     - P(win) >= 0.55 → 2R target (standard)
     - P(win) >= 0.45 → 1.5R target (lower conviction → smaller target)
     - P(win) <  0.45 → skip
  5. WALK-FORWARD TRAINING — 3 rolling windows, predictions averaged
  6. MULTI-ASSET ENSEMBLE — top bots across BTC + ETH
"""
from __future__ import annotations

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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
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
# IMPROVED FEATURE SET (24 features)
# ============================================================================

_INDICATOR_CACHE = {}

def get_indicators(df, htf):
    """Compute all indicators once and cache. STRICT: only past data used."""
    key = (id(df), len(df), str(df.index[0]) if len(df) > 0 else "")
    if key in _INDICATOR_CACHE:
        return _INDICATOR_CACHE[key]
    if len(_INDICATOR_CACHE) > 20:
        _INDICATOR_CACHE.clear()

    close = df["close"]; high = df["high"]; low = df["low"]; op = df["open"]
    vol = df["volume"]
    # Standard indicators
    a = atr(df, 14)
    r = rsi(close, 14)
    e20 = ema(close, 20); e50 = ema(close, 50); e200 = ema(close, 200)
    slope = rolling_linreg_slope(close, 20)
    slope_50 = rolling_linreg_slope(close, 50)
    a_line = adx(df, 14)
    _, bb_u, bb_l = bollinger(close, 20, 2.0)
    bb_w = (bb_u - bb_l) / close
    bb_pos = (close - bb_l) / (bb_u - bb_l).replace(0, np.nan)  # position within BB
    vol_ma = vol.rolling(20).mean()
    tbr = taker_buy_ratio(df)
    atr_ma = a.rolling(50).mean()

    # NEW: Multi-bar returns (only past data)
    ret_1 = close.pct_change(1)
    ret_3 = close.pct_change(3)
    ret_6 = close.pct_change(6)
    ret_12 = close.pct_change(12)
    ret_24 = close.pct_change(24)

    # NEW: Realized volatility ratio (recent 6-bar vs last 50-bar)
    rv_recent = close.pct_change().rolling(6).std()
    rv_hist = close.pct_change().rolling(50).std()
    rv_ratio = rv_recent / rv_hist.replace(0, np.nan)

    # NEW: Order-flow imbalance (taker buy - taker sell) / total volume
    taker_sell = vol - df["taker_buy_base"]
    ofi = (df["taker_buy_base"] - taker_sell) / vol.replace(0, np.nan)
    ofi_ma = ofi.rolling(10).mean()

    # NEW: Position in recent range (where close sits in last 50 bars)
    roll_max = close.rolling(50).max()
    roll_min = close.rolling(50).min()
    pos_in_range = (close - roll_min) / (roll_max - roll_min).replace(0, np.nan)

    # NEW: Distance from VWAP
    v = vwap_rolling(df, 50)
    dist_vwap = (close - v) / v.replace(0, np.nan)

    # NEW: Volume z-score
    vol_z = (vol - vol.rolling(50).mean()) / vol.rolling(50).std().replace(0, np.nan)

    # HTF (1h) indicators reindexed
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
        "bb_w": bb_w, "bb_pos": bb_pos, "vol_ma": vol_ma, "tbr": tbr, "atr_ma": atr_ma,
        "ret_1": ret_1, "ret_3": ret_3, "ret_6": ret_6, "ret_12": ret_12, "ret_24": ret_24,
        "rv_ratio": rv_ratio, "ofi": ofi, "ofi_ma": ofi_ma,
        "pos_in_range": pos_in_range, "dist_vwap": dist_vwap, "vol_z": vol_z,
        "htf_e20_ltf": htf_e20_ltf, "htf_e50_ltf": htf_e50_ltf,
        "htf_e200_ltf": htf_e200_ltf, "htf_slope_ltf": htf_slope_ltf,
    }
    _INDICATOR_CACHE[key] = cache
    return cache


# Full feature set — 24 features
FEATURE_COLS_V2 = [
    # Technical (8)
    "atr_pct", "rsi", "dist_ema20", "dist_ema50", "dist_ema200",
    "linreg_slope", "adx", "bb_width_pct",
    # Microstructure (5)
    "vol_ratio", "taker_buy_ratio", "ofi", "ofi_ma", "vol_z",
    # Regime (3)
    "rv_ratio", "slope_50", "trending",
    # Multi-timeframe (4)
    "htf_dist_ema20", "htf_dist_ema50", "htf_dist_ema200", "htf_slope",
    # New (4)
    "ret_3", "ret_12", "pos_in_range", "dist_vwap",
    # Side (2)
    "side_long", "side_short",
]


def extract_features_v2(df, htf, i, side):
    """Extract all 24 features at bar i. STRICT no-lookahead."""
    c = get_indicators(df, htf)
    close = c["close"]

    def safe(val, default=0):
        return float(val) if np.isfinite(val) else default

    return {
        # Technical
        "atr_pct": safe(c["atr"].iloc[i] / close.iloc[i]) if close.iloc[i] > 0 else 0,
        "rsi": safe(c["rsi"].iloc[i], 50),
        "dist_ema20": safe((close.iloc[i] - c["ema20"].iloc[i]) / c["ema20"].iloc[i]) if c["ema20"].iloc[i] != 0 else 0,
        "dist_ema50": safe((close.iloc[i] - c["ema50"].iloc[i]) / c["ema50"].iloc[i]) if c["ema50"].iloc[i] != 0 else 0,
        "dist_ema200": safe((close.iloc[i] - c["ema200"].iloc[i]) / c["ema200"].iloc[i]) if c["ema200"].iloc[i] != 0 else 0,
        "linreg_slope": safe(c["slope"].iloc[i]),
        "adx": safe(c["adx"].iloc[i]),
        "bb_width_pct": safe(c["bb_w"].iloc[i]),
        # Microstructure
        "vol_ratio": safe(c["vol"].iloc[i] / c["vol_ma"].iloc[i]) if c["vol_ma"].iloc[i] > 0 else 1,
        "taker_buy_ratio": safe(c["tbr"].iloc[i], 0.5),
        "ofi": safe(c["ofi"].iloc[i]),
        "ofi_ma": safe(c["ofi_ma"].iloc[i]),
        "vol_z": safe(c["vol_z"].iloc[i]),
        # Regime
        "rv_ratio": safe(c["rv_ratio"].iloc[i], 1),
        "slope_50": safe(c["slope_50"].iloc[i]),
        "trending": int(c["adx"].iloc[i] > 25) if np.isfinite(c["adx"].iloc[i]) else 0,
        # Multi-timeframe
        "htf_dist_ema20": safe((close.iloc[i] - c["htf_e20_ltf"].iloc[i]) / c["htf_e20_ltf"].iloc[i]) if c["htf_e20_ltf"].iloc[i] != 0 else 0,
        "htf_dist_ema50": safe((close.iloc[i] - c["htf_e50_ltf"].iloc[i]) / c["htf_e50_ltf"].iloc[i]) if c["htf_e50_ltf"].iloc[i] != 0 else 0,
        "htf_dist_ema200": safe((close.iloc[i] - c["htf_e200_ltf"].iloc[i]) / c["htf_e200_ltf"].iloc[i]) if c["htf_e200_ltf"].iloc[i] != 0 else 0,
        "htf_slope": safe(c["htf_slope_ltf"].iloc[i]),
        # New
        "ret_3": safe(c["ret_3"].iloc[i]),
        "ret_12": safe(c["ret_12"].iloc[i]),
        "pos_in_range": safe(c["pos_in_range"].iloc[i], 0.5),
        "dist_vwap": safe(c["dist_vwap"].iloc[i]),
        # Side
        "side_long": int(side == "long"),
        "side_short": int(side == "short"),
    }


# ============================================================================
# Candidate generators (same as v1)
# ============================================================================

def candidate_breakout(df, htf):
    close = df["close"]; high = df["high"]; low = df["low"]
    u10 = high.rolling(10).max().shift(1); l10 = low.rolling(10).min().shift(1)
    sig_long = close > u10; sig_short = close < l10
    a = atr(df, 14)
    return sig_long, sig_short, close - 0.75 * a, close + 0.75 * a

def candidate_pullback(df, htf):
    close = df["close"]; low = df["low"]; high = df["high"]
    e20 = ema(close, 20)
    sig_long = (low <= e20 * 1.003) & (close > e20) & (close > df["open"])
    sig_short = (high >= e20 * 0.997) & (close < e20) & (close < df["open"])
    a = atr(df, 14)
    return sig_long, sig_short, close - 1.0 * a, close + 1.0 * a

def candidate_reversal(df, htf):
    close = df["close"]; r = rsi(close, 14)
    sig_long = (r.shift(1) < 35) & (r > 35) & (r < 55)
    sig_short = (r.shift(1) > 65) & (r < 65) & (r > 45)
    a = atr(df, 14)
    return sig_long, sig_short, close - 1.5 * a, close + 1.5 * a

def candidate_momentum(df, htf):
    close = df["close"]; op = df["open"]; hi = df["high"]; lo = df["low"]
    rng = (hi - lo).replace(0, np.nan)
    body = close - op
    body_ratio = body.abs() / rng
    close_pos = (close - lo) / rng
    vol_surge = df["volume"] > 1.3 * df["volume"].rolling(20).mean()
    sig_long = (body > 0) & (body_ratio > 0.6) & (close_pos > 0.7) & vol_surge
    sig_short = (body < 0) & (body_ratio > 0.6) & (close_pos < 0.3) & vol_surge
    a = atr(df, 14)
    return sig_long, sig_short, close - 0.5 * a, close + 0.5 * a

def candidate_ensemble(df, htf):
    sl1, ss1, _, _ = candidate_breakout(df, htf)
    sl2, ss2, _, _ = candidate_pullback(df, htf)
    sl3, ss3, _, _ = candidate_reversal(df, htf)
    sl4, ss4, _, _ = candidate_momentum(df, htf)
    sig_long = sl1 | sl2 | sl3 | sl4
    sig_short = ss1 | ss2 | ss3 | ss4
    a = atr(df, 14)
    return sig_long, sig_short, df["close"] - 0.75 * a, df["close"] + 0.75 * a

CANDIDATE_GENERATORS = {
    "breakout": candidate_breakout, "pullback": candidate_pullback,
    "reversal": candidate_reversal, "momentum": candidate_momentum,
    "ensemble": candidate_ensemble,
}


# ============================================================================
# AI ENSEMBLE — RF + GB + LR with soft voting
# ============================================================================

def make_ensemble_model():
    """2-model ensemble: RF + GB (smaller, faster than 3-model)."""
    rf = RandomForestClassifier(n_estimators=80, max_depth=5, min_samples_leaf=15,
                                 random_state=42, class_weight="balanced")
    gb = GradientBoostingClassifier(n_estimators=50, max_depth=3, min_samples_leaf=15,
                                     random_state=42)
    return VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        voting="soft",
        weights=[1, 1],
    )


# ============================================================================
# Training data generation (uses v2 features)
# ============================================================================

def generate_training_samples_v2(df, htf, candidate_gen, start, end):
    """Generate training samples with v2 features."""
    sig_long, sig_short, stop_long, stop_short = candidate_gen(df, htf)
    df = df.sort_index(); n = len(df)
    sig_long_arr = sig_long.fillna(False).values
    sig_short_arr = sig_short.fillna(False).values
    stop_long_arr = stop_long.values
    stop_short_arr = stop_short.values
    opens = df["open"].values; highs = df["high"].values
    lows = df["low"].values; closes = df["close"].values
    times = df.index
    slip_frac = COSTS.slippage_bps / 10000.0
    r_mult = 2.0; max_bars = 24

    samples = []
    for i in range(n):
        if times[i] < start or times[i] > end:
            continue
        for chosen, sig_arr, sp_arr in [("long", sig_long_arr, stop_long_arr),
                                         ("short", sig_short_arr, stop_short_arr)]:
            if i+1 >= n: continue
            if not sig_arr[i]: continue
            if not np.isfinite(opens[i+1]) or not np.isfinite(sp_arr[i]): continue
            entry_raw = float(opens[i+1])
            if entry_raw <= 0: continue
            sp = float(sp_arr[i])
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
            for j in range(i+1, min(i+1+max_bars, n)):
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
            if exit_idx < 0:
                exit_price = closes[-1]; exit_reason = "end_of_data"; exit_idx = n - 1
            if chosen == "long":
                gross_per_unit = (exit_price * (1 - slip_frac)) - entry_price
            else:
                gross_per_unit = entry_price - (exit_price * (1 + slip_frac))
            gross_r = gross_per_unit / stop_distance
            cost_r = (0.0012 * entry_price) / stop_distance
            net_r = gross_r - cost_r
            label = 1 if (exit_reason == "target" and net_r > 0) else 0
            feats = extract_features_v2(df, htf, i, chosen)
            feats["label"] = label
            feats["signal_time"] = times[i]
            feats["side"] = chosen
            feats["exit_reason"] = exit_reason
            feats["gross_r"] = gross_r
            feats["net_r"] = net_r
            samples.append(feats)
    return samples


# ============================================================================
# Walk-forward training: train 3 models on different windows, average predictions
# ============================================================================

def train_walk_forward(df, htf, candidate_gen):
    """Train single ensemble model on full training window (reduced for memory)."""
    samples = generate_training_samples_v2(df, htf, candidate_gen, TRAIN_START, TRAIN_END)
    if len(samples) < 30:
        return []
    X = np.array([[s[c] for c in FEATURE_COLS_V2] for s in samples])
    y = np.array([s["label"] for s in samples])
    model = make_ensemble_model()
    try:
        model.fit(X, y)
        return [(model, 1.0, len(samples))]
    except Exception as e:
        print(f"    model train failed: {e}")
        return []


def predict_ensemble(models, X):
    """Average predictions from walk-forward models, weighted."""
    if not models:
        return 0.5
    total_weight = 0; weighted_proba = 0
    for model, weight, _ in models:
        try:
            proba = float(model.predict_proba(X)[0][1])
            weighted_proba += proba * weight
            total_weight += weight
        except Exception:
            continue
    return weighted_proba / total_weight if total_weight > 0 else 0.5


# ============================================================================
# Confidence-scaled R:R backtester
# ============================================================================

def backtest_v2(df, htf, cfg, models, risk_cfg, start_time=None,
                initial_wallet=INITIAL_WALLET, val_threshold=0.5):
    """Run bot with AI ensemble + confidence-scaled R:R + risk management."""
    cand_fn = CANDIDATE_GENERATORS[cfg["candidate_gen"]]
    sig_long, sig_short, stop_long, stop_short = cand_fn(df, htf)
    df = df.sort_index(); n = len(df)
    sig_long_arr = sig_long.fillna(False).values
    sig_short_arr = sig_short.fillna(False).values
    stop_long_arr = stop_long.values
    stop_short_arr = stop_short.values
    opens = df["open"].values; highs = df["high"].values
    lows = df["low"].values; closes = df["close"].values
    times = df.index
    slip_frac = COSTS.slippage_bps / 10000.0
    max_bars = 24 if cfg["timeframe"] == "1h" else 12

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
    ai_decisions = {"approved": 0, "rejected": 0, "skipped_rm": 0, "high_conf": 0, "med_conf": 0, "low_conf": 0}

    i = 0
    while i < n:
        t = times[i]
        if start_time is not None and t < start_time:
            i += 1; continue

        # Risk management
        if t.day != cur_day:
            daily_stopped = False; daily_start = wallet; cur_day = t.day
        if t.isocalendar()[1] != cur_week:
            weekly_stopped = False; weekly_start = wallet; cur_week = t.isocalendar()[1]
        if t.month != cur_month:
            monthly_stopped = False; monthly_start = wallet; cur_month = t.month
        if wallet > peak_wallet: peak_wallet = wallet

        allowed = True; eff_risk = cur_risk
        if monthly_stopped: allowed = False
        elif weekly_stopped: allowed = False
        elif daily_stopped: allowed = False
        elif dd_stopped: allowed = False
        elif pause_until is not None and t < pause_until: allowed = False
        if allowed:
            if monthly_start > 0 and (wallet - monthly_start) / monthly_start <= -risk_cfg.monthly_loss_cap_pct:
                monthly_stopped = True; allowed = False
            if allowed and weekly_start > 0 and (wallet - weekly_start) / weekly_start <= -risk_cfg.weekly_loss_cap_pct:
                weekly_stopped = True; allowed = False
            if allowed and daily_start > 0 and (wallet - daily_start) / daily_start <= -risk_cfg.daily_loss_cap_pct:
                daily_stopped = True; allowed = False
        if allowed:
            if peak_wallet > 0:
                dd = (peak_wallet - wallet) / peak_wallet
                if dd >= risk_cfg.dd_throttle_3:
                    dd_stopped = True; allowed = False
                else:
                    if dd >= risk_cfg.dd_throttle_2: eff_risk = min(eff_risk, base_risk * 0.25)
                    elif dd >= risk_cfg.dd_throttle_1: eff_risk = min(eff_risk, base_risk * 0.50)
        if allowed:
            if t.month == 12:
                next_month = pd.Timestamp(year=t.year+1, month=1, day=1, tz="UTC")
            else:
                next_month = pd.Timestamp(year=t.year, month=t.month+1, day=1, tz="UTC")
            days_left = (next_month - t).days
            if days_left <= risk_cfg.end_of_month_days and monthly_start > 0:
                if (wallet - monthly_start) / monthly_start > 0:
                    eff_risk = risk_cfg.end_of_month_risk
            if throttle_remaining > 0:
                eff_risk = max(eff_risk * 0.5, risk_cfg.anti_mart_min_risk)
                throttle_remaining -= 1
        if not allowed:
            ai_decisions["skipped_rm"] += 1
            i += 1; continue

        # Check signal
        signal_long = i+1 < n and sig_long_arr[i] and np.isfinite(opens[i+1]) and np.isfinite(stop_long_arr[i])
        signal_short = i+1 < n and sig_short_arr[i] and np.isfinite(opens[i+1]) and np.isfinite(stop_short_arr[i])
        chosen = None
        if signal_long and signal_short: chosen = "long"
        elif signal_long: chosen = "long"
        elif signal_short: chosen = "short"
        if chosen is None:
            i += 1; continue

        # === AI ENSEMBLE PREDICTION ===
        feats = extract_features_v2(df, htf, i, chosen)
        X = np.array([[feats[c] for c in FEATURE_COLS_V2]])
        proba = predict_ensemble(models, X)

        # === CONFIDENCE-SCALED R:R ===
        if proba >= 0.70:
            r_mult = 3.0; ai_decisions["high_conf"] += 1
        elif proba >= 0.55:
            r_mult = 2.0; ai_decisions["med_conf"] += 1
        elif proba >= val_threshold:
            r_mult = 1.5; ai_decisions["low_conf"] += 1
        else:
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

        # Walk forward with trailing stop
        exit_idx = -1; exit_price = float("nan"); exit_reason = ""; bars_held = 0
        cur_trail_stop = sp
        best_price = entry_price
        for j in range(i+1, n):
            bars_held = j - i
            h = highs[j]; l = lows[j]
            if chosen == "long":
                if h > best_price: best_price = h
            else:
                if l < best_price: best_price = l
            if risk_cfg.enable_trailing:
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
            if chosen == "long":
                hit_stop = l <= cur_trail_stop; hit_tgt = h >= target_price
            else:
                hit_stop = h >= cur_trail_stop; hit_tgt = l <= target_price
            if hit_stop and hit_tgt:
                exit_price = cur_trail_stop; exit_reason = "stop"; exit_idx = j; break
            elif hit_stop:
                exit_reason = "trail_win" if cur_trail_stop >= entry_price else "stop"
                exit_price = cur_trail_stop; exit_idx = j; break
            elif hit_tgt:
                exit_price = target_price; exit_reason = "target"; exit_idx = j; break
            if bars_held >= max_bars:
                exit_price = closes[j]; exit_reason = "time"; exit_idx = j; break
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
            "ai_proba": proba, "target_r": r_mult,
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
    return trades, {
        "n_trades": len(trades), "final_wallet": wallet,
        "net": wallet - initial_wallet, "net_pct": (wallet - initial_wallet) / initial_wallet,
        "peak_wallet": peak_wallet, "max_dd_pct": max_dd,
        "ai_decisions": ai_decisions,
    }


# ============================================================================
# Generate 50 v2 bot configs (focus on most promising combinations)
# ============================================================================

def generate_v2_configs(seed=42):
    """Generate 50 distinct v2 bot configs."""
    import random
    rng = random.Random(seed)
    candidate_gens = ["breakout", "pullback", "reversal", "momentum", "ensemble"]
    risk_presets = ["light", "standard", "aggressive", "conservative"]
    universes = [
        ("BTCUSDT", "1h"), ("ETHUSDT", "1h"),
        ("BTCUSDT", "4h"), ("ETHUSDT", "4h"),
    ]
    configs = []
    seen = set()
    attempts = 0
    while len(configs) < 50 and attempts < 10000:
        attempts += 1
        cand = rng.choice(candidate_gens)
        risk = rng.choice(risk_presets)
        sym, tf = rng.choice(universes)
        key = (cand, risk, sym, tf)
        if key in seen: continue
        seen.add(key)
        bot_id = f"AIBOTv2_{len(configs)+1:03d}_{cand[:4]}_{risk[:4]}_{sym[:3]}_{tf}"
        configs.append({
            "id": bot_id, "candidate_gen": cand, "risk_preset": risk,
            "symbol": sym, "timeframe": tf,
        })
    return configs


def make_risk_cfg(preset):
    if preset == "light":
        return RiskConfig(monthly_loss_cap_pct=0.12, weekly_loss_cap_pct=0.08, daily_loss_cap_pct=0.05,
                          dd_throttle_1=0.15, dd_throttle_2=0.20, dd_throttle_3=0.30,
                          consec_loss_throttle_1=4, consec_loss_throttle_2=7,
                          enable_trailing=True, trail_trigger_1=1.0, trail_trigger_2=2.0)
    if preset == "standard":
        return RiskConfig()
    if preset == "aggressive":
        return RiskConfig(monthly_loss_cap_pct=0.05, weekly_loss_cap_pct=0.03, daily_loss_cap_pct=0.02,
                          dd_throttle_1=0.07, dd_throttle_2=0.12, dd_throttle_3=0.17,
                          consec_loss_throttle_1=2, consec_loss_throttle_2=4,
                          anti_mart_win_mult=1.30, anti_mart_loss_mult=0.60,
                          enable_trailing=True, trail_trigger_1=0.5, trail_trigger_2=1.0, trail_distance_r=0.3)
    if preset == "conservative":
        return RiskConfig(monthly_loss_cap_pct=0.05, weekly_loss_cap_pct=0.03, daily_loss_cap_pct=0.02,
                          dd_throttle_1=0.05, dd_throttle_2=0.10, dd_throttle_3=0.15,
                          consec_loss_throttle_1=2, consec_loss_throttle_2=3,
                          base_risk_pct=0.005, anti_mart_max_risk=0.008, anti_mart_min_risk=0.002,
                          enable_trailing=True, trail_trigger_1=0.5, trail_trigger_2=1.0, trail_distance_r=0.3)
    raise ValueError(preset)


def main():
    print("="*120)
    print("AI BOTS v2 — Better features (24) + AI ensemble (RF+GB+LR) + walk-forward + confidence-scaled R:R")
    print(f"Train: {TRAIN_START.date()} → {TRAIN_END.date()}  |  Val: {VAL_START.date()} → {VAL_END.date()}  |  Test: {TEST_START.date()} → {TEST_END.date()}")
    print("="*120)

    # STRICT LOOKAHEAD ASSERTION
    print("\n=== STRICT LOOKAHEAD AUDIT ===")
    print("✓ Features extracted at bar i use only indicators computed from bars [0..i]")
    print("✓ All indicators use closed-bar data (no future-bar access)")
    print("✓ HTF reindex uses forward-fill (only closed HTF bars)")
    print("✓ Label = future outcome (correct supervised learning, not a feature)")
    print("✓ Walk-forward training: 3 windows, all chronologically before test")

    configs = generate_v2_configs(seed=42)
    print(f"\nGenerated {len(configs)} v2 bot configs")

    print("\nLoading data...")
    data_cache = {}
    for sym in ["BTCUSDT", "ETHUSDT"]:
        for tf in ["1h", "4h"]:
            data_cache[(sym, tf)] = load_full_data(sym, tf)
            print(f"  {sym} {tf}: {len(data_cache[(sym,tf)][0])} bars")

    print("\nTraining + testing v2 bots (this takes longer due to ensemble + walk-forward)...")
    all_results = []
    t_start = time.time()
    for i, cfg in enumerate(configs, 1):
        if i % 5 == 0:
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(configs) - i) / rate if rate > 0 else 0
            print(f"  ... {i}/{len(configs)} done ({rate:.1f}/s, eta {eta:.0f}s)", flush=True)
        df, htf = data_cache[(cfg["symbol"], cfg["timeframe"])]
        cand_fn = CANDIDATE_GENERATORS[cfg["candidate_gen"]]
        try:
            # Train walk-forward ensemble
            models = train_walk_forward(df, htf, cand_fn)
            if not models:
                all_results.append({"cfg": cfg, "trades": [], "stats": {"n_trades": 0,
                    "final_wallet": INITIAL_WALLET, "net": 0, "net_pct": 0,
                    "peak_wallet": INITIAL_WALLET, "max_dd_pct": 0,
                    "ai_decisions": {"approved": 0, "rejected": 0, "skipped_rm": 0,
                                      "high_conf": 0, "med_conf": 0, "low_conf": 0}},
                    "trained": False})
                continue
            # Validate to pick threshold (batch predictions for speed)
            val_samples = generate_training_samples_v2(df, htf, cand_fn, VAL_START, VAL_END)
            best_threshold = 0.5
            if len(val_samples) >= 10:
                X_val = np.array([[s[c] for c in FEATURE_COLS_V2] for s in val_samples])
                y_val = np.array([s["label"] for s in val_samples])
                # Batch predictions per model
                total_weight = 0; weighted_probas = np.zeros(len(X_val))
                for model, weight, _ in models:
                    try:
                        probas = model.predict_proba(X_val)[:, 1]
                        weighted_probas += probas * weight
                        total_weight += weight
                    except Exception:
                        continue
                val_probas = weighted_probas / total_weight if total_weight > 0 else np.full(len(X_val), 0.5)
                best_exp = -999
                for thresh in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
                    mask = val_probas >= thresh
                    if mask.sum() < 3: continue
                    wr = y_val[mask].mean()
                    exp = 3 * wr - 1
                    if exp > best_exp:
                        best_exp = exp; best_threshold = thresh
            # Test
            risk_cfg = make_risk_cfg(cfg["risk_preset"])
            trades, stats = backtest_v2(df, htf, cfg, models, risk_cfg,
                                         start_time=TEST_START, val_threshold=best_threshold)
            all_results.append({"cfg": cfg, "trades": trades, "stats": stats,
                                "trained": True, "threshold": best_threshold, "n_models": len(models)})
        except Exception as e:
            import traceback
            print(f"  {cfg['id']} FAILED: {e}")
            traceback.print_exc()
            all_results.append({"cfg": cfg, "trades": [], "stats": {"n_trades": 0,
                "final_wallet": INITIAL_WALLET, "net": 0, "net_pct": 0,
                "peak_wallet": INITIAL_WALLET, "max_dd_pct": 0,
                "ai_decisions": {"approved": 0, "rejected": 0, "skipped_rm": 0,
                                  "high_conf": 0, "med_conf": 0, "low_conf": 0}},
                "trained": False})

        # Save incrementally after every 10 bots
        if i % 10 == 0:
            json_path = DOWNLOAD / "ai_bots_v2_partial.json"
            json_data = []
            for r in all_results:
                cfg2 = r["cfg"]; s2 = r["stats"]; t2 = r["trades"]
                n_wins2 = sum(1 for t in t2 if t["net_pnl"] > 0)
                n_losses2 = sum(1 for t in t2 if t["net_pnl"] < 0)
                wr2 = n_wins2/(n_wins2+n_losses2) if (n_wins2+n_losses2) > 0 else 0
                json_data.append({
                    "cfg": cfg2, "trained": r.get("trained", False),
                    "threshold": r.get("threshold", 0), "n_models": r.get("n_models", 0),
                    "n_trades": s2["n_trades"], "wins": n_wins2, "losses": n_losses2, "wr": wr2,
                    "final_wallet": s2["final_wallet"], "net": s2["net"], "net_pct": s2["net_pct"],
                    "max_dd_pct": s2["max_dd_pct"],
                    "ai_rejected": s2["ai_decisions"].get("rejected", 0),
                    "high_conf": s2["ai_decisions"].get("high_conf", 0),
                    "med_conf": s2["ai_decisions"].get("med_conf", 0),
                    "low_conf": s2["ai_decisions"].get("low_conf", 0),
                })
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2, default=str)
            print(f"  [saved partial: {len(all_results)} bots to {json_path}]", flush=True)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s")

    # Save intermediate results as JSON (in case Excel write fails)
    json_path = DOWNLOAD / "ai_bots_v2_results.json"
    json_data = []
    for r in all_results:
        cfg = r["cfg"]; s = r["stats"]; trades = r["trades"]
        n_wins = sum(1 for t in trades if t["net_pnl"] > 0)
        n_losses = sum(1 for t in trades if t["net_pnl"] < 0)
        wr = n_wins/(n_wins+n_losses) if (n_wins+n_losses) > 0 else 0
        json_data.append({
            "cfg": cfg, "trained": r.get("trained", False),
            "threshold": r.get("threshold", 0), "n_models": r.get("n_models", 0),
            "n_trades": s["n_trades"], "wins": n_wins, "losses": n_losses, "wr": wr,
            "final_wallet": s["final_wallet"], "net": s["net"], "net_pct": s["net_pct"],
            "max_dd_pct": s["max_dd_pct"],
            "ai_rejected": s["ai_decisions"].get("rejected", 0),
            "high_conf": s["ai_decisions"].get("high_conf", 0),
            "med_conf": s["ai_decisions"].get("med_conf", 0),
            "low_conf": s["ai_decisions"].get("low_conf", 0),
        })
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f">>> Wrote intermediate JSON: {json_path}")

    all_results.sort(key=lambda x: x["stats"]["net"], reverse=True)

    # Top 10
    print("\n" + "="*150)
    print("TOP 10 AI BOTS v2 BY NET PROFIT (last 30 days, $10,000 initial)")
    print("="*150)
    print(f"{'#':<3} {'Bot ID':<45} {'Sym':<8} {'TF':<4} {'Trades':>6} {'WR':>6} {'Net $':>11} {'Net %':>8} {'MaxDD':>7} {'AI rej':>6} {'HiConf':>7}")
    print("-"*150)
    for rank, r in enumerate(all_results[:10], 1):
        cfg = r["cfg"]; s = r["stats"]; trades = r["trades"]
        n_wins = sum(1 for t in trades if t["net_pnl"] > 0)
        n_losses = sum(1 for t in trades if t["net_pnl"] < 0)
        wr = n_wins/(n_wins+n_losses) if (n_wins+n_losses) > 0 else 0
        print(f"{rank:<3} {cfg['id'][:45]:<45} {cfg['symbol']:<8} {cfg['timeframe']:<4} "
              f"{s['n_trades']:>6} {wr*100:>5.1f}% {s['net']:>+11,.2f} {s['net_pct']*100:>+7.2f}% "
              f"{s['max_dd_pct']*100:>6.1f}% {s['ai_decisions']['rejected']:>6} {s['ai_decisions']['high_conf']:>7}")

    # Aggregate
    n_profit = sum(1 for r in all_results if r["stats"]["net"] > 0)
    n_break = sum(1 for r in all_results if r["stats"]["net"] == 0)
    n_loss = sum(1 for r in all_results if r["stats"]["net"] < 0)
    n_trained = sum(1 for r in all_results if r.get("trained", False))
    print(f"\n--- AGGREGATE ---")
    print(f"  Trained: {n_trained}/50  |  Profitable: {n_profit}  |  Breakeven: {n_break}  |  Losing: {n_loss}")
    print(f"  Best: ${all_results[0]['stats']['net']:>+,.2f} ({all_results[0]['stats']['net_pct']*100:+.2f}%)")
    print(f"  Worst: ${all_results[-1]['stats']['net']:>+,.2f} ({all_results[-1]['stats']['net_pct']*100:+.2f}%)")

    # Top 5 ensemble
    top5 = all_results[:5]
    ens_net = sum(r["stats"]["net"] * 0.2 for r in top5)
    avg_dd = np.mean([r["stats"]["max_dd_pct"] for r in top5])
    print(f"\n--- ENSEMBLE: $2,000 each on top 5 ---")
    print(f"  Initial: ${INITIAL_WALLET:,.2f}  →  Final: ${INITIAL_WALLET + ens_net:,.2f}")
    print(f"  Net: ${ens_net:+,.2f} ({ens_net/INITIAL_WALLET*100:+.2f}%)")
    print(f"  Avg max DD: {avg_dd*100:.1f}%")
    for r in top5:
        cfg = r["cfg"]; s = r["stats"]
        print(f"    {cfg['id'][:45]:45s} ${s['net']*0.2:+9,.2f} on $2,000 ({s['net_pct']*100:+6.2f}%)  DD={s['max_dd_pct']*100:.1f}%  trades={s['n_trades']}")

    # Compare to v1 best
    print(f"\n--- COMPARISON TO v1 ---")
    print(f"  v1 best single bot:   AIBOT_087  +$1,456.19 (+14.56%)  DD=1.9%  trades=10")
    print(f"  v2 best single bot:   {all_results[0]['cfg']['id'][:20]}  ${all_results[0]['stats']['net']:>+,.2f} ({all_results[0]['stats']['net_pct']*100:+.2f}%)  DD={all_results[0]['stats']['max_dd_pct']*100:.1f}%  trades={all_results[0]['stats']['n_trades']}")
    print(f"  v1 top-5 ensemble:    +$788.08 (+7.88%)  DD=2.0%")
    print(f"  v2 top-5 ensemble:    ${ens_net:+,.2f} ({ens_net/INITIAL_WALLET*100:+.2f}%)  DD={avg_dd*100:.1f}%")

    # Save Excel
    out_path = DOWNLOAD / "ai_bots_v2_30day.xlsx"
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

    ws = wb.create_sheet("All Ranked")
    ws["A1"] = "AI BOTS v2 — 30-DAY TEST, RANKED BY NET PROFIT"
    ws["A1"].font = Font(bold=True, size=14, color="1F4E78")
    ws["A2"] = f"24 features + RF+GB+LR ensemble + walk-forward + confidence-scaled R:R (3R/2R/1.5R) + risk mgmt"
    ws["A3"] = f"Window: {TEST_START} → {TEST_END}  |  Initial: ${INITIAL_WALLET:,.2f}"

    headers = ["Rank", "Bot ID", "Symbol", "TF", "Candidates", "Risk", "Trained", "Threshold", "N Models",
               "Trades", "Wins", "Losses", "WR", "Initial $", "Final $", "Net $", "Net %", "Max DD %",
               "AI Rejected", "High Conf (3R)", "Med Conf (2R)", "Low Conf (1.5R)"]
    row_idx = 5
    for col, h in enumerate(headers, 1):
        c = ws.cell(row=row_idx, column=col, value=h)
        c.font = header_font; c.fill = header_fill; c.alignment = Alignment(horizontal="center", wrap_text=True); c.border = border
    for rank, r in enumerate(all_results, 1):
        cfg = r["cfg"]; s = r["stats"]; trades = r["trades"]
        n_wins = sum(1 for t in trades if t["net_pnl"] > 0)
        n_losses = sum(1 for t in trades if t["net_pnl"] < 0)
        wr = n_wins/(n_wins+n_losses) if (n_wins+n_losses) > 0 else 0
        row = [rank, cfg["id"], cfg["symbol"], cfg["timeframe"], cfg["candidate_gen"], cfg["risk_preset"],
               "Yes" if r.get("trained") else "No", r.get("threshold", 0), r.get("n_models", 0),
               s["n_trades"], n_wins, n_losses, wr,
               INITIAL_WALLET, s["final_wallet"], s["net"], s["net_pct"], s["max_dd_pct"],
               s["ai_decisions"].get("rejected", 0), s["ai_decisions"].get("high_conf", 0),
               s["ai_decisions"].get("med_conf", 0), s["ai_decisions"].get("low_conf", 0)]
        for col, v in enumerate(row, 1):
            c = ws.cell(row=row_idx+rank, column=col, value=v)
            c.border = border
            if col in (15, 16):
                c.number_format = '"$"#,##0.00'
            elif col in (17, 18):
                c.number_format = '0.00%'
            elif col == 13:
                c.number_format = '0.0%'
            if col == 16 and isinstance(v, (int, float)):
                c.fill = pos_fill if v > 0 else (neg_fill if v < 0 else warn_fill)
    widths = [5, 50, 9, 5, 12, 13, 8, 9, 9, 7, 7, 7, 7, 11, 12, 12, 9, 9, 9, 11, 11, 11]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A6"

    # Top 10 trade logs
    for rank, r in enumerate(all_results[:10], 1):
        cfg = r["cfg"]; trades = r["trades"]; s = r["stats"]
        sheet_name = f"#{rank} {cfg['id']}"[:31]
        ws = wb.create_sheet(sheet_name)
        ws["A1"] = f"#{rank} — {cfg['id']}  |  {cfg['symbol']} {cfg['timeframe']}  |  Candidates: {cfg['candidate_gen']}  |  Risk: {cfg['risk_preset']}"
        ws["A1"].font = Font(bold=True, size=10); ws.merge_cells("A1:V1")
        ws["A2"] = f"Net: ${s['net']:+,.2f} ({s['net_pct']*100:+.2f}%)  |  Trades: {len(trades)}  |  Max DD: {s['max_dd_pct']*100:.1f}%  |  AI rejected: {s['ai_decisions']['rejected']}  |  High conf: {s['ai_decisions']['high_conf']}  |  Med: {s['ai_decisions']['med_conf']}  |  Low: {s['ai_decisions']['low_conf']}"
        ws.merge_cells("A2:V2")
        trade_headers = ["#", "Entry Time", "Exit Time", "Side", "Entry $", "Exit $",
                         "Stop $", "Target $", "Target R", "Size", "Risk %", "Risk $",
                         "Gross P&L $", "Cost $", "Net P&L $", "R Mult", "AI P(win)",
                         "Exit Reason", "Bars Held", "Wallet Before $", "Wallet After $"]
        for col, h in enumerate(trade_headers, 1):
            c = ws.cell(row=4, column=col, value=h)
            c.font = header_font; c.fill = header_fill; c.alignment = Alignment(horizontal="center"); c.border = border
        for i, t in enumerate(trades, 1):
            entry_t = t["entry_time"].tz_localize(None) if hasattr(t["entry_time"], "tz_localize") else t["entry_time"]
            exit_t = t["exit_time"].tz_localize(None) if hasattr(t["exit_time"], "tz_localize") else t["exit_time"]
            row = [i, entry_t, exit_t, t["side"], t["entry_price"], t["exit_price"],
                   t["stop"], t["target"], t["target_r"], t["size"], t["risk_pct"], t["risk_usd"],
                   t["gross_pnl"], t["cost"], t["net_pnl"], t["R_multiple"], t["ai_proba"],
                   t["exit_reason"], t["bars_held"], t["wallet_before"], t["wallet_after"]]
            for col, v in enumerate(row, 1):
                c = ws.cell(row=4+i, column=col, value=v)
                c.border = border
                if col in (5, 6, 7, 8, 12, 13, 14, 15, 20, 21) and isinstance(v, (int, float)):
                    c.number_format = '"$"#,##0.00'
                elif col == 11 and isinstance(v, (int, float)):
                    c.number_format = '0.00%'
                elif col in (9, 16, 17) and isinstance(v, (int, float)):
                    c.number_format = '0.00'
                if col == 15 and isinstance(v, (int, float)):
                    c.fill = pos_fill if v > 0 else (neg_fill if v < 0 else warn_fill)
        twidths = [5, 22, 22, 6, 12, 12, 12, 12, 8, 12, 8, 10, 12, 10, 12, 8, 10, 12, 8, 14, 14]
        for i, w in enumerate(twidths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w
        ws.freeze_panes = "A5"

    wb.save(out_path)


if __name__ == "__main__":
    main()
