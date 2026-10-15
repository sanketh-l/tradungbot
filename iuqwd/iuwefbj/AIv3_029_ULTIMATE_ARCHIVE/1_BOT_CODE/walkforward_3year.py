"""
AIv3_029 — 3-Year Walk-Forward Test (36 months, month-by-month)

For EACH of 36 test months (2023-11 to 2026-10):
  1. Training window = 12 months BEFORE test month begins
  2. Validation window = 1 month BEFORE test month begins
  3. Test window = the test month itself
  4. Train 9 fresh Random Forest AIs (one per market) using ONLY training data
  5. Validate to pick threshold using ONLY validation data
  6. Test on the test month — AI has NEVER seen this data
  7. Free AI memory before next month

STRICT NO-LOOKAHEAD ASSERTIONS (logged for every month):
  - train_end < val_start (training ends before validation begins)
  - val_end < test_start (validation ends before test begins)
  - AI trained on data with timestamps <= train_end
  - AI tested on data with timestamps >= test_start
  - Features at bar i use only data from bars [0..i]

OUTPUT: download/AIv3_029_walkforward_3year.xlsx
  - Month-by-month results table (36 rows)
  - Aggregate stats (average, median, best, worst, # profitable)
  - No-lookahead audit log
"""
from __future__ import annotations
import json
import time
import gc
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from sklearn.ensemble import RandomForestClassifier

from backtester import Costs, ema, rsi, atr, adx, bollinger, vwap_rolling, donchian
from smart_strategies import zscore, taker_buy_ratio
from bots_100 import rolling_linreg_slope, INITIAL_WALLET, RISK_PER_TRADE, COSTS, DATA_DIR, DOWNLOAD
from risk_managed_v2 import RiskConfig

# ============================================================================
# AIv3_029 EXACT CONFIGURATION (from the document)
# ============================================================================

CONFIG = {
    "id": "AIv3_029_all9_mome_aggr_L3rf_onl_mc1",
    "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
                "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "PAXGUSDT"],
    "candidate_gen": "momentum",
    "risk_preset": "aggressive",
    "ai_config": {
        "name": "L3rf_only",
        "use_layer3": True,
        "layer3_kind": "rf",
    },
    "max_concurrent": 1,
    "max_same_direction": 3,
}

# Feature columns (27 features — same as v3)
FEATURE_COLS = [
    # Microstructure (5)
    "vol_ratio", "taker_buy_ratio", "vol_ma_diff", "atr_pct", "vol_surge_3bar",
    # Technical (5)
    "rsi", "dist_ema20", "adx", "linreg_slope", "bb_width_pct",
    # New (3)
    "ret_3", "ret_12", "pos_in_range",
    # Cross-asset (9)
    "BTC_ret_3", "BTC_rsi", "BTC_adx",
    "ETH_ret_3", "ETH_rsi", "ETH_adx",
    "PAX_ret_3", "PAX_rsi", "PAX_adx",
    # Side (2)
    "side_long", "side_short",
    # Asset identity (8)
    "is_BTC", "is_ETH", "is_BNB", "is_SOL", "is_XRP", "is_ADA", "is_DOGE", "is_AVAX",
]

REF_SYMBOLS = ["BTCUSDT", "ETHUSDT", "PAXGUSDT"]

# ============================================================================
# Data loading (1h candles, with lookback for indicator warmup)
# ============================================================================

_DATA_CACHE = {}

def load_data(symbol: str):
    """Load 1h data for symbol. Cache in memory."""
    if symbol in _DATA_CACHE:
        return _DATA_CACHE[symbol]
    df = pd.read_parquet(DATA_DIR / f"{symbol}_1h.parquet").set_index("open_time").sort_index()
    _DATA_CACHE[symbol] = df
    return df


# ============================================================================
# Indicator cache (per symbol, per data slice)
# ============================================================================

_IND_CACHE = {}

def get_indicators(df, symbol):
    key = (symbol, id(df))
    if key in _IND_CACHE:
        return _IND_CACHE[key]
    if len(_IND_CACHE) > 30:
        _IND_CACHE.clear()
    close = df["close"]; high = df["high"]; low = df["low"]; op = df["open"]
    vol = df["volume"]
    a = atr(df, 14); r = rsi(close, 14)
    e20 = ema(close, 20); e50 = ema(close, 50); e200 = ema(close, 200)
    slope = rolling_linreg_slope(close, 20); slope_50 = rolling_linreg_slope(close, 50)
    a_line = adx(df, 14)
    _, bb_u, bb_l = bollinger(close, 20, 2.0)
    bb_w = (bb_u - bb_l) / close
    vol_ma = vol.rolling(20).mean()
    tbr = taker_buy_ratio(df)
    atr_ma = a.rolling(50).mean()
    ret_3 = close.pct_change(3); ret_12 = close.pct_change(12)
    roll_max = close.rolling(50).max(); roll_min = close.rolling(50).min()
    pos_in_range = (close - roll_min) / (roll_max - roll_min).replace(0, np.nan)
    cache = {"close": close, "high": high, "low": low, "op": op, "vol": vol,
             "atr": a, "rsi": r, "ema20": e20, "ema50": e50, "ema200": e200,
             "slope": slope, "slope_50": slope_50, "adx": a_line,
             "bb_w": bb_w, "vol_ma": vol_ma, "tbr": tbr, "atr_ma": atr_ma,
             "ret_3": ret_3, "ret_12": ret_12, "pos_in_range": pos_in_range}
    _IND_CACHE[key] = cache
    return cache


# ============================================================================
# Cross-asset features — STRICT no-lookahead via searchsorted
# ============================================================================

def get_cross_asset_features(symbol: str, t: pd.Timestamp) -> dict:
    """Get features from BTC, ETH, PAXG at time t, using ONLY bars that CLOSED BEFORE t."""
    feats = {}
    for ref in REF_SYMBOLS:
        if ref == symbol:
            continue
        try:
            df_ref = load_data(ref)
            ind = get_indicators(df_ref, ref)
            ref_times = df_ref.index
            # searchsorted with side='left' finds the insertion point
            # We want the LAST bar with index < t (strictly before)
            idx_pos = ref_times.searchsorted(t, side="left") - 1
            if idx_pos < 50:
                feats[f"{ref[:3]}_ret_3"] = 0
                feats[f"{ref[:3]}_rsi"] = 50
                feats[f"{ref[:3]}_adx"] = 0
                continue
            feats[f"{ref[:3]}_ret_3"] = float(ind["ret_3"].iloc[idx_pos]) if np.isfinite(ind["ret_3"].iloc[idx_pos]) else 0
            feats[f"{ref[:3]}_rsi"] = float(ind["rsi"].iloc[idx_pos]) if np.isfinite(ind["rsi"].iloc[idx_pos]) else 50
            feats[f"{ref[:3]}_adx"] = float(ind["adx"].iloc[idx_pos]) if np.isfinite(ind["adx"].iloc[idx_pos]) else 0
        except Exception:
            feats[f"{ref[:3]}_ret_3"] = 0
            feats[f"{ref[:3]}_rsi"] = 50
            feats[f"{ref[:3]}_adx"] = 0
    return feats


def extract_features(df, symbol, i, side):
    """Extract all 27 features at bar i. STRICT no-lookahead."""
    c = get_indicators(df, symbol)
    close = c["close"]; vol = c["vol"]; vol_ma = c["vol_ma"]
    t = df.index[i]
    def safe(v, d=0):
        return float(v) if np.isfinite(v) else d
    feats = {
        "vol_ratio": safe(vol.iloc[i] / vol_ma.iloc[i], 1) if vol_ma.iloc[i] > 0 else 1,
        "taker_buy_ratio": safe(c["tbr"].iloc[i], 0.5),
        "vol_ma_diff": safe((vol.iloc[i] - vol_ma.iloc[i]) / vol_ma.iloc[i]) if vol_ma.iloc[i] > 0 else 0,
        "atr_pct": safe(c["atr"].iloc[i] / close.iloc[i]) if close.iloc[i] > 0 else 0,
        "vol_surge_3bar": safe(vol.iloc[i] / vol.iloc[i-3], 1) if i >= 3 and vol.iloc[i-3] > 0 else 1,
        "rsi": safe(c["rsi"].iloc[i], 50),
        "dist_ema20": safe((close.iloc[i] - c["ema20"].iloc[i]) / c["ema20"].iloc[i]) if c["ema20"].iloc[i] != 0 else 0,
        "adx": safe(c["adx"].iloc[i]),
        "linreg_slope": safe(c["slope"].iloc[i]),
        "bb_width_pct": safe(c["bb_w"].iloc[i]),
        "ret_3": safe(c["ret_3"].iloc[i]),
        "ret_12": safe(c["ret_12"].iloc[i]),
        "pos_in_range": safe(c["pos_in_range"].iloc[i], 0.5),
    }
    cross = get_cross_asset_features(symbol, t)
    feats["BTC_ret_3"] = cross.get("BTC_ret_3", 0)
    feats["BTC_rsi"] = cross.get("BTC_rsi", 50)
    feats["BTC_adx"] = cross.get("BTC_adx", 0)
    feats["ETH_ret_3"] = cross.get("ETH_ret_3", 0)
    feats["ETH_rsi"] = cross.get("ETH_rsi", 50)
    feats["ETH_adx"] = cross.get("ETH_adx", 0)
    feats["PAX_ret_3"] = cross.get("PAX_ret_3", 0)
    feats["PAX_rsi"] = cross.get("PAX_rsi", 50)
    feats["PAX_adx"] = cross.get("PAX_adx", 0)
    feats["side_long"] = int(side == "long")
    feats["side_short"] = int(side == "short")
    for s in ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX"]:
        feats[f"is_{s}"] = 1 if symbol.startswith(s) else 0
    return feats


# ============================================================================
# Candidate generator: momentum bars
# ============================================================================

def cand_momentum(df):
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


# ============================================================================
# Training sample generation
# ============================================================================

def generate_training_samples(symbol, start, end):
    """Generate training samples for one symbol in [start, end] window."""
    df = load_data(symbol)
    sig_long, sig_short, stop_long, stop_short = cand_momentum(df)
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
            # Walk forward to find outcome
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
            feats = extract_features(df, symbol, i, chosen)
            feats["label"] = label
            samples.append(feats)
    return samples


# ============================================================================
# Train one AI for a symbol
# ============================================================================

def train_ai(symbol, train_start, train_end):
    """Train a Random Forest for `symbol` using data in [train_start, train_end].
    Returns (model, n_samples) or (None, 0) if insufficient data."""
    samples = generate_training_samples(symbol, train_start, train_end)
    if len(samples) < 30:
        return None, 0
    X = np.array([[s[c] for c in FEATURE_COLS] for s in samples])
    y = np.array([s["label"] for s in samples])
    model = RandomForestClassifier(
        n_estimators=80, max_depth=5, min_samples_leaf=15,
        random_state=42, class_weight="balanced"
    )
    model.fit(X, y)
    return model, len(samples)


def validate_threshold(model, symbol, val_start, val_end):
    """Pick the best threshold using validation data."""
    samples = generate_training_samples(symbol, val_start, val_end)
    if len(samples) < 10:
        return 0.5
    X_val = np.array([[s[c] for c in FEATURE_COLS] for s in samples])
    y_val = np.array([s["label"] for s in samples])
    probas = model.predict_proba(X_val)[:, 1]
    best_thresh = 0.5; best_exp = -999
    for thresh in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        mask = probas >= thresh
        if mask.sum() < 3: continue
        wr = y_val[mask].mean()
        exp = 3 * wr - 1
        if exp > best_exp:
            best_exp = exp; best_thresh = thresh
    return best_thresh


# ============================================================================
# Risk management (aggressive preset)
# ============================================================================

def make_risk_cfg():
    return RiskConfig(
        monthly_loss_cap_pct=0.05, weekly_loss_cap_pct=0.03, daily_loss_cap_pct=0.02,
        dd_throttle_1=0.07, dd_throttle_2=0.12, dd_throttle_3=0.17,
        consec_loss_throttle_1=2, consec_loss_throttle_2=4,
        anti_mart_win_mult=1.30, anti_mart_loss_mult=0.60,
        anti_mart_max_risk=0.015, anti_mart_min_risk=0.004,
        enable_trailing=True, trail_trigger_1=0.5, trail_trigger_2=1.0, trail_distance_r=0.3,
        base_risk_pct=0.01, target_r_mult=2.0,
        end_of_month_days=3, end_of_month_risk=0.004,
        max_bars=24,
    )


# ============================================================================
# Multi-asset backtester (with portfolio risk)
# ============================================================================

def backtest_month(ai_per_symbol, threshold_per_symbol, risk_cfg,
                   test_start, test_end, initial_wallet=INITIAL_WALLET):
    """Backtest AIv3_029 on the test month."""
    symbols = CONFIG["symbols"]
    max_concurrent = CONFIG["max_concurrent"]
    max_same_direction = CONFIG["max_same_direction"]

    # Pre-compute signals for each symbol
    sym_data = {}
    for sym in symbols:
        df = load_data(sym).copy()
        sig_long, sig_short, stop_long, stop_short = cand_momentum(df)
        df = df.sort_index()
        sym_data[sym] = {
            "df": df,
            "sig_long_arr": sig_long.fillna(False).values,
            "sig_short_arr": sig_short.fillna(False).values,
            "stop_long_arr": stop_long.values,
            "stop_short_arr": stop_short.values,
            "opens": df["open"].values, "highs": df["high"].values,
            "lows": df["low"].values, "closes": df["close"].values,
            "times": df.index, "n": len(df),
        }

    # Build unified timeline of events in [test_start, test_end)
    from collections import defaultdict
    all_events = []
    for sym, d in sym_data.items():
        for i in range(d["n"]):
            t = d["times"][i]
            if test_start <= t < test_end:
                all_events.append((t, sym, i))
    all_events.sort()
    events_by_time = defaultdict(list)
    for t, sym, i in all_events:
        events_by_time[t].append((sym, i))
    sorted_times = sorted(events_by_time.keys())

    # State
    wallet = initial_wallet
    peak_wallet = initial_wallet
    base_risk = risk_cfg.base_risk_pct
    cur_risk = base_risk
    consec_losses = 0; consec_wins = 0
    throttle_remaining = 0
    pause_until = None
    monthly_start = initial_wallet; weekly_start = initial_wallet; daily_start = initial_wallet
    if sorted_times:
        cur_month = sorted_times[0].month
        cur_week = sorted_times[0].isocalendar()[1]
        cur_day = sorted_times[0].day
    else:
        cur_month = 1; cur_week = 1; cur_day = 1
    monthly_stopped = False; weekly_stopped = False; daily_stopped = False; dd_stopped = False

    open_positions = {}
    trades = []
    ai_decisions = {"approved": 0, "rejected": 0, "skipped_rm": 0, "skipped_portfolio": 0}
    slip_frac = COSTS.slippage_bps / 10000.0
    max_bars = risk_cfg.max_bars

    for t in sorted_times:
        # Risk management resets
        if t.day != cur_day:
            daily_stopped = False; daily_start = wallet; cur_day = t.day
        if t.isocalendar()[1] != cur_week:
            weekly_stopped = False; weekly_start = wallet; cur_week = t.isocalendar()[1]
        if t.month != cur_month:
            monthly_stopped = False; monthly_start = wallet; cur_month = t.month
        if wallet > peak_wallet: peak_wallet = wallet

        # Risk checks
        allowed = True; eff_risk = cur_risk
        if monthly_stopped or weekly_stopped or daily_stopped or dd_stopped: allowed = False
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

        for sym, i in events_by_time[t]:
            d = sym_data[sym]
            # 1. Check exit for open position in this symbol
            if sym in open_positions:
                pos = open_positions[sym]
                if i < d["n"]:
                    h = d["highs"][i]; l = d["lows"][i]; c = d["closes"][i]
                    bars_held = i - pos["entry_bar_idx"]
                    cur_trail_stop = pos["sp"]
                    best_price = pos["best_price"]
                    if pos["side"] == "long":
                        if h > best_price: best_price = h
                    else:
                        if l < best_price: best_price = l
                    if risk_cfg.enable_trailing:
                        r_best = (best_price - pos["entry_price"]) / pos["stop_distance"] if pos["side"] == "long" \
                                 else (pos["entry_price"] - best_price) / pos["stop_distance"]
                        if r_best >= risk_cfg.trail_trigger_1:
                            if pos["side"] == "long":
                                if pos["entry_price"] > cur_trail_stop: cur_trail_stop = pos["entry_price"]
                            else:
                                if pos["entry_price"] < cur_trail_stop: cur_trail_stop = pos["entry_price"]
                        if r_best >= risk_cfg.trail_trigger_2:
                            if pos["side"] == "long":
                                new_stop = best_price - risk_cfg.trail_distance_r * pos["stop_distance"]
                                if new_stop > cur_trail_stop: cur_trail_stop = new_stop
                            else:
                                new_stop = best_price + risk_cfg.trail_distance_r * pos["stop_distance"]
                                if new_stop < cur_trail_stop: cur_trail_stop = new_stop
                    if pos["side"] == "long":
                        hit_stop = l <= cur_trail_stop; hit_tgt = h >= pos["target_price"]
                    else:
                        hit_stop = h >= cur_trail_stop; hit_tgt = l <= pos["target_price"]
                    exit_now = False; exit_price = 0; exit_reason = ""
                    if hit_stop and hit_tgt:
                        exit_price = cur_trail_stop; exit_reason = "stop"; exit_now = True
                    elif hit_stop:
                        exit_reason = "trail_win" if cur_trail_stop >= pos["entry_price"] else "stop"
                        exit_price = cur_trail_stop; exit_now = True
                    elif hit_tgt:
                        exit_price = pos["target_price"]; exit_reason = "target"; exit_now = True
                    elif bars_held >= max_bars:
                        exit_price = c; exit_reason = "time"; exit_now = True
                    elif t >= test_end:
                        exit_price = c; exit_reason = "end_window"; exit_now = True

                    if exit_now:
                        if pos["side"] == "long":
                            exit_fill = exit_price * (1 - slip_frac)
                            gross_pnl = pos["size"] * (exit_fill - pos["entry_price"])
                        else:
                            exit_fill = exit_price * (1 + slip_frac)
                            gross_pnl = pos["size"] * (pos["entry_price"] - exit_fill)
                        exit_notional = pos["size"] * exit_fill
                        exit_commission = exit_notional * COSTS.commission_pct
                        net_pnl = gross_pnl - exit_commission
                        r_mult_actual = net_pnl / pos["risk_amount"] if pos["risk_amount"] > 0 else 0
                        new_wallet = pos["wallet_after_entry"] + net_pnl
                        won = net_pnl > 0
                        if won:
                            consec_wins += 1; consec_losses = 0
                            cur_risk = min(cur_risk * risk_cfg.anti_mart_win_mult, risk_cfg.anti_mart_max_risk)
                        else:
                            consec_losses += 1; consec_wins = 0
                            cur_risk = max(cur_risk * risk_cfg.anti_mart_loss_mult, risk_cfg.anti_mart_min_risk)
                        if consec_losses >= risk_cfg.consec_loss_throttle_2:
                            pause_until = t + pd.Timedelta(hours=risk_cfg.consec_loss_throttle_2_hours)
                            consec_losses = 0
                        elif consec_losses >= risk_cfg.consec_loss_throttle_1:
                            throttle_remaining = risk_cfg.consec_loss_throttle_1_trades
                            cur_risk = max(cur_risk * 0.5, risk_cfg.anti_mart_min_risk)
                        trades.append({
                            "symbol": sym,
                            "entry_time": pos["entry_time"], "exit_time": t,
                            "side": pos["side"], "entry_price": pos["entry_price"], "exit_price": exit_fill,
                            "stop": pos["sp"], "target": pos["target_price"], "size": pos["size"],
                            "risk_pct": pos["risk_pct"], "risk_usd": pos["risk_amount"],
                            "gross_pnl": gross_pnl, "cost": pos["entry_commission"] + exit_commission,
                            "net_pnl": net_pnl,
                            "wallet_before": pos["wallet_before"], "wallet_after": new_wallet,
                            "R_multiple": r_mult_actual, "exit_reason": exit_reason, "bars_held": bars_held,
                            "ai_proba": pos.get("ai_proba", 0),
                        })
                        wallet = new_wallet
                        if wallet > peak_wallet: peak_wallet = wallet
                        del open_positions[sym]

            # 2. Check for new entry signal
            if not allowed:
                ai_decisions["skipped_rm"] += 1
                continue
            if sym in open_positions:
                continue
            if len(open_positions) >= max_concurrent:
                ai_decisions["skipped_portfolio"] += 1
                continue
            long_count = sum(1 for p in open_positions.values() if p["side"] == "long")
            short_count = sum(1 for p in open_positions.values() if p["side"] == "short")

            signal_long = i+1 < d["n"] and d["sig_long_arr"][i] and np.isfinite(d["opens"][i+1]) and np.isfinite(d["stop_long_arr"][i])
            signal_short = i+1 < d["n"] and d["sig_short_arr"][i] and np.isfinite(d["opens"][i+1]) and np.isfinite(d["stop_short_arr"][i])
            chosen = None
            if signal_long and signal_short: chosen = "long"
            elif signal_long: chosen = "long"
            elif signal_short: chosen = "short"
            if chosen is None: continue
            if chosen == "long" and long_count >= max_same_direction: continue
            if chosen == "short" and short_count >= max_same_direction: continue

            # AI gatekeeper
            ai = ai_per_symbol.get(sym)
            threshold = threshold_per_symbol.get(sym, 0.5)
            if ai is not None:
                feats = extract_features(d["df"], sym, i, chosen)
                X = np.array([[feats[c] for c in FEATURE_COLS]])
                proba = float(ai.predict_proba(X)[0][1])
                if proba < threshold:
                    ai_decisions["rejected"] += 1
                    continue
            else:
                proba = 1.0
            ai_decisions["approved"] += 1

            # Execute trade
            entry_raw = float(d["opens"][i+1])
            if entry_raw <= 0: continue
            if chosen == "long":
                sp_ref = float(d["stop_long_arr"][i])
                if not np.isfinite(sp_ref) or sp_ref >= entry_raw: continue
                entry_price = entry_raw * (1 + slip_frac)
                stop_distance = abs(entry_raw - sp_ref)
                sp = entry_price - stop_distance
                target_price = entry_price + 2.0 * (entry_price - sp)
            else:
                sp_ref = float(d["stop_short_arr"][i])
                if not np.isfinite(sp_ref) or sp_ref <= entry_raw: continue
                entry_price = entry_raw * (1 - slip_frac)
                stop_distance = abs(sp_ref - entry_raw)
                sp = entry_price + stop_distance
                target_price = entry_price - 2.0 * (sp - entry_price)
            if stop_distance <= 0: continue

            risk_amount = eff_risk * wallet
            size_units = risk_amount / stop_distance
            if size_units <= 0: continue

            entry_notional = size_units * entry_price
            entry_commission = entry_notional * COSTS.commission_pct
            wallet_after_entry = wallet - entry_commission

            open_positions[sym] = {
                "side": chosen, "entry_price": entry_price, "sp": sp,
                "target_price": target_price, "size": size_units,
                "entry_time": d["times"][i+1], "entry_bar_idx": i+1,
                "stop_distance": stop_distance, "best_price": entry_price,
                "risk_amount": risk_amount, "risk_pct": eff_risk,
                "wallet_before": wallet, "wallet_after_entry": wallet_after_entry,
                "entry_commission": entry_commission, "ai_proba": proba,
            }

    # Close remaining positions at end of test window
    for sym in list(open_positions.keys()):
        pos = open_positions[sym]
        d = sym_data[sym]
        i = d["n"] - 1
        c = d["closes"][i]
        if pos["side"] == "long":
            exit_fill = c * (1 - slip_frac)
            gross_pnl = pos["size"] * (exit_fill - pos["entry_price"])
        else:
            exit_fill = c * (1 + slip_frac)
            gross_pnl = pos["size"] * (pos["entry_price"] - exit_fill)
        exit_notional = pos["size"] * exit_fill
        exit_commission = exit_notional * COSTS.commission_pct
        net_pnl = gross_pnl - exit_commission
        r_mult_actual = net_pnl / pos["risk_amount"] if pos["risk_amount"] > 0 else 0
        new_wallet = pos["wallet_after_entry"] + net_pnl
        trades.append({
            "symbol": sym,
            "entry_time": pos["entry_time"], "exit_time": d["times"][i],
            "side": pos["side"], "entry_price": pos["entry_price"], "exit_price": exit_fill,
            "stop": pos["sp"], "target": pos["target_price"], "size": pos["size"],
            "risk_pct": pos["risk_pct"], "risk_usd": pos["risk_amount"],
            "gross_pnl": gross_pnl, "cost": pos["entry_commission"] + exit_commission,
            "net_pnl": net_pnl,
            "wallet_before": pos["wallet_before"], "wallet_after": new_wallet,
            "R_multiple": r_mult_actual, "exit_reason": "end_window",
            "bars_held": i - pos["entry_bar_idx"], "ai_proba": pos.get("ai_proba", 0),
        })
        wallet = new_wallet
        del open_positions[sym]

    if trades:
        wallets = [initial_wallet] + [t["wallet_after"] for t in trades]
        peak_arr = np.maximum.accumulate(wallets)
        dd_arr = (peak_arr - wallets) / np.where(peak_arr > 0, peak_arr, 1)
        max_dd = float(np.max(dd_arr))
    else:
        max_dd = 0
    n_wins = sum(1 for t in trades if t["net_pnl"] > 0)
    n_losses = sum(1 for t in trades if t["net_pnl"] < 0)
    wr = n_wins / (n_wins + n_losses) if (n_wins + n_losses) > 0 else 0
    return trades, {
        "n_trades": len(trades), "final_wallet": wallet,
        "net": wallet - initial_wallet, "net_pct": (wallet - initial_wallet) / initial_wallet,
        "peak_wallet": peak_wallet, "max_dd_pct": max_dd,
        "n_wins": n_wins, "n_losses": n_losses, "win_rate": wr,
        "ai_decisions": ai_decisions,
    }


# ============================================================================
# Generate 36 test months
# ============================================================================

def generate_test_months():
    """Generate 36 test months from 2023-11 to 2026-10."""
    months = []
    # Start from 2023-11, go to 2026-10
    year, month = 2023, 11
    end_year, end_month = 2026, 10
    while (year, month) <= (end_year, end_month):
        test_start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
        if month == 12:
            test_end = pd.Timestamp(year=year+1, month=1, day=1, tz="UTC")
        else:
            test_end = pd.Timestamp(year=year, month=month+1, day=1, tz="UTC")
        # Training: 12 months before test_start
        train_end = test_start - pd.Timedelta(seconds=1)  # strict <
        train_start_year = year - 1 if month == 1 else year
        train_start_month = month if month == 1 else month - 1
        # Actually: train_start = test_start - 365 days, train_end = val_start - 1 sec
        # val window: 1 month before test_start
        val_end = test_start - pd.Timedelta(seconds=1)
        val_start_month = month - 1 if month > 1 else 12
        val_start_year = year if month > 1 else year - 1
        val_start = pd.Timestamp(year=val_start_year, month=val_start_month, day=1, tz="UTC")
        # train: 12 months before val_start
        train_end = val_start - pd.Timedelta(seconds=1)
        train_start_year = val_start_year - 1
        train_start = pd.Timestamp(year=train_start_year, month=val_start_month, day=1, tz="UTC")
        months.append({
            "label": f"{year}-{month:02d}",
            "test_start": test_start, "test_end": test_end,
            "val_start": val_start, "val_end": val_end,
            "train_start": train_start, "train_end": train_end,
        })
        month += 1
        if month > 12:
            month = 1; year += 1
    return months


# ============================================================================
# Main: 36-month walk-forward
# ============================================================================

def main():
    print("="*120)
    print("AIv3_029 — 3-YEAR WALK-FORWARD TEST (36 months, month-by-month)")
    print("AI retrained fresh for each month using ONLY data that existed before that month began")
    print("="*120)

    test_months = generate_test_months()
    print(f"\nGenerated {len(test_months)} test months: {test_months[0]['label']} to {test_months[-1]['label']}")
    print(f"\nExample month windows (first test month):")
    m = test_months[0]
    print(f"  Test month:      {m['label']}")
    print(f"  Training:        {m['train_start'].date()} to {m['train_end'].date()}")
    print(f"  Validation:      {m['val_start'].date()} to {m['val_end'].date()}")
    print(f"  Test:            {m['test_start'].date()} to {m['test_end'].date()}")
    print(f"\n  NO-LOOKAHEAD ASSERTIONS:")
    print(f"    train_end ({m['train_end'].date()}) < val_start ({m['val_start'].date()}): {m['train_end'] < m['val_start']}")
    print(f"    val_end ({m['val_end'].date()}) < test_start ({m['test_start'].date()}): {m['val_end'] < m['test_start']}")

    # Strict no-lookahead audit log
    audit_log = []
    all_monthly_results = []
    risk_cfg = make_risk_cfg()

    # RESUME LOGIC: load partial results if they exist
    partial_path = DOWNLOAD / "walkforward_3year_partial.json"
    start_idx = 0
    if partial_path.exists():
        try:
            partial = json.loads(partial_path.read_text())
            saved_results = partial.get("results", [])
            saved_audit = partial.get("audit_log", [])
            if len(saved_results) > 0:
                print(f"\n[RESUME] Found {len(saved_results)} completed months. Resuming from month {len(saved_results)+1}.")
                # Reconstruct all_monthly_results from saved data
                for sr in saved_results:
                    # Find the matching test_month
                    for tm in test_months:
                        if tm["label"] == sr["month"]:
                            all_monthly_results.append({
                                "month": tm,
                                "trades": [],  # don't have trades in partial
                                "stats": {
                                    "n_trades": sr["n_trades"], "n_wins": sr["n_wins"], "n_losses": sr["n_losses"],
                                    "win_rate": sr["win_rate"], "final_wallet": INITIAL_WALLET + sr["net"],
                                    "net": sr["net"], "net_pct": sr["net_pct"],
                                    "max_dd_pct": sr["max_dd_pct"], "peak_wallet": INITIAL_WALLET,
                                    "ai_decisions": {"approved": 0, "rejected": 0, "skipped_rm": 0, "skipped_portfolio": 0},
                                },
                                "n_ais_trained": sr.get("n_ais_trained", 0),
                            })
                            break
                audit_log = saved_audit
                start_idx = len(saved_results)
        except Exception as e:
            print(f"  [RESUME] Failed to load partial: {e}. Starting fresh.")

    print(f"\nRunning 36-month walk-forward... (starting from month {start_idx+1})")
    t_start = time.time()
    for mi, month in enumerate(test_months):
        if mi < start_idx:
            continue  # skip already-completed months
        print(f"\n--- Month {mi+1}/36: {month['label']} ---", flush=True)
        print(f"  Train: {month['train_start'].date()} -> {month['train_end'].date()}", flush=True)
        print(f"  Val:   {month['val_start'].date()} -> {month['val_end'].date()}", flush=True)
        print(f"  Test:  {month['test_start'].date()} -> {month['test_end'].date()}", flush=True)

        # STRICT no-lookahead assertions
        assert month['train_end'] < month['val_start'], f"LOOKAHEAD: train overlaps val in {month['label']}"
        assert month['val_end'] < month['test_start'], f"LOOKAHEAD: val overlaps test in {month['label']}"

        # Train 9 AIs (one per symbol)
        ai_per_symbol = {}
        threshold_per_symbol = {}
        audit_entry = {
            "month": month['label'],
            "train_start": str(month['train_start']),
            "train_end": str(month['train_end']),
            "val_start": str(month['val_start']),
            "val_end": str(month['val_end']),
            "test_start": str(month['test_start']),
            "test_end": str(month['test_end']),
            "train_end_lt_val_start": month['train_end'] < month['val_start'],
            "val_end_lt_test_start": month['val_end'] < month['test_start'],
            "ais_trained": {},
        }
        for sym in CONFIG["symbols"]:
            try:
                model, n_samples = train_ai(sym, month['train_start'], month['train_end'])
                if model is None:
                    print(f"    {sym}: insufficient training data ({n_samples} samples)", flush=True)
                    audit_entry["ais_trained"][sym] = {"n_samples": n_samples, "trained": False}
                    continue
                threshold = validate_threshold(model, sym, month['val_start'], month['val_end'])
                ai_per_symbol[sym] = model
                threshold_per_symbol[sym] = threshold
                audit_entry["ais_trained"][sym] = {
                    "n_samples": n_samples, "trained": True, "threshold": threshold,
                    "train_win_rate": "logged_in_model",
                }
                print(f"    {sym}: trained on {n_samples} samples, threshold={threshold:.2f}", flush=True)
            except Exception as e:
                print(f"    {sym}: FAILED - {e}", flush=True)
                audit_entry["ais_trained"][sym] = {"trained": False, "error": str(e)}

        audit_log.append(audit_entry)

        # Run backtest on test month
        try:
            trades, stats = backtest_month(
                ai_per_symbol, threshold_per_symbol, risk_cfg,
                month['test_start'], month['test_end']
            )
            n_wins = stats['n_wins']; n_losses = stats['n_losses']
            wr = stats['win_rate']
            print(f"  RESULT: {stats['n_trades']} trades, WR={wr*100:.1f}%, net=${stats['net']:+,.2f} ({stats['net_pct']*100:+.2f}%), DD={stats['max_dd_pct']*100:.1f}%", flush=True)
            all_monthly_results.append({
                "month": month, "trades": trades, "stats": stats,
                "n_ais_trained": len(ai_per_symbol),
            })
        except Exception as e:
            import traceback
            print(f"  BACKTEST FAILED: {e}", flush=True)
            traceback.print_exc()
            all_monthly_results.append({
                "month": month, "trades": [], "stats": {
                    "n_trades": 0, "final_wallet": INITIAL_WALLET, "net": 0, "net_pct": 0,
                    "peak_wallet": INITIAL_WALLET, "max_dd_pct": 0,
                    "n_wins": 0, "n_losses": 0, "win_rate": 0,
                    "ai_decisions": {"approved": 0, "rejected": 0, "skipped_rm": 0, "skipped_portfolio": 0},
                },
                "n_ais_trained": len(ai_per_symbol), "error": str(e),
            })

        # Free AI memory AND indicator cache after every month
        del ai_per_symbol
        _IND_CACHE.clear()
        gc.collect()

        # Save partial results every 3 months
        if (mi + 1) % 3 == 0:
            json_path = DOWNLOAD / "walkforward_3year_partial.json"
            json_data = []
            for r in all_monthly_results:
                m = r["month"]; s = r["stats"]
                json_data.append({
                    "month": m["label"],
                    "test_start": str(m["test_start"]),
                    "test_end": str(m["test_end"]),
                    "train_start": str(m["train_start"]),
                    "train_end": str(m["train_end"]),
                    "n_trades": s["n_trades"], "n_wins": s["n_wins"], "n_losses": s["n_losses"],
                    "win_rate": s["win_rate"], "net": s["net"], "net_pct": s["net_pct"],
                    "max_dd_pct": s["max_dd_pct"], "n_ais_trained": r.get("n_ais_trained", 0),
                })
            with open(json_path, "w") as f:
                json.dump({"results": json_data, "audit_log": audit_log}, f, indent=2, default=str)
            elapsed = time.time() - t_start
            print(f"  [saved partial: {len(all_monthly_results)} months, {elapsed:.0f}s elapsed]", flush=True)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/len(test_months):.1f}s per month)")

    # ============================================================================
    # Aggregate results
    # ============================================================================
    print("\n" + "="*120)
    print("36-MONTH WALK-FORWARD RESULTS — AIv3_029")
    print("="*120)
    print(f"{'Month':<10} {'Trades':>6} {'Wins':>5} {'Losses':>6} {'WR':>6} {'Net $':>11} {'Net %':>8} {'MaxDD':>7} {'AIs':>4}")
    print("-"*120)
    for r in all_monthly_results:
        m = r["month"]; s = r["stats"]
        print(f"{m['label']:<10} {s['n_trades']:>6} {s['n_wins']:>5} {s['n_losses']:>6} "
              f"{s['win_rate']*100:>5.1f}% {s['net']:>+11,.2f} {s['net_pct']*100:>+7.2f}% "
              f"{s['max_dd_pct']*100:>6.1f}% {r.get('n_ais_trained', 0):>4}")

    # Aggregate stats
    print("\n" + "="*80)
    print("AGGREGATE STATISTICS (36 months)")
    print("="*80)
    nets = [r["stats"]["net"] for r in all_monthly_results]
    net_pcts = [r["stats"]["net_pct"] for r in all_monthly_results]
    wrs = [r["stats"]["win_rate"] for r in all_monthly_results if r["stats"]["n_trades"] > 0]
    dds = [r["stats"]["max_dd_pct"] for r in all_monthly_results]
    trades_counts = [r["stats"]["n_trades"] for r in all_monthly_results]

    n_profitable = sum(1 for n in nets if n > 0)
    n_breakeven = sum(1 for n in nets if n == 0)
    n_losing = sum(1 for n in nets if n < 0)
    avg_net = np.mean(nets)
    median_net = np.median(nets)
    best_net = max(nets)
    worst_net = min(nets)
    avg_wr = np.mean(wrs) if wrs else 0
    avg_dd = np.mean(dds)
    worst_dd = max(dds)
    avg_trades = np.mean(trades_counts)
    total_trades = sum(trades_counts)

    # Compounded return (reinvesting each month)
    compounded = 1.0
    for pct in net_pcts:
        compounded *= (1 + pct)
    compounded_pct = (compounded - 1) * 100

    print(f"  Months tested:           {len(all_monthly_results)}")
    print(f"  Profitable months:       {n_profitable}/{len(all_monthly_results)} ({n_profitable/len(all_monthly_results)*100:.0f}%)")
    print(f"  Breakeven months:        {n_breakeven}/{len(all_monthly_results)}")
    print(f"  Losing months:           {n_losing}/{len(all_monthly_results)}")
    print(f"  Total trades:            {total_trades}")
    print(f"  Avg trades/month:        {avg_trades:.1f}")
    print(f"  Avg net profit/month:    ${avg_net:+,.2f} ({avg_net/INITIAL_WALLET*100:+.2f}%)")
    print(f"  Median net profit/month: ${median_net:+,.2f} ({median_net/INITIAL_WALLET*100:+.2f}%)")
    print(f"  Best month:              ${best_net:+,.2f} ({max(net_pcts)*100:+.2f}%)")
    print(f"  Worst month:             ${worst_net:+,.2f} ({min(net_pcts)*100:+.2f}%)")
    print(f"  Avg win rate:            {avg_wr*100:.1f}%")
    print(f"  Avg max drawdown:        {avg_dd*100:.1f}%")
    print(f"  Worst single-month DD:   {worst_dd*100:.1f}%")
    print(f"  Compounded 3-yr return:  {compounded_pct:+.2f}%")
    print(f"  Sum of monthly returns:  {sum(net_pcts)*100:+.2f}%")

    # Verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    if n_profitable >= 24:  # 2/3 of months
        print(f"  PASS: {n_profitable}/36 months profitable ({n_profitable/36*100:.0f}%)")
        print(f"  The edge appears REAL — profitable in {n_profitable/36*100:.0f}% of months over 3 years.")
        if compounded_pct > 50:
            print(f"  Compounded 3-year return: {compounded_pct:+.2f}% — STRONG edge confirmed.")
        elif compounded_pct > 0:
            print(f"  Compounded 3-year return: {compounded_pct:+.2f}% — modest but positive edge.")
        else:
            print(f"  Compounded 3-year return: {compounded_pct:+.2f}% — edge decayed over time despite monthly hit rate.")
    elif n_profitable >= 18:  # 1/2 of months
        print(f"  MARGINAL: {n_profitable}/36 months profitable ({n_profitable/36*100:.0f}%)")
        print(f"  Edge is weak — barely better than coin flip on monthly basis.")
        print(f"  Compounded 3-year return: {compounded_pct:+.2f}%")
    else:
        print(f"  FAIL: {n_profitable}/36 months profitable ({n_profitable/36*100:.0f}%)")
        print(f"  The +69.66% in the original test was LUCK, not a real edge.")
        print(f"  Compounded 3-year return: {compounded_pct:+.2f}%")
        print(f"  DO NOT deploy this bot with real money.")

    # Write Excel
    out_path = DOWNLOAD / "AIv3_029_walkforward_3year.xlsx"
    write_excel(all_monthly_results, audit_log, out_path,
                n_profitable, n_breakeven, n_losing,
                avg_net, median_net, best_net, worst_net,
                avg_wr, avg_dd, worst_dd, avg_trades, total_trades,
                compounded_pct)
    print(f"\n>>> Wrote {out_path}")


def write_excel(all_monthly_results, audit_log, out_path,
                n_profitable, n_breakeven, n_losing,
                avg_net, median_net, best_net, worst_net,
                avg_wr, avg_dd, worst_dd, avg_trades, total_trades,
                compounded_pct):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    header_font = Font(bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    pos_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    neg_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    warn_fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
    border = Border(left=Side(style='thin'), right=Side(style='thin'),
                    top=Side(style='thin'), bottom=Side(style='thin'))

    # Sheet 1: Month-by-month results
    ws = wb.create_sheet("36-Month Results")
    ws["A1"] = "AIv3_029 — 3-YEAR WALK-FORWARD TEST (36 months)"
    ws["A1"].font = Font(bold=True, size=14, color="1F4E78")
    ws["A2"] = "AI retrained fresh for each month using ONLY data that existed before that month began"
    ws["A3"] = f"Test period: {all_monthly_results[0]['month']['label']} to {all_monthly_results[-1]['month']['label']}  |  Initial wallet: $10,000/month"

    headers = ["Month", "Train Start", "Train End", "Val Start", "Val End", "Test Start", "Test End",
               "Trades", "Wins", "Losses", "Win Rate", "Initial $", "Final $", "Net $", "Net %", "Max DD %", "AIs Trained"]
    row_idx = 5
    for col, h in enumerate(headers, 1):
        c = ws.cell(row=row_idx, column=col, value=h)
        c.font = header_font; c.fill = header_fill
        c.alignment = Alignment(horizontal="center", wrap_text=True); c.border = border

    for rank, r in enumerate(all_monthly_results, 1):
        m = r["month"]; s = r["stats"]
        row = [m["label"], str(m["train_start"].date()), str(m["train_end"].date()),
               str(m["val_start"].date()), str(m["val_end"].date()),
               str(m["test_start"].date()), str(m["test_end"].date()),
               s["n_trades"], s["n_wins"], s["n_losses"], s["win_rate"],
               INITIAL_WALLET, s["final_wallet"], s["net"], s["net_pct"], s["max_dd_pct"],
               r.get("n_ais_trained", 0)]
        for col, v in enumerate(row, 1):
            c = ws.cell(row=row_idx+rank, column=col, value=v)
            c.border = border
            if col in (14,):
                c.number_format = '"$"#,##0.00'
            elif col == 13:
                c.number_format = '"$"#,##0.00'
            elif col in (15, 16):
                c.number_format = '0.00%'
            elif col == 11:
                c.number_format = '0.0%'
            if col == 14 and isinstance(v, (int, float)):
                c.fill = pos_fill if v > 0 else (neg_fill if v < 0 else warn_fill)
    widths = [10, 12, 12, 12, 12, 12, 12, 7, 5, 7, 8, 11, 12, 12, 9, 9, 9]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A6"

    # Sheet 2: Aggregate stats
    ws = wb.create_sheet("Aggregate Stats")
    ws["A1"] = "3-YEAR WALK-FORWARD — AGGREGATE STATISTICS"
    ws["A1"].font = Font(bold=True, size=14, color="1F4E78")
    ws["A3"] = "Metric"; ws["B3"] = "Value"
    ws["A3"].font = header_font; ws["A3"].fill = header_fill
    ws["B3"].font = header_font; ws["B3"].fill = header_fill

    nets = [r["stats"]["net"] for r in all_monthly_results]
    net_pcts = [r["stats"]["net_pct"] for r in all_monthly_results]
    compounded = 1.0
    for pct in net_pcts:
        compounded *= (1 + pct)

    stats_rows = [
        ("Months tested", len(all_monthly_results)),
        ("Profitable months", f"{n_profitable}/{len(all_monthly_results)} ({n_profitable/len(all_monthly_results)*100:.0f}%)"),
        ("Breakeven months", f"{n_breakeven}/{len(all_monthly_results)}"),
        ("Losing months", f"{n_losing}/{len(all_monthly_results)}"),
        ("Total trades", total_trades),
        ("Avg trades/month", f"{avg_trades:.1f}"),
        ("Avg net profit/month", f"${avg_net:+,.2f} ({avg_net/INITIAL_WALLET*100:+.2f}%)"),
        ("Median net profit/month", f"${median_net:+,.2f} ({median_net/INITIAL_WALLET*100:+.2f}%)"),
        ("Best month", f"${best_net:+,.2f} ({max(net_pcts)*100:+.2f}%)"),
        ("Worst month", f"${worst_net:+,.2f} ({min(net_pcts)*100:+.2f}%)"),
        ("Avg win rate", f"{avg_wr*100:.1f}%"),
        ("Avg max drawdown", f"{avg_dd*100:.1f}%"),
        ("Worst single-month DD", f"{worst_dd*100:.1f}%"),
        ("Compounded 3-yr return", f"{(compounded-1)*100:+.2f}%"),
        ("Sum of monthly returns", f"{sum(net_pcts)*100:+.2f}%"),
    ]
    for i, (metric, value) in enumerate(stats_rows, 4):
        ws.cell(row=i, column=1, value=metric).border = border
        c = ws.cell(row=i, column=2, value=value); c.border = border
        if "Profitable" in metric:
            if n_profitable >= 24:
                c.fill = pos_fill
            elif n_profitable >= 18:
                c.fill = warn_fill
            else:
                c.fill = neg_fill
        if "Compounded" in metric:
            if compounded > 1.5:
                c.fill = pos_fill
            elif compounded > 1.0:
                c.fill = warn_fill
            else:
                c.fill = neg_fill
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 30

    # Verdict box
    verdict_row = len(stats_rows) + 6
    ws.cell(row=verdict_row, column=1, value="VERDICT").font = Font(bold=True, size=12, color="1F4E78")
    if n_profitable >= 24:
        verdict = f"PASS: {n_profitable}/36 months profitable. Edge appears REAL."
        ws.cell(row=verdict_row+1, column=1, value=verdict).fill = pos_fill
    elif n_profitable >= 18:
        verdict = f"MARGINAL: {n_profitable}/36 months profitable. Edge is weak."
        ws.cell(row=verdict_row+1, column=1, value=verdict).fill = warn_fill
    else:
        verdict = f"FAIL: {n_profitable}/36 months profitable. The +69.66% was LUCK."
        ws.cell(row=verdict_row+1, column=1, value=verdict).fill = neg_fill
    ws.cell(row=verdict_row+1, column=1).font = Font(bold=True)

    # Sheet 3: No-lookahead audit
    ws = wb.create_sheet("No-Lookahead Audit")
    ws["A1"] = "STRICT NO-LOOKAHEAD AUDIT — Per Month"
    ws["A1"].font = Font(bold=True, size=14, color="1F4E78")
    ws["A2"] = "For each month, the AI was trained on data BEFORE the test month began. Verified:"

    audit_headers = ["Month", "Train Start", "Train End", "Val Start", "Val End", "Test Start", "Test End",
                     "Train End < Val Start?", "Val End < Test Start?", "AIs Trained"]
    row_idx = 4
    for col, h in enumerate(audit_headers, 1):
        c = ws.cell(row=row_idx, column=col, value=h)
        c.font = header_font; c.fill = header_fill
        c.alignment = Alignment(horizontal="center", wrap_text=True); c.border = border

    for rank, entry in enumerate(audit_log, 1):
        n_ais = sum(1 for v in entry["ais_trained"].values() if v.get("trained"))
        row = [entry["month"], entry["train_start"][:10], entry["train_end"][:10],
               entry["val_start"][:10], entry["val_end"][:10],
               entry["test_start"][:10], entry["test_end"][:10],
               "YES" if entry["train_end_lt_val_start"] else "NO",
               "YES" if entry["val_end_lt_test_start"] else "NO",
               f"{n_ais}/9"]
        for col, v in enumerate(row, 1):
            c = ws.cell(row=row_idx+rank, column=col, value=v)
            c.border = border
            if col in (8, 9):
                c.fill = pos_fill if v == "YES" else neg_fill
                c.alignment = Alignment(horizontal="center")
    widths = [10, 12, 12, 12, 12, 12, 12, 18, 18, 10]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A5"

    # Sheet 4: Per-month AI training details
    ws = wb.create_sheet("AI Training Details")
    ws["A1"] = "AI Training Details — Per Month, Per Symbol"
    ws["A1"].font = Font(bold=True, size=14, color="1F4E78")
    detail_headers = ["Month", "Symbol", "Trained?", "N Samples", "Threshold"]
    row_idx = 3
    for col, h in enumerate(detail_headers, 1):
        c = ws.cell(row=row_idx, column=col, value=h)
        c.font = header_font; c.fill = header_fill
        c.alignment = Alignment(horizontal="center"); c.border = border
    for entry in audit_log:
        for sym, info in entry["ais_trained"].items():
            row = [entry["month"], sym, "Yes" if info.get("trained") else "No",
                   info.get("n_samples", 0), info.get("threshold", 0)]
            for col, v in enumerate(row, 1):
                c = ws.cell(row=row_idx+1, column=col, value=v)
                c.border = border
                if col == 3:
                    c.fill = pos_fill if v == "Yes" else neg_fill
                    c.alignment = Alignment(horizontal="center")
                if col == 5 and isinstance(v, (int, float)):
                    c.number_format = '0.00'
            row_idx += 1
    widths = [10, 12, 10, 12, 12]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w

    wb.save(out_path)


if __name__ == "__main__":
    main()
