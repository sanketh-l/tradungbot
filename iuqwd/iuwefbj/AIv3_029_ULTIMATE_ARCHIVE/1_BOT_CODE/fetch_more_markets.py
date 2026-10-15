"""
Fetch additional market data for the multi-asset AI bot expansion.

Symbols to fetch (all Binance USDT spot):
  - BNBUSDT, SOLUSDT, XRPUSDT, ADAUSDT, DOGEUSDT, AVAXUSDT (top alts)
  - PAXGUSDT (PAX Gold — tokenized gold, tracks XAU/USD closely)

Plus existing BTCUSDT, ETHUSDT = 9 markets total.

Timeframes: 1h and 4h (4h resampled from 1h on the fly).
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd
import requests

BASE_URL = "https://api.binance.com/api/v3/klines"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

NEW_SYMBOLS = ["BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "PAXGUSDT"]
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def fetch_klines_paginated(symbol, interval, start_ms, end_ms, session):
    all_rows = []
    cur = start_ms
    while cur < end_ms:
        params = {"symbol": symbol, "interval": interval, "startTime": cur, "endTime": end_ms, "limit": 1000}
        for attempt in range(5):
            try:
                r = session.get(BASE_URL, params=params, timeout=30)
                if r.status_code in (429, 418):
                    time.sleep(2 ** (attempt + 2)); continue
                r.raise_for_status(); break
            except Exception as e:
                time.sleep(2 ** (attempt + 1))
        else:
            raise RuntimeError(f"Failed {symbol} {interval}")
        rows = r.json()
        if not rows: break
        all_rows.extend(rows)
        cur = rows[-1][6] + 1
        time.sleep(0.12)
        if len(rows) < 1000: break
    return all_rows


def build_dataframe(rows):
    df = pd.DataFrame(rows, columns=KLINE_COLS).drop(columns=["ignore"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume","quote_volume","taker_buy_base","taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c]).astype("float64")
    df["trades"] = df["trades"].astype("int64")
    return df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)


def fetch_one(symbol, interval, start_str, session):
    cache_path = DATA_DIR / f"{symbol}_{interval}.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  [cache] {symbol} {interval}: {len(df)} rows  {df['open_time'].min()} → {df['open_time'].max()}")
        return df
    start_dt = pd.Timestamp(start_str, tz="UTC")
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    print(f"  Fetching {symbol} {interval} from {start_dt}")
    rows = fetch_klines_paginated(symbol, interval, start_ms, end_ms, session)
    if not rows: raise RuntimeError(f"No rows for {symbol} {interval}")
    df = build_dataframe(rows)
    df.to_parquet(cache_path, index=False)
    print(f"  [saved] {cache_path} | {len(df)} rows")
    return df


def main():
    session = requests.Session()
    session.headers.update({"User-Agent": "multi-asset-bot-fetch/1.0"})
    # Use 2022-01-01 as start (most alts launched by then; PAXG existed since 2020)
    START = "2022-01-01"
    print(f"=== Fetching 7 new markets at 1h (start={START}) ===")
    for sym in NEW_SYMBOLS:
        try:
            fetch_one(sym, "1h", START, session)
        except Exception as e:
            print(f"  FAILED {sym} 1h: {e}")


if __name__ == "__main__":
    main()
