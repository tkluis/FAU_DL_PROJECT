"""scripts/fetch_yf.py

Standalone Yahoo Finance downloader + cache builder.

Usage:
  python scripts/fetch_yf.py --tickers tickers.txt --start 2018-01-01 --end 2024-12-31

This script caches raw OHLCV to data_cache/yf_<start>_<end>.parquet
so training runs can be repeated without re-downloading.
"""

import argparse
import os
import pandas as pd
import yfinance as yf

def normalize(t: str) -> str:
    return t.strip().replace(".", "-")

def read_tickers(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [normalize(x) for x in f.read().splitlines() if x.strip() and not x.strip().startswith("#")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="tickers.txt")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tickers = read_tickers(args.tickers)
    if not tickers:
        raise ValueError("No tickers found in tickers file.")

    out = args.out or os.path.join("data_cache", f"yf_{args.start}_{args.end}.parquet")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    df = yf.download(
        tickers=tickers,
        start=args.start,
        end=args.end,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=True,
    )
    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.levels[0].tolist()
        if "Open" in level0 or "Close" in level0:
            df = df.swaplevel(0, 1, axis=1).sort_index(axis=1)

    df.to_parquet(out)
    print("Saved:", out, "shape:", df.shape)

if __name__ == "__main__":
    main()
