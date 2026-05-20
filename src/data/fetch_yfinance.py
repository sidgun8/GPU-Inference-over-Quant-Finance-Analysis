from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_yfinance_data(tickers: list[str], start_date: str, end_date: str | None, output_dir: str | Path) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for ticker in tickers:
        target = output_path / f"{ticker}.csv"
        frame = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
        if frame.empty:
            if target.exists() and target.stat().st_size > 0:
                saved[ticker] = target
                continue
            raise RuntimeError(f"No yfinance data returned for ticker: {ticker}")
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [str(col[0]) for col in frame.columns]
        frame.to_csv(target)
        saved[ticker] = target
    return saved
