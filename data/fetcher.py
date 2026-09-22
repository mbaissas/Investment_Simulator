"""
data/fetcher.py
---------------
Fetches historical price data for ETFs and assets using yfinance.
Supports caching to avoid redundant API calls.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta

# Mapping of friendly names to Yahoo Finance tickers
TICKER_MAP = {
    # ETFs & Indices
    "MSCI World":           "IWDA.AS",    # iShares Core MSCI World (Amsterdam)
    "CAC 40":               "^FCHI",
    "S&P 500":              "SPY",
    "Core S&P 500":         "CSPX.L",     # iShares Core S&P 500 (London)
    "S&P 500 Info Tech":    "IYW",        # iShares US Technology ETF
    "MSCI ACWI":            "ACWI",
    "Bourso Monde":         "IWDA.AS",    # Proxy (même indice sous-jacent)
    "Bitcoin":              "BTC-USD",
}

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


def _cache_path(ticker: str, period_years: int) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{ticker.replace('/', '_')}_{period_years}y.csv")


def fetch_historical(
    ticker: str,
    period_years: int = 20,
    use_cache: bool = True,
    cache_ttl_days: int = 1,
) -> pd.Series:
    """
    Returns a Series of monthly log-returns for the given ticker.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g. 'IWDA.AS', 'SPY', 'BTC-USD').
    period_years : int
        How many years of history to fetch.
    use_cache : bool
        Whether to use a local CSV cache.
    cache_ttl_days : int
        How many days before the cache is considered stale.

    Returns
    -------
    pd.Series
        Monthly log-returns, indexed by date.
    """
    cache_file = _cache_path(ticker, period_years)

    # Try cache first
    if use_cache and os.path.exists(cache_file):
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - mtime < timedelta(days=cache_ttl_days):
            prices = pd.read_csv(cache_file, index_col=0, parse_dates=True).squeeze()
            print(f"[cache] {ticker} ({len(prices)} monthly returns)")
            return _to_log_returns(prices)

    # Fetch from Yahoo Finance
    print(f"[fetch] Downloading {ticker} ({period_years} years)...")
    end = datetime.now()
    start = end - timedelta(days=period_years * 365)
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if raw.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Check the symbol.")

    # Resample to monthly (last close of each month)
    prices = raw["Close"].resample("ME").last().dropna()
    prices.to_csv(cache_file)
    print(f"[fetch] {ticker}: {len(prices)} monthly data points ({prices.index[0].date()} → {prices.index[-1].date()})")

    return _to_log_returns(prices)


def _to_log_returns(prices: pd.Series) -> pd.Series:
    """Convert a price series to monthly log-returns."""
    return np.log(prices / prices.shift(1)).dropna()


def fetch_by_name(asset_name: str, period_years: int = 20) -> pd.Series:
    """
    Convenience wrapper: fetch returns using a friendly asset name.

    Parameters
    ----------
    asset_name : str
        A key from TICKER_MAP (e.g. 'MSCI World', 'Bitcoin').

    Returns
    -------
    pd.Series
        Monthly log-returns.
    """
    ticker = TICKER_MAP.get(asset_name)
    if ticker is None:
        raise KeyError(
            f"Unknown asset '{asset_name}'. "
            f"Available: {list(TICKER_MAP.keys())}"
        )
    return fetch_historical(ticker, period_years=period_years)


def fetch_portfolio_returns(portfolio: dict, period_years: int = 20) -> dict[str, pd.Series]:
    """
    Fetch returns for all unique assets in a portfolio config.

    Parameters
    ----------
    portfolio : dict
        Portfolio configuration (see config/portfolio_example.json).

    Returns
    -------
    dict[str, pd.Series]
        Mapping of asset_name → monthly log-returns.
    """
    asset_names = set()
    for envelope in portfolio["envelopes"]:
        for holding in envelope["holdings"]:
            asset_names.add(holding["asset"])

    results = {}
    for name in sorted(asset_names):
        try:
            results[name] = fetch_by_name(name, period_years=period_years)
        except (KeyError, ValueError) as e:
            print(f"[warn] Could not fetch '{name}': {e}. Skipping.")

    return results


if __name__ == "__main__":
    # Quick test
    returns = fetch_by_name("MSCI World", period_years=10)
    print(f"\nMSCI World monthly returns (last 5):\n{returns.tail()}")
    print(f"\nAnnualized mean: {returns.mean() * 12:.2%}")
    print(f"Annualized vol:  {returns.std() * np.sqrt(12):.2%}")
