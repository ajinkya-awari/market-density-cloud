from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

BASE_PERIODS_PER_YEAR = {
    "stock": 252,
    "forex": 252,
    "crypto": 365,
}
TRADING_HOURS_PER_DAY = {
    "stock": 6.5,
    "forex": 24.0,
    "crypto": 24.0,
}
CRYPTO_QUOTES = {
    "AUD",
    "BRL",
    "BTC",
    "BUSD",
    "CAD",
    "DAI",
    "ETH",
    "EUR",
    "GBP",
    "JPY",
    "TRY",
    "USD",
    "USDC",
    "USDT",
}
INTERVAL_PATTERN = re.compile(r"^(?P<count>\d+)(?P<unit>m|h|d|wk|mo)$")


def normalize_symbols(symbols: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    for symbol in symbols:
        value = symbol.strip().upper()
        if value and value not in seen:
            ordered.append(value)
            seen.add(value)

    return ordered


def infer_asset_class(symbol: str) -> str:
    value = symbol.strip().upper()
    if value.endswith("=X"):
        return "forex"

    if "-" in value:
        base, quote = value.rsplit("-", maxsplit=1)
        if base and quote in CRYPTO_QUOTES:
            return "crypto"

    return "stock"


def annualization_factor(asset_class: str, interval: str = "1d") -> float:
    """Return the number of periods per year for the selected interval."""
    if asset_class not in BASE_PERIODS_PER_YEAR:
        raise ValueError(f"Unsupported asset class '{asset_class}'.")

    value = interval.strip().lower()
    match = INTERVAL_PATTERN.fullmatch(value)
    if match is None:
        return float(BASE_PERIODS_PER_YEAR[asset_class])

    count = int(match.group("count"))
    unit = match.group("unit")
    if count < 1:
        raise ValueError("Interval multiplier must be at least 1.")

    base_periods = float(BASE_PERIODS_PER_YEAR[asset_class])
    if unit == "m":
        minutes_per_day = TRADING_HOURS_PER_DAY[asset_class] * 60.0
        return base_periods * (minutes_per_day / count)
    if unit == "h":
        return base_periods * (TRADING_HOURS_PER_DAY[asset_class] / count)
    if unit == "d":
        return base_periods / count
    if unit == "wk":
        trading_days_per_week = 7.0 if asset_class == "crypto" else 5.0
        return base_periods / (trading_days_per_week * count)
    if unit == "mo":
        return 12.0 / count

    return base_periods


def build_asset_metadata(
    symbols: Sequence[str],
    overrides: Mapping[str, str] | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    index = normalize_symbols(symbols)
    metadata = pd.DataFrame(index=index)

    resolved: dict[str, str] = {}
    for symbol in index:
        asset_class = None
        if overrides is not None:
            asset_class = overrides.get(symbol) or overrides.get(symbol.upper())
        asset_class = asset_class or infer_asset_class(symbol)

        if asset_class not in BASE_PERIODS_PER_YEAR:
            raise ValueError(f"Unsupported asset class '{asset_class}' for '{symbol}'.")
        resolved[symbol] = asset_class

    metadata["asset_class"] = pd.Series(resolved)
    metadata["annualization_factor"] = metadata["asset_class"].map(
        lambda asset_class: annualization_factor(asset_class, interval=interval)
    )
    return metadata


def download_prices(
    symbols: Sequence[str],
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    requested = normalize_symbols(symbols)
    if not requested:
        raise ValueError("At least one symbol is required.")

    raw = yf.download(
        tickers=requested,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise ValueError("No data was returned by yfinance.")

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw.xs("Close", axis=1, level=0)
    else:
        if "Close" not in raw.columns:
            raise ValueError("Close prices were not found in the yfinance response.")
        prices = raw[["Close"]].rename(columns={"Close": requested[0]})

    prices = prices.dropna(how="all").sort_index()
    if prices.empty:
        raise ValueError("Price history is empty after removing missing rows.")

    return prices


def build_features(
    prices: pd.DataFrame,
    window: int = 20,
    asset_overrides: Mapping[str, str] | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    if window < 2:
        raise ValueError("Rolling volatility window must be at least 2.")

    metadata = build_asset_metadata(prices.columns, asset_overrides, interval=interval)
    rows: dict[str, dict[str, float]] = {}

    for symbol in prices.columns:
        series = prices[symbol].dropna()
        returns = series.pct_change(fill_method=None).dropna()
        factor = float(metadata.at[symbol, "annualization_factor"])
        volatility = (
            returns.rolling(window=window, min_periods=window).std() * np.sqrt(factor)
        ).dropna()

        rows[symbol] = {
            "mean_return": returns.mean(),
            "return_std": returns.std(),
            "annualized_return": returns.mean() * factor,
            "average_volatility": volatility.mean(),
            "volatility_std": volatility.std(),
            "latest_volatility": volatility.iloc[-1] if not volatility.empty else np.nan,
        }

    features = pd.DataFrame.from_dict(rows, orient="index")
    features.index.name = "Ticker"
    features = features.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if len(features) < 2:
        raise ValueError("Need at least two symbols with complete features.")

    return features
