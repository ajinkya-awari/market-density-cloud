from __future__ import annotations

import math

import numpy as np
import pandas as pd

from market_density.data import (
    annualization_factor,
    build_asset_metadata,
    build_features,
    infer_asset_class,
)


def test_infer_asset_class_is_strict_about_crypto_quotes() -> None:
    assert infer_asset_class("BTC-USD") == "crypto"
    assert infer_asset_class("EURUSD=X") == "forex"
    assert infer_asset_class("BRK-B") == "stock"
    assert infer_asset_class("BF-B") == "stock"


def test_annualization_factor_changes_with_interval() -> None:
    assert annualization_factor("stock", "1d") == 252.0
    assert annualization_factor("stock", "1wk") == 252.0 / 5.0
    assert annualization_factor("crypto", "1wk") == 365.0 / 7.0
    assert annualization_factor("stock", "1mo") == 12.0


def test_build_asset_metadata_uses_interval_adjusted_factor() -> None:
    metadata = build_asset_metadata(["AAPL", "BTC-USD"], interval="1wk")

    assert math.isclose(metadata.loc["AAPL", "annualization_factor"], 252.0 / 5.0)
    assert math.isclose(metadata.loc["BTC-USD", "annualization_factor"], 365.0 / 7.0)


def test_build_features_uses_interval_adjusted_annualization() -> None:
    index = pd.date_range("2025-01-05", periods=8, freq="W")
    aapl_prices = 100 * pd.Series(
        np.cumprod([1.0, 1.02, 0.98, 1.03, 0.98, 1.04, 0.98, 1.03]),
        index=index,
    )
    msft_prices = 200 * pd.Series(
        np.cumprod([1.0, 0.99, 1.02, 0.985, 1.025, 0.98, 1.03, 0.98]),
        index=index,
    )
    prices = pd.DataFrame(
        {
            "AAPL": aapl_prices,
            "MSFT": msft_prices,
        },
    )
    features = build_features(prices, window=3, interval="1wk")

    aapl_returns = prices["AAPL"].pct_change(fill_method=None).dropna()
    expected_factor = 252.0 / 5.0
    expected_annualized_return = aapl_returns.mean() * expected_factor

    assert math.isclose(
        features.loc["AAPL", "annualized_return"],
        expected_annualized_return,
        rel_tol=1e-9,
    )
