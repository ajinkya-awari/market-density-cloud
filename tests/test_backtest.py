from __future__ import annotations

import pandas as pd

from market_density.backtest import run_train_test_backtest


def test_run_train_test_backtest_generates_positive_result_for_helpful_signals() -> None:
    index = pd.date_range("2025-01-01", periods=10, freq="D")
    prices = pd.DataFrame(
        {
            "AAPL": [100, 102, 104, 106, 108, 110, 114, 118, 122, 126],
            "MSFT": [100, 98, 96, 94, 92, 90, 87, 84, 81, 78],
            "GOOGL": [100, 100.5, 101, 100.8, 101.2, 101.1, 101.0, 101.2, 101.1, 101.0],
        },
        index=index,
    )

    result = run_train_test_backtest(
        prices,
        interval="1d",
        feature_window=2,
        clusters=3,
        random_state=42,
        holdout_periods=3,
    )

    assert not result.signals.empty
    assert not result.returns.empty
    assert result.summary.loc[0, "observations"] == 3
    assert result.summary.loc[0, "active_symbols"] >= 2
    assert result.summary.loc[0, "gross_exposure"] > 0
    assert "strategy_return" in result.returns.columns
