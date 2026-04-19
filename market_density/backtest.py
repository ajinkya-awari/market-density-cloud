from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analysis import cluster_projection, project_features
from .data import build_asset_metadata, build_features
from .signals import SignalBook, build_cluster_signals


@dataclass(slots=True)
class BacktestResult:
    signals: pd.DataFrame
    returns: pd.DataFrame
    summary: pd.DataFrame


def _periods_per_year(index: pd.Index) -> float:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 252.0

    elapsed_days = (index[-1] - index[0]).total_seconds() / 86400.0
    if elapsed_days <= 0:
        return 252.0

    interval_count = max(len(index) - 1, 1)
    return interval_count / (elapsed_days / 365.25)


def summarize_returns(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    periods = len(strategy_returns)
    periods_per_year = _periods_per_year(strategy_returns.index)
    equity_curve = (1.0 + strategy_returns).cumprod()

    total_return = equity_curve.iloc[-1] - 1.0
    annualized_return = equity_curve.iloc[-1] ** (periods_per_year / periods) - 1.0
    annualized_volatility = strategy_returns.std(ddof=0) * np.sqrt(periods_per_year)
    sharpe_ratio = (
        annualized_return / annualized_volatility
        if annualized_volatility > 0
        else np.nan
    )
    running_peak = equity_curve.cummax()
    max_drawdown = (equity_curve / running_peak - 1.0).min()

    benchmark_curve = (1.0 + benchmark_returns).cumprod()
    benchmark_return = benchmark_curve.iloc[-1] - 1.0

    summary = pd.DataFrame(
        [
            {
                "observations": periods,
                "periods_per_year": periods_per_year,
                "total_return": total_return,
                "annualized_return": annualized_return,
                "annualized_volatility": annualized_volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "hit_rate": float((strategy_returns > 0).mean()),
                "benchmark_return": benchmark_return,
                "benchmark_annualized_return": (
                    benchmark_curve.iloc[-1] ** (periods_per_year / periods) - 1.0
                ),
            }
        ]
    )
    return summary


def _portfolio_returns(
    returns: pd.DataFrame,
    signal_book: SignalBook,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    weights = signal_book.signals["position_weight"].astype(float)
    active_returns = returns.reindex(columns=weights.index).fillna(0.0)

    strategy_returns = active_returns.mul(weights, axis=1).sum(axis=1)
    if transaction_cost_bps > 0:
        cost = weights.abs().sum() * (transaction_cost_bps / 10000.0)
        if not strategy_returns.empty:
            strategy_returns.iloc[0] -= cost

    benchmark_returns = active_returns.mean(axis=1)
    equity_curve = (1.0 + strategy_returns).cumprod()
    benchmark_curve = (1.0 + benchmark_returns).cumprod()
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0

    return pd.DataFrame(
        {
            "strategy_return": strategy_returns,
            "benchmark_return": benchmark_returns,
            "equity_curve": equity_curve,
            "benchmark_curve": benchmark_curve,
            "drawdown": drawdown,
        }
    )


def run_train_test_backtest(
    prices: pd.DataFrame,
    *,
    asset_overrides: dict[str, str] | None = None,
    interval: str = "1d",
    feature_window: int = 20,
    clusters: int = 3,
    random_state: int = 42,
    holdout_periods: int = 60,
    transaction_cost_bps: float = 0.0,
) -> BacktestResult:
    if holdout_periods < 2:
        raise ValueError("Backtest holdout must contain at least 2 periods.")
    if len(prices) <= holdout_periods + feature_window:
        raise ValueError(
            "Not enough history for a train/test backtest. Increase the downloaded "
            "period or reduce the holdout window."
        )

    train_prices = prices.iloc[:-holdout_periods]
    test_prices = prices.iloc[-(holdout_periods + 1) :]
    if len(test_prices) < 2:
        raise ValueError("Not enough prices in the holdout window to compute returns.")

    metadata = build_asset_metadata(
        train_prices.columns,
        overrides=asset_overrides,
        interval=interval,
    )
    features = build_features(
        train_prices,
        window=feature_window,
        asset_overrides=asset_overrides,
        interval=interval,
    )
    projection = project_features(features, components=2)
    clustering = cluster_projection(
        projection.points,
        clusters=clusters,
        random_state=random_state,
    )
    point_table = metadata.loc[clustering.points.index].join(clustering.points)
    point_table = point_table.join(
        features.loc[point_table.index, ["annualized_return", "latest_volatility"]]
    )
    signal_book = build_cluster_signals(point_table, centers=clustering.centers)

    returns = test_prices.pct_change(fill_method=None).iloc[1:]
    performance = _portfolio_returns(
        returns,
        signal_book,
        transaction_cost_bps=transaction_cost_bps,
    )
    summary = summarize_returns(
        performance["strategy_return"],
        performance["benchmark_return"],
    )
    summary["train_start"] = train_prices.index[0]
    summary["train_end"] = train_prices.index[-1]
    summary["test_start"] = performance.index[0]
    summary["test_end"] = performance.index[-1]
    summary["active_symbols"] = int((signal_book.signals["signal"] != 0).sum())
    summary["gross_exposure"] = float(
        signal_book.signals["position_weight"].abs().sum()
    )
    summary["net_exposure"] = float(signal_book.signals["position_weight"].sum())

    return BacktestResult(
        signals=signal_book.signals,
        returns=performance,
        summary=summary,
    )
