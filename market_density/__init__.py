"""Market density dashboard package."""

from .analysis import cluster_projection, project_features
from .backtest import BacktestResult, run_train_test_backtest, summarize_returns
from .dashboard import DashboardArtifacts, build_dashboard, write_dashboard
from .data import (
    annualization_factor,
    build_asset_metadata,
    build_features,
    download_prices,
    infer_asset_class,
    normalize_symbols,
)
from .signals import SignalBook, build_cluster_signals

__all__ = [
    "BacktestResult",
    "DashboardArtifacts",
    "SignalBook",
    "annualization_factor",
    "build_asset_metadata",
    "build_dashboard",
    "build_features",
    "build_cluster_signals",
    "cluster_projection",
    "download_prices",
    "infer_asset_class",
    "normalize_symbols",
    "project_features",
    "run_train_test_backtest",
    "summarize_returns",
    "write_dashboard",
]
