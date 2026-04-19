from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import cluster_projection, project_features
from .backtest import run_train_test_backtest
from .dashboard import write_dashboard
from .data import (
    build_asset_metadata,
    build_features,
    download_prices,
    infer_asset_class,
)
from .signals import build_cluster_signals

DEFAULT_SYMBOLS = {
    "stock": ["AAPL", "MSFT", "NVDA", "GOOGL", "JPM", "JNJ"],
    "crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
    "forex": ["EURUSD=X", "GBPUSD=X", "JPY=X"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a clustered PCA dashboard for stocks, crypto, and forex."
    )
    parser.add_argument(
        "symbols",
        nargs="*",
        help="Mixed symbols such as AAPL, BTC-USD, and EURUSD=X.",
    )
    parser.add_argument(
        "--stocks",
        nargs="+",
        action="extend",
        default=[],
        help="Additional stock symbols.",
    )
    parser.add_argument(
        "--crypto",
        nargs="+",
        action="extend",
        default=[],
        help="Additional crypto symbols.",
    )
    parser.add_argument(
        "--forex",
        nargs="+",
        action="extend",
        default=[],
        help="Additional forex symbols.",
    )
    parser.add_argument("--period", default="1y", help="yfinance period, for example 1y or 2y.")
    parser.add_argument("--interval", default="1d", help="yfinance interval, for example 1d or 1wk.")
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling window used to calculate volatility.",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=3,
        help="Requested number of KMeans clusters.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for KMeans.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for generated CSV and HTML files.",
    )
    parser.add_argument(
        "--backtest-window",
        type=int,
        default=60,
        help="Number of periods reserved for the out-of-sample backtest.",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=0.0,
        help="One-way transaction cost, applied on the first backtest rebalance.",
    )
    parser.add_argument(
        "--preview-path",
        default="assets/dashboard-preview.png",
        help="Path for the static screenshot used in the repository README.",
    )
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="Skip PNG preview export.",
    )
    return parser.parse_args()


def collect_symbols(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    requested: list[tuple[str, str]] = []

    for symbol in args.symbols:
        requested.append((symbol, infer_asset_class(symbol)))
    for symbol in args.stocks:
        requested.append((symbol, "stock"))
    for symbol in args.crypto:
        requested.append((symbol, "crypto"))
    for symbol in args.forex:
        requested.append((symbol, "forex"))

    if not requested:
        for asset_class, values in DEFAULT_SYMBOLS.items():
            requested.extend((symbol, asset_class) for symbol in values)

    symbols: list[str] = []
    asset_map: dict[str, str] = {}
    seen: set[str] = set()
    for symbol, asset_class in requested:
        value = symbol.strip().upper()
        if value and value not in seen:
            symbols.append(value)
            asset_map[value] = asset_class
            seen.add(value)

    return symbols, asset_map


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    symbols, asset_map = collect_symbols(args)
    prices = download_prices(symbols, period=args.period, interval=args.interval)
    metadata = build_asset_metadata(prices.columns, asset_map, interval=args.interval)
    features = build_features(
        prices,
        window=args.window,
        asset_overrides=asset_map,
        interval=args.interval,
    )
    projection = project_features(features, components=2)
    clusters = cluster_projection(
        projection.points,
        clusters=args.clusters,
        random_state=args.random_state,
    )

    feature_table = metadata.loc[features.index].join(features)
    point_table = metadata.loc[clusters.points.index].join(clusters.points)
    point_table = point_table.join(
        features.loc[point_table.index, ["annualized_return", "latest_volatility"]]
    )
    signal_book = build_cluster_signals(point_table, centers=clusters.centers)

    features_path = output_dir / "features.csv"
    points_path = output_dir / "pca_clusters.csv"
    signals_path = output_dir / "signals.csv"
    backtest_signals_path = output_dir / "backtest_signals.csv"
    backtest_returns_path = output_dir / "backtest_returns.csv"
    backtest_summary_path = output_dir / "backtest_summary.csv"
    dashboard_path = output_dir / "dashboard.html"
    preview_path = None if args.skip_preview else Path(args.preview_path)

    feature_table.to_csv(features_path, index_label="Ticker")
    point_table.to_csv(points_path, index_label="Ticker")
    signal_book.signals.to_csv(signals_path, index_label="Ticker")
    dashboard_artifacts = write_dashboard(
        point_table,
        html_path=dashboard_path,
        preview_path=preview_path,
        centers=clusters.centers,
        explained_variance=projection.explained_variance,
        silhouette=clusters.silhouette,
        title="Market Density Cloud",
    )
    backtest_result = None
    backtest_warning = None
    try:
        backtest_result = run_train_test_backtest(
            prices,
            asset_overrides=asset_map,
            interval=args.interval,
            feature_window=args.window,
            clusters=args.clusters,
            random_state=args.random_state,
            holdout_periods=args.backtest_window,
            transaction_cost_bps=args.transaction_cost_bps,
        )
    except ValueError as exc:
        backtest_warning = str(exc)
    else:
        backtest_result.signals.to_csv(backtest_signals_path, index_label="Ticker")
        backtest_result.returns.to_csv(backtest_returns_path, index_label="Date")
        backtest_result.summary.to_csv(backtest_summary_path, index=False)

    explained = ", ".join(
        f"PC{index + 1}: {ratio:.1%}"
        for index, ratio in enumerate(projection.explained_variance)
    )

    print(f"Saved features to {features_path.resolve()}")
    print(f"Saved PCA clusters to {points_path.resolve()}")
    print(f"Saved signals to {signals_path.resolve()}")
    print(f"Saved dashboard to {dashboard_path.resolve()}")
    if dashboard_artifacts.preview_path is not None:
        print(f"Saved repo preview to {dashboard_artifacts.preview_path.resolve()}")
    elif dashboard_artifacts.preview_warning is not None:
        print(dashboard_artifacts.preview_warning)
    if backtest_result is not None:
        print(f"Saved backtest signals to {backtest_signals_path.resolve()}")
        print(f"Saved backtest returns to {backtest_returns_path.resolve()}")
        print(f"Saved backtest summary to {backtest_summary_path.resolve()}")
        print(
            "Backtest annualized return: "
            f"{backtest_result.summary.loc[0, 'annualized_return']:.2%}"
        )
        print(
            "Backtest Sharpe ratio: "
            f"{backtest_result.summary.loc[0, 'sharpe_ratio']:.2f}"
        )
    elif backtest_warning is not None:
        print(f"Backtest skipped: {backtest_warning}")
    print(f"Downloaded symbols: {', '.join(symbols)}")
    print(f"Explained variance: {explained}")
    print(f"KMeans clusters: {clusters.count}")
    print(f"KMeans inertia: {clusters.inertia:.3f}")
    if clusters.silhouette is not None:
        print(f"Silhouette score: {clusters.silhouette:.3f}")
