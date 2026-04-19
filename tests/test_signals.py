from __future__ import annotations

import math

import pandas as pd

from market_density.signals import build_cluster_signals


def test_build_cluster_signals_assigns_long_short_roles_and_weights() -> None:
    points = pd.DataFrame(
        {
            "Ticker": ["AAPL", "MSFT", "JNJ", "BTC-USD"],
            "Cluster": ["Cluster 1", "Cluster 1", "Cluster 2", "Cluster 3"],
            "asset_class": ["stock", "stock", "stock", "crypto"],
            "PC1": [-2.0, -1.0, 0.3, 1.8],
            "PC2": [0.1, -0.2, 0.4, -0.5],
            "latest_volatility": [0.2, 0.4, 0.3, 0.5],
        }
    ).set_index("Ticker")
    centers = pd.DataFrame(
        {
            "PC1": [-1.5, 0.3, 1.8],
            "PC2": [-0.05, 0.4, -0.5],
            "size": [2, 1, 1],
        },
        index=["Cluster 1", "Cluster 2", "Cluster 3"],
    )

    signal_book = build_cluster_signals(points, centers=centers)
    signals = signal_book.signals

    assert signal_book.cluster_roles["Cluster 1"] == "long"
    assert signal_book.cluster_roles["Cluster 3"] == "short"
    assert signals.loc["JNJ", "signal"] == 0
    assert signals.loc["BTC-USD", "signal"] == -1
    assert math.isclose(signals.loc[signals["signal"] == 1, "position_weight"].sum(), 0.5)
    assert math.isclose(
        signals.loc[signals["signal"] == -1, "position_weight"].sum(),
        -0.5,
    )
    assert signals.loc["AAPL", "position_weight"] > signals.loc["MSFT", "position_weight"]
