from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SignalBook:
    signals: pd.DataFrame
    cluster_roles: dict[str, str]


def _cluster_order(
    points: pd.DataFrame,
    centers: pd.DataFrame | None = None,
) -> list[str]:
    if centers is not None and not centers.empty:
        center_frame = centers.reset_index().rename(columns={"index": "Cluster"})
        ordered = (
            center_frame.sort_values("PC1", ascending=True)["Cluster"].astype(str).tolist()
        )
        if ordered:
            return ordered

    return (
        points.groupby("Cluster", observed=False)["PC1"]
        .mean()
        .sort_values(ascending=True)
        .index.astype(str)
        .tolist()
    )


def _inverse_volatility_weights(frame: pd.DataFrame) -> pd.Series:
    if "latest_volatility" not in frame.columns:
        return pd.Series(1.0, index=frame.index, dtype=float)

    volatility = frame["latest_volatility"].astype(float).replace(0.0, np.nan)
    inverse = 1.0 / volatility
    if inverse.notna().sum() == 0:
        return pd.Series(1.0, index=frame.index, dtype=float)

    return inverse.fillna(inverse[inverse.notna()].mean())


def build_cluster_signals(
    points: pd.DataFrame,
    centers: pd.DataFrame | None = None,
) -> SignalBook:
    if points.empty:
        raise ValueError("Need clustered PCA points to create trading signals.")
    if "Cluster" not in points.columns:
        raise ValueError("Trading signals require a 'Cluster' column.")

    signal_frame = points.copy()
    signal_frame["Cluster"] = signal_frame["Cluster"].astype(str)
    signal_frame["signal"] = 0
    signal_frame["signal_label"] = "Neutral"
    signal_frame["cluster_role"] = "neutral"
    signal_frame["position_weight"] = 0.0

    ordered_clusters = _cluster_order(signal_frame, centers=centers)
    if len(ordered_clusters) < 2:
        return SignalBook(signals=signal_frame, cluster_roles={})

    long_cluster = ordered_clusters[0]
    short_cluster = ordered_clusters[-1]
    cluster_roles = {
        cluster_name: (
            "long"
            if cluster_name == long_cluster
            else "short"
            if cluster_name == short_cluster
            else "neutral"
        )
        for cluster_name in ordered_clusters
    }
    signal_frame["cluster_role"] = signal_frame["Cluster"].map(cluster_roles).fillna(
        "neutral"
    )

    long_mask = signal_frame["cluster_role"] == "long"
    short_mask = signal_frame["cluster_role"] == "short"
    signal_frame.loc[long_mask, "signal"] = 1
    signal_frame.loc[short_mask, "signal"] = -1
    signal_frame["signal_label"] = signal_frame["signal"].map(
        {1: "Long", -1: "Short", 0: "Neutral"}
    )

    if long_mask.any():
        long_strength = _inverse_volatility_weights(signal_frame.loc[long_mask])
        signal_frame.loc[long_mask, "position_weight"] = (
            long_strength / long_strength.sum()
        ) * 0.5

    if short_mask.any():
        short_strength = _inverse_volatility_weights(signal_frame.loc[short_mask])
        signal_frame.loc[short_mask, "position_weight"] = (
            short_strength / short_strength.sum()
        ) * -0.5

    if long_mask.any() and not short_mask.any():
        signal_frame.loc[long_mask, "position_weight"] *= 2.0
    if short_mask.any() and not long_mask.any():
        signal_frame.loc[short_mask, "position_weight"] *= 2.0

    return SignalBook(signals=signal_frame, cluster_roles=cluster_roles)
