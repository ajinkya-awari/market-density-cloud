from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from market_density.dashboard import build_dashboard, write_dashboard


def test_marker_sizes_are_scaled_globally_across_clusters() -> None:
    points = pd.DataFrame(
        {
            "Ticker": ["AAPL", "MSFT", "BTC-USD"],
            "Cluster": ["Cluster 1", "Cluster 1", "Cluster 2"],
            "asset_class": ["stock", "stock", "crypto"],
            "PC1": [-1.0, -0.4, 1.2],
            "PC2": [0.2, 0.4, -0.3],
            "latest_volatility": [0.40, 0.50, 0.90],
        }
    )

    figure = build_dashboard(points)
    scatter_traces = {
        trace.name: list(trace.marker.size)
        for trace in figure.data
        if trace.type == "scatter"
    }

    assert scatter_traces["Cluster 2"][0] > max(scatter_traces["Cluster 1"])


def test_write_dashboard_keeps_html_when_preview_export_fails(
    tmp_path,
    monkeypatch,
) -> None:
    def raise_missing_chrome(self, *args, **kwargs) -> None:
        raise RuntimeError("Chrome not found")

    monkeypatch.setattr(go.Figure, "write_image", raise_missing_chrome)

    points = pd.DataFrame(
        {
            "Ticker": ["AAPL", "MSFT"],
            "Cluster": ["Cluster 1", "Cluster 2"],
            "asset_class": ["stock", "stock"],
            "PC1": [0.1, 0.8],
            "PC2": [0.2, -0.2],
        }
    )

    html_path = tmp_path / "dashboard.html"
    preview_path = tmp_path / "preview.png"
    artifacts = write_dashboard(
        points,
        html_path=html_path,
        preview_path=preview_path,
    )

    assert html_path.exists()
    assert artifacts.preview_path is None
    assert artifacts.preview_warning is not None
