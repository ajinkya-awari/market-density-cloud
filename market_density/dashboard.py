from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CLUSTER_COLORS = [
    "#125b9a",
    "#ca6702",
    "#2a9d8f",
    "#b5179e",
    "#6c757d",
    "#d62828",
    "#3a86ff",
    "#588157",
]
ASSET_COLORS = {
    "stock": "#125b9a",
    "crypto": "#ca6702",
    "forex": "#2a9d8f",
}
ASSET_SYMBOLS = {
    "stock": "circle",
    "crypto": "diamond",
    "forex": "square",
}


@dataclass(slots=True)
class DashboardArtifacts:
    figure: go.Figure
    preview_path: Path | None
    preview_warning: str | None = None


def _marker_sizes(frame: pd.DataFrame) -> pd.Series:
    if "latest_volatility" not in frame.columns:
        return pd.Series(18.0, index=frame.index, dtype=float)

    values = frame["latest_volatility"].astype(float)
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        return pd.Series(18.0, index=frame.index, dtype=float)

    scaled = (values - low) / (high - low)
    return pd.Series((14 + scaled * 12).round(2), index=frame.index, dtype=float)


def build_dashboard(
    points: pd.DataFrame,
    *,
    centers: pd.DataFrame | None = None,
    explained_variance: list[float] | None = None,
    silhouette: float | None = None,
    title: str = "Market Density Cloud",
) -> go.Figure:
    frame = points.copy()
    if "Ticker" not in frame.columns:
        frame = frame.reset_index().rename(columns={frame.index.name or "index": "Ticker"})
    if "Cluster" not in frame.columns:
        frame["Cluster"] = "All Assets"
    if "asset_class" not in frame.columns:
        frame["asset_class"] = "unknown"

    subtitle_bits = [
        f"{len(frame)} assets",
        f"{frame['Cluster'].astype(str).nunique()} clusters",
    ]
    if explained_variance:
        subtitle_bits.append(f"PCA variance {sum(explained_variance):.1%}")
    if silhouette is not None:
        subtitle_bits.append(f"Silhouette {silhouette:.2f}")
    subtitle = " | ".join(subtitle_bits)

    show_return = "annualized_return" in frame.columns
    show_volatility = "latest_volatility" in frame.columns
    ordered_clusters = sorted(frame["Cluster"].astype(str).unique())
    size_map = _marker_sizes(frame)
    cluster_colors = {
        name: CLUSTER_COLORS[index % len(CLUSTER_COLORS)]
        for index, name in enumerate(ordered_clusters)
    }

    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.74, 0.26],
        row_heights=[0.52, 0.48],
        horizontal_spacing=0.08,
        vertical_spacing=0.16,
        specs=[
            [{"type": "xy", "rowspan": 2}, {"type": "domain"}],
            [None, {"type": "xy"}],
        ],
        subplot_titles=("Clustered PCA Cloud", "Asset Mix", "Cluster Sizes"),
    )

    has_density = (
        len(frame) >= 3
        and frame["PC1"].nunique() > 1
        and frame["PC2"].nunique() > 1
    )
    if has_density:
        fig.add_trace(
            go.Histogram2dContour(
                x=frame["PC1"],
                y=frame["PC2"],
                colorscale="Teal",
                ncontours=12,
                opacity=0.35,
                showscale=False,
                contours={"coloring": "fill", "showlines": False},
                hoverinfo="skip",
                name="Density",
            ),
            row=1,
            col=1,
        )

    for cluster_name in ordered_clusters:
        cluster_frame = frame[frame["Cluster"].astype(str) == cluster_name]
        custom_columns = ["asset_class"]
        if show_return:
            custom_columns.append("annualized_return")
        if show_volatility:
            custom_columns.append("latest_volatility")
        marker_sizes = size_map.loc[cluster_frame.index].tolist()

        hover_lines = [
            "<b>%{text}</b>",
            "Cluster: %{fullData.name}",
            "Asset Class: %{customdata[0]}",
            "PC1: %{x:.3f}",
            "PC2: %{y:.3f}",
        ]
        if show_return:
            hover_lines.append("Annualized Return: %{customdata[1]:.2%}")
        if show_volatility:
            hover_index = 2 if show_return else 1
            hover_lines.append(f"Latest Volatility: %{{customdata[{hover_index}]:.2%}}")

        fig.add_trace(
            go.Scatter(
                x=cluster_frame["PC1"],
                y=cluster_frame["PC2"],
                mode="markers",
                name=cluster_name,
                text=cluster_frame["Ticker"],
                customdata=cluster_frame[custom_columns].astype(object).to_numpy(),
                hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
                marker={
                    "size": marker_sizes,
                    "symbol": cluster_frame["asset_class"]
                    .map(ASSET_SYMBOLS)
                    .fillna("circle")
                    .tolist(),
                    "color": cluster_colors[cluster_name],
                    "line": {"color": "#fdf7ee", "width": 1.4},
                    "opacity": 0.95,
                },
            ),
            row=1,
            col=1,
        )

    if centers is not None and not centers.empty:
        center_frame = centers.reset_index().rename(columns={"index": "Cluster"})
        for _, center in center_frame.iterrows():
            cluster_name = str(center["Cluster"])
            fig.add_trace(
                go.Scatter(
                    x=[center["PC1"]],
                    y=[center["PC2"]],
                    mode="markers+text",
                    text=[cluster_name],
                    textposition="top center",
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{cluster_name} centroid</b><br>"
                        f"PC1: {center['PC1']:.3f}<br>"
                        f"PC2: {center['PC2']:.3f}<br>"
                        f"Members: {int(center.get('size', 0))}<extra></extra>"
                    ),
                    marker={
                        "symbol": "star-diamond",
                        "size": 20,
                        "color": cluster_colors.get(cluster_name, "#264653"),
                        "line": {"color": "#0d1b2a", "width": 1.6},
                    },
                    textfont={"size": 11, "color": "#1f2933"},
                ),
                row=1,
                col=1,
            )

    asset_counts = frame["asset_class"].astype(str).value_counts().sort_index()
    fig.add_trace(
        go.Pie(
            labels=[label.title() for label in asset_counts.index],
            values=asset_counts.values,
            hole=0.55,
            sort=False,
            textinfo="label+percent",
            textfont={"size": 12, "color": "#22313f"},
            marker={
                "colors": [ASSET_COLORS.get(label, "#6c757d") for label in asset_counts.index],
                "line": {"color": "#fffaf2", "width": 3},
            },
            hovertemplate="%{label}: %{value} assets (%{percent})<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    cluster_sizes = frame["Cluster"].astype(str).value_counts().reindex(ordered_clusters)
    fig.add_trace(
        go.Bar(
            x=cluster_sizes.values,
            y=cluster_sizes.index,
            orientation="h",
            marker={
                "color": [cluster_colors[name] for name in cluster_sizes.index],
                "line": {"color": "#f6efe5", "width": 1.5},
            },
            text=cluster_sizes.values,
            textposition="outside",
            hovertemplate="%{y}: %{x} assets<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    x_min, x_max = float(frame["PC1"].min()), float(frame["PC1"].max())
    y_min, y_max = float(frame["PC2"].min()), float(frame["PC2"].max())
    fig.add_shape(
        type="line",
        x0=x_min,
        x1=x_max,
        y0=0,
        y1=0,
        xref="x",
        yref="y",
        line={"dash": "dash", "color": "#98a2b3", "width": 1},
    )
    fig.add_shape(
        type="line",
        x0=0,
        x1=0,
        y0=y_min,
        y1=y_max,
        xref="x",
        yref="y",
        line={"dash": "dash", "color": "#98a2b3", "width": 1},
    )

    fig.update_xaxes(
        title_text="Principal Component 1",
        zeroline=False,
        showgrid=True,
        gridcolor="rgba(62, 86, 104, 0.12)",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Principal Component 2",
        zeroline=False,
        showgrid=True,
        gridcolor="rgba(62, 86, 104, 0.12)",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="Asset Count",
        showgrid=False,
        zeroline=False,
        row=2,
        col=2,
    )
    fig.update_yaxes(
        title_text="",
        autorange="reversed",
        showgrid=False,
        row=2,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        height=840,
        title={
            "text": (
                f"{title}<br>"
                f"<span style='font-size:14px;color:#51606f;font-weight:500'>{subtitle}</span>"
            ),
            "x": 0.03,
            "y": 0.97,
        },
        font={"family": "Segoe UI, Arial, sans-serif", "color": "#14213d"},
        paper_bgcolor="#f4efe6",
        plot_bgcolor="#fffaf2",
        hoverlabel={
            "bgcolor": "#fffdf8",
            "bordercolor": "#1f2933",
            "font_size": 13,
            "font_family": "Segoe UI, Arial, sans-serif",
        },
        legend={
            "title": {"text": "KMeans Cluster"},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.03,
            "bgcolor": "rgba(255,250,242,0.75)",
            "bordercolor": "rgba(20,33,61,0.08)",
            "borderwidth": 1,
        },
        margin={"l": 65, "r": 40, "t": 120, "b": 60},
    )

    if not has_density:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.05,
            y=0.02,
            showarrow=False,
            text="Density contour skipped: need at least 3 non-collinear points.",
            font={"size": 12, "color": "#4a5568"},
        )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.03,
        y=1.09,
        showarrow=False,
        align="left",
        text=(
            "<b>Visual guide</b><br>"
            "Color = cluster<br>"
            "Shape = asset class<br>"
            "Size = latest volatility"
        ),
        font={"size": 12, "color": "#405261"},
        bgcolor="rgba(255,250,242,0.82)",
        bordercolor="rgba(20,33,61,0.1)",
        borderwidth=1,
        borderpad=8,
    )

    return fig


def write_dashboard(
    points: pd.DataFrame,
    *,
    html_path: str | Path,
    preview_path: str | Path | None = None,
    centers: pd.DataFrame | None = None,
    explained_variance: list[float] | None = None,
    silhouette: float | None = None,
    title: str = "Market Density Cloud",
) -> DashboardArtifacts:
    html_target = Path(html_path)
    html_target.parent.mkdir(parents=True, exist_ok=True)

    fig = build_dashboard(
        points,
        centers=centers,
        explained_variance=explained_variance,
        silhouette=silhouette,
        title=title,
    )
    fig.write_html(
        html_target,
        full_html=True,
        include_plotlyjs=True,
        config={"responsive": True, "displaylogo": False},
    )

    preview_warning = None
    written_preview_path: Path | None = None
    if preview_path is not None:
        preview_target = Path(preview_path)
        preview_target.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(preview_target, width=1600, height=900, scale=2)
            written_preview_path = preview_target
        except (OSError, RuntimeError, ValueError) as exc:
            preview_warning = (
                "Preview image export was skipped. HTML output was written successfully, "
                "but static image export requires a working Kaleido and Chrome/Chromium setup. "
                f"Details: {exc}"
            )

    return DashboardArtifacts(
        figure=fig,
        preview_path=written_preview_path,
        preview_warning=preview_warning,
    )
