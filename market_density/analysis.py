from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class Projection:
    points: pd.DataFrame
    explained_variance: list[float]


@dataclass(slots=True)
class ClusterResult:
    points: pd.DataFrame
    centers: pd.DataFrame
    count: int
    inertia: float
    silhouette: float | None


def project_features(features: pd.DataFrame, components: int = 2) -> Projection:
    if len(features) < components:
        raise ValueError(f"Need at least {components} rows to compute PCA.")

    scaled = StandardScaler().fit_transform(features)
    model = PCA(n_components=components)
    projected = model.fit_transform(scaled)

    columns = [f"PC{index + 1}" for index in range(components)]
    points = pd.DataFrame(projected, index=features.index, columns=columns)
    return Projection(
        points=points,
        explained_variance=model.explained_variance_ratio_.tolist(),
    )


def cluster_projection(
    points: pd.DataFrame,
    clusters: int = 3,
    random_state: int = 42,
) -> ClusterResult:
    if clusters < 1:
        raise ValueError("Cluster count must be at least 1.")
    if points.empty:
        raise ValueError("Need PCA coordinates before clustering.")

    count = min(clusters, len(points))
    model = KMeans(n_clusters=count, n_init=10, random_state=random_state)
    labels = model.fit_predict(points[["PC1", "PC2"]])
    names = [f"Cluster {label + 1}" for label in labels]

    clustered = points.copy()
    clustered["Cluster"] = pd.Categorical(names)

    centers = pd.DataFrame(
        model.cluster_centers_,
        columns=["PC1", "PC2"],
        index=[f"Cluster {index + 1}" for index in range(count)],
    )
    centers["size"] = (
        clustered["Cluster"].value_counts().reindex(centers.index).fillna(0).astype(int)
    )

    silhouette = None
    if count > 1 and len(points) > count:
        silhouette = float(silhouette_score(points[["PC1", "PC2"]], labels))

    return ClusterResult(
        points=clustered,
        centers=centers,
        count=count,
        inertia=float(model.inertia_),
        silhouette=silhouette,
    )
