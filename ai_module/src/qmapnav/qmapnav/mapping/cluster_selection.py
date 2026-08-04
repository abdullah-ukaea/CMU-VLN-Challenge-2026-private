"""Deterministic distance-aware DBSCAN and explicit cluster selection."""

from collections import deque
from dataclasses import dataclass
from math import exp, isfinite
from types import MappingProxyType
from typing import Mapping

import numpy as np

from qmapnav.mapping.object_candidate import readonly_array
from qmapnav.perception.contracts import Detection2D


@dataclass(frozen=True)
class ClusterSelectionConfig:
    """Distance-aware DBSCAN and primary-cluster score policy."""

    base_epsilon_m: float = 0.07
    range_epsilon_slope: float = 0.015
    minimum_epsilon_m: float = 0.07
    maximum_epsilon_m: float = 0.30
    minimum_samples: int = 5
    ambiguity_score_delta: float = 0.06
    maximum_cluster_extent_m: float = 4.0

    def __post_init__(self) -> None:
        for name in (
            'base_epsilon_m',
            'range_epsilon_slope',
            'minimum_epsilon_m',
            'maximum_epsilon_m',
            'maximum_cluster_extent_m',
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f'{name} must be finite and non-negative')
        if self.minimum_epsilon_m > self.maximum_epsilon_m:
            raise ValueError('epsilon limits are invalid')
        if self.minimum_samples <= 0:
            raise ValueError('minimum_samples must be positive')
        if not 0.0 <= self.ambiguity_score_delta <= 1.0:
            raise ValueError('ambiguity_score_delta must lie in [0, 1]')


@dataclass(frozen=True)
class ClusterSummary:
    """Auditable score terms for one spatial cluster."""

    cluster_id: int
    point_count: int
    median_depth_m: float
    centre_panorama_uv: tuple[float, float]
    extent_xyz: np.ndarray
    point_support_score: float
    foreground_score: float
    centre_alignment_score: float
    compactness_score: float
    total_score: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, 'extent_xyz', readonly_array('extent_xyz', self.extent_xyz, (3,))
        )


@dataclass(frozen=True)
class ClusterSelectionResult:
    """DBSCAN labels, selected indices and ambiguity diagnostics."""

    labels: np.ndarray
    selected_indices: np.ndarray
    noise_indices: np.ndarray
    selected_cluster_id: int | None
    summaries: tuple[ClusterSummary, ...]
    epsilon_m: float
    ambiguous: bool
    reason: str
    alternative_scores: Mapping[int, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, 'labels', readonly_array('labels', self.labels, (None,), dtype=np.int64)
        )
        for name in ('selected_indices', 'noise_indices'):
            object.__setattr__(
                self,
                name,
                readonly_array(name, getattr(self, name), (None,), dtype=np.int64),
            )
        object.__setattr__(self, 'summaries', tuple(self.summaries))
        object.__setattr__(
            self, 'alternative_scores', MappingProxyType(dict(self.alternative_scores))
        )


def cluster_and_select(
    points_map_xyz: np.ndarray,
    depth_m: np.ndarray,
    panorama_uv: np.ndarray,
    detection: Detection2D,
    config: ClusterSelectionConfig | None = None,
) -> ClusterSelectionResult:
    """Cluster filtered points and select the nearest substantial compact set."""
    policy = config or ClusterSelectionConfig()
    points = np.asarray(points_map_xyz, dtype=np.float64)
    depths = np.asarray(depth_m, dtype=np.float64)
    uv = np.asarray(panorama_uv, dtype=np.float64)
    count = points.shape[0]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points_map_xyz must have shape (N, 3)')
    if depths.shape != (count,) or uv.shape != (count, 2):
        raise ValueError('depth and UV arrays must align with points')
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(depths)):
        raise ValueError('cluster inputs must be finite')
    if count == 0:
        return ClusterSelectionResult(
            labels=np.empty(0, dtype=np.int64),
            selected_indices=np.empty(0, dtype=np.int64),
            noise_indices=np.empty(0, dtype=np.int64),
            selected_cluster_id=None,
            summaries=(),
            epsilon_m=policy.minimum_epsilon_m,
            ambiguous=False,
            reason='empty',
            alternative_scores={},
        )
    epsilon = float(np.clip(
        policy.base_epsilon_m + policy.range_epsilon_slope * np.median(depths),
        policy.minimum_epsilon_m,
        policy.maximum_epsilon_m,
    ))
    labels = deterministic_dbscan(points, epsilon, policy.minimum_samples)
    summaries = []
    cluster_ids = sorted(int(value) for value in np.unique(labels) if value >= 0)
    minimum_depth = min(
        (
            float(np.median(depths[labels == value]))
            for value in cluster_ids
        ),
        default=0.0,
    )
    for cluster_id in cluster_ids:
        mask = labels == cluster_id
        cluster_points = points[mask]
        cluster_depths = depths[mask]
        cluster_uv = uv[mask]
        point_count = cluster_points.shape[0]
        median_depth = float(np.median(cluster_depths))
        centre_u = _circular_mean_u(cluster_uv[:, 0], detection.panorama_box.panorama_width)
        centre_v = float(np.median(cluster_uv[:, 1]))
        extent = np.ptp(cluster_points, axis=0)
        point_score = min(1.0, point_count / 80.0)
        foreground_score = exp(-max(0.0, median_depth - minimum_depth) / 1.5)
        centre_score = _centre_alignment_score(
            centre_u, centre_v, detection
        )
        compactness = exp(-float(np.max(extent)) / 2.0)
        if np.max(extent) > policy.maximum_cluster_extent_m:
            compactness *= 0.05
        total = (
            0.25 * point_score
            + 0.35 * foreground_score
            + 0.25 * centre_score
            + 0.15 * compactness
        )
        summaries.append(
            ClusterSummary(
                cluster_id=cluster_id,
                point_count=point_count,
                median_depth_m=median_depth,
                centre_panorama_uv=(centre_u, centre_v),
                extent_xyz=extent,
                point_support_score=float(point_score),
                foreground_score=float(foreground_score),
                centre_alignment_score=float(centre_score),
                compactness_score=float(compactness),
                total_score=float(total),
            )
        )
    ranked = sorted(
        summaries,
        key=lambda item: (
            -item.total_score,
            item.median_depth_m,
            item.cluster_id,
        ),
    )
    selected_id = ranked[0].cluster_id if ranked else None
    selected = (
        np.flatnonzero(labels == selected_id)
        if selected_id is not None
        else np.empty(0, dtype=np.int64)
    )
    ambiguous = (
        len(ranked) > 1
        and ranked[0].total_score - ranked[1].total_score <= policy.ambiguity_score_delta
    )
    return ClusterSelectionResult(
        labels=labels,
        selected_indices=selected,
        noise_indices=np.flatnonzero(labels < 0),
        selected_cluster_id=selected_id,
        summaries=tuple(sorted(summaries, key=lambda item: item.cluster_id)),
        epsilon_m=epsilon,
        ambiguous=ambiguous,
        reason=(
            'multiple_plausible_clusters'
            if ambiguous
            else ('selected' if ranked else 'no_cluster')
        ),
        alternative_scores={item.cluster_id: item.total_score for item in ranked[1:]},
    )


def deterministic_dbscan(
    points_xyz: np.ndarray,
    epsilon_m: float,
    minimum_samples: int,
) -> np.ndarray:
    """Run bounded-neighborhood DBSCAN with lexicographic deterministic order."""
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points_xyz must have shape (N, 3)')
    if epsilon_m <= 0.0 or minimum_samples <= 0:
        raise ValueError('DBSCAN parameters must be positive')
    count = points.shape[0]
    if count == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((np.arange(count), points[:, 2], points[:, 1], points[:, 0]))
    sorted_points = points[order]
    cells = np.floor(sorted_points / epsilon_m).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, cell in enumerate(cells):
        buckets.setdefault(tuple(cell), []).append(index)

    def neighbours(index: int) -> list[int]:
        cell = cells[index]
        candidates = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    candidates.extend(buckets.get(tuple(cell + (dx, dy, dz)), ()))
        candidate_array = np.asarray(sorted(set(candidates)), dtype=np.int64)
        distances = np.linalg.norm(sorted_points[candidate_array] - sorted_points[index], axis=1)
        return candidate_array[distances <= epsilon_m + 1e-12].tolist()

    unvisited = -2
    noise = -1
    sorted_labels = np.full(count, unvisited, dtype=np.int64)
    cluster_id = 0
    for index in range(count):
        if sorted_labels[index] != unvisited:
            continue
        adjacent = neighbours(index)
        if len(adjacent) < minimum_samples:
            sorted_labels[index] = noise
            continue
        sorted_labels[index] = cluster_id
        queue = deque(adjacent)
        queued = set(adjacent)
        while queue:
            neighbour = queue.popleft()
            if sorted_labels[neighbour] == noise:
                sorted_labels[neighbour] = cluster_id
            if sorted_labels[neighbour] != unvisited:
                continue
            sorted_labels[neighbour] = cluster_id
            expanded = neighbours(neighbour)
            if len(expanded) >= minimum_samples:
                for candidate in expanded:
                    if candidate not in queued:
                        queue.append(candidate)
                        queued.add(candidate)
        cluster_id += 1
    labels = np.empty_like(sorted_labels)
    labels[order] = sorted_labels
    return labels


def _circular_mean_u(values: np.ndarray, width: int) -> float:
    angles = np.asarray(values) * (2.0 * np.pi / width)
    angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    return float(np.mod(angle, 2.0 * np.pi) * width / (2.0 * np.pi))


def _centre_alignment_score(u: float, v: float, detection: Detection2D) -> float:
    width = detection.panorama_box.panorama_width
    du = abs(u - detection.centre_panorama_uv[0])
    du = min(du, width - du)
    box_width = sum(end - start for start, end in detection.panorama_box.x_intervals)
    box_height = detection.panorama_box.y_max - detection.panorama_box.y_min
    normalized = np.hypot(
        du / max(box_width / 2.0, 1.0),
        (v - detection.centre_panorama_uv[1])
        / max(box_height / 2.0, 1.0),
    )
    return float(exp(-(normalized ** 2)))


__all__ = [
    'cluster_and_select',
    'ClusterSelectionConfig',
    'ClusterSelectionResult',
    'ClusterSummary',
    'deterministic_dbscan',
]
