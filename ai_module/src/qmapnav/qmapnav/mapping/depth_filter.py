"""Deterministic foreground depth-layer selection."""

from dataclasses import dataclass
from math import ceil, isfinite

import numpy as np

from qmapnav.mapping.object_candidate import readonly_array


@dataclass(frozen=True)
class DepthFilterConfig:
    """Histogram support and foreground-band policy."""

    bin_width_m: float = 0.15
    minimum_mode_points: int = 5
    minimum_mode_fraction: float = 0.08
    maximum_band_width_m: float = 1.50
    fallback_front_percentile: float = 15.0
    fallback_band_width_m: float = 1.0

    def __post_init__(self) -> None:
        if not isfinite(self.bin_width_m) or self.bin_width_m <= 0.0:
            raise ValueError('bin_width_m must be positive')
        if self.minimum_mode_points <= 0:
            raise ValueError('minimum_mode_points must be positive')
        if not 0.0 <= self.minimum_mode_fraction <= 1.0:
            raise ValueError('minimum_mode_fraction must lie in [0, 1]')
        if self.maximum_band_width_m <= 0.0 or self.fallback_band_width_m <= 0.0:
            raise ValueError('depth band widths must be positive')
        if not 0.0 <= self.fallback_front_percentile <= 100.0:
            raise ValueError('fallback percentile must lie in [0, 100]')


@dataclass(frozen=True)
class DepthFilterResult:
    """Foreground indices and depth-layer diagnostics."""

    kept_indices: np.ndarray
    removed_indices: np.ndarray
    lower_depth_m: float | None
    upper_depth_m: float | None
    mode_support: int
    contamination_fraction: float
    used_fallback: bool
    reason: str

    def __post_init__(self) -> None:
        for name in ('kept_indices', 'removed_indices'):
            object.__setattr__(
                self,
                name,
                readonly_array(name, getattr(self, name), (None,), dtype=np.int64),
            )


def select_foreground_depth_layer(
    depth_m: np.ndarray,
    config: DepthFilterConfig | None = None,
) -> DepthFilterResult:
    """Keep the nearest substantial contiguous histogram depth mode."""
    policy = config or DepthFilterConfig()
    depths = np.asarray(depth_m, dtype=np.float64)
    if depths.ndim != 1:
        raise ValueError('depth_m must be one-dimensional')
    if not np.all(np.isfinite(depths)) or np.any(depths <= 0.0):
        raise ValueError('depth_m must contain finite positive values')
    indices = np.arange(depths.size, dtype=np.int64)
    if depths.size == 0:
        return DepthFilterResult(indices, indices, None, None, 0, 0.0, False, 'empty')
    minimum_support = max(
        policy.minimum_mode_points,
        int(ceil(depths.size * policy.minimum_mode_fraction)),
    )
    origin = np.floor(np.min(depths) / policy.bin_width_m) * policy.bin_width_m
    bin_ids = np.floor((depths - origin) / policy.bin_width_m).astype(np.int64)
    counts = np.bincount(bin_ids)
    occupied = np.flatnonzero(counts)
    groups = []
    if occupied.size:
        splits = np.flatnonzero(np.diff(occupied) > 1) + 1
        for group in np.split(occupied, splits):
            support = int(np.sum(counts[group]))
            groups.append((int(group[0]), int(group[-1]), support))
    substantial = [group for group in groups if group[2] >= minimum_support]
    used_fallback = not substantial
    if substantial:
        first_bin, last_bin, support = substantial[0]
        lower = origin + first_bin * policy.bin_width_m
        upper = min(
            origin + (last_bin + 1) * policy.bin_width_m,
            lower + policy.maximum_band_width_m,
        )
        reason = 'nearest_substantial_histogram_mode'
    else:
        front = float(np.percentile(depths, policy.fallback_front_percentile))
        lower = max(0.0, front - policy.bin_width_m)
        upper = front + policy.fallback_band_width_m
        support = int(np.count_nonzero((depths >= lower) & (depths <= upper)))
        reason = 'robust_front_percentile_fallback'
    keep = (depths >= lower - 1e-12) & (depths <= upper + 1e-12)
    kept = indices[keep]
    removed = indices[~keep]
    return DepthFilterResult(
        kept_indices=kept,
        removed_indices=removed,
        lower_depth_m=float(lower),
        upper_depth_m=float(upper),
        mode_support=support,
        contamination_fraction=float(removed.size / depths.size),
        used_fallback=used_fallback,
        reason=reason,
    )


__all__ = ['DepthFilterConfig', 'DepthFilterResult', 'select_foreground_depth_layer']
