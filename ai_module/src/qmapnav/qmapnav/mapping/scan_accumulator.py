"""Bounded map-frame registered-scan accumulation and free-space queries."""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import atan2
from math import cos
from math import floor
from math import hypot
from math import isfinite
from math import pi
from math import sin
from threading import RLock

import numpy as np


@dataclass(frozen=True)
class ScanAccumulatorConfig:
    """Memory, geometry, and frame policy for registered-scan accumulation."""

    frame_id: str = 'map'
    voxel_size: float = 0.20
    max_range: float = 30.0
    max_age_seconds: float = 120.0
    max_voxels: int = 200_000
    max_scan_views: int = 16
    ray_angle_resolution: float = pi / 90.0
    navigation_min_z: float = 0.10
    navigation_max_z: float = 1.80
    max_view_origin_distance: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ValueError('frame_id must be a non-empty string')
        positive_values = {
            'voxel_size': self.voxel_size,
            'max_range': self.max_range,
            'max_age_seconds': self.max_age_seconds,
            'ray_angle_resolution': self.ray_angle_resolution,
            'max_view_origin_distance': self.max_view_origin_distance,
        }
        for name, value in positive_values.items():
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        for name, value in {
            'max_voxels': self.max_voxels,
            'max_scan_views': self.max_scan_views,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if not isfinite(self.navigation_min_z):
            raise ValueError('navigation_min_z must be finite')
        if not isfinite(self.navigation_max_z):
            raise ValueError('navigation_max_z must be finite')
        if self.navigation_min_z >= self.navigation_max_z:
            raise ValueError('navigation_min_z must be below navigation_max_z')
        if self.ray_angle_resolution > 2.0 * pi:
            raise ValueError('ray_angle_resolution must not exceed 2*pi')


class AccumulationStatus(Enum):
    """Outcome of one attempted registered-scan accumulation."""

    ACCEPTED = 'accepted'
    EMPTY = 'empty'
    REJECTED_FRAME = 'rejected_frame'
    STALE = 'stale'


@dataclass(frozen=True)
class AccumulationResult:
    """Summary of one scan update without exposing mutable map state."""

    status: AccumulationStatus
    input_point_count: int
    finite_point_count: int
    unique_voxels_observed: int
    voxel_count: int
    evicted_voxel_count: int


@dataclass(frozen=True)
class ScanAccumulatorStats:
    """Read-only bounded accumulator counters for monitoring and traces."""

    frame_id: str
    voxel_count: int
    scan_view_count: int
    accepted_scan_count: int
    empty_scan_count: int
    rejected_scan_count: int
    stale_scan_count: int
    invalid_point_count: int
    evicted_voxel_count: int


@dataclass(frozen=True)
class _VoxelRecord:
    last_seen: float


@dataclass(frozen=True)
class _ScanView:
    timestamp: float
    origin_x: float
    origin_y: float
    nearest_ranges: dict[int, float]


class RegisteredScanAccumulator:
    """Voxelize registered XYZ points under explicit bounded-map policies."""

    def __init__(
        self,
        config: ScanAccumulatorConfig | None = None,
    ) -> None:
        self._config = config or ScanAccumulatorConfig()
        self._lock = RLock()
        self._voxels: dict[tuple[int, int, int], _VoxelRecord] = {}
        self._scan_views: deque[_ScanView] = deque()
        self._accepted_scan_count = 0
        self._empty_scan_count = 0
        self._rejected_scan_count = 0
        self._stale_scan_count = 0
        self._invalid_point_count = 0
        self._evicted_voxel_count = 0
        self._latest_timestamp: float | None = None

    @property
    def config(self) -> ScanAccumulatorConfig:
        """Return the immutable accumulator configuration."""
        return self._config

    def add_scan(
        self,
        points_xyz: np.ndarray,
        *,
        frame_id: str,
        timestamp: float,
        sensor_origin_xy: tuple[float, float] | None = None,
    ) -> AccumulationResult:
        """Merge one registered XYZ scan after validating frame and points."""
        points = np.asarray(points_xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('points_xyz must have shape (N, 3)')
        if not isfinite(timestamp) or timestamp < 0.0:
            raise ValueError('timestamp must be finite and non-negative')
        origin = self._validate_origin(sensor_origin_xy)
        input_count = int(points.shape[0])

        with self._lock:
            if frame_id != self._config.frame_id:
                self._rejected_scan_count += 1
                return self._result(
                    AccumulationStatus.REJECTED_FRAME,
                    input_count,
                    0,
                    0,
                    0,
                )
            if (
                self._latest_timestamp is not None
                and timestamp < self._latest_timestamp
            ):
                self._stale_scan_count += 1
                return self._result(
                    AccumulationStatus.STALE,
                    input_count,
                    0,
                    0,
                    0,
                )

            self._latest_timestamp = timestamp
            finite_mask = np.all(np.isfinite(points), axis=1)
            invalid_count = input_count - int(np.count_nonzero(finite_mask))
            self._invalid_point_count += invalid_count
            finite_points = points[finite_mask]
            if origin is not None and finite_points.size:
                offsets = finite_points[:, :2] - np.asarray(origin)
                ranges = np.linalg.norm(offsets, axis=1)
                finite_points = finite_points[ranges <= self._config.max_range]

            finite_count = int(finite_points.shape[0])
            self._accepted_scan_count += 1
            evicted = self._evict(timestamp, origin)
            if finite_count == 0:
                self._empty_scan_count += 1
                return self._result(
                    AccumulationStatus.EMPTY,
                    input_count,
                    finite_count,
                    0,
                    evicted,
                )

            voxel_keys = np.floor(
                finite_points / self._config.voxel_size
            ).astype(np.int64)
            unique_keys = np.unique(voxel_keys, axis=0)
            for key_values in unique_keys:
                key = tuple(int(value) for value in key_values)
                self._voxels[key] = _VoxelRecord(timestamp)

            if origin is not None:
                self._record_scan_view(finite_points, timestamp, origin)
            evicted += self._enforce_cap()
            return self._result(
                AccumulationStatus.ACCEPTED,
                input_count,
                finite_count,
                int(unique_keys.shape[0]),
                evicted,
            )

    def stats(self) -> ScanAccumulatorStats:
        """Return a consistent immutable statistics snapshot."""
        with self._lock:
            return ScanAccumulatorStats(
                frame_id=self._config.frame_id,
                voxel_count=len(self._voxels),
                scan_view_count=len(self._scan_views),
                accepted_scan_count=self._accepted_scan_count,
                empty_scan_count=self._empty_scan_count,
                rejected_scan_count=self._rejected_scan_count,
                stale_scan_count=self._stale_scan_count,
                invalid_point_count=self._invalid_point_count,
                evicted_voxel_count=self._evicted_voxel_count,
            )

    def snapshot_points(self) -> np.ndarray:
        """Return sorted voxel-centre XYZ points as a defensive array copy."""
        with self._lock:
            keys = sorted(self._voxels)
        if not keys:
            return np.empty((0, 3), dtype=np.float64)
        key_array = np.asarray(keys, dtype=np.float64)
        return (key_array + 0.5) * self._config.voxel_size

    def reset(self) -> None:
        """Clear map contents and counters at an episode/process boundary."""
        with self._lock:
            self._voxels.clear()
            self._scan_views.clear()
            self._accepted_scan_count = 0
            self._empty_scan_count = 0
            self._rejected_scan_count = 0
            self._stale_scan_count = 0
            self._invalid_point_count = 0
            self._evicted_voxel_count = 0
            self._latest_timestamp = None

    def is_known_free(
        self,
        x: float,
        y: float,
        *,
        clearance: float,
    ) -> bool:
        """Return whether scan-ray evidence proves a clear planar candidate."""
        if not all(isfinite(value) for value in (x, y, clearance)):
            return False
        if clearance <= 0.0:
            return False
        with self._lock:
            if not self._candidate_has_free_ray(x, y, clearance):
                return False
            return not self._has_occupied_voxel_near(x, y, clearance)

    def select_safe_offset(
        self,
        current_x: float,
        current_y: float,
        goal_x: float,
        goal_y: float,
        *,
        offset_distance: float = 0.75,
        clearance: float = 0.35,
    ) -> tuple[float, float, float] | None:
        """Choose a deterministic observed-free lateral/backward recovery pose."""
        values = (
            current_x,
            current_y,
            goal_x,
            goal_y,
            offset_distance,
            clearance,
        )
        if not all(isfinite(value) for value in values):
            return None
        if offset_distance <= 0.0 or clearance <= 0.0:
            return None

        goal_heading = atan2(goal_y - current_y, goal_x - current_x)
        recovery_angles = (
            goal_heading + pi / 2.0,
            goal_heading - pi / 2.0,
            goal_heading + 3.0 * pi / 4.0,
            goal_heading - 3.0 * pi / 4.0,
            goal_heading + pi,
        )
        for angle in recovery_angles:
            candidate_x = current_x + offset_distance * cos(angle)
            candidate_y = current_y + offset_distance * sin(angle)
            if self.is_known_free(
                candidate_x,
                candidate_y,
                clearance=clearance,
            ):
                return candidate_x, candidate_y, goal_heading
        return None

    def _record_scan_view(
        self,
        points: np.ndarray,
        timestamp: float,
        origin: tuple[float, float],
    ) -> None:
        navigation_mask = (
            (points[:, 2] >= self._config.navigation_min_z)
            & (points[:, 2] <= self._config.navigation_max_z)
        )
        navigation_points = points[navigation_mask]
        nearest_ranges: dict[int, float] = {}
        if navigation_points.size:
            dx = navigation_points[:, 0] - origin[0]
            dy = navigation_points[:, 1] - origin[1]
            ranges = np.hypot(dx, dy)
            angles = np.arctan2(dy, dx)
            bins = np.floor(
                (angles + pi) / self._config.ray_angle_resolution
            ).astype(np.int64)
            for bin_index, distance in zip(bins, ranges):
                if distance <= 0.0:
                    continue
                key = int(bin_index)
                previous = nearest_ranges.get(key)
                if previous is None or distance < previous:
                    nearest_ranges[key] = float(distance)

        self._scan_views.append(
            _ScanView(
                timestamp=timestamp,
                origin_x=origin[0],
                origin_y=origin[1],
                nearest_ranges=nearest_ranges,
            )
        )
        while len(self._scan_views) > self._config.max_scan_views:
            self._scan_views.popleft()

    def _candidate_has_free_ray(
        self,
        x: float,
        y: float,
        clearance: float,
    ) -> bool:
        for view in reversed(self._scan_views):
            dx = x - view.origin_x
            dy = y - view.origin_y
            candidate_range = hypot(dx, dy)
            if candidate_range > self._config.max_range:
                continue
            if candidate_range > self._config.max_view_origin_distance:
                continue
            angle = atan2(dy, dx)
            bin_index = int(
                floor((angle + pi) / self._config.ray_angle_resolution)
            )
            observed_range = view.nearest_ranges.get(bin_index)
            if observed_range is not None and observed_range >= (
                candidate_range + clearance
            ):
                return True
        return False

    def _has_occupied_voxel_near(
        self,
        x: float,
        y: float,
        clearance: float,
    ) -> bool:
        size = self._config.voxel_size
        min_x = floor((x - clearance) / size)
        max_x = floor((x + clearance) / size)
        min_y = floor((y - clearance) / size)
        max_y = floor((y + clearance) / size)
        min_z = floor(self._config.navigation_min_z / size)
        max_z = floor(self._config.navigation_max_z / size)
        clearance_squared = clearance * clearance
        for key_x in range(min_x, max_x + 1):
            for key_y in range(min_y, max_y + 1):
                centre_x = (key_x + 0.5) * size
                centre_y = (key_y + 0.5) * size
                if (
                    (centre_x - x) ** 2 + (centre_y - y) ** 2
                    > clearance_squared
                ):
                    continue
                for key_z in range(min_z, max_z + 1):
                    if (key_x, key_y, key_z) in self._voxels:
                        return True
        return False

    def _evict(
        self,
        timestamp: float,
        origin: tuple[float, float] | None,
    ) -> int:
        oldest_allowed = timestamp - self._config.max_age_seconds
        removed_keys = {
            key
            for key, record in self._voxels.items()
            if record.last_seen < oldest_allowed
        }
        if origin is not None:
            max_range_squared = self._config.max_range ** 2
            size = self._config.voxel_size
            removed_keys.update(
                key
                for key in self._voxels
                if (
                    ((key[0] + 0.5) * size - origin[0]) ** 2
                    + ((key[1] + 0.5) * size - origin[1]) ** 2
                    > max_range_squared
                )
            )
        for key in removed_keys:
            del self._voxels[key]

        while (
            self._scan_views
            and self._scan_views[0].timestamp < oldest_allowed
        ):
            self._scan_views.popleft()
        self._evicted_voxel_count += len(removed_keys)
        return len(removed_keys)

    def _enforce_cap(self) -> int:
        excess = len(self._voxels) - self._config.max_voxels
        if excess <= 0:
            return 0
        oldest = sorted(
            self._voxels,
            key=lambda key: (self._voxels[key].last_seen, key),
        )[:excess]
        for key in oldest:
            del self._voxels[key]
        self._evicted_voxel_count += len(oldest)
        return len(oldest)

    def _result(
        self,
        status: AccumulationStatus,
        input_count: int,
        finite_count: int,
        unique_count: int,
        evicted_count: int,
    ) -> AccumulationResult:
        return AccumulationResult(
            status=status,
            input_point_count=input_count,
            finite_point_count=finite_count,
            unique_voxels_observed=unique_count,
            voxel_count=len(self._voxels),
            evicted_voxel_count=evicted_count,
        )

    @staticmethod
    def _validate_origin(
        origin: tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        if origin is None:
            return None
        if len(origin) != 2 or not all(isfinite(value) for value in origin):
            raise ValueError('sensor_origin_xy must contain two finite values')
        return float(origin[0]), float(origin[1])


__all__ = [
    'AccumulationResult',
    'AccumulationStatus',
    'RegisteredScanAccumulator',
    'ScanAccumulatorConfig',
    'ScanAccumulatorStats',
]
