"""Bounded dense map-frame scan accumulation with centroid voxel filtering."""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from threading import RLock

import numpy as np


@dataclass(frozen=True)
class DenseScanAccumulatorConfig:
    """Temporal, spatial, memory, and voxel policy for projection geometry."""

    frame_id: str = 'map'
    voxel_size_m: float = 0.04
    max_age_seconds: float = 15.0
    max_radius_m: float = 12.0
    max_points: int = 1_000_000

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ValueError('frame_id must be non-empty')
        for name in ('voxel_size_m', 'max_age_seconds', 'max_radius_m'):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        if isinstance(self.max_points, bool) or not isinstance(self.max_points, int):
            raise ValueError('max_points must be a positive integer')
        if self.max_points <= 0:
            raise ValueError('max_points must be a positive integer')


class DenseAccumulationStatus(Enum):
    """Outcome of one dense scan update."""

    ACCEPTED = 'accepted'
    EMPTY = 'empty'
    REJECTED_FRAME = 'rejected_frame'
    STALE = 'stale'


@dataclass(frozen=True)
class DenseAccumulationResult:
    """Summary of one dense scan insertion."""

    status: DenseAccumulationStatus
    input_point_count: int
    retained_scan_point_count: int
    raw_accumulated_point_count: int
    evicted_point_count: int


@dataclass(frozen=True)
class DenseAccumulatorStats:
    """Read-only dense accumulation counters and current bounds."""

    frame_id: str
    scan_count: int
    raw_point_count: int
    accepted_scan_count: int
    empty_scan_count: int
    rejected_scan_count: int
    stale_scan_count: int
    invalid_point_count: int
    evicted_point_count: int


@dataclass(frozen=True)
class DenseScanSnapshot:
    """Deterministic centroid voxels and their latest source timestamps."""

    points_xyz: np.ndarray
    last_seen_timestamp_ns: np.ndarray
    observation_count: np.ndarray
    raw_point_count: int
    voxel_size_m: float

    def __post_init__(self) -> None:
        points = np.asarray(self.points_xyz, dtype=np.float64).copy()
        stamps = np.asarray(self.last_seen_timestamp_ns, dtype=np.int64).copy()
        counts = np.asarray(self.observation_count, dtype=np.int64).copy()
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('points_xyz must have shape (N, 3)')
        if stamps.shape != (points.shape[0],) or counts.shape != (points.shape[0],):
            raise ValueError('snapshot metadata shape does not match points')
        for array in (points, stamps, counts):
            array.setflags(write=False)
        object.__setattr__(self, 'points_xyz', points)
        object.__setattr__(self, 'last_seen_timestamp_ns', stamps)
        object.__setattr__(self, 'observation_count', counts)

    def age_seconds(self, timestamp_ns: int) -> np.ndarray:
        """Return non-negative voxel ages relative to a query timestamp."""
        ages = (timestamp_ns - self.last_seen_timestamp_ns) / 1_000_000_000.0
        return np.maximum(0.0, ages)


@dataclass(frozen=True)
class _ScanChunk:
    timestamp_ns: int
    points_xyz: np.ndarray


def voxel_filter_centroids(
    points_xyz: np.ndarray,
    timestamps_ns: np.ndarray,
    voxel_size_m: float,
) -> DenseScanSnapshot:
    """Return lexicographically ordered voxel centroids and latest timestamps."""
    points = np.asarray(points_xyz, dtype=np.float64)
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points_xyz must have shape (N, 3)')
    if timestamps.shape != (points.shape[0],):
        raise ValueError('timestamps_ns must have one value per point')
    if not isfinite(voxel_size_m) or voxel_size_m <= 0.0:
        raise ValueError('voxel_size_m must be finite and positive')
    if not np.all(np.isfinite(points)):
        raise ValueError('points_xyz must contain only finite values')
    if points.shape[0] == 0:
        return DenseScanSnapshot(
            points_xyz=np.empty((0, 3), dtype=np.float64),
            last_seen_timestamp_ns=np.empty((0,), dtype=np.int64),
            observation_count=np.empty((0,), dtype=np.int64),
            raw_point_count=0,
            voxel_size_m=voxel_size_m,
        )
    keys = np.floor(points / voxel_size_m).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    voxel_count = int(np.max(inverse)) + 1
    sums = np.zeros((voxel_count, 3), dtype=np.float64)
    counts = np.zeros(voxel_count, dtype=np.int64)
    latest = np.full(voxel_count, np.iinfo(np.int64).min, dtype=np.int64)
    np.add.at(sums, inverse, points)
    np.add.at(counts, inverse, 1)
    np.maximum.at(latest, inverse, timestamps)
    return DenseScanSnapshot(
        points_xyz=sums / counts[:, None],
        last_seen_timestamp_ns=latest,
        observation_count=counts,
        raw_point_count=int(points.shape[0]),
        voxel_size_m=voxel_size_m,
    )


class DenseRegisteredScanAccumulator:
    """Retain recent local map points and expose centroid-voxel snapshots."""

    def __init__(self, config: DenseScanAccumulatorConfig | None = None) -> None:
        self.config = config or DenseScanAccumulatorConfig()
        self._chunks: deque[_ScanChunk] = deque()
        self._lock = RLock()
        self._latest_timestamp_ns: int | None = None
        self._accepted = 0
        self._empty = 0
        self._rejected = 0
        self._stale = 0
        self._invalid = 0
        self._evicted = 0

    def add_scan(
        self,
        points_xyz: np.ndarray,
        *,
        frame_id: str,
        timestamp_ns: int,
        sensor_origin_xyz: np.ndarray | None = None,
    ) -> DenseAccumulationResult:
        """Insert one finite/range-filtered scan and enforce all hard bounds."""
        points = np.asarray(points_xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('points_xyz must have shape (N, 3)')
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int):
            raise ValueError('timestamp_ns must be a non-negative integer')
        if timestamp_ns < 0:
            raise ValueError('timestamp_ns must be a non-negative integer')
        if sensor_origin_xyz is not None:
            origin = np.asarray(sensor_origin_xyz, dtype=np.float64)
            if origin.shape != (3,) or not np.all(np.isfinite(origin)):
                raise ValueError('sensor_origin_xyz must contain three finite values')
        else:
            origin = None
        input_count = int(points.shape[0])
        with self._lock:
            if frame_id != self.config.frame_id:
                self._rejected += 1
                return self._result(
                    DenseAccumulationStatus.REJECTED_FRAME,
                    input_count,
                    0,
                    0,
                )
            if self._latest_timestamp_ns is not None and timestamp_ns < self._latest_timestamp_ns:
                self._stale += 1
                return self._result(
                    DenseAccumulationStatus.STALE,
                    input_count,
                    0,
                    0,
                )
            self._latest_timestamp_ns = timestamp_ns
            finite = np.all(np.isfinite(points), axis=1)
            self._invalid += input_count - int(np.count_nonzero(finite))
            retained = points[finite]
            if origin is not None and retained.size:
                ranges = np.linalg.norm(retained - origin, axis=1)
                retained = retained[ranges <= self.config.max_radius_m]
            evicted = self._evict_locked(timestamp_ns, origin)
            self._accepted += 1
            if retained.shape[0] == 0:
                self._empty += 1
                return self._result(
                    DenseAccumulationStatus.EMPTY,
                    input_count,
                    0,
                    evicted,
                )
            copied = np.ascontiguousarray(retained).copy()
            copied.setflags(write=False)
            self._chunks.append(_ScanChunk(timestamp_ns, copied))
            evicted += self._enforce_cap_locked()
            return self._result(
                DenseAccumulationStatus.ACCEPTED,
                input_count,
                int(copied.shape[0]),
                evicted,
            )

    def snapshot(self, timestamp_ns: int | None = None) -> DenseScanSnapshot:
        """Voxel-filter chunks no newer than an optional image source time."""
        if timestamp_ns is not None and (
            isinstance(timestamp_ns, bool)
            or not isinstance(timestamp_ns, int)
            or timestamp_ns < 0
        ):
            raise ValueError('timestamp_ns must be a non-negative integer')
        with self._lock:
            if timestamp_ns is None:
                chunks = tuple(self._chunks)
            else:
                maximum_age_ns = int(
                    round(self.config.max_age_seconds * 1_000_000_000)
                )
                oldest_ns = timestamp_ns - maximum_age_ns
                chunks = tuple(
                    chunk
                    for chunk in self._chunks
                    if oldest_ns <= chunk.timestamp_ns <= timestamp_ns
                )
        if not chunks:
            return voxel_filter_centroids(
                np.empty((0, 3)),
                np.empty((0,), dtype=np.int64),
                self.config.voxel_size_m,
            )
        points = np.concatenate([chunk.points_xyz for chunk in chunks], axis=0)
        timestamps = np.concatenate(
            [
                np.full(chunk.points_xyz.shape[0], chunk.timestamp_ns, dtype=np.int64)
                for chunk in chunks
            ]
        )
        return voxel_filter_centroids(points, timestamps, self.config.voxel_size_m)

    def stats(self) -> DenseAccumulatorStats:
        """Return current raw bounds and lifetime counters."""
        with self._lock:
            return DenseAccumulatorStats(
                frame_id=self.config.frame_id,
                scan_count=len(self._chunks),
                raw_point_count=self._raw_count_locked(),
                accepted_scan_count=self._accepted,
                empty_scan_count=self._empty,
                rejected_scan_count=self._rejected,
                stale_scan_count=self._stale,
                invalid_point_count=self._invalid,
                evicted_point_count=self._evicted,
            )

    def reset(self) -> None:
        """Clear all geometry and counters at an episode boundary."""
        with self._lock:
            self._chunks.clear()
            self._latest_timestamp_ns = None
            self._accepted = 0
            self._empty = 0
            self._rejected = 0
            self._stale = 0
            self._invalid = 0
            self._evicted = 0

    def _evict_locked(self, timestamp_ns: int, origin: np.ndarray | None) -> int:
        max_age_ns = int(round(self.config.max_age_seconds * 1_000_000_000))
        oldest = timestamp_ns - max_age_ns
        removed = 0
        while self._chunks and self._chunks[0].timestamp_ns < oldest:
            removed += int(self._chunks.popleft().points_xyz.shape[0])
        if origin is not None and self._chunks:
            filtered_chunks = deque()
            for chunk in self._chunks:
                ranges = np.linalg.norm(chunk.points_xyz - origin, axis=1)
                kept = chunk.points_xyz[ranges <= self.config.max_radius_m]
                removed += int(chunk.points_xyz.shape[0] - kept.shape[0])
                if kept.shape[0]:
                    copied = np.ascontiguousarray(kept).copy()
                    copied.setflags(write=False)
                    filtered_chunks.append(_ScanChunk(chunk.timestamp_ns, copied))
            self._chunks = filtered_chunks
        self._evicted += removed
        return removed

    def _enforce_cap_locked(self) -> int:
        removed = 0
        while self._chunks and self._raw_count_locked() > self.config.max_points:
            excess = self._raw_count_locked() - self.config.max_points
            first = self._chunks[0]
            if first.points_xyz.shape[0] <= excess:
                removed += int(self._chunks.popleft().points_xyz.shape[0])
            else:
                kept = np.ascontiguousarray(first.points_xyz[excess:]).copy()
                kept.setflags(write=False)
                self._chunks[0] = _ScanChunk(first.timestamp_ns, kept)
                removed += excess
        self._evicted += removed
        return removed

    def _raw_count_locked(self) -> int:
        return sum(chunk.points_xyz.shape[0] for chunk in self._chunks)

    def _result(
        self,
        status: DenseAccumulationStatus,
        input_count: int,
        retained_count: int,
        evicted_count: int,
    ) -> DenseAccumulationResult:
        return DenseAccumulationResult(
            status=status,
            input_point_count=input_count,
            retained_scan_point_count=retained_count,
            raw_accumulated_point_count=self._raw_count_locked(),
            evicted_point_count=evicted_count,
        )


__all__ = [
    'DenseAccumulationResult',
    'DenseAccumulationStatus',
    'DenseAccumulatorStats',
    'DenseRegisteredScanAccumulator',
    'DenseScanAccumulatorConfig',
    'DenseScanSnapshot',
    'voxel_filter_centroids',
]
