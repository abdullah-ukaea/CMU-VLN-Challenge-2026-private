"""Immutable sensor records and bounded source-timestamp association."""

from bisect import bisect_left
from dataclasses import dataclass
from threading import RLock
from typing import Generic, TypeVar

import numpy as np

from qmapnav.mapping.pose_interpolation import normalize_quaternion_xyzw
from qmapnav.mapping.pose_interpolation import slerp_quaternion_xyzw


def _timestamp(value: int, name: str = 'timestamp_ns') -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a non-negative integer')
    return value


def _non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')
    return value


def _readonly(value: np.ndarray, shape: tuple[int | None, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape)
    ):
        raise ValueError(f'array must have shape {shape}')
    if not np.all(np.isfinite(array)):
        raise ValueError('array must contain only finite values')
    copied = np.ascontiguousarray(array).copy()
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class TimedPanorama:
    """One immutable RGB panorama with source and receipt timestamps."""

    image_id: str
    timestamp_ns: int
    frame_id: str
    image_rgb: np.ndarray
    receipt_timestamp_ns: int | None = None

    def __post_init__(self) -> None:
        _non_empty(self.image_id, 'image_id')
        _timestamp(self.timestamp_ns)
        _non_empty(self.frame_id, 'frame_id')
        if self.receipt_timestamp_ns is not None:
            _timestamp(self.receipt_timestamp_ns, 'receipt_timestamp_ns')
        image = _readonly(self.image_rgb, (None, None, 3))
        if image.shape[0] < 2 or image.shape[1] < 2:
            raise ValueError('image_rgb dimensions must be at least 2 x 2')
        object.__setattr__(self, 'image_rgb', image)


@dataclass(frozen=True)
class TimedRegisteredScan:
    """One finite map-frame registered scan and optional intensity."""

    timestamp_ns: int
    frame_id: str
    points_xyz: np.ndarray
    intensity: np.ndarray | None = None
    receipt_timestamp_ns: int | None = None

    def __post_init__(self) -> None:
        _timestamp(self.timestamp_ns)
        _non_empty(self.frame_id, 'frame_id')
        if self.receipt_timestamp_ns is not None:
            _timestamp(self.receipt_timestamp_ns, 'receipt_timestamp_ns')
        points = _readonly(self.points_xyz, (None, 3)).astype(
            np.float64,
            copy=False,
        )
        points.setflags(write=False)
        object.__setattr__(self, 'points_xyz', points)
        if self.intensity is not None:
            intensity = _readonly(self.intensity, (points.shape[0],)).astype(
                np.float64,
                copy=False,
            )
            intensity.setflags(write=False)
            object.__setattr__(self, 'intensity', intensity)


@dataclass(frozen=True)
class TimedPose:
    """One ROS-style parent-from-child pose at a source timestamp."""

    timestamp_ns: int
    parent_frame_id: str
    child_frame_id: str
    position_xyz: np.ndarray
    orientation_xyzw: np.ndarray
    receipt_timestamp_ns: int | None = None

    def __post_init__(self) -> None:
        _timestamp(self.timestamp_ns)
        _non_empty(self.parent_frame_id, 'parent_frame_id')
        _non_empty(self.child_frame_id, 'child_frame_id')
        if self.receipt_timestamp_ns is not None:
            _timestamp(self.receipt_timestamp_ns, 'receipt_timestamp_ns')
        position = _readonly(self.position_xyz, (3,)).astype(np.float64, copy=False)
        orientation = normalize_quaternion_xyzw(self.orientation_xyzw)
        position.setflags(write=False)
        orientation.setflags(write=False)
        object.__setattr__(self, 'position_xyz', position)
        object.__setattr__(self, 'orientation_xyzw', orientation)


def interpolate_pose(
    before: TimedPose,
    after: TimedPose,
    timestamp_ns: int,
) -> tuple[TimedPose, float]:
    """Interpolate a pose at ``timestamp_ns`` and return its interpolation ratio."""
    _timestamp(timestamp_ns)
    if before.parent_frame_id != after.parent_frame_id:
        raise ValueError('pose parent frames do not match')
    if before.child_frame_id != after.child_frame_id:
        raise ValueError('pose child frames do not match')
    if after.timestamp_ns < before.timestamp_ns:
        raise ValueError('after pose precedes before pose')
    if not before.timestamp_ns <= timestamp_ns <= after.timestamp_ns:
        raise ValueError('timestamp_ns must lie between the pose samples')
    duration = after.timestamp_ns - before.timestamp_ns
    ratio = 0.0 if duration == 0 else (timestamp_ns - before.timestamp_ns) / duration
    position = before.position_xyz + ratio * (
        after.position_xyz - before.position_xyz
    )
    orientation = slerp_quaternion_xyzw(
        before.orientation_xyzw,
        after.orientation_xyzw,
        ratio,
    )
    return TimedPose(
        timestamp_ns=timestamp_ns,
        parent_frame_id=before.parent_frame_id,
        child_frame_id=before.child_frame_id,
        position_xyz=position,
        orientation_xyzw=orientation,
    ), float(ratio)


T = TypeVar('T')


class TimedBuffer(Generic[T]):
    """Thread-safe sorted buffer bounded by source-time span and item count."""

    def __init__(self, *, duration_ns: int, max_items: int) -> None:
        if isinstance(duration_ns, bool) or not isinstance(duration_ns, int) or duration_ns <= 0:
            raise ValueError('duration_ns must be a positive integer')
        if isinstance(max_items, bool) or not isinstance(max_items, int) or max_items <= 0:
            raise ValueError('max_items must be a positive integer')
        self._duration_ns = duration_ns
        self._max_items = max_items
        self._items: list[T] = []
        self._lock = RLock()

    def add(self, item: T) -> bool:
        """Insert by timestamp; replace an equal timestamp deterministically."""
        timestamp_ns = _timestamp(getattr(item, 'timestamp_ns'))
        with self._lock:
            stamps = [getattr(existing, 'timestamp_ns') for existing in self._items]
            index = bisect_left(stamps, timestamp_ns)
            replaced = index < len(self._items) and stamps[index] == timestamp_ns
            if replaced:
                self._items[index] = item
            else:
                self._items.insert(index, item)
            self._evict_locked()
            return not replaced

    def snapshot(self) -> tuple[T, ...]:
        """Return an immutable item snapshot in source-time order."""
        with self._lock:
            return tuple(self._items)

    def nearest(self, timestamp_ns: int, max_delta_ns: int) -> T | None:
        """Return nearest acceptable item, choosing the earlier item on a tie."""
        _timestamp(timestamp_ns)
        _timestamp(max_delta_ns, 'max_delta_ns')
        with self._lock:
            if not self._items:
                return None
            return min(
                self._items,
                key=lambda item: (
                    abs(getattr(item, 'timestamp_ns') - timestamp_ns),
                    getattr(item, 'timestamp_ns'),
                ),
            ) if min(
                abs(getattr(item, 'timestamp_ns') - timestamp_ns)
                for item in self._items
            ) <= max_delta_ns else None

    def bracket(self, timestamp_ns: int) -> tuple[T | None, T | None]:
        """Return nearest samples at or before and at or after a timestamp."""
        _timestamp(timestamp_ns)
        with self._lock:
            stamps = [getattr(item, 'timestamp_ns') for item in self._items]
            index = bisect_left(stamps, timestamp_ns)
            after = self._items[index] if index < len(self._items) else None
            if after is not None and getattr(after, 'timestamp_ns') == timestamp_ns:
                return after, after
            before = self._items[index - 1] if index > 0 else None
            return before, after

    def clear(self) -> None:
        """Remove every buffered item."""
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def _evict_locked(self) -> None:
        if not self._items:
            return
        newest = getattr(self._items[-1], 'timestamp_ns')
        oldest_allowed = newest - self._duration_ns
        while self._items and getattr(self._items[0], 'timestamp_ns') < oldest_allowed:
            self._items.pop(0)
        excess = len(self._items) - self._max_items
        if excess > 0:
            del self._items[:excess]


@dataclass(frozen=True)
class AssociationConfig:
    """Measured bounds for image, scan, and pose source-time association."""

    max_pose_delta_ns: int = 50_000_000
    max_scan_delta_ns: int = 150_000_000
    buffer_duration_ns: int = 5_000_000_000
    max_pose_items: int = 2_000
    max_scan_items: int = 64

    def __post_init__(self) -> None:
        for name in ('max_pose_delta_ns', 'max_scan_delta_ns', 'buffer_duration_ns'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        for name in ('max_pose_items', 'max_scan_items'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')


@dataclass(frozen=True)
class AssociationResult:
    """One accepted image/scan/pose association with timing diagnostics."""

    panorama: TimedPanorama
    scan: TimedRegisteredScan
    pose: TimedPose
    pose_mode: str
    image_scan_delta_ns: int
    pose_before_delta_ns: int | None
    pose_after_delta_ns: int | None
    interpolation_ratio: float | None


@dataclass(frozen=True)
class AssociationFailure:
    """Explicit reason why an image could not be associated."""

    panorama: TimedPanorama
    reason: str


class ProjectionSynchronizer:
    """Associate panorama keyframes to map scans and image-time sensor poses."""

    def __init__(self, config: AssociationConfig | None = None) -> None:
        self.config = config or AssociationConfig()
        self.scans = TimedBuffer[TimedRegisteredScan](
            duration_ns=self.config.buffer_duration_ns,
            max_items=self.config.max_scan_items,
        )
        self.poses = TimedBuffer[TimedPose](
            duration_ns=self.config.buffer_duration_ns,
            max_items=self.config.max_pose_items,
        )

    def associate(
        self,
        panorama: TimedPanorama,
    ) -> AssociationResult | AssociationFailure:
        """Associate by source timestamp, interpolating pose when bracketed."""
        scan = self.scans.nearest(
            panorama.timestamp_ns,
            self.config.max_scan_delta_ns,
        )
        if scan is None:
            return AssociationFailure(panorama, 'no_scan_within_threshold')
        before, after = self.poses.bracket(panorama.timestamp_ns)
        before_delta = (
            panorama.timestamp_ns - before.timestamp_ns
            if before is not None
            else None
        )
        after_delta = (
            after.timestamp_ns - panorama.timestamp_ns
            if after is not None
            else None
        )
        pose = None
        mode = ''
        ratio = None
        if before is not None and after is not None:
            if (
                before_delta <= self.config.max_pose_delta_ns
                and after_delta <= self.config.max_pose_delta_ns
            ):
                pose, ratio = interpolate_pose(
                    before,
                    after,
                    panorama.timestamp_ns,
                )
                mode = 'exact' if before is after else 'interpolated'
        if pose is None:
            nearest = self.poses.nearest(
                panorama.timestamp_ns,
                self.config.max_pose_delta_ns,
            )
            if nearest is None:
                return AssociationFailure(panorama, 'no_pose_within_threshold')
            pose = nearest
            mode = 'nearest_fallback'
        return AssociationResult(
            panorama=panorama,
            scan=scan,
            pose=pose,
            pose_mode=mode,
            image_scan_delta_ns=scan.timestamp_ns - panorama.timestamp_ns,
            pose_before_delta_ns=before_delta,
            pose_after_delta_ns=after_delta,
            interpolation_ratio=ratio,
        )

    def clear(self) -> None:
        """Clear both sensor buffers at an episode boundary."""
        self.scans.clear()
        self.poses.clear()


__all__ = [
    'AssociationConfig',
    'AssociationFailure',
    'AssociationResult',
    'ProjectionSynchronizer',
    'TimedBuffer',
    'TimedPanorama',
    'TimedPose',
    'TimedRegisteredScan',
    'interpolate_pose',
]
