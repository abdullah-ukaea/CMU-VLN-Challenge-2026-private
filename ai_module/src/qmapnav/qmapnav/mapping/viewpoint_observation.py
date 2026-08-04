"""Episode-local metadata for one 3D object observation."""

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping

import numpy as np

from qmapnav.mapping.object_candidate import readonly_array


VISIBILITY_VALUES = frozenset({'full', 'partial', 'sparse'})


@dataclass(frozen=True)
class ViewpointObservation:
    """Pose, time, visibility, and optional crop for one candidate."""

    viewpoint_id: str
    robot_pose_xyz_yaw: np.ndarray
    timestamp_ns: int
    detection_id: str
    point_count: int
    geometry_confidence: float
    visibility: str
    best_crop: np.ndarray | None = None
    best_crop_score: float = 0.0
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        for name in ('viewpoint_id', 'detection_id'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{name} must be a non-empty string')
        pose = readonly_array(
            'robot_pose_xyz_yaw', self.robot_pose_xyz_yaw, (4,)
        )
        object.__setattr__(self, 'robot_pose_xyz_yaw', pose)
        if (
            isinstance(self.timestamp_ns, bool)
            or not isinstance(self.timestamp_ns, int)
            or self.timestamp_ns < 0
        ):
            raise ValueError('timestamp_ns must be a non-negative integer')
        if (
            isinstance(self.point_count, bool)
            or not isinstance(self.point_count, int)
            or self.point_count < 0
        ):
            raise ValueError('point_count must be a non-negative integer')
        if not isfinite(self.geometry_confidence) or not (
            0.0 <= self.geometry_confidence <= 1.0
        ):
            raise ValueError('geometry_confidence must lie in [0, 1]')
        if self.visibility not in VISIBILITY_VALUES:
            raise ValueError('visibility must be full, partial, or sparse')
        if not isfinite(self.best_crop_score) or not (
            0.0 <= self.best_crop_score <= 1.0
        ):
            raise ValueError('best_crop_score must lie in [0, 1]')
        if self.best_crop is not None:
            crop = np.asarray(self.best_crop)
            if crop.ndim != 3 or crop.shape[2] != 3 or crop.size == 0:
                raise ValueError('best_crop must have shape (H, W, 3)')
            copied = np.ascontiguousarray(crop).copy()
            copied.setflags(write=False)
            object.__setattr__(self, 'best_crop', copied)
        object.__setattr__(
            self,
            'metadata',
            MappingProxyType(dict(self.metadata or {})),
        )


__all__ = ['VISIBILITY_VALUES', 'ViewpointObservation']
