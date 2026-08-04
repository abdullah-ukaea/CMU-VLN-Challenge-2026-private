"""Contract tests for single-observation candidates."""

import numpy as np
import pytest

from qmapnav.mapping.object_candidate import ConfidenceComponents
from qmapnav.mapping.object_candidate import GeometrySource
from qmapnav.mapping.object_candidate import GeometryStatus
from qmapnav.mapping.object_candidate import LiftingCounts
from qmapnav.mapping.object_candidate import ObjectCandidate3D


def _components() -> ConfidenceComponents:
    return ConfidenceComponents(*(0.8 for _ in range(9)))


def _candidate() -> ObjectCandidate3D:
    points = np.array([[0.0, 0.0, 0.1], [1.0, 0.5, 1.1], [0.5, 0.2, 0.6]])
    counts = LiftingCounts(10, 8, 0, 7, 5, 3, 3)
    return ObjectCandidate3D(
        candidate_id='image:chair:0',
        detection_id='chair:0',
        class_name='chair',
        detection_confidence=0.8,
        source=GeometrySource.CURRENT,
        source_timestamp_ns=1,
        image_timestamp_ns=1,
        scan_timestamp_ns=1,
        pose_timestamp_ns=1,
        pose_mode='interpolated',
        image_scan_delta_ms=0.0,
        pose_before_delta_ms=0.0,
        pose_after_delta_ms=0.0,
        timing_warning=False,
        points_map_xyz=points,
        source_projection_indices=np.array([1, 3, 5]),
        point_centroid_xyz=np.median(points, axis=0),
        aabb_min_xyz=np.array([0.0, 0.0, 0.1]),
        aabb_max_xyz=np.array([1.0, 0.5, 1.1]),
        obb_centre_xyz=np.array([0.5, 0.25, 0.6]),
        obb_dimensions_xyz=np.array([1.0, 0.5, 1.0]),
        obb_yaw_rad=0.0,
        estimated_yaw_rad=0.0,
        orientation_confidence=0.8,
        geometry_confidence=0.7,
        geometry_status=GeometryStatus.GOOD,
        partial_geometry=False,
        low_orientation_fallback=False,
        counts=counts,
        confidence_components=_components(),
        diagnostics={'mode': 'box'},
    )


def test_candidate_defensively_copies_arrays_and_mapping() -> None:
    candidate = _candidate()

    assert candidate.point_count == 3
    assert not candidate.points_map_xyz.flags.writeable
    assert not candidate.source_projection_indices.flags.writeable
    with pytest.raises(ValueError):
        candidate.points_map_xyz[0, 0] = 2.0
    with pytest.raises(TypeError):
        candidate.diagnostics['mode'] = 'mask'


def test_candidate_rejects_invalid_box_and_count() -> None:
    candidate = _candidate()
    values = dict(candidate.__dict__)
    values['aabb_max_xyz'] = values['aabb_min_xyz']
    with pytest.raises(ValueError, match='AABB'):
        ObjectCandidate3D(**values)

    values = dict(candidate.__dict__)
    values['pose_mode'] = ''
    with pytest.raises(ValueError, match='pose_mode'):
        ObjectCandidate3D(**values)
