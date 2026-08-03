"""Tests for bounded source-time association and pose interpolation."""

from math import pi

import numpy as np
import pytest

from qmapnav.mapping.timed_buffers import AssociationConfig
from qmapnav.mapping.timed_buffers import AssociationFailure
from qmapnav.mapping.timed_buffers import interpolate_pose
from qmapnav.mapping.timed_buffers import ProjectionSynchronizer
from qmapnav.mapping.timed_buffers import TimedBuffer
from qmapnav.mapping.timed_buffers import TimedPanorama
from qmapnav.mapping.timed_buffers import TimedPose
from qmapnav.mapping.timed_buffers import TimedRegisteredScan
from qmapnav.mapping.transforms import quaternion_xyzw_to_rotation


def _pose(timestamp: int, x: float, quaternion: np.ndarray | None = None) -> TimedPose:
    return TimedPose(
        timestamp_ns=timestamp,
        parent_frame_id='map',
        child_frame_id='sensor',
        position_xyz=np.array([x, 0.0, 0.75]),
        orientation_xyzw=(
            np.array([0.0, 0.0, 0.0, 1.0])
            if quaternion is None
            else quaternion
        ),
    )


def _scan(timestamp: int) -> TimedRegisteredScan:
    return TimedRegisteredScan(
        timestamp_ns=timestamp,
        frame_id='map',
        points_xyz=np.array([[1.0, 0.0, 1.0]]),
    )


def _image(timestamp: int) -> TimedPanorama:
    return TimedPanorama(
        image_id=f'image-{timestamp}',
        timestamp_ns=timestamp,
        frame_id='camera',
        image_rgb=np.zeros((4, 8, 3), dtype=np.uint8),
    )


def test_pose_interpolation_uses_translation_and_shortest_arc_slerp() -> None:
    before = _pose(0, 0.0)
    after = _pose(
        10,
        2.0,
        np.array([0.0, 0.0, np.sin(pi / 4.0), np.cos(pi / 4.0)]),
    )

    pose, ratio = interpolate_pose(before, after, 5)

    assert ratio == pytest.approx(0.5)
    np.testing.assert_allclose(pose.position_xyz, [1.0, 0.0, 0.75])
    rotation = quaternion_xyzw_to_rotation(pose.orientation_xyzw)
    np.testing.assert_allclose(
        rotation @ np.array([1.0, 0.0, 0.0]),
        [np.sqrt(0.5), np.sqrt(0.5), 0.0],
        atol=1e-7,
    )


def test_quaternion_sign_equivalence_does_not_spin() -> None:
    before = _pose(0, 0.0, np.array([0.0, 0.0, 0.0, 1.0]))
    after = _pose(10, 0.0, np.array([0.0, 0.0, 0.0, -1.0]))

    pose, _ = interpolate_pose(before, after, 5)

    np.testing.assert_allclose(pose.orientation_xyzw, [0.0, 0.0, 0.0, 1.0])


def test_timed_buffer_orders_replaces_and_bounds_items() -> None:
    buffer = TimedBuffer[TimedPose](duration_ns=10, max_items=3)
    buffer.add(_pose(10, 1.0))
    buffer.add(_pose(0, 0.0))
    buffer.add(_pose(5, 0.5))
    assert buffer.add(_pose(5, 0.6)) is False
    buffer.add(_pose(20, 2.0))

    assert [item.timestamp_ns for item in buffer.snapshot()] == [10, 20]


def test_nearest_tie_chooses_earlier_sample() -> None:
    buffer = TimedBuffer[TimedRegisteredScan](duration_ns=100, max_items=10)
    buffer.add(_scan(40))
    buffer.add(_scan(60))

    assert buffer.nearest(50, 10).timestamp_ns == 40
    assert buffer.nearest(50, 9) is None


def test_synchronizer_interpolates_pose_and_records_signed_scan_delta() -> None:
    synchronizer = ProjectionSynchronizer(
        AssociationConfig(max_pose_delta_ns=20, max_scan_delta_ns=20)
    )
    synchronizer.poses.add(_pose(40, 0.0))
    synchronizer.poses.add(_pose(60, 2.0))
    synchronizer.scans.add(_scan(55))

    result = synchronizer.associate(_image(50))

    assert not isinstance(result, AssociationFailure)
    assert result.pose_mode == 'interpolated'
    assert result.interpolation_ratio == pytest.approx(0.5)
    assert result.image_scan_delta_ns == 5
    np.testing.assert_allclose(result.pose.position_xyz, [1.0, 0.0, 0.75])


def test_synchronizer_has_explicit_scan_and_pose_failures() -> None:
    synchronizer = ProjectionSynchronizer(
        AssociationConfig(max_pose_delta_ns=5, max_scan_delta_ns=5)
    )
    no_scan = synchronizer.associate(_image(50))
    assert no_scan.reason == 'no_scan_within_threshold'

    synchronizer.scans.add(_scan(50))
    synchronizer.poses.add(_pose(0, 0.0))
    no_pose = synchronizer.associate(_image(50))
    assert no_pose.reason == 'no_pose_within_threshold'


def test_exact_and_nearest_pose_modes_are_explicit() -> None:
    synchronizer = ProjectionSynchronizer(
        AssociationConfig(max_pose_delta_ns=10, max_scan_delta_ns=10)
    )
    synchronizer.scans.add(_scan(50))
    synchronizer.poses.add(_pose(50, 1.0))
    exact = synchronizer.associate(_image(50))
    assert exact.pose_mode == 'exact'

    synchronizer.clear()
    synchronizer.scans.add(_scan(50))
    synchronizer.poses.add(_pose(45, 1.0))
    fallback = synchronizer.associate(_image(50))
    assert fallback.pose_mode == 'nearest_fallback'
