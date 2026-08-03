"""Tests for dense rolling centroid-voxel scan accumulation."""

import numpy as np

from qmapnav.mapping.dense_scan_accumulator import DenseAccumulationStatus
from qmapnav.mapping.dense_scan_accumulator import DenseRegisteredScanAccumulator
from qmapnav.mapping.dense_scan_accumulator import DenseScanAccumulatorConfig
from qmapnav.mapping.dense_scan_accumulator import voxel_filter_centroids


def test_voxel_filter_returns_centroids_counts_and_latest_stamps() -> None:
    snapshot = voxel_filter_centroids(
        np.array([[0.01, 0.01, 0.01], [0.03, 0.03, 0.03], [0.11, 0.0, 0.0]]),
        np.array([10, 20, 30]),
        0.1,
    )

    np.testing.assert_allclose(
        snapshot.points_xyz,
        [[0.02, 0.02, 0.02], [0.11, 0.0, 0.0]],
    )
    np.testing.assert_array_equal(snapshot.observation_count, [2, 1])
    np.testing.assert_array_equal(snapshot.last_seen_timestamp_ns, [20, 30])


def test_repeated_scans_densify_without_unbounded_raw_growth() -> None:
    accumulator = DenseRegisteredScanAccumulator(
        DenseScanAccumulatorConfig(voxel_size_m=0.05, max_points=20)
    )
    points = np.array([[1.0, 0.0, 0.5], [1.01, 0.0, 0.5]])
    for timestamp in range(20):
        accumulator.add_scan(
            points,
            frame_id='map',
            timestamp_ns=timestamp,
            sensor_origin_xyz=np.zeros(3),
        )

    assert accumulator.stats().raw_point_count == 20
    snapshot = accumulator.snapshot()
    assert snapshot.points_xyz.shape == (1, 3)
    assert snapshot.observation_count[0] == 20


def test_age_radius_cap_frame_and_stale_policies() -> None:
    accumulator = DenseRegisteredScanAccumulator(
        DenseScanAccumulatorConfig(
            voxel_size_m=0.1,
            max_age_seconds=1e-9,
            max_radius_m=2.0,
            max_points=2,
        )
    )
    wrong = accumulator.add_scan(
        np.array([[0.0, 0.0, 0.0]]),
        frame_id='sensor',
        timestamp_ns=0,
    )
    assert wrong.status is DenseAccumulationStatus.REJECTED_FRAME
    accumulator.add_scan(
        np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        frame_id='map',
        timestamp_ns=1,
        sensor_origin_xyz=np.zeros(3),
    )
    stale = accumulator.add_scan(
        np.array([[0.0, 0.0, 0.0]]),
        frame_id='map',
        timestamp_ns=0,
    )
    assert stale.status is DenseAccumulationStatus.STALE
    accumulator.add_scan(
        np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]),
        frame_id='map',
        timestamp_ns=3,
        sensor_origin_xyz=np.zeros(3),
    )
    stats = accumulator.stats()
    assert stats.raw_point_count <= 2
    assert stats.evicted_point_count >= 2


def test_empty_nonfinite_reset_and_deterministic_snapshot() -> None:
    accumulator = DenseRegisteredScanAccumulator()
    result = accumulator.add_scan(
        np.array([[np.nan, 0.0, 0.0]]),
        frame_id='map',
        timestamp_ns=0,
    )
    assert result.status is DenseAccumulationStatus.EMPTY
    accumulator.add_scan(
        np.array([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        frame_id='map',
        timestamp_ns=1,
    )
    first = accumulator.snapshot().points_xyz.copy()
    second = accumulator.snapshot().points_xyz.copy()
    np.testing.assert_array_equal(first, second)
    accumulator.reset()
    assert accumulator.stats().raw_point_count == 0


def test_image_time_snapshot_excludes_future_scans() -> None:
    accumulator = DenseRegisteredScanAccumulator(
        DenseScanAccumulatorConfig(max_age_seconds=1.0)
    )
    accumulator.add_scan(
        np.array([[1.0, 0.0, 0.0]]),
        frame_id='map',
        timestamp_ns=100,
    )
    accumulator.add_scan(
        np.array([[2.0, 0.0, 0.0]]),
        frame_id='map',
        timestamp_ns=200,
    )

    snapshot = accumulator.snapshot(timestamp_ns=150)

    np.testing.assert_allclose(snapshot.points_xyz, [[1.0, 0.0, 0.0]])
