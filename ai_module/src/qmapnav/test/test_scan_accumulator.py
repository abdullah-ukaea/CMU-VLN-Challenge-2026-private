"""Tests for bounded persistent registered-scan accumulation."""

from math import nan

import numpy as np
import pytest

from qmapnav.mapping import AccumulationStatus
from qmapnav.mapping import RegisteredScanAccumulator
from qmapnav.mapping import ScanAccumulatorConfig


def test_empty_and_non_finite_scans_are_safe() -> None:
    accumulator = RegisteredScanAccumulator()

    empty = accumulator.add_scan(
        np.empty((0, 3)),
        frame_id='map',
        timestamp=0.0,
    )
    invalid = accumulator.add_scan(
        np.array([[nan, 0.0, 1.0], [0.0, np.inf, 1.0]]),
        frame_id='map',
        timestamp=1.0,
    )

    assert empty.status is AccumulationStatus.EMPTY
    assert invalid.status is AccumulationStatus.EMPTY
    assert accumulator.stats().voxel_count == 0
    assert accumulator.stats().invalid_point_count == 2


def test_wrong_frame_is_rejected_without_mixing_map_data() -> None:
    accumulator = RegisteredScanAccumulator()

    result = accumulator.add_scan(
        np.array([[1.0, 2.0, 0.5]]),
        frame_id='lidar',
        timestamp=0.0,
    )

    assert result.status is AccumulationStatus.REJECTED_FRAME
    assert accumulator.stats().rejected_scan_count == 1
    assert accumulator.snapshot_points().shape == (0, 3)


def test_repeated_scans_and_same_voxel_observations_do_not_grow_map() -> None:
    accumulator = RegisteredScanAccumulator(
        ScanAccumulatorConfig(voxel_size=0.5)
    )
    points = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [1.1, 0.1, 0.1],
        ]
    )

    first = accumulator.add_scan(
        points,
        frame_id='map',
        timestamp=0.0,
    )
    second = accumulator.add_scan(
        points,
        frame_id='map',
        timestamp=1.0,
    )

    assert first.unique_voxels_observed == 2
    assert first.voxel_count == second.voxel_count == 2
    assert accumulator.stats().accepted_scan_count == 2


def test_age_spatial_and_hard_cap_policies_bound_voxels() -> None:
    accumulator = RegisteredScanAccumulator(
        ScanAccumulatorConfig(
            voxel_size=0.1,
            max_range=2.0,
            max_age_seconds=2.0,
            max_voxels=3,
        )
    )
    accumulator.add_scan(
        np.array([[0.0, 0.0, 0.5], [0.5, 0.0, 0.5]]),
        frame_id='map',
        timestamp=0.0,
        sensor_origin_xy=(0.0, 0.0),
    )
    accumulator.add_scan(
        np.array(
            [
                [1.0, 0.0, 0.5],
                [1.1, 0.0, 0.5],
                [1.2, 0.0, 0.5],
                [1.3, 0.0, 0.5],
            ]
        ),
        frame_id='map',
        timestamp=1.0,
        sensor_origin_xy=(0.0, 0.0),
    )
    assert accumulator.stats().voxel_count == 3

    accumulator.add_scan(
        np.array([[10.0, 0.0, 0.5], [5.0, 0.0, 0.5]]),
        frame_id='map',
        timestamp=4.0,
        sensor_origin_xy=(5.0, 0.0),
    )

    stats = accumulator.stats()
    assert stats.voxel_count <= 3
    assert stats.evicted_voxel_count >= 3


def test_stale_scan_is_rejected_deterministically() -> None:
    accumulator = RegisteredScanAccumulator()
    accumulator.add_scan(
        np.array([[0.0, 0.0, 0.5]]),
        frame_id='map',
        timestamp=2.0,
    )

    stale = accumulator.add_scan(
        np.array([[1.0, 0.0, 0.5]]),
        frame_id='map',
        timestamp=1.0,
    )

    assert stale.status is AccumulationStatus.STALE
    assert accumulator.stats().stale_scan_count == 1


def test_safe_offset_requires_observed_free_ray_and_obstacle_clearance() -> None:
    accumulator = RegisteredScanAccumulator()
    assert accumulator.select_safe_offset(0.0, 0.0, 2.0, 0.0) is None

    accumulator.add_scan(
        np.array([[0.0, 2.0, 1.0]]),
        frame_id='map',
        timestamp=0.0,
        sensor_origin_xy=(0.0, 0.0),
    )

    candidate = accumulator.select_safe_offset(0.0, 0.0, 2.0, 0.0)
    assert candidate == pytest.approx((0.0, 0.75, 0.0))

    accumulator.add_scan(
        np.array([[0.0, 0.75, 1.0]]),
        frame_id='map',
        timestamp=1.0,
        sensor_origin_xy=(0.0, 0.0),
    )

    assert accumulator.select_safe_offset(0.0, 0.0, 2.0, 0.0) is None


def test_reset_clears_episode_state_and_statistics() -> None:
    accumulator = RegisteredScanAccumulator()
    accumulator.add_scan(
        np.array([[1.0, 0.0, 0.5]]),
        frame_id='map',
        timestamp=0.0,
    )

    accumulator.reset()

    assert accumulator.snapshot_points().shape == (0, 3)
    assert accumulator.stats().accepted_scan_count == 0
    assert accumulator.stats().voxel_count == 0


def test_long_synthetic_stream_never_exceeds_hard_voxel_cap() -> None:
    accumulator = RegisteredScanAccumulator(
        ScanAccumulatorConfig(
            voxel_size=0.1,
            max_range=100.0,
            max_voxels=25,
        )
    )
    for timestamp in range(100):
        accumulator.add_scan(
            np.array([[timestamp * 0.1, 0.0, 0.5]]),
            frame_id='map',
            timestamp=float(timestamp),
        )

    assert accumulator.stats().voxel_count <= 25


def test_accumulator_rejects_malformed_arrays_and_configuration() -> None:
    accumulator = RegisteredScanAccumulator()
    with pytest.raises(ValueError, match='shape'):
        accumulator.add_scan(
            np.array([1.0, 2.0, 3.0]),
            frame_id='map',
            timestamp=0.0,
        )
    with pytest.raises(ValueError, match='voxel_size'):
        ScanAccumulatorConfig(voxel_size=0.0)
