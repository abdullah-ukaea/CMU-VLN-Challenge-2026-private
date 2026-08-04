"""Tests for deterministic clustering and primary-cluster selection."""

import numpy as np

from qmapnav.mapping.cluster_selection import cluster_and_select
from qmapnav.mapping.cluster_selection import deterministic_dbscan
from qmapnav.perception.contracts import Detection2D
from qmapnav.perception.contracts import PanoramaBox


def _detection() -> Detection2D:
    box = PanoramaBox(
        360,
        120,
        ((100.0, 220.0),),
        20.0,
        100.0,
        np.array([[100.0, 20.0], [220.0, 20.0], [220.0, 100.0], [100.0, 100.0]]),
    )
    return Detection2D(
        'chair:0', 'chair', 'chair', 0.8, box, (0,),
        ((0.0, 0.0, 10.0, 10.0),), (160.0, 60.0),
        np.array([1.0, 0.0, 0.0]),
    )


def test_dbscan_is_deterministic_under_input_shuffle() -> None:
    rng = np.random.default_rng(3)
    first = rng.normal([0.0, 0.0, 0.5], 0.02, (30, 3))
    second = rng.normal([1.0, 1.0, 0.5], 0.02, (30, 3))
    points = np.vstack((first, second))
    permutation = rng.permutation(len(points))

    labels = deterministic_dbscan(points, 0.10, 5)
    shuffled = deterministic_dbscan(points[permutation], 0.10, 5)

    assert len(set(labels)) == 2
    assert len(set(shuffled)) == 2
    for original_index, shuffled_index in enumerate(permutation):
        same_original = labels == labels[shuffled_index]
        same_shuffled = shuffled == shuffled[original_index]
        assert np.count_nonzero(same_original) == np.count_nonzero(same_shuffled)


def test_selector_prefers_near_centred_substantial_cluster_not_largest_wall() -> None:
    rng = np.random.default_rng(5)
    object_points = rng.normal([2.0, 0.0, 0.8], 0.03, (30, 3))
    wall_points = rng.normal([5.0, 1.0, 1.0], 0.03, (80, 3))
    points = np.vstack((object_points, wall_points))
    depths = np.concatenate((np.full(30, 2.0), np.full(80, 5.0)))
    uv = np.vstack(
        (
            rng.normal([160.0, 60.0], 2.0, (30, 2)),
            rng.normal([205.0, 70.0], 3.0, (80, 2)),
        )
    )

    result = cluster_and_select(points, depths, uv, _detection())

    assert result.selected_indices.size == 30
    assert np.max(result.selected_indices) < 30
    assert len(result.summaries) == 2


def test_two_similar_clusters_report_ambiguity() -> None:
    rng = np.random.default_rng(7)
    first = rng.normal([2.0, -0.2, 0.8], 0.02, (30, 3))
    second = rng.normal([2.0, 0.2, 0.8], 0.02, (30, 3))
    points = np.vstack((first, second))
    depths = np.full(60, 2.0)
    uv = np.vstack(
        (
            rng.normal([145.0, 60.0], 1.0, (30, 2)),
            rng.normal([175.0, 60.0], 1.0, (30, 2)),
        )
    )

    result = cluster_and_select(points, depths, uv, _detection())

    assert result.ambiguous
    assert result.reason == 'multiple_plausible_clusters'
