"""End-to-end synthetic tests for single-observation object lifting."""

import numpy as np

from qmapnav.mapping.ground_filter import GroundPlane
from qmapnav.mapping.lidar_camera_projection import ProjectionDiagnostics
from qmapnav.mapping.lidar_camera_projection import ProjectionResult
from qmapnav.mapping.object_candidate import GeometrySource
from qmapnav.mapping.object_candidate import GeometryStatus
from qmapnav.mapping.object_lifting import ObjectLifter
from qmapnav.perception.contracts import Detection2D
from qmapnav.perception.contracts import PanoramaBox


def _detection(metadata=None) -> Detection2D:
    box = PanoramaBox(
        360,
        120,
        ((100.0, 220.0),),
        15.0,
        105.0,
        np.array([[100.0, 15.0], [220.0, 15.0], [220.0, 105.0], [100.0, 105.0]]),
    )
    return Detection2D(
        'frame:chair:0',
        'chair',
        'chair',
        0.9,
        box,
        (0,),
        ((1.0, 1.0, 10.0, 10.0),),
        (160.0, 60.0),
        np.array([1.0, 0.0, 0.0]),
        metadata=metadata or {},
    )


def _projection(
    points: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
) -> ProjectionResult:
    count = len(points)
    return ProjectionResult(
        image_id='frame',
        image_timestamp_ns=100,
        scan_timestamp_ns=100,
        transform_camera_internal_from_map=np.eye(4),
        source_valid_mask=np.ones(count, dtype=np.bool_),
        source_point_indices=np.arange(count),
        points_map_xyz=points,
        points_camera_xyz=points,
        panorama_uv=uv,
        euclidean_range_m=depth,
        forward_depth_m=points[:, 0],
        intensity=None,
        diagnostics=ProjectionDiagnostics(
            count, count, count, count, 0.0, 'exact', 0.0, 0.0, False
        ),
    )


def _scene() -> tuple[ProjectionResult, float]:
    rng = np.random.default_rng(12)
    yaw = np.deg2rad(30.0)
    local = rng.uniform(
        [-0.6, -0.25, 0.10],
        [0.6, 0.25, 1.10],
        size=(900, 3),
    )
    rotation = np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    )
    object_points = local.copy()
    object_points[:, :2] = local[:, :2] @ rotation.T
    object_points += np.array([2.0, 0.0, 0.0])
    floor = np.column_stack(
        (rng.uniform(1.5, 2.5, 60), rng.uniform(-0.6, 0.6, 60), np.zeros(60))
    )
    wall = np.column_stack(
        (np.full(180, 5.0), rng.uniform(-1.0, 1.0, 180), rng.uniform(0.1, 2.0, 180))
    )
    points = np.vstack((object_points, floor, wall))
    uv = np.vstack(
        (
            rng.normal([160.0, 60.0], [18.0, 20.0], (900, 2)),
            rng.normal([160.0, 98.0], [25.0, 2.0], (60, 2)),
            rng.normal([165.0, 60.0], [30.0, 25.0], (180, 2)),
        )
    )
    depth = np.concatenate(
        (
            np.linalg.norm(object_points, axis=1),
            np.linalg.norm(floor, axis=1),
            np.linalg.norm(wall, axis=1),
        )
    )
    return _projection(points, uv, depth), yaw


def test_lifter_removes_ground_wall_and_estimates_oriented_candidate() -> None:
    projection, expected_yaw = _scene()
    lifter = ObjectLifter()

    result = lifter.lift(
        _detection(),
        projection,
        source=GeometrySource.CURRENT,
        ground_plane=GroundPlane(np.array([0.0, 0.0, 1.0]), 0.0),
    )

    assert result.candidate is not None
    candidate = result.candidate
    assert result.status in {GeometryStatus.GOOD, GeometryStatus.UNSTABLE_ORIENTATION}
    assert candidate.point_count > 100
    assert candidate.counts.post_ground < candidate.counts.box_selected
    assert candidate.counts.post_depth < candidate.counts.post_ground
    np.testing.assert_allclose(candidate.obb_centre_xyz[:2], [2.0, 0.0], atol=0.12)
    assert candidate.obb_dimensions_xyz[0] > candidate.obb_dimensions_xyz[1]
    if not candidate.low_orientation_fallback:
        delta = abs(candidate.obb_yaw_rad - expected_yaw)
        assert min(delta, np.pi - delta) < np.deg2rad(12.0)


def test_mask_selection_reduces_background_proposals() -> None:
    projection, _ = _scene()
    polygon = ((125.0, 25.0), (195.0, 25.0), (195.0, 90.0), (125.0, 90.0))
    detection = _detection({'mask_polygons_panorama_uv': (polygon,)})

    with_mask = ObjectLifter().lift(
        detection,
        projection,
        source=GeometrySource.CURRENT,
        ground_plane=GroundPlane(np.array([0.0, 0.0, 1.0]), 0.0),
        use_mask=True,
    )
    box_only = ObjectLifter().lift(
        detection,
        projection,
        source=GeometrySource.CURRENT,
        ground_plane=GroundPlane(np.array([0.0, 0.0, 1.0]), 0.0),
        use_mask=False,
    )

    assert with_mask.counts.mask_selected > 0
    assert with_mask.counts.mask_selected < box_only.counts.box_selected
    assert with_mask.candidate is not None


def test_no_points_and_ground_dominated_are_structured_failures() -> None:
    empty = _projection(
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[10.0, 10.0]]),
        np.array([1.0]),
    )
    ground = _projection(
        np.array([[2.0, 0.0, 0.0], [2.0, 0.1, 0.0], [2.0, -0.1, 0.0]]),
        np.array([[160.0, 60.0], [161.0, 60.0], [159.0, 60.0]]),
        np.array([2.0, 2.0, 2.0]),
    )
    plane = GroundPlane(np.array([0.0, 0.0, 1.0]), 0.0)

    no_points = ObjectLifter().lift(
        _detection(), empty, source=GeometrySource.CURRENT, ground_plane=plane
    )
    dominated = ObjectLifter().lift(
        _detection(), ground, source=GeometrySource.CURRENT, ground_plane=plane
    )

    assert no_points.status is GeometryStatus.NO_POINTS
    assert no_points.candidate is None
    assert dominated.status is GeometryStatus.GROUND_DOMINATED
    assert dominated.candidate is None


def test_three_point_sparse_geometry_fails_gracefully_without_false_box() -> None:
    points = np.array([[2.0, 0.0, 0.2], [2.1, 0.1, 0.7], [2.2, -0.1, 1.0]])
    projection = _projection(
        points,
        np.array([[158.0, 60.0], [160.0, 58.0], [162.0, 62.0]]),
        np.linalg.norm(points, axis=1),
    )

    result = ObjectLifter().lift(
        _detection(),
        projection,
        source=GeometrySource.CURRENT,
        ground_plane=GroundPlane(np.array([0.0, 0.0, 1.0]), 0.0),
    )

    assert result.status in {GeometryStatus.SPARSE, GeometryStatus.INVALID_GEOMETRY}
