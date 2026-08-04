"""Wall extraction and repeated structural-anchor regression tests."""

import numpy as np

from qmapnav.mapping.ray_wall_intersection import intersect_ray_with_wall
from qmapnav.mapping.structural_map import StructuralAnchor
from qmapnav.mapping.structural_map import StructuralMap
from qmapnav.mapping.structural_map import StructuralMapConfig
from qmapnav.mapping.wall_extraction import extract_wall_candidates
from qmapnav.mapping.wall_extraction import WallExtractionConfig
from qmapnav.perception.contracts import Detection2D
from qmapnav.perception.contracts import PanoramaBox


def _wall_points(
    x_intervals: tuple[tuple[float, float], ...] = ((-2.0, 2.0),),
) -> np.ndarray:
    points = []
    for x_min, x_max in x_intervals:
        for x_value in np.linspace(x_min, x_max, 41):
            for z_value in np.linspace(0.0, 2.4, 9):
                points.append([x_value, 2.0, z_value])
    for x_value in np.linspace(-2.0, 2.0, 15):
        for y_value in np.linspace(-1.0, 3.0, 15):
            points.append([x_value, y_value, 0.0])
    points.extend([
        [0.3, -0.4, 0.4],
        [-0.7, 0.8, 0.7],
        [1.4, 1.1, 0.3],
    ])
    return np.asarray(points, dtype=np.float64)


def _wall_anchor(
    y_value: float,
    *,
    anchor_id: str = 'candidate',
    timestamp_ns: int = 1,
) -> StructuralAnchor:
    return StructuralAnchor(
        anchor_id=anchor_id,
        anchor_type='wall',
        semantic_class='wall',
        position_xyz=np.array([0.0, y_value, 1.2]),
        line_segment_xy=np.array([[-2.0, y_value], [2.0, y_value]]),
        polygon_xy=None,
        plane_parameters=np.array([0.0, 1.0, 0.0, -y_value]),
        extent_xyz=np.array([4.0, 0.05, 2.4]),
        yaw_rad=0.0,
        supporting_wall_id=None,
        confidence=0.9,
        first_seen_ns=timestamp_ns,
        last_seen_ns=timestamp_ns,
        source_viewpoint_ids=('scan_a',),
        source_detection_ids=(),
    )


def _detection(
    detection_id: str,
    ray: np.ndarray,
    *,
    class_name: str = 'window',
    viewpoint_id: str = 'view_a',
    timestamp_ns: int = 1,
    lidar_depth_m: float | None = None,
) -> Detection2D:
    boundary = np.array([
        [10.0, 10.0],
        [20.0, 10.0],
        [20.0, 20.0],
        [10.0, 20.0],
    ])
    return Detection2D(
        detection_id=detection_id,
        class_name=class_name,
        prompt_used=class_name,
        confidence=0.9,
        panorama_box=PanoramaBox(
            100,
            50,
            ((10.0, 20.0),),
            10.0,
            20.0,
            boundary,
        ),
        crop_ids=(0,),
        crop_boxes_xyxy=((10.0, 10.0, 20.0, 20.0),),
        centre_panorama_uv=(15.0, 15.0),
        centre_camera_ray=np.asarray(ray) / np.linalg.norm(ray),
        metadata={
            'viewpoint_id': viewpoint_id,
            'timestamp_ns': timestamp_ns,
            'lidar_depth_m': lidar_depth_m,
        },
    )


def test_wall_extraction_rejects_ground_and_fits_vertical_segment() -> None:
    walls = extract_wall_candidates(
        _wall_points(),
        timestamp_ns=10,
        config=WallExtractionConfig(minimum_support_points=20),
    )

    assert walls
    wall = max(walls, key=lambda item: item.length_m)
    assert wall.length_m > 3.5
    assert wall.supporting_point_count >= 20
    assert wall.fit_residual_m < 0.03
    assert abs(abs(wall.plane_parameters[1]) - 1.0) < 0.05
    assert wall.vertical_extent_m[1] - wall.vertical_extent_m[0] > 1.5


def test_wall_extraction_preserves_doorway_sized_opening() -> None:
    points = _wall_points(((-2.0, -0.5), (0.5, 2.0)))
    walls = extract_wall_candidates(
        points,
        timestamp_ns=10,
        config=WallExtractionConfig(minimum_support_points=20),
    )

    long_walls = [wall for wall in walls if wall.length_m > 1.2]
    assert len(long_walls) >= 2
    assert max(wall.length_m for wall in long_walls) < 2.0


def test_ray_wall_intersection_rejects_parallel_behind_and_outside() -> None:
    plane = np.array([0.0, 1.0, 0.0, -2.0])
    segment = np.array([[-1.0, 2.0], [1.0, 2.0]])
    origin = np.array([0.0, 0.0, 1.0])

    hit = intersect_ray_with_wall(
        origin, np.array([0.0, 1.0, 0.0]), plane, segment
    )

    np.testing.assert_allclose(hit.position_xyz, [0.0, 2.0, 1.0])
    assert hit.distance_m == 2.0
    assert intersect_ray_with_wall(
        origin, np.array([1.0, 0.0, 0.0]), plane, segment
    ) is None
    assert intersect_ray_with_wall(
        origin, np.array([0.0, -1.0, 0.0]), plane, segment
    ) is None
    assert intersect_ray_with_wall(
        np.array([3.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        plane,
        segment,
    ) is None


def test_window_reobserved_from_two_poses_keeps_wall_and_anchor() -> None:
    structural_map = StructuralMap()
    wall_id = structural_map.add_or_update_wall(
        _wall_anchor(2.0),
        supporting_point_count=100,
        fit_residual_m=0.01,
        vertical_extent_m=(0.0, 2.4),
    )
    first_pose = np.eye(4)
    first_pose[:3, 3] = [0.0, 0.0, 1.0]
    first = structural_map.anchor_detection_to_wall(
        _detection('window_a', np.array([0.0, 1.0, 0.0])),
        first_pose,
    )
    second_pose = np.eye(4)
    second_pose[:3, 3] = [1.0, 0.0, 1.0]
    second = structural_map.anchor_detection_to_wall(
        _detection(
            'window_b',
            np.array([-1.0, 2.0, 0.0]),
            viewpoint_id='view_b',
            timestamp_ns=2,
        ),
        second_pose,
    )

    assert wall_id == 'wall_0000'
    assert first.anchor_id == second.anchor_id == 'anchor_0000'
    assert first.supporting_wall_id == second.supporting_wall_id == wall_id
    np.testing.assert_allclose(second.position_xyz, [0.0, 2.0, 1.0])
    record = structural_map.record('anchor_0000')
    assert record.observation_count == 2
    assert record.anchor.source_viewpoint_ids == ('view_a', 'view_b')


def test_nearest_parallel_wall_is_selected_and_reset_is_deterministic() -> None:
    structural_map = StructuralMap()
    assert structural_map.add_or_update_wall(_wall_anchor(2.0)) == 'wall_0000'
    assert structural_map.add_or_update_wall(_wall_anchor(4.0)) == 'wall_0001'
    pose = np.eye(4)
    pose[:3, 3] = [0.0, 0.0, 1.0]

    anchor = structural_map.anchor_detection_to_wall(
        _detection('picture', np.array([0.0, 1.0, 0.0]),
                   class_name='picture'),
        pose,
    )

    assert anchor.supporting_wall_id == 'wall_0000'
    structural_map.reset_episode()
    assert structural_map.walls() == []
    assert structural_map.anchors() == []
    assert structural_map.add_or_update_wall(_wall_anchor(2.0)) == 'wall_0000'


def test_lidar_depth_selects_consistent_forward_wall_when_available() -> None:
    structural_map = StructuralMap()
    structural_map.add_or_update_wall(_wall_anchor(2.0))
    structural_map.add_or_update_wall(_wall_anchor(4.0))
    pose = np.eye(4)
    pose[:3, 3] = [0.0, 0.0, 1.0]

    anchor = structural_map.anchor_detection_to_wall(
        _detection(
            'screen_depth',
            np.array([0.0, 1.0, 0.0]),
            class_name='screen',
            lidar_depth_m=3.9,
        ),
        pose,
    )

    assert anchor.supporting_wall_id == 'wall_0001'
    assert structural_map.last_events[0].lidar_depth_m == 3.9
    assert structural_map.last_events[0].reason == (
        'lidar_depth_consistent_forward_wall'
    )


def test_non_structural_detection_is_rejected_with_reason() -> None:
    structural_map = StructuralMap()
    structural_map.add_or_update_wall(_wall_anchor(2.0))

    result = structural_map.anchor_detection_to_wall(
        _detection('chair', np.array([0.0, 1.0, 0.0]),
                   class_name='chair'),
        np.eye(4),
    )

    assert result is None
    assert structural_map.last_events[0].reason == 'class_is_not_structural'


def test_structural_wall_and_anchor_counts_are_hard_bounded() -> None:
    structural_map = StructuralMap(StructuralMapConfig(
        max_walls=2,
        max_anchors=1,
    ))
    structural_map.add_or_update_wall(_wall_anchor(2.0))
    structural_map.add_or_update_wall(_wall_anchor(4.0))
    structural_map.add_or_update_wall(_wall_anchor(6.0))

    assert len(structural_map.walls()) == 2
    pose = np.eye(4)
    pose[:3, 3] = [0.0, 0.0, 1.0]
    structural_map.anchor_detection_to_wall(
        _detection('window', np.array([0.0, 1.0, 0.0])), pose
    )
    structural_map.anchor_detection_to_wall(
        _detection(
            'picture', np.array([0.0, 1.0, 0.0]), class_name='picture'
        ),
        pose,
    )

    assert len(structural_map.anchors()) == 1
