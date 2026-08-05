"""Tests for perceived near regions, goal poses, and the adapter fallback."""

from math import hypot

from day11_helpers import add_wall
from day11_helpers import make_instance
from day11_helpers import open_grid
import numpy as np
import pytest

from qmapnav.mapping.perceived_geometry import perceived_box
from qmapnav.navigation.semantic_regions import GoalPoseScoringConfig
from qmapnav.navigation.semantic_regions import NearRegionConfig
from qmapnav.navigation.semantic_regions import perceived_near_region
from qmapnav.navigation.semantic_regions import pose_faces
from qmapnav.navigation.semantic_regions import region_excludes_footprint
from qmapnav.navigation.semantic_regions import sample_goal_poses
from qmapnav.navigation.semantic_regions import select_goal_pose
from qmapnav.navigation.semantic_regions import semantic_region_satisfied


def test_adapter_uses_the_obb_when_orientation_is_trusted() -> None:
    instance = make_instance(
        1, 'table', (2.0, 0.0, 0.4), (1.2, 0.8, 0.75),
        yaw=0.5, orientation_confidence=0.9,
    )
    box = perceived_box(instance)
    assert box.used_axis_aligned_fallback is False
    assert box.yaw == pytest.approx(0.5)
    assert box.dimensions_xyz == pytest.approx((1.2, 0.8, 0.75))
    assert box.object_id == '1'
    assert box.class_name == 'table'


def test_adapter_degrades_to_the_aabb_when_orientation_is_weak() -> None:
    instance = make_instance(
        2, 'table', (2.0, 0.0, 0.4), (1.2, 0.8, 0.75),
        yaw=0.5, orientation_confidence=0.05,
    )
    box = perceived_box(instance)
    assert box.used_axis_aligned_fallback is True
    # Never invent a precise orientation from weak evidence.
    assert box.yaw == 0.0
    extent = instance.aabb_max_xyz - instance.aabb_min_xyz
    assert box.dimensions_xyz == pytest.approx(tuple(extent))


def test_adapter_inflation_grows_the_footprint_symmetrically() -> None:
    instance = make_instance(3, 'table', (0.0, 0.0, 0.4), (1.0, 1.0, 0.75))
    plain = perceived_box(instance)
    inflated = perceived_box(instance, inflation_m=0.25)
    assert inflated.dimensions_xyz[0] == pytest.approx(
        plain.dimensions_xyz[0] + 0.5
    )
    assert inflated.dimensions_xyz[2] == pytest.approx(
        plain.dimensions_xyz[2]
    )


def test_adapter_rejects_bad_input() -> None:
    with pytest.raises(TypeError, match='ObjectInstance'):
        perceived_box(object())
    instance = make_instance(4, 'table', (0.0, 0.0, 0.4))
    with pytest.raises(ValueError, match='inflation_m'):
        perceived_box(instance, inflation_m=-1.0)


def test_near_region_excludes_the_object_footprint() -> None:
    instance = make_instance(5, 'table', (2.0, 0.0, 0.4), (1.2, 0.8, 0.75))
    region = perceived_near_region(instance)
    assert region_excludes_footprint(region, instance) is True
    # The object centre is never a valid goal.
    assert semantic_region_satisfied((2.0, 0.0), region) is False
    # A point inside the annulus is. The exclusion boundary itself counts as
    # excluded, so this sits just beyond the 0.6 m clearance ring.
    assert semantic_region_satisfied((2.0, 1.05), region) is True
    assert semantic_region_satisfied((2.0, 1.4), region) is True
    # A point well outside is not.
    assert semantic_region_satisfied((2.0, 5.0), region) is False


def test_near_region_respects_configured_distances() -> None:
    instance = make_instance(6, 'chair', (0.0, 0.0, 0.4), (0.5, 0.5, 0.9))
    region = perceived_near_region(
        instance,
        config=NearRegionConfig(
            near_min_clearance_m=0.6, near_max_distance_m=1.5
        ),
    )
    # Just outside the footprint but inside the clearance ring: excluded.
    assert semantic_region_satisfied((0.5, 0.0), region) is False
    assert semantic_region_satisfied((1.2, 0.0), region) is True
    assert semantic_region_satisfied((2.5, 0.0), region) is False


def test_low_orientation_confidence_produces_a_conservative_region() -> None:
    weak = make_instance(
        7, 'table', (0.0, 0.0, 0.4), (1.0, 1.0, 0.75),
        orientation_confidence=0.05,
    )
    region = perceived_near_region(weak)
    # The inflated exclusion keeps the robot further from an uncertain box.
    assert semantic_region_satisfied((0.62, 0.0), region) is False


def test_goal_poses_exclude_occupied_cells_and_face_the_object() -> None:
    grid = open_grid(half_extent=8.0)
    instance = make_instance(8, 'table', (2.0, 0.0, 0.4), (1.0, 1.0, 0.75))
    add_wall(grid, (2.0, 0.6, 4.0, 3.0))
    region = perceived_near_region(instance)
    box = perceived_box(instance)
    poses = sample_goal_poses(
        region, box, grid=grid, start_xy=(-3.0, 0.0)
    )
    assert poses
    for candidate in poses:
        x, y, _ = candidate.pose_xy_yaw
        assert grid.is_free(x, y, clearance=0.35)
        assert semantic_region_satisfied((x, y), region)
        assert pose_faces(candidate.pose_xy_yaw, (2.0, 0.0))
        # No pose may sit inside the walled-off block.
        assert not (2.0 <= x <= 4.0 and 0.6 <= y <= 3.0)


def test_stage_a_pose_accounts_for_travel_onward_to_stage_b() -> None:
    grid = open_grid(half_extent=10.0)
    stage_a = make_instance(9, 'plant', (0.0, 0.0, 0.4), (0.6, 0.6, 1.0))
    region = perceived_near_region(stage_a)
    box = perceived_box(stage_a)
    start = (-4.0, 0.0)
    stage_b_xy = (6.0, 0.0)

    without_b = select_goal_pose(
        region, box, grid=grid, start_xy=start
    )
    with_b = select_goal_pose(
        region, box, grid=grid, start_xy=start, next_stage_xy=stage_b_xy
    )
    assert without_b is not None and with_b is not None
    # Onward cost is priced whenever the next stage is known.
    assert without_b.transition_cost_m is None
    assert with_b.transition_cost_m is not None

    # Under the default weights approach cost dominates, so the near side of
    # A still wins; raising the transition weight moves the pose to the far
    # side of A, proving the term genuinely drives the choice.
    transition_heavy = select_goal_pose(
        region,
        box,
        grid=grid,
        start_xy=start,
        next_stage_xy=stage_b_xy,
        scoring=GoalPoseScoringConfig(
            approach_weight=0.2, transition_weight=2.0
        ),
    )
    assert transition_heavy is not None
    assert transition_heavy.pose_xy_yaw[0] > with_b.pose_xy_yaw[0]
    assert transition_heavy.transition_cost_m < with_b.transition_cost_m


def test_goal_pose_selection_returns_none_when_region_is_blocked() -> None:
    grid = open_grid(half_extent=6.0)
    instance = make_instance(10, 'table', (0.0, 0.0, 0.4), (1.0, 1.0, 0.75))
    # Bury the entire near annulus in obstacles.
    add_wall(grid, (-3.0, -3.0, 3.0, 3.0))
    region = perceived_near_region(instance)
    box = perceived_box(instance)
    assert select_goal_pose(
        region, box, grid=grid, start_xy=(-5.0, 0.0)
    ) is None


def test_goal_poses_prefer_line_of_sight_to_the_target() -> None:
    grid = open_grid(half_extent=8.0)
    instance = make_instance(11, 'chair', (0.0, 0.0, 0.4), (0.5, 0.5, 0.9))
    region = perceived_near_region(instance)
    box = perceived_box(instance)
    poses = sample_goal_poses(
        region, box, grid=grid, start_xy=(-3.0, 0.0)
    )
    assert poses[0].has_line_of_sight is True


def test_semantic_satisfaction_is_stricter_than_waypoint_tolerance() -> None:
    instance = make_instance(12, 'chair', (0.0, 0.0, 0.4), (0.4, 0.4, 0.9))
    region = perceived_near_region(
        instance,
        config=NearRegionConfig(
            near_min_clearance_m=0.6, near_max_distance_m=1.5
        ),
    )
    goal = (1.2, 0.0)
    # A pose within the 0.75 m executor arrival radius of the goal can still
    # sit outside the semantic near region.
    stray = (2.0, 0.6)
    assert hypot(stray[0] - goal[0], stray[1] - goal[1]) < 0.75 + 0.3
    assert semantic_region_satisfied(goal, region) is True
    assert semantic_region_satisfied(stray, region) is False


def test_region_config_validates_distances() -> None:
    with pytest.raises(ValueError, match='near_max_distance_m'):
        NearRegionConfig(near_min_clearance_m=2.0, near_max_distance_m=1.0)
    with pytest.raises(ValueError, match='robot_clearance_m'):
        NearRegionConfig(robot_clearance_m=0.0)


def test_region_satisfaction_validates_input() -> None:
    instance = make_instance(13, 'chair', (0.0, 0.0, 0.4))
    region = perceived_near_region(instance)
    with pytest.raises(TypeError, match='SemanticRegion'):
        semantic_region_satisfied((0.0, 0.0), object())
    with pytest.raises(ValueError, match='finite'):
        semantic_region_satisfied((float('nan'), 0.0), region)


def test_adapter_picks_the_dominant_class_deterministically() -> None:
    instance = make_instance(14, 'table', (0.0, 0.0, 0.4))
    instance.class_scores = {'table': 0.6, 'desk': 0.9}
    box = perceived_box(instance)
    assert box.class_name == 'desk'
    override = perceived_box(instance, class_name='table')
    assert override.class_name == 'table'


def test_unreachable_goal_poses_are_dropped() -> None:
    grid = open_grid(half_extent=8.0)
    instance = make_instance(15, 'table', (4.0, 0.0, 0.4), (1.0, 1.0, 0.75))
    # Seal the object off from the start pose entirely.
    add_wall(grid, (1.0, -8.0, 1.5, 8.0))
    region = perceived_near_region(instance)
    box = perceived_box(instance)
    assert sample_goal_poses(
        region, box, grid=grid, start_xy=(-3.0, 0.0)
    ) == ()


def test_geometry_matches_a_directly_constructed_instance() -> None:
    instance = make_instance(16, 'table', (1.0, 2.0, 0.4), (2.0, 1.0, 0.8))
    box = perceived_box(instance)
    assert np.allclose(box.centre_xyz, (1.0, 2.0, 0.4))
    assert box.footprint_radius == pytest.approx(
        hypot(2.0, 1.0) / 2.0
    )
