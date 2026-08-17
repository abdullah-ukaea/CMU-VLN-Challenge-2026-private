"""Tests for occupancy, grid planning, and viewpoint candidate generation."""

from math import atan2
from math import cos
from math import degrees
from math import hypot
from math import pi
from math import sin

from fixtures import add_unknown
from fixtures import add_wall
from fixtures import open_grid
import pytest

from qmapnav.exploration import generate_frontier_viewpoints
from qmapnav.exploration import generate_object_annulus_viewpoints
from qmapnav.exploration import generate_occluder_offset_viewpoints
from qmapnav.exploration import is_novel
from qmapnav.exploration import ViewpointGenerationConfig
from qmapnav.exploration import VisitedViewpoint
from qmapnav.mapping.grid_planning import cost_field
from qmapnav.mapping.grid_planning import is_reachable
from qmapnav.mapping.grid_planning import planned_distance
from qmapnav.mapping.grid_planning import shortest_path
from qmapnav.mapping.occupancy_grid import CELL_FREE
from qmapnav.mapping.occupancy_grid import CELL_OCCUPIED
from qmapnav.mapping.occupancy_grid import CELL_UNKNOWN
from qmapnav.mapping.occupancy_grid import OccupancyGrid2D


def test_grid_reports_three_states_and_treats_outside_as_unknown() -> None:
    grid = open_grid(half_extent=2.0)
    assert grid.state_at_point(0.0, 0.0) == CELL_FREE
    add_wall(grid, (0.5, -0.5, 1.0, 0.5))
    assert grid.state_at_point(0.75, 0.0) == CELL_OCCUPIED
    assert grid.state((-100, -100)) == CELL_UNKNOWN


def test_grid_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError, match='resolution'):
        OccupancyGrid2D(0.0, (0.0, 0.0), 4, 4)
    with pytest.raises(ValueError, match='width'):
        OccupancyGrid2D(0.25, (0.0, 0.0), 0, 4)
    with pytest.raises(ValueError, match='origin_xy'):
        OccupancyGrid2D(0.25, (0.0,), 4, 4)


def test_clearance_rejects_poses_hugging_an_obstacle() -> None:
    grid = open_grid(half_extent=3.0)
    add_wall(grid, (1.0, -1.0, 1.5, 1.0))
    assert grid.is_free(0.2, 0.0, clearance=0.35) is True
    assert grid.is_free(0.8, 0.0, clearance=0.35) is False


def test_line_of_sight_is_blocked_by_a_wall() -> None:
    grid = open_grid(half_extent=4.0)
    add_wall(grid, (-0.25, -2.0, 0.25, 2.0))
    assert grid.line_of_sight((-2.0, 0.0), (-1.0, 0.0)) is True
    assert grid.line_of_sight((-2.0, 0.0), (2.0, 0.0)) is False


def test_unknown_counting_stops_at_obstacles() -> None:
    grid = open_grid(half_extent=6.0)
    add_unknown(grid, (2.0, -3.0, 6.0, 3.0))
    open_view = grid.count_visible_unknown((0.0, 0.0, 0.0), max_range=5.0)
    assert open_view > 0
    add_wall(grid, (1.0, -3.0, 1.4, 3.0))
    blocked_view = grid.count_visible_unknown((0.0, 0.0, 0.0), max_range=5.0)
    assert blocked_view < open_view


def test_frontier_clusters_drop_tiny_isolated_boundaries() -> None:
    grid = open_grid(half_extent=4.0)
    add_unknown(grid, (2.0, -4.0, 4.0, 4.0))
    clusters = grid.frontier_clusters(minimum_cells=3)
    assert clusters
    assert all(item.size >= 3 for item in clusters)
    assert clusters[0].centroid_xy[0] == pytest.approx(1.875, abs=0.3)

    strict = grid.frontier_clusters(minimum_cells=10_000)
    assert strict == ()


def test_cost_field_and_path_respect_obstacles() -> None:
    grid = open_grid(half_extent=4.0)
    add_wall(grid, (-0.3, -4.0, 0.3, 2.0))
    costs = cost_field(grid, (-2.0, 0.0), clearance=0.3)
    direct = planned_distance(
        grid, (-2.0, 0.0), (2.0, 0.0), clearance=0.3, costs=costs
    )
    assert direct is not None
    # The detour around the wall must exceed the 4 m straight-line distance.
    assert direct > 4.0
    path = shortest_path(grid, (-2.0, 0.0), (2.0, 0.0), clearance=0.3)
    assert path is not None
    assert max(point[1] for point in path) > 1.5


def test_unreachable_goal_is_rejected_not_penalized() -> None:
    grid = open_grid(half_extent=4.0)
    add_wall(grid, (1.0, -4.0, 1.5, 4.0))
    assert is_reachable(grid, (-2.0, 0.0), (3.0, 0.0), clearance=0.3) is False
    assert planned_distance(
        grid, (-2.0, 0.0), (3.0, 0.0), clearance=0.3
    ) is None
    assert shortest_path(grid, (-2.0, 0.0), (3.0, 0.0), clearance=0.3) is None


def test_annulus_places_candidates_at_radius_and_faces_target() -> None:
    grid = open_grid(half_extent=8.0)
    focus = (2.0, 0.0)
    outcome = generate_object_annulus_viewpoints(
        focus,
        grid=grid,
        current_pose_xy_yaw=(-2.0, 0.0, 0.0),
        prefix='plant_1',
        config=ViewpointGenerationConfig(
            object_min_radius_m=1.5, object_max_radius_m=3.0, radius_steps=2
        ),
    )
    assert outcome.candidates
    for candidate in outcome.candidates:
        x, y, yaw = candidate.pose_xy_yaw
        radius = hypot(x - focus[0], y - focus[1])
        assert 1.5 - 1e-6 <= radius <= 3.0 + 1e-6
        expected_yaw = atan2(focus[1] - y, focus[0] - x)
        assert abs(
            degrees(atan2(sin(yaw - expected_yaw), cos(yaw - expected_yaw)))
        ) < 1e-6
        assert candidate.source == 'object_annulus'


def test_annulus_rejects_occupied_candidates_with_counted_reason() -> None:
    grid = open_grid(half_extent=8.0)
    # Wall the entire far side of the annulus.
    add_wall(grid, (3.0, -6.0, 8.0, 6.0))
    outcome = generate_object_annulus_viewpoints(
        (2.0, 0.0),
        grid=grid,
        current_pose_xy_yaw=(-2.0, 0.0, 0.0),
        prefix='plant_1',
    )
    assert outcome.rejected_counts.get('occupied', 0) > 0
    for candidate in outcome.candidates:
        assert candidate.pose_xy_yaw[0] < 3.0


def test_annulus_rejects_candidates_outside_the_travel_budget() -> None:
    grid = open_grid(half_extent=10.0)
    outcome = generate_object_annulus_viewpoints(
        (6.0, 0.0),
        grid=grid,
        current_pose_xy_yaw=(-6.0, 0.0, 0.0),
        prefix='plant_1',
        max_travel_m=2.0,
    )
    assert outcome.candidates == ()
    assert outcome.rejected_counts.get('out_of_budget', 0) > 0


def test_novelty_rejects_a_nearly_identical_repeat_view() -> None:
    config = ViewpointGenerationConfig()
    visited = (VisitedViewpoint((1.0, 1.0, 0.0), focus_key='plant_1'),)
    assert is_novel(
        (1.2, 1.0, 0.05), visited, config, focus_key='plant_1'
    ) is False
    # A large translation is a genuinely new baseline.
    assert is_novel(
        (2.5, 1.0, 0.05), visited, config, focus_key='plant_1'
    ) is True
    # So is a large yaw change from the same spot.
    assert is_novel(
        (1.2, 1.0, 1.2), visited, config, focus_key='plant_1'
    ) is True
    # A different focus is never redundant.
    assert is_novel(
        (1.2, 1.0, 0.05), visited, config, focus_key='chair_2'
    ) is True


def test_generation_counts_redundant_rejections() -> None:
    grid = open_grid(half_extent=8.0)
    first = generate_object_annulus_viewpoints(
        (2.0, 0.0),
        grid=grid,
        current_pose_xy_yaw=(-2.0, 0.0, 0.0),
        prefix='plant_1',
    )
    visited = tuple(
        VisitedViewpoint(item.pose_xy_yaw, focus_key='plant_1')
        for item in first.candidates
    )
    second = generate_object_annulus_viewpoints(
        (2.0, 0.0),
        grid=grid,
        current_pose_xy_yaw=(-2.0, 0.0, 0.0),
        prefix='plant_1',
        visited=visited,
    )
    assert second.rejected_counts.get('redundant', 0) > 0


def test_occluder_offsets_produce_two_lateral_baselines() -> None:
    grid = open_grid(half_extent=8.0)
    outcome = generate_occluder_offset_viewpoints(
        (2.0, 0.0),
        grid=grid,
        current_pose_xy_yaw=(-2.0, 0.0, 0.0),
        prefix='chair_3',
    )
    assert len(outcome.candidates) == 2
    lateral = sorted(item.pose_xy_yaw[1] for item in outcome.candidates)
    assert lateral[0] < 0.0 < lateral[1]
    assert all(
        item.source == 'occluder_offset' for item in outcome.candidates
    )


def test_frontier_viewpoints_face_the_unknown_region() -> None:
    grid = open_grid(half_extent=5.0)
    add_unknown(grid, (2.0, -5.0, 5.0, 5.0))
    outcome = generate_frontier_viewpoints(
        grid=grid,
        current_pose_xy_yaw=(-2.0, 0.0, 0.0),
    )
    assert outcome.candidates
    best = outcome.candidates[0]
    assert best.source == 'frontier'
    # Facing +x, toward the unobserved half of the map.
    assert abs(best.pose_xy_yaw[2]) < pi / 4.0


def test_generation_outcomes_merge_rejection_counts() -> None:
    grid = open_grid(half_extent=8.0)
    first = generate_object_annulus_viewpoints(
        (2.0, 0.0),
        grid=grid,
        current_pose_xy_yaw=(-2.0, 0.0, 0.0),
        prefix='a',
    )
    second = generate_occluder_offset_viewpoints(
        (2.0, 0.0),
        grid=grid,
        current_pose_xy_yaw=(-2.0, 0.0, 0.0),
        prefix='b',
    )
    merged = first.merge(second)
    assert len(merged.candidates) == len(first.candidates) + len(
        second.candidates
    )
    for key in set(first.rejected_counts) | set(second.rejected_counts):
        assert merged.rejected_counts[key] == (
            first.rejected_counts.get(key, 0)
            + second.rejected_counts.get(key, 0)
        )
