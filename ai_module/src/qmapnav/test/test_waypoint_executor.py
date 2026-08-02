"""Tests for sequential single-active-waypoint execution."""

from math import nan

import pytest

from qmapnav.navigation import DEFAULT_ARRIVAL_RADIUS
from qmapnav.navigation import SequentialWaypointExecutor
from qmapnav.navigation import Waypoint2D
from qmapnav.navigation import WaypointExecutorState


ROUTE = [
    Waypoint2D(1.0, 0.0, 0.0),
    Waypoint2D(2.0, 1.0, 1.57),
    Waypoint2D(3.0, 1.0, 0.0),
]


def test_executor_uses_measured_provisional_arrival_radius() -> None:
    executor = SequentialWaypointExecutor()

    assert executor.arrival_radius == DEFAULT_ARRIVAL_RADIUS == 0.75


def test_start_owns_route_and_activates_only_first_goal() -> None:
    executor = SequentialWaypointExecutor()
    supplied_route = list(ROUTE)

    first_goal = executor.start(supplied_route)
    supplied_route.clear()

    assert first_goal == ROUTE[0]
    assert executor.route == tuple(ROUTE)
    assert executor.active_waypoint == ROUTE[0]
    assert executor.active_index == 0
    assert executor.state is WaypointExecutorState.ACTIVE


def test_pose_outside_arrival_radius_does_not_advance_route() -> None:
    executor = SequentialWaypointExecutor()
    executor.start(ROUTE)

    next_goal = executor.update_pose(0.0, 0.0)

    assert next_goal is None
    assert executor.active_waypoint == ROUTE[0]
    assert executor.active_index == 0


def test_pose_at_arrival_boundary_activates_next_goal() -> None:
    executor = SequentialWaypointExecutor(arrival_radius=0.75)
    executor.start(ROUTE)

    next_goal = executor.update_pose(0.25, 0.0)

    assert next_goal == ROUTE[1]
    assert executor.active_waypoint == ROUTE[1]
    assert executor.active_index == 1


def test_three_waypoints_advance_one_at_a_time_and_complete() -> None:
    executor = SequentialWaypointExecutor()
    published = [executor.start(ROUTE)]

    outside_updates = [
        executor.update_pose(0.0, 0.0),
        executor.update_pose(1.0, 0.9),
        executor.update_pose(2.0, 0.0),
    ]
    second = executor.update_pose(1.0, 0.0)
    third = executor.update_pose(2.0, 1.0)
    final_update = executor.update_pose(3.0, 1.0)
    published.extend(goal for goal in (second, third) if goal is not None)

    assert outside_updates == [None, None, None]
    assert published == ROUTE
    assert final_update is None
    assert executor.active_waypoint is None
    assert executor.active_index is None
    assert executor.state is WaypointExecutorState.COMPLETE


def test_single_waypoint_route_completes_without_another_goal() -> None:
    executor = SequentialWaypointExecutor()
    goal = Waypoint2D(1.0, 2.0)
    executor.start([goal])

    next_goal = executor.update_pose(1.0, 2.0)

    assert next_goal is None
    assert executor.state is WaypointExecutorState.COMPLETE


def test_active_route_cannot_be_replaced() -> None:
    executor = SequentialWaypointExecutor()
    executor.start(ROUTE)

    with pytest.raises(RuntimeError, match='active route'):
        executor.start([Waypoint2D(9.0, 9.0)])

    assert executor.active_waypoint == ROUTE[0]


@pytest.mark.parametrize('arrival_radius', [0.0, -0.1, nan])
def test_executor_rejects_invalid_arrival_radius(arrival_radius: float) -> None:
    with pytest.raises(ValueError, match='arrival_radius'):
        SequentialWaypointExecutor(arrival_radius)


def test_executor_rejects_empty_or_invalid_route() -> None:
    executor = SequentialWaypointExecutor()

    with pytest.raises(ValueError, match='at least one'):
        executor.start([])
    with pytest.raises(TypeError, match='Waypoint2D'):
        executor.start([(1.0, 2.0)])


def test_waypoint_and_pose_values_must_be_finite() -> None:
    with pytest.raises(ValueError, match='finite'):
        Waypoint2D(nan, 0.0)

    executor = SequentialWaypointExecutor()
    executor.start(ROUTE)
    with pytest.raises(ValueError, match='finite'):
        executor.update_pose(0.0, nan)

    assert executor.active_waypoint == ROUTE[0]


def test_pose_updates_are_noop_while_idle_or_complete() -> None:
    executor = SequentialWaypointExecutor()

    assert executor.update_pose(0.0, 0.0) is None
    executor.start([Waypoint2D(0.0, 0.0)])
    executor.update_pose(0.0, 0.0)

    assert executor.update_pose(0.0, 0.0) is None
    assert executor.state is WaypointExecutorState.COMPLETE
