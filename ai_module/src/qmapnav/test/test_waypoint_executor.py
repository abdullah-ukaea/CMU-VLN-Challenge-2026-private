"""Tests for sequential single-active-waypoint execution."""

from math import nan

import pytest

from qmapnav.navigation import DEFAULT_ARRIVAL_RADIUS
from qmapnav.navigation import DEFAULT_DIRECT_REPUBLISH_LIMIT
from qmapnav.navigation import DEFAULT_NO_PROGRESS_TIMEOUT
from qmapnav.navigation import DEFAULT_PROGRESS_EPSILON
from qmapnav.navigation import DEFAULT_SAFE_OFFSET_LIMIT
from qmapnav.navigation import ExecutorEventType
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
    assert executor.progress_epsilon == DEFAULT_PROGRESS_EPSILON == 0.15
    assert executor.no_progress_timeout == DEFAULT_NO_PROGRESS_TIMEOUT == 12.0
    assert executor.direct_republish_limit == DEFAULT_DIRECT_REPUBLISH_LIMIT == 1
    assert executor.safe_offset_limit == DEFAULT_SAFE_OFFSET_LIMIT == 1


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


def test_two_waypoint_route_completes_with_final_event_metadata() -> None:
    executor = SequentialWaypointExecutor()
    route = [Waypoint2D(1.0, 0.0), Waypoint2D(2.0, 0.0)]
    executor.start(route, now=0.0)

    assert executor.update_pose(1.0, 0.0, now=1.0) == route[1]
    assert executor.update_pose(2.0, 0.0, now=2.0) is None

    completion = executor.drain_events()[-1]
    assert completion.event_type is ExecutorEventType.ROUTE_COMPLETED
    assert completion.route_index == 1
    assert completion.direct_republish_count == 0
    assert completion.recovery_count == 0


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


def test_small_noisy_motion_does_not_postpone_watchdog() -> None:
    executor = SequentialWaypointExecutor(
        progress_epsilon=0.15,
        no_progress_timeout=12.0,
    )
    goal = Waypoint2D(10.0, 0.0)
    executor.start([goal], now=0.0)
    executor.update_pose(0.0, 0.0, now=1.0)

    executor.update_pose(0.05, 0.0, now=8.0)
    executor.update_pose(0.10, 0.0, now=12.0)

    assert executor.tick(now=12.9) is None
    assert executor.tick(now=13.0) == goal
    assert executor.direct_republish_count == 1


def test_meaningful_progress_resets_watchdog() -> None:
    executor = SequentialWaypointExecutor(no_progress_timeout=12.0)
    goal = Waypoint2D(10.0, 0.0)
    executor.start([goal], now=0.0)
    executor.update_pose(0.0, 0.0, now=1.0)

    executor.update_pose(0.2, 0.0, now=12.0)

    assert executor.tick(now=23.9) is None
    assert executor.tick(now=24.0) == goal


def test_republish_recovery_and_failure_are_strictly_bounded() -> None:
    recovery = Waypoint2D(0.0, 1.0, 0.0)
    selections = []

    def select_offset(x: float, y: float, goal: Waypoint2D) -> Waypoint2D:
        selections.append((x, y, goal))
        return recovery

    executor = SequentialWaypointExecutor(
        safe_offset_selector=select_offset,
        no_progress_timeout=12.0,
    )
    goal = Waypoint2D(10.0, 0.0)
    executor.start([goal], now=0.0)
    executor.update_pose(0.0, 0.0, now=0.0)

    assert executor.tick(now=12.0) == goal
    assert executor.tick(now=24.0) == recovery
    assert executor.state is WaypointExecutorState.RECOVERING
    assert executor.update_pose(0.0, 1.0, now=25.0) == goal
    assert executor.state is WaypointExecutorState.ACTIVE
    assert executor.tick(now=37.0) is None

    assert selections == [(0.0, 0.0, goal)]
    assert executor.direct_republish_count == 1
    assert executor.recovery_count == 1
    assert executor.state is WaypointExecutorState.FAILED
    events = executor.drain_events()
    assert [event.event_type for event in events].count(
        ExecutorEventType.GOAL_REPUBLISHED
    ) == 1
    assert [event.event_type for event in events].count(
        ExecutorEventType.RECOVERY_STARTED
    ) == 1
    assert events[-1].event_type is ExecutorEventType.ROUTE_FAILED
    assert events[-1].route_index == 0


def test_recovery_timeout_fails_instead_of_cycling() -> None:
    recovery = Waypoint2D(0.0, 1.0)
    executor = SequentialWaypointExecutor(
        direct_republish_limit=0,
        safe_offset_selector=lambda x, y, goal: recovery,
    )
    executor.start([Waypoint2D(10.0, 0.0)], now=0.0)
    executor.update_pose(0.0, 0.0, now=0.0)

    assert executor.tick(now=12.0) == recovery
    assert executor.tick(now=24.0) is None
    assert executor.state is WaypointExecutorState.FAILED


def test_unknown_or_unavailable_safe_offset_fails_safely() -> None:
    executor = SequentialWaypointExecutor(direct_republish_limit=0)
    executor.start([Waypoint2D(10.0, 0.0)], now=0.0)
    executor.update_pose(0.0, 0.0, now=0.0)

    assert executor.tick(now=12.0) is None
    assert executor.state is WaypointExecutorState.FAILED
    assert executor.drain_events()[-1].reason == (
        'safe_offset_selector_unavailable'
    )


def test_no_pose_timeout_is_bounded() -> None:
    executor = SequentialWaypointExecutor(
        direct_republish_limit=0,
        safe_offset_selector=lambda x, y, goal: Waypoint2D(x, y),
    )
    executor.start([Waypoint2D(10.0, 0.0)], now=0.0)

    assert executor.tick(now=12.0) is None
    assert executor.state is WaypointExecutorState.FAILED
    assert executor.drain_events()[-1].reason == (
        'safe_offset_unavailable_without_pose'
    )


def test_cancel_with_pose_emits_one_hold_and_is_idempotent() -> None:
    executor = SequentialWaypointExecutor()
    executor.start(ROUTE, now=0.0)
    executor.update_pose(0.2, -0.3, 1.2, now=1.0)

    hold = executor.cancel(now=2.0)

    assert hold == Waypoint2D(0.2, -0.3, 1.2)
    assert executor.cancel(now=3.0) is None
    assert executor.state is WaypointExecutorState.CANCELLED
    assert executor.active_waypoint is None
    assert executor.update_pose(1.0, 0.0, now=4.0) is None
    events = executor.drain_events()
    assert events[-1].event_type is ExecutorEventType.ROUTE_CANCELLED
    assert events[-1].route_index == 0


def test_cancel_without_pose_and_after_completion_are_safe() -> None:
    executor = SequentialWaypointExecutor()
    executor.start(ROUTE, now=0.0)

    assert executor.cancel(now=1.0) is None
    assert executor.state is WaypointExecutorState.CANCELLED

    completed = SequentialWaypointExecutor()
    completed.start([Waypoint2D(0.0, 0.0)], now=0.0)
    completed.update_pose(0.0, 0.0, now=1.0)

    assert completed.cancel(now=2.0) is None
    assert completed.state is WaypointExecutorState.COMPLETE


def test_cancel_during_recovery_stops_interrupted_route() -> None:
    recovery = Waypoint2D(0.0, 1.0)
    executor = SequentialWaypointExecutor(
        direct_republish_limit=0,
        safe_offset_selector=lambda x, y, goal: recovery,
    )
    executor.start([Waypoint2D(10.0, 0.0)], now=0.0)
    executor.update_pose(0.0, 0.0, now=0.0)
    assert executor.tick(now=12.0) == recovery

    assert executor.cancel(now=13.0) == Waypoint2D(0.0, 0.0)
    assert executor.state is WaypointExecutorState.CANCELLED
    assert executor.update_pose(0.0, 1.0, now=14.0) is None
    assert executor.tick(now=30.0) is None


@pytest.mark.parametrize(
    ('keyword', 'value'),
    [
        ('progress_epsilon', 0.0),
        ('no_progress_timeout', -1.0),
        ('direct_republish_limit', -1),
        ('safe_offset_limit', -1),
    ],
)
def test_executor_rejects_invalid_watchdog_configuration(
    keyword: str,
    value: float,
) -> None:
    with pytest.raises(ValueError):
        SequentialWaypointExecutor(**{keyword: value})
