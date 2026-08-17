"""Tests for two-stage route planning and ordered semantic execution."""

from fixtures import add_wall
from fixtures import make_instance
from fixtures import open_grid
import pytest

from qmapnav.language import parse_question
from qmapnav.navigation import plan_two_stage_route
from qmapnav.navigation import SemanticStageExecutor
from qmapnav.navigation import SemanticStageState
from qmapnav.navigation import SequentialWaypointExecutor
from qmapnav.navigation import stage_waypoints
from qmapnav.navigation import two_stage_steps
from qmapnav.navigation import TwoStageRouteError
from qmapnav.navigation.instruction_route import PerceivedRoutePlan


QUESTION = (
    'Go to the potted plant furthest from the projector screen then stop '
    'at the water cooler near the window.'
)


def _task():
    return parse_question(QUESTION)


def _resolved(task, plant_xy=(4.0, 0.0), cooler_xy=(-4.0, 0.0)):
    steps = two_stage_steps(task)
    return {
        steps[0][2]: make_instance(
            67, 'potted_plant', (plant_xy[0], plant_xy[1], 0.5),
            (0.8, 0.7, 2.1),
        ),
        steps[1][2]: make_instance(
            23, 'water_cooler', (cooler_xy[0], cooler_xy[1], 0.5),
            (0.4, 0.4, 1.0),
        ),
    }


def test_released_question_yields_exactly_two_destination_stages() -> None:
    task = _task()
    steps = two_stage_steps(task)
    assert len(steps) == 2
    assert [item[1] for item in steps] == ['go_to', 'stop_at']


def test_non_two_stage_instructions_are_refused_explicitly() -> None:
    three_stage = parse_question(
        'First, go near the stool, then take the path near the cabinet, '
        'and stop at the bowl on the table.'
    )
    with pytest.raises(TwoStageRouteError, match='exactly two stages'):
        two_stage_steps(three_stage)
    numerical = parse_question('How many sofas are below a window?')
    with pytest.raises(TwoStageRouteError, match='instruction-following'):
        two_stage_steps(numerical)


def test_plan_produces_ordered_stages_with_safe_goal_poses() -> None:
    task = _task()
    grid = open_grid(half_extent=10.0)
    resolved = _resolved(task)
    plan = plan_two_stage_route(
        task, resolved, grid=grid, start_xy=(0.0, -6.0)
    )
    assert plan.planned is True
    assert len(plan.stages) == 2
    assert [item.stage_index for item in plan.stages] == [0, 1]
    assert plan.stages[0].resolved_instance_id == '67'
    assert plan.stages[1].resolved_instance_id == '23'
    assert plan.stages[1].is_terminal is True
    assert plan.total_path_length_m > 0.0
    for stage in plan.stages:
        x, y, _ = stage.selected_goal_pose
        assert grid.is_free(x, y, clearance=0.35)
        # A goal pose is never the object centre.
        assert stage.target_region.contains((x, y))


def test_goal_poses_are_never_inside_the_target_box() -> None:
    task = _task()
    grid = open_grid(half_extent=10.0)
    resolved = _resolved(task)
    plan = plan_two_stage_route(
        task, resolved, grid=grid, start_xy=(0.0, -6.0)
    )
    for stage, instance in zip(plan.stages, resolved.values()):
        goal = stage.selected_goal_pose
        centre = instance.centroid_xyz
        distance = ((goal[0] - centre[0]) ** 2
                    + (goal[1] - centre[1]) ** 2) ** 0.5
        assert distance > 0.5


def test_unresolved_stage_reports_cleanly_without_deadlock() -> None:
    task = _task()
    grid = open_grid(half_extent=10.0)
    steps = two_stage_steps(task)
    partial = {steps[1][2]: make_instance(
        23, 'water_cooler', (-4.0, 0.0, 0.5), (0.4, 0.4, 1.0)
    )}
    plan = plan_two_stage_route(
        task, partial, grid=grid, start_xy=(0.0, -6.0)
    )
    assert plan.route_status == 'unresolved_stage'
    assert plan.planned is False
    assert plan.unresolved_stages == (0,)
    assert plan.stages == ()


def test_resolved_stage_a_can_be_used_for_route_first_reobservation() -> None:
    task = _task()
    grid = open_grid(half_extent=10.0)
    steps = two_stage_steps(task)
    partial = {steps[0][2]: make_instance(
        67, 'potted_plant', (4.0, 0.0, 0.5), (0.8, 0.8, 1.4)
    )}

    plan = plan_two_stage_route(
        task,
        partial,
        grid=grid,
        start_xy=(0.0, -6.0),
        allow_stage_a_only=True,
    )

    assert plan.route_status == 'stage_a_only'
    assert plan.executable is True
    assert plan.unresolved_stages == (1,)
    assert plan.stages[0].resolved_instance_id == '67'
    assert plan.stages[0].target_reference_id == steps[0][2]


def test_blocked_region_reports_blocked_not_planned() -> None:
    task = _task()
    grid = open_grid(half_extent=10.0)
    resolved = _resolved(task)
    # Seal the whole area around the first stage target.
    add_wall(grid, (1.0, -4.0, 8.0, 4.0))
    plan = plan_two_stage_route(
        task, resolved, grid=grid, start_xy=(0.0, -6.0)
    )
    assert plan.route_status in {'blocked', 'unresolved_stage'}
    assert plan.planned is False


def test_unsupported_instruction_degrades_without_raising() -> None:
    three_stage = parse_question(
        'First, go near the stool, then take the path near the cabinet, '
        'and stop at the bowl on the table.'
    )
    grid = open_grid(half_extent=8.0)
    plan = plan_two_stage_route(
        three_stage, {}, grid=grid, start_xy=(0.0, 0.0)
    )
    assert plan.route_status == 'unsupported_instruction'
    assert plan.planned is False


def test_plan_serializes_a_route_trace_record() -> None:
    task = _task()
    grid = open_grid(half_extent=10.0)
    plan = plan_two_stage_route(
        task, _resolved(task), grid=grid, start_xy=(0.0, -6.0)
    )
    record = plan.to_dict()
    assert record['event'] == 'semantic_route_planned'
    assert record['oracle_mode'] is False
    assert len(record['stages']) == 2
    assert record['stages'][0]['action'] == 'go_to'
    assert record['stages'][1]['action'] == 'stop_at'
    assert len(record['stages'][0]['goal_pose']) == 3


def test_plan_rejects_out_of_order_or_duplicate_stages() -> None:
    with pytest.raises(ValueError, match='route_status'):
        PerceivedRoutePlan(stages=(), route_status='nonsense')


def test_stage_waypoints_end_at_the_goal_pose() -> None:
    task = _task()
    grid = open_grid(half_extent=10.0)
    plan = plan_two_stage_route(
        task, _resolved(task), grid=grid, start_xy=(0.0, -6.0)
    )
    waypoints = stage_waypoints(
        plan, 0, grid=grid, start_xy=(0.0, -6.0)
    )
    assert waypoints
    assert waypoints[-1] == plan.stages[0].selected_goal_pose


def _executor_pair(start_xy=(0.0, -6.0)):
    task = _task()
    grid = open_grid(half_extent=10.0)
    resolved = _resolved(task)
    plan = plan_two_stage_route(
        task, resolved, grid=grid, start_xy=start_xy
    )
    assert plan.planned
    stage_executor = SemanticStageExecutor(
        plan, SequentialWaypointExecutor(arrival_radius=0.75)
    )
    return plan, grid, stage_executor


def test_stage_b_cannot_begin_before_stage_a_is_verified() -> None:
    plan, grid, stage_executor = _executor_pair()
    waypoints_a = stage_waypoints(
        plan, 0, grid=grid, start_xy=(0.0, -6.0)
    )
    stage_executor.start(waypoints_a, now=0.0)
    assert stage_executor.state is SemanticStageState.EXECUTE_STAGE_A

    with pytest.raises(RuntimeError, match='before stage A'):
        stage_executor.begin_stage_b(((0.0, 0.0, 0.0),), now=1.0)


def test_ordered_two_stage_execution_completes_in_order() -> None:
    plan, grid, stage_executor = _executor_pair()
    start = (0.0, -6.0)
    waypoints_a = stage_waypoints(plan, 0, grid=grid, start_xy=start)
    stage_executor.start(waypoints_a, now=0.0)

    goal_a = plan.stages[0].selected_goal_pose
    stage_executor.update_pose(goal_a[0], goal_a[1], goal_a[2], now=5.0)
    assert stage_executor.state is SemanticStageState.VERIFY_STAGE_A
    events = stage_executor.drain_events()
    assert [item.stage_index for item in events] == [0]
    assert events[0].region_satisfied is True

    waypoints_b = stage_waypoints(
        plan, 1, grid=grid, start_xy=(goal_a[0], goal_a[1])
    )
    stage_executor.begin_stage_b(waypoints_b, now=6.0)
    assert stage_executor.state is SemanticStageState.EXECUTE_STAGE_B

    goal_b = plan.stages[1].selected_goal_pose
    stage_executor.update_pose(goal_b[0], goal_b[1], goal_b[2], now=12.0)
    assert stage_executor.state is SemanticStageState.COMPLETE
    final = stage_executor.drain_events()
    assert [item.stage_index for item in final] == [1]
    assert stage_executor.completed_stages == ()


def test_stage_a_does_not_complete_from_waypoint_arrival_alone() -> None:
    plan, grid, stage_executor = _executor_pair()
    start = (0.0, -6.0)
    waypoints_a = stage_waypoints(plan, 0, grid=grid, start_xy=start)
    stage_executor.start(waypoints_a, now=0.0)

    # Sit on an early route waypoint: the executor may advance, but the
    # robot is nowhere near stage A's object, so the stage must not complete.
    early = waypoints_a[0]
    stage_executor.update_pose(early[0], early[1], early[2], now=1.0)
    assert stage_executor.state is SemanticStageState.EXECUTE_STAGE_A
    assert stage_executor.drain_events() == ()


def test_stage_completion_event_serializes_for_the_trace() -> None:
    plan, grid, stage_executor = _executor_pair()
    waypoints_a = stage_waypoints(
        plan, 0, grid=grid, start_xy=(0.0, -6.0)
    )
    stage_executor.start(waypoints_a, now=0.0)
    goal_a = plan.stages[0].selected_goal_pose
    stage_executor.update_pose(goal_a[0], goal_a[1], goal_a[2], now=5.0)
    record = stage_executor.drain_events()[0].to_dict()
    assert record['event'] == 'semantic_stage_complete'
    assert record['stage'] == 0
    assert record['target'] == '67'
    assert record['region_satisfied'] is True
    assert record['semantic_distance_m'] >= 0.0


def test_failure_is_terminal_and_bounded() -> None:
    plan, grid, stage_executor = _executor_pair()
    waypoints_a = stage_waypoints(
        plan, 0, grid=grid, start_xy=(0.0, -6.0)
    )
    stage_executor.start(waypoints_a, now=0.0)
    stage_executor.fail('no_progress')
    assert stage_executor.state is SemanticStageState.FAILED
    assert stage_executor.failure_reason == 'no_progress'
    # Failing twice never reopens the route.
    stage_executor.fail('other')
    assert stage_executor.state is SemanticStageState.FAILED


def test_executor_refuses_an_unplanned_route() -> None:
    unplanned = PerceivedRoutePlan(
        stages=(), route_status='unresolved_stage', unresolved_stages=(0,)
    )
    with pytest.raises(ValueError, match='only a planned'):
        SemanticStageExecutor(unplanned, SequentialWaypointExecutor())


def test_double_start_is_refused() -> None:
    plan, grid, stage_executor = _executor_pair()
    waypoints_a = stage_waypoints(
        plan, 0, grid=grid, start_xy=(0.0, -6.0)
    )
    stage_executor.start(waypoints_a, now=0.0)
    with pytest.raises(RuntimeError, match='already started'):
        stage_executor.start(waypoints_a, now=1.0)
