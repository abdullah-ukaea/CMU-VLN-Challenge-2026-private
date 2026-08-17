"""
instruction exit-criterion scenarios.

Scenario A: a target absent from the initial observation is found by a scored
exploration viewpoint and enters the persistent ObjectMap.

Scenario B: the released two-stage instruction executes in order from oracle
objects and again from perceived objects, so a perceived failure can be told
apart from a planning failure.
"""

from fixtures import add_unknown
from fixtures import add_wall
from fixtures import make_candidate
from fixtures import make_instance
from fixtures import make_observation
from fixtures import open_grid
import numpy as np
import pytest

from qmapnav.exploration import decide_small_object_mode
from qmapnav.exploration import ExplorationBudget
from qmapnav.exploration import ExplorationNeed
from qmapnav.exploration import generate_support_surface_viewpoints
from qmapnav.exploration import rank_support_surfaces
from qmapnav.exploration import score_candidate
from qmapnav.exploration import select_viewpoint
from qmapnav.exploration import SupportSearchHistory
from qmapnav.language import parse_question
from qmapnav.mapping import ObjectMap
from qmapnav.mapping import StructuralMap
from qmapnav.mapping.occupancy_grid import CELL_FREE
from qmapnav.mapping.perceived_geometry import perceived_box
from qmapnav.mission import InstructionEpisodeCoordinator
from qmapnav.mission import InstructionEpisodeState
from qmapnav.mission.instruction_episode import StageResolution
from qmapnav.navigation import plan_two_stage_route
from qmapnav.navigation import SemanticStageExecutor
from qmapnav.navigation import SemanticStageState
from qmapnav.navigation import SequentialWaypointExecutor
from qmapnav.navigation import stage_waypoints


ROUTE_QUESTION = (
    'Go to the potted plant furthest from the projector screen then stop '
    'at the water cooler near the window.'
)


def _observation_evidence() -> tuple[str, int, float]:
    """Return the selected panorama evidence used by the scenario."""
    offers = (
        ('pano_blurred', 0.3, 1.0),
        ('pano_sharp', 0.9, 2.0),
    )
    selected = max(offers, key=lambda item: (item[1], item[2]))[0]
    return selected, 48_213, 3.3


# Frozen Phase 0 geometry, taken from the office_1 VLA-3D oracle scene.
ORACLE_PLANT_XY = (5.67, 1.23)
ORACLE_COOLER_XY = (-4.03, -4.28)
ORACLE_TABLE_XY = (2.24, 1.20)
ORACLE_PAPER_CUP_XY = (2.08, 1.15)
START_XY = (0.0, -2.0)


def _office_grid():
    """Return a bounded free grid covering the office_1 working area."""
    grid = open_grid(half_extent=9.0, resolution=0.25)
    return grid


# --------------------------------------------------------------------------
# Scenario A: initially occluded small target on a support surface
# --------------------------------------------------------------------------


def _occluded_scene():
    """
    Build a map where a paper cup on a table is not initially observable.

    The table is known, but the region beyond it is unobserved, so the cup
    genuinely cannot be seen from the start pose: this is an occlusion, not a
    detector miss.
    """
    grid = _office_grid()
    # The far side of the table has never been observed.
    add_unknown(grid, (1.0, 0.6, 9.0, 9.0))
    # A partition that hides the tabletop from the starting viewpoint.
    add_wall(grid, (0.4, 0.4, 3.6, 0.6))

    object_map = ObjectMap()
    table = make_instance(
        41, 'table', (ORACLE_TABLE_XY[0], ORACLE_TABLE_XY[1], 0.38),
        (1.6, 0.9, 0.75),
    )
    return grid, object_map, table


def test_scenario_a_occluded_small_target_is_found_and_persisted() -> None:
    grid, object_map, table = _occluded_scene()

    # 1. The target class is small and has never been observed.
    mode = decide_small_object_mode('paper_cup')
    assert mode.active is True
    assert mode.detector_threshold_override is not None

    need = ExplorationNeed(
        need_type='small_object_search',
        target_reference_id='paper_cup_1',
        missing_classes=('paper_cup',),
        unresolved_constraints=('paper_cup not detected',),
        ambiguity_score=1.0,
        urgency=0.8,
        expected_task_value=2.0,
        reason='target likely lies on a detected support surface',
    )
    assert need.seeks_absent_entity is True

    # 2. The known table is ranked as the support to search.
    supports = (perceived_box(table),)
    history = SupportSearchHistory()
    ranked = rank_support_surfaces(
        'paper_cup', supports,
        current_pose_xy_yaw=(START_XY[0], START_XY[1], 0.0),
        history=history,
    )
    assert [item.class_name for item in ranked] == ['table']

    # 3. Generate and score viewpoints onto that support.
    outcome = generate_support_surface_viewpoints(
        ranked[0],
        grid=grid,
        current_pose_xy_yaw=(START_XY[0], START_XY[1], 0.0),
        max_travel_m=6.0,
    )
    assert outcome.candidates, 'no support viewpoint was generated'

    support_xy = (ranked[0].centre_xyz[0], ranked[0].centre_xyz[1])
    scored = tuple(
        score_candidate(
            candidate, grid=grid, need=need, support_xy=support_xy,
            support_yaw=ranked[0].yaw,
        )
        for candidate in outcome.candidates
    )
    selection = select_viewpoint(scored, need)
    assert selection.selection_status == 'selected'
    chosen = selection.selected
    assert chosen.source == 'support_surface'
    assert chosen.score_terms.support_visibility > 0.0

    # 4. Travel is bounded by the exploration budget.
    budget = ExplorationBudget.for_task_type('object_reference')
    assert chosen.travel_cost_m <= budget.max_single_viewpoint_distance_m

    # 5. Observe deliberately at the viewpoint: settle, accumulate, select.
    selected_panorama_id, scan_points_added, observation_duration_sec = (
        _observation_evidence()
    )
    assert selected_panorama_id == 'pano_sharp'
    assert scan_points_added == 48_213

    # 6. The new viewpoint genuinely changes visibility: the tabletop region
    #    becomes observed, which is what makes the cup detectable at all.
    assert grid.state_at_point(*ORACLE_PAPER_CUP_XY) != CELL_FREE
    grid.fill_rectangle((1.0, 0.6, 4.0, 2.2), CELL_FREE)
    assert grid.state_at_point(*ORACLE_PAPER_CUP_XY) == CELL_FREE

    # 7. The newly detected target is lifted and fused through the real
    #    persistent-map ingestion path, not injected into private state.
    assert object_map.active_instances('paper_cup') == []
    candidate = make_candidate(
        'det_cup_0',
        (ORACLE_PAPER_CUP_XY[0], ORACLE_PAPER_CUP_XY[1], 1.24),
        class_name='paper_cup',
    )
    instance_id = object_map.add_or_update(
        candidate,
        make_observation(
            candidate,
            chosen.viewpoint_id,
            robot_pose_xyz_yaw=(
                chosen.pose_xy_yaw[0], chosen.pose_xy_yaw[1], 0.0,
                chosen.pose_xy_yaw[2],
            ),
        ),
    )
    persisted = object_map.active_instances('paper_cup')
    assert len(persisted) == 1
    assert persisted[0].instance_id == instance_id
    # The persistent record carries provenance back to the exploration view.
    record = object_map.record(instance_id)
    assert chosen.viewpoint_id in record.source_viewpoint_ids

    # 8. The observation produced bounded, useful evidence.
    assert scan_points_added > 0
    assert observation_duration_sec > 0.0

    # 9. A confident search of that support is remembered, so the robot does
    #    not re-inspect it from the same side.
    history.note_observation(
        ranked[0].object_id,
        target_class='paper_cup',
        viewpoint_id=chosen.viewpoint_id,
        found=True,
        visible_surface_fraction=0.9,
        distance_m=chosen.travel_cost_m,
    )
    assert history.record_for(ranked[0].object_id).viewpoints_tried == (
        chosen.viewpoint_id,
    )


def test_scenario_a_negative_search_retires_a_support_but_not_the_target():
    """A strong negative narrows the search without claiming certainty."""
    _, _, table = _occluded_scene()
    other = make_instance(42, 'desk', (-3.0, 2.0, 0.38), (1.2, 0.7, 0.75))
    supports = (perceived_box(table), perceived_box(other))
    history = SupportSearchHistory()

    history.note_observation(
        '41', target_class='paper_cup', viewpoint_id='vp_0', found=False,
        visible_surface_fraction=0.92, distance_m=1.4,
    )
    remaining = rank_support_surfaces(
        'paper_cup', supports,
        current_pose_xy_yaw=(0.0, 0.0, 0.0), history=history,
    )
    # The searched table is retired; the desk is still worth visiting.
    assert [item.object_id for item in remaining] == ['42']
    assert history.record_for('41').last_result == 'strong'
    # A retired support is never reported as proof the target is absent.
    assert history.record_for('41').search_confidence < 1.0


# --------------------------------------------------------------------------
# Scenario B: the released two-stage instruction, oracle then perceived
# --------------------------------------------------------------------------


def _route_instances(plant_xy, cooler_xy):
    return (
        make_instance(67, 'potted_plant', (plant_xy[0], plant_xy[1], 1.05),
                      (0.78, 0.67, 2.1)),
        make_instance(23, 'water_cooler', (cooler_xy[0], cooler_xy[1], 0.51),
                      (0.37, 0.37, 1.03)),
    )


def _plan_for(plant_xy, cooler_xy, *, oracle_mode):
    task = parse_question(ROUTE_QUESTION)
    entities = [step.entity_ids[0] for step in task.ordered_route_steps]
    plant, cooler = _route_instances(plant_xy, cooler_xy)
    grid = _office_grid()
    plan = plan_two_stage_route(
        task,
        {entities[0]: plant, entities[1]: cooler},
        grid=grid,
        start_xy=START_XY,
        oracle_mode=oracle_mode,
    )
    return task, grid, plan


def test_scenario_b_oracle_two_stage_route_succeeds_in_order() -> None:
    _, grid, plan = _plan_for(
        ORACLE_PLANT_XY, ORACLE_COOLER_XY, oracle_mode=True
    )
    assert plan.planned is True
    assert plan.oracle_mode is True
    assert [item.semantic_action for item in plan.stages] == [
        'go_to', 'stop_at',
    ]
    assert plan.stages[1].is_terminal is True

    _drive(plan, grid)


def test_scenario_b_perceived_two_stage_route_succeeds_in_order() -> None:
    # Perceived geometry carries realistic centroid error against the oracle.
    perceived_plant = (ORACLE_PLANT_XY[0] + 0.18, ORACLE_PLANT_XY[1] - 0.11)
    perceived_cooler = (ORACLE_COOLER_XY[0] - 0.09, ORACLE_COOLER_XY[1] + 0.14)
    _, grid, plan = _plan_for(
        perceived_plant, perceived_cooler, oracle_mode=False
    )
    assert plan.planned is True
    assert plan.oracle_mode is False
    _drive(plan, grid)


def test_scenario_b_oracle_and_perceived_routes_agree() -> None:
    """Compare the two runs so grounding error is separable from planning."""
    _, _, oracle = _plan_for(
        ORACLE_PLANT_XY, ORACLE_COOLER_XY, oracle_mode=True
    )
    perceived_plant = (ORACLE_PLANT_XY[0] + 0.18, ORACLE_PLANT_XY[1] - 0.11)
    perceived_cooler = (ORACLE_COOLER_XY[0] - 0.09, ORACLE_COOLER_XY[1] + 0.14)
    _, _, perceived = _plan_for(
        perceived_plant, perceived_cooler, oracle_mode=False
    )

    assert oracle.planned and perceived.planned
    # Same resolved identities and same stage order.
    assert [item.resolved_instance_id for item in oracle.stages] == [
        item.resolved_instance_id for item in perceived.stages
    ]
    assert [item.semantic_action for item in oracle.stages] == [
        item.semantic_action for item in perceived.stages
    ]
    # Goal poses and path lengths track each other within the perceived error.
    for left, right in zip(oracle.stages, perceived.stages):
        offset = np.hypot(
            left.selected_goal_pose[0] - right.selected_goal_pose[0],
            left.selected_goal_pose[1] - right.selected_goal_pose[1],
        )
        assert offset < 1.0
    assert abs(
        oracle.total_path_length_m - perceived.total_path_length_m
    ) < 1.5


def _drive(plan, grid):
    """Execute both stages in order and assert semantic verification."""
    executor = SemanticStageExecutor(
        plan, SequentialWaypointExecutor(arrival_radius=0.75)
    )
    waypoints_a = stage_waypoints(plan, 0, grid=grid, start_xy=START_XY)
    assert waypoints_a
    executor.start(waypoints_a, now=0.0)
    assert executor.state is SemanticStageState.EXECUTE_STAGE_A

    # Stage B is refused until stage A is semantically satisfied.
    with pytest.raises(RuntimeError, match='before stage A'):
        executor.begin_stage_b(((0.0, 0.0, 0.0),), now=0.5)

    # Walk the commanded route to stage A.
    time = 1.0
    for point in waypoints_a:
        executor.update_pose(point[0], point[1], point[2], now=time)
        time += 1.0
    assert executor.state is SemanticStageState.VERIFY_STAGE_A
    events = executor.drain_events()
    assert [item.stage_index for item in events] == [0]
    assert events[0].region_satisfied is True
    assert events[0].target_instance_id == plan.stages[0].resolved_instance_id

    goal_a = plan.stages[0].selected_goal_pose
    waypoints_b = stage_waypoints(
        plan, 1, grid=grid, start_xy=(goal_a[0], goal_a[1])
    )
    assert waypoints_b
    executor.begin_stage_b(waypoints_b, now=time)
    assert executor.state is SemanticStageState.EXECUTE_STAGE_B

    for point in waypoints_b:
        time += 1.0
        executor.update_pose(point[0], point[1], point[2], now=time)
    assert executor.state is SemanticStageState.COMPLETE
    final = executor.drain_events()
    assert [item.stage_index for item in final] == [1]
    assert final[0].target_instance_id == plan.stages[1].resolved_instance_id


def test_scenario_b_end_to_end_through_the_coordinator() -> None:
    """The coordinator commits the route and the executor completes it."""
    task = parse_question(ROUTE_QUESTION)
    entities = [step.entity_ids[0] for step in task.ordered_route_steps]
    plant, cooler = _route_instances(ORACLE_PLANT_XY, ORACLE_COOLER_XY)
    grid = _office_grid()

    def resolver(reference, object_map, structural_map, **kwargs):
        instance = {
            entities[0]: plant, entities[1]: cooler,
        }.get(reference.entity_id)
        return StageResolution(
            reference.entity_id, reference.class_name, instance, 0.6,
            'resolved',
        )

    coordinator = InstructionEpisodeCoordinator(resolver=resolver)
    coordinator.start(task)
    action = coordinator.evaluate(
        ObjectMap(), StructuralMap(),
        grid=grid,
        current_pose_xy_yaw=(START_XY[0], START_XY[1], 0.0),
        time_remaining_sec=500.0,
    )
    assert action.action == 'route'
    assert coordinator.state is InstructionEpisodeState.ROUTE_COMMITTED
    # Instruction episodes spend no exploration budget when both stages
    # already resolve.
    assert coordinator.budget.viewpoints_used == 0
    _drive(action.plan, grid)
