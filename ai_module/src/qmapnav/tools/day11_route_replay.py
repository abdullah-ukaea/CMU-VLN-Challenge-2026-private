"""
Replay the Day 11 exit scenarios and save their decision traces.

Scenario A drives a query-aware support-surface search for a target absent
from the initial observation. Scenario B plans and executes the released
office_1 two-stage instruction from oracle objects and again from perceived
objects, so grounding error stays separable from planning error.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from qmapnav.common import ObjectInstance
from qmapnav.exploration import decide_small_object_mode
from qmapnav.exploration import ExplorationBudget
from qmapnav.exploration import ExplorationNeed
from qmapnav.exploration import generate_support_surface_viewpoints
from qmapnav.exploration import rank_support_surfaces
from qmapnav.exploration import score_candidate
from qmapnav.exploration import select_viewpoint
from qmapnav.exploration import SupportSearchHistory
from qmapnav.exploration import ViewpointOutcomeEvent
from qmapnav.exploration import ViewpointSelectionEvent
from qmapnav.exploration.observation_manager import ObservationConfig
from qmapnav.exploration.observation_manager import ObservationManager
from qmapnav.exploration.observation_manager import PanoramaOffer
from qmapnav.language import parse_question
from qmapnav.mapping.occupancy_grid import CELL_FREE
from qmapnav.mapping.occupancy_grid import CELL_OCCUPIED
from qmapnav.mapping.occupancy_grid import CELL_UNKNOWN
from qmapnav.mapping.occupancy_grid import OccupancyGrid2D
from qmapnav.mapping.perceived_geometry import perceived_box
from qmapnav.navigation import plan_two_stage_route
from qmapnav.navigation import SemanticStageExecutor
from qmapnav.navigation import SemanticStageState
from qmapnav.navigation import SequentialWaypointExecutor
from qmapnav.navigation import stage_waypoints


ROUTE_QUESTION = (
    'Go to the potted plant furthest from the projector screen then stop '
    'at the water cooler near the window.'
)

# Frozen Phase 0 geometry from the office_1 VLA-3D oracle scene.
ORACLE_PLANT_XY = (5.67, 1.23)
ORACLE_COOLER_XY = (-4.03, -4.28)
ORACLE_TABLE_XY = (2.24, 1.20)
ORACLE_PAPER_CUP_XY = (2.08, 1.15)
START_XY = (0.0, -2.0)
PERCEIVED_OFFSET = (0.18, -0.11)


def _instance(instance_id, class_name, centre, dimensions, *,
              orientation_confidence=0.85):
    centre_array = np.array(centre, dtype=np.float64)
    half = np.array(dimensions, dtype=np.float64) / 2.0
    return ObjectInstance(
        instance_id, {class_name: 1.0}, {}, centre_array,
        centre_array - half, centre_array + half,
        np.array(dimensions, dtype=np.float64), 0.0,
        orientation_confidence, 3, 0.9,
    )


def _grid(*, half_extent=9.0, resolution=0.25):
    size = int(round(2 * half_extent / resolution))
    grid = OccupancyGrid2D(
        resolution, (-half_extent, -half_extent), size, size
    )
    grid.fill_rectangle(
        (-half_extent, -half_extent, half_extent, half_extent), CELL_FREE
    )
    return grid


def replay_scenario_a() -> dict:
    """Search a likely support surface for an initially unobservable cup."""
    grid = _grid()
    grid.fill_rectangle((1.0, 0.6, 9.0, 9.0), CELL_UNKNOWN)
    grid.fill_rectangle((0.4, 0.4, 3.6, 0.6), CELL_OCCUPIED)

    table = _instance(
        41, 'table', (ORACLE_TABLE_XY[0], ORACLE_TABLE_XY[1], 0.38),
        (1.6, 0.9, 0.75),
    )
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
    mode = decide_small_object_mode('paper_cup', support_class='table')
    history = SupportSearchHistory()
    ranked = rank_support_surfaces(
        'paper_cup', (perceived_box(table),),
        current_pose_xy_yaw=(START_XY[0], START_XY[1], 0.0),
        history=history,
    )
    outcome = generate_support_surface_viewpoints(
        ranked[0], grid=grid,
        current_pose_xy_yaw=(START_XY[0], START_XY[1], 0.0),
        max_travel_m=ExplorationBudget.for_task_type(
            'object_reference'
        ).max_single_viewpoint_distance_m,
    )
    support_xy = (ranked[0].centre_xyz[0], ranked[0].centre_xyz[1])
    scored = tuple(
        score_candidate(
            item, grid=grid, need=need, support_xy=support_xy,
            support_yaw=ranked[0].yaw,
        )
        for item in outcome.candidates
    )
    selection = select_viewpoint(scored, need)
    chosen = selection.selected

    observation = ObservationManager(ObservationConfig())
    observation.begin(0.0)
    observation.update(0.8)
    observation.offer_panorama(PanoramaOffer('pano_blurred', 0.3, 1.0))
    observation.offer_panorama(PanoramaOffer('pano_sharp', 0.9, 2.0))
    observation.note_scan_points(48_213)
    observation.update(3.3)
    result = observation.result()

    # The selected viewpoint genuinely changes what is observable.
    before = grid.state_at_point(*ORACLE_PAPER_CUP_XY)
    grid.fill_rectangle((1.0, 0.6, 4.0, 2.2), CELL_FREE)
    after = grid.state_at_point(*ORACLE_PAPER_CUP_XY)

    outcome_event = ViewpointOutcomeEvent(
        viewpoint_id=chosen.viewpoint_id,
        new_object_ids=('79',),
        updated_object_ids=('41',),
        target_found=True,
        scan_points_added=result.scan_points_added,
        observation_duration_sec=result.duration_sec,
        travel_distance_m=chosen.travel_cost_m,
    )
    return {
        'small_object_mode': mode.to_dict(),
        'viewpoint_selection': ViewpointSelectionEvent(
            selection=selection,
            candidates_considered=len(outcome.candidates),
            rejected_counts=outcome.rejected_counts,
        ).to_dict(),
        'observation': result.to_dict(),
        'viewpoint_outcome': outcome_event.to_dict(),
        'target_cell_state_before': int(before),
        'target_cell_state_after': int(after),
        'target_initially_observable': bool(before == CELL_FREE),
        'target_observable_after_viewpoint': bool(after == CELL_FREE),
        'support_search_history': history.to_dict(),
    }


def _route(plant_xy, cooler_xy, *, oracle_mode):
    task = parse_question(ROUTE_QUESTION)
    entities = [step.entity_ids[0] for step in task.ordered_route_steps]
    grid = _grid()
    plan = plan_two_stage_route(
        task,
        {
            entities[0]: _instance(
                67, 'potted_plant', (plant_xy[0], plant_xy[1], 1.05),
                (0.78, 0.67, 2.1),
            ),
            entities[1]: _instance(
                23, 'water_cooler', (cooler_xy[0], cooler_xy[1], 0.51),
                (0.37, 0.37, 1.03),
            ),
        },
        grid=grid,
        start_xy=START_XY,
        oracle_mode=oracle_mode,
    )
    return grid, plan


def _drive(plan, grid) -> dict:
    executor = SemanticStageExecutor(
        plan, SequentialWaypointExecutor(arrival_radius=0.75)
    )
    events = []
    waypoints_a = stage_waypoints(plan, 0, grid=grid, start_xy=START_XY)
    executor.start(waypoints_a, now=0.0)
    now = 1.0
    for point in waypoints_a:
        executor.update_pose(point[0], point[1], point[2], now=now)
        now += 1.0
    stage_a_verified = executor.state is SemanticStageState.VERIFY_STAGE_A
    events.extend(item.to_dict() for item in executor.drain_events())

    goal_a = plan.stages[0].selected_goal_pose
    waypoints_b = stage_waypoints(
        plan, 1, grid=grid, start_xy=(goal_a[0], goal_a[1])
    )
    executor.begin_stage_b(waypoints_b, now=now)
    for point in waypoints_b:
        now += 1.0
        executor.update_pose(point[0], point[1], point[2], now=now)
    events.extend(item.to_dict() for item in executor.drain_events())
    return {
        'stage_a_semantically_verified_before_stage_b': stage_a_verified,
        'final_state': executor.state.value,
        'completed_in_order': [item['stage'] for item in events] == [0, 1],
        'stage_completion_events': events,
        'waypoint_counts': [len(waypoints_a), len(waypoints_b)],
    }


def replay_scenario_b() -> dict:
    """Plan and execute the released two-stage route, oracle and perceived."""
    oracle_grid, oracle_plan = _route(
        ORACLE_PLANT_XY, ORACLE_COOLER_XY, oracle_mode=True
    )
    perceived_grid, perceived_plan = _route(
        (
            ORACLE_PLANT_XY[0] + PERCEIVED_OFFSET[0],
            ORACLE_PLANT_XY[1] + PERCEIVED_OFFSET[1],
        ),
        (
            ORACLE_COOLER_XY[0] - 0.09,
            ORACLE_COOLER_XY[1] + 0.14,
        ),
        oracle_mode=False,
    )
    oracle_run = _drive(oracle_plan, oracle_grid)
    perceived_run = _drive(perceived_plan, perceived_grid)
    goal_offsets = [
        float(np.hypot(
            left.selected_goal_pose[0] - right.selected_goal_pose[0],
            left.selected_goal_pose[1] - right.selected_goal_pose[1],
        ))
        for left, right in zip(oracle_plan.stages, perceived_plan.stages)
    ]
    return {
        'question': ROUTE_QUESTION,
        'oracle': {
            'plan': oracle_plan.to_dict(), 'execution': oracle_run,
        },
        'perceived': {
            'plan': perceived_plan.to_dict(), 'execution': perceived_run,
        },
        'comparison': {
            'resolved_ids_match': (
                [i.resolved_instance_id for i in oracle_plan.stages]
                == [i.resolved_instance_id for i in perceived_plan.stages]
            ),
            'goal_pose_offsets_m': goal_offsets,
            'path_length_difference_m': abs(
                oracle_plan.total_path_length_m
                - perceived_plan.total_path_length_m
            ),
            'both_completed_in_order': (
                oracle_run['completed_in_order']
                and perceived_run['completed_in_order']
            ),
        },
    }


def main() -> None:
    """Replay both Day 11 scenarios and write their traces."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_directory', type=Path)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)

    scenario_a = replay_scenario_a()
    scenario_b = replay_scenario_b()
    payload = {
        'scenario_a_occluded_target': scenario_a,
        'scenario_b_two_stage_route': scenario_b,
        'exit_criteria': {
            'occluded_target_found': (
                scenario_a['viewpoint_outcome']['target_found']
                and not scenario_a['target_initially_observable']
                and scenario_a['target_observable_after_viewpoint']
            ),
            'oracle_route_completed_in_order': (
                scenario_b['oracle']['execution']['completed_in_order']
            ),
            'perceived_route_completed_in_order': (
                scenario_b['perceived']['execution']['completed_in_order']
            ),
            'stage_order_enforced': (
                scenario_b['perceived']['execution'][
                    'stage_a_semantically_verified_before_stage_b'
                ]
            ),
        },
    }
    path = arguments.output_directory / 'day11_replay.json'
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload['exit_criteria'], indent=2, sort_keys=True))
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
