"""Tests for the bounded two-stage instruction episode coordinator."""

from fixtures import add_unknown
from fixtures import make_instance
from fixtures import open_grid
import pytest

from qmapnav.exploration import ExplorationBudget
from qmapnav.language import parse_question
from qmapnav.mapping import ObjectMap
from qmapnav.mapping import StructuralMap
from qmapnav.mission import InstructionEpisodeCoordinator
from qmapnav.mission import InstructionEpisodeState
from qmapnav.mission.instruction_episode import StageResolution
from qmapnav.navigation import PerceivedRoutePlan


QUESTION = (
    'Go to the potted plant furthest from the projector screen then stop '
    'at the water cooler near the window.'
)


def _task():
    return parse_question(QUESTION)


def _stub_resolver(mapping):
    """Return a resolver yielding preset instances per entity id."""
    def resolver(reference, object_map, structural_map, **kwargs):
        return StageResolution(
            reference_id=reference.entity_id,
            class_name=reference.class_name,
            instance=mapping.get(reference.entity_id),
            confidence_margin=0.5,
            status=(
                'resolved' if mapping.get(reference.entity_id)
                else 'no_candidates'
            ),
        )
    return resolver


def _entities(task):
    return [step.entity_ids[0] for step in task.ordered_route_steps]


def test_both_stages_resolved_commits_a_route_without_exploring() -> None:
    task = _task()
    first, second = _entities(task)
    coordinator = InstructionEpisodeCoordinator(
        resolver=_stub_resolver({
            first: make_instance(67, 'potted_plant', (4.0, 0.0, 0.5)),
            second: make_instance(23, 'water_cooler', (-4.0, 0.0, 0.5)),
        })
    )
    coordinator.start(task)
    action = coordinator.evaluate(
        ObjectMap(), StructuralMap(),
        grid=open_grid(half_extent=10.0),
        current_pose_xy_yaw=(0.0, -6.0, 0.0),
        time_remaining_sec=500.0,
    )
    assert action.action == 'route'
    assert action.plan.planned is True
    assert len(action.plan.stages) == 2
    assert coordinator.state is InstructionEpisodeState.ROUTE_COMMITTED
    # No motion was spent on evidence the episode did not need.
    assert coordinator.budget.viewpoints_used == 0


def test_unresolved_stage_requests_one_bounded_viewpoint() -> None:
    task = _task()
    first, second = _entities(task)
    object_map = ObjectMap()
    coordinator = InstructionEpisodeCoordinator(
        resolver=_stub_resolver({
            second: make_instance(23, 'water_cooler', (-4.0, 0.0, 0.5)),
        })
    )
    coordinator.start(task)
    action = coordinator.evaluate(
        object_map, StructuralMap(),
        grid=_partly_unknown_grid(),
        current_pose_xy_yaw=(0.0, 0.0, 0.0),
        time_remaining_sec=500.0,
    )
    assert action.action == 'explore'
    assert action.selection.selection_status == 'selected'
    assert action.need.need_type == 'missing_target'
    # The need names the specific unresolved stage entity, not a generic
    # instruction to explore.
    assert 'potted_plant' in action.need.reason
    assert action.need.missing_classes == ('potted_plant',)
    assert coordinator.state is InstructionEpisodeState.VIEWPOINT_ACTIVE
    # The chosen viewpoint faces the unobserved half of the map.
    selected = action.selection.selected
    assert selected.source == 'frontier'
    assert selected.score_terms.unexplored_gain > 0.0


def _partly_unknown_grid():
    grid = open_grid(half_extent=8.0)
    add_unknown(grid, (3.0, -8.0, 8.0, 8.0))
    return grid


def test_exhausted_budget_falls_back_to_the_terminal_target() -> None:
    task = _task()
    first, second = _entities(task)
    coordinator = InstructionEpisodeCoordinator(
        budget=ExplorationBudget(
            max_targeted_viewpoints=1,
            max_single_viewpoint_distance_m=2.0,
            max_total_exploration_distance_m=2.0,
            minimum_time_remaining_sec=400.0,
        ),
        resolver=_stub_resolver({
            second: make_instance(23, 'water_cooler', (-4.0, 0.0, 0.5)),
        }),
    )
    coordinator.start(task)
    action = coordinator.evaluate(
        ObjectMap(), StructuralMap(),
        grid=open_grid(half_extent=10.0),
        # Too little time left to justify any exploration.
        current_pose_xy_yaw=(0.0, -6.0, 0.0),
        time_remaining_sec=100.0,
    )
    assert action.action == 'fallback'
    # The frozen partial-credit policy always attempts the terminal target.
    assert action.plan.route_status == 'terminal_only'
    assert action.plan.executable is True
    assert action.plan.stages[0].resolved_instance_id == '23'
    assert action.plan.unresolved_stages == (0,)


def test_missing_terminal_routes_to_resolved_stage_a_then_reobserves() -> None:
    task = _task()
    first, _ = _entities(task)
    coordinator = InstructionEpisodeCoordinator(
        resolver=_stub_resolver({
            first: make_instance(67, 'potted_plant', (4.0, 0.0, 0.5)),
        })
    )
    coordinator.start(task)

    action = coordinator.evaluate(
        ObjectMap(),
        StructuralMap(),
        grid=open_grid(half_extent=10.0),
        current_pose_xy_yaw=(0.0, -6.0, 0.0),
        time_remaining_sec=500.0,
    )

    assert action.action == 'fallback'
    assert action.plan.route_status == 'stage_a_only'
    assert action.plan.stages[0].resolved_instance_id == '67'
    coordinator.notify_stage_a_information_arrived(
        pose_xy_yaw=(3.0, 0.0, 0.0)
    )
    assert coordinator.state is InstructionEpisodeState.REOBSERVATION

    second = coordinator.evaluate(
        ObjectMap(),
        StructuralMap(),
        grid=open_grid(half_extent=10.0),
        current_pose_xy_yaw=(3.0, 0.0, 0.0),
        time_remaining_sec=450.0,
    )
    assert second.action == 'abort'


def test_blocked_stage_a_uses_safe_stage_a_observation_viewpoint(
    monkeypatch,
) -> None:
    """A blocked strict annulus must not make the live robot stay silent."""
    task = _task()
    first, _ = _entities(task)
    stage_a = make_instance(67, 'potted_plant', (2.0, 0.0, 0.5))
    coordinator = InstructionEpisodeCoordinator(
        resolver=_stub_resolver({first: stage_a})
    )
    coordinator.start(task)

    monkeypatch.setattr(
        'qmapnav.mission.instruction_episode.plan_two_stage_route',
        lambda *args, **kwargs: PerceivedRoutePlan((), 'blocked'),
    )
    action = coordinator.evaluate(
        ObjectMap(),
        StructuralMap(),
        grid=open_grid(half_extent=10.0),
        current_pose_xy_yaw=(0.0, 0.0, 0.0),
        time_remaining_sec=500.0,
    )

    assert action.action == 'explore'
    assert action.selection.selection_status == 'selected'
    assert action.selection.selected.source == 'object_annulus'
    assert action.selection.selected.target_instance_ids == ('67',)
    assert coordinator.state is InstructionEpisodeState.VIEWPOINT_ACTIVE


def test_no_resolvable_stage_aborts_cleanly_without_deadlock() -> None:
    task = _task()
    coordinator = InstructionEpisodeCoordinator(
        budget=ExplorationBudget(minimum_time_remaining_sec=400.0),
        resolver=_stub_resolver({}),
    )
    coordinator.start(task)
    action = coordinator.evaluate(
        ObjectMap(), StructuralMap(),
        grid=open_grid(half_extent=8.0),
        current_pose_xy_yaw=(0.0, 0.0, 0.0),
        time_remaining_sec=50.0,
    )
    assert action.action == 'abort'
    assert action.plan is None
    assert coordinator.state is InstructionEpisodeState.ROUTE_COMMITTED


def test_viewpoint_arrival_consumes_budget_and_opens_reobservation() -> None:
    task = _task()
    first, second = _entities(task)
    coordinator = InstructionEpisodeCoordinator(
        resolver=_stub_resolver({
            second: make_instance(23, 'water_cooler', (-4.0, 0.0, 0.5)),
        })
    )
    coordinator.start(task)
    action = coordinator.evaluate(
        ObjectMap(), StructuralMap(),
        grid=_partly_unknown_grid(),
        current_pose_xy_yaw=(0.0, 0.0, 0.0),
        time_remaining_sec=500.0,
    )
    assert action.action == 'explore'
    coordinator.notify_viewpoint_arrived(
        pose_xy_yaw=(1.0, 0.0, 0.0), distance_m=1.5, duration_sec=4.0
    )
    assert coordinator.state is InstructionEpisodeState.REOBSERVATION
    assert coordinator.budget.viewpoints_used == 1
    assert coordinator.budget.distance_travelled_m == pytest.approx(1.5)


def test_instruction_budget_is_conservative_by_default() -> None:
    coordinator = InstructionEpisodeCoordinator()
    budget = coordinator.budget.budget
    reference = ExplorationBudget.for_task_type('object_reference')
    assert (
        budget.max_total_exploration_distance_m
        < reference.max_total_exploration_distance_m
    )


def test_non_two_stage_instruction_is_refused_at_start() -> None:
    coordinator = InstructionEpisodeCoordinator()
    three_stage = parse_question(
        'First, go near the stool, then take the path near the cabinet, '
        'and stop at the bowl on the table.'
    )
    with pytest.raises(Exception):
        coordinator.start(three_stage)


def test_double_start_and_premature_arrival_are_refused() -> None:
    coordinator = InstructionEpisodeCoordinator()
    task = _task()
    coordinator.start(task)
    with pytest.raises(RuntimeError, match='already started'):
        coordinator.start(task)
    with pytest.raises(RuntimeError, match='no exploration viewpoint'):
        coordinator.notify_viewpoint_arrived(
            pose_xy_yaw=(0.0, 0.0, 0.0), distance_m=1.0, duration_sec=1.0
        )


def test_decision_serializes_for_the_trace() -> None:
    task = _task()
    first, second = _entities(task)
    coordinator = InstructionEpisodeCoordinator(
        resolver=_stub_resolver({
            first: make_instance(67, 'potted_plant', (4.0, 0.0, 0.5)),
            second: make_instance(23, 'water_cooler', (-4.0, 0.0, 0.5)),
        })
    )
    coordinator.start(task)
    action = coordinator.evaluate(
        ObjectMap(), StructuralMap(),
        grid=open_grid(half_extent=10.0),
        current_pose_xy_yaw=(0.0, -6.0, 0.0),
        time_remaining_sec=500.0,
    )
    record = action.to_dict()
    assert record['event'] == 'two_stage_route_decision'
    assert record['action'] == 'route'
    assert len(record['stage_resolutions']) == 2
    assert record['plan']['event'] == 'semantic_route_planned'
    assert coordinator.last_action is action


def test_evaluate_before_start_is_refused() -> None:
    coordinator = InstructionEpisodeCoordinator()
    with pytest.raises(RuntimeError, match='not awaiting evidence'):
        coordinator.evaluate(
            ObjectMap(), StructuralMap(),
            grid=open_grid(half_extent=4.0),
            current_pose_xy_yaw=(0.0, 0.0, 0.0),
            time_remaining_sec=500.0,
        )
