"""
Bounded Day 11 two-stage instruction episode coordinator.

Mirrors the shape of
:class:`~qmapnav.mission.object_reference_episode.ObjectReferenceEpisodeCoordinator`:
constructor takes policy plus an injectable resolver, ``evaluate_*`` methods
return actions, and nothing here imports ROS. The composition root owns
transport; this class owns only the decision of whether to route, explore, or
fall back.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from qmapnav.common import ObjectInstance
from qmapnav.common import TaskSpecification
from qmapnav.exploration.exploration_budget import ExplorationBudget
from qmapnav.exploration.exploration_budget import ExplorationBudgetTracker
from qmapnav.exploration.exploration_need import ExplorationNeed
from qmapnav.exploration.small_object_mode import decide_small_object_mode
from qmapnav.exploration.small_object_mode import is_support_surface
from qmapnav.exploration.small_object_mode import likely_supports
from qmapnav.exploration.support_surface_search import (
    generate_support_surface_viewpoints,
)
from qmapnav.exploration.support_surface_search import rank_support_surfaces
from qmapnav.exploration.support_surface_search import SupportSearchHistory
from qmapnav.exploration.viewpoint_candidate import ViewpointSelection
from qmapnav.exploration.viewpoint_generation import (
    CandidateGenerationOutcome,
)
from qmapnav.exploration.viewpoint_generation import (
    generate_frontier_viewpoints,
)
from qmapnav.exploration.viewpoint_generation import (
    generate_object_annulus_viewpoints,
)
from qmapnav.exploration.viewpoint_generation import (
    ViewpointGenerationConfig,
)
from qmapnav.exploration.viewpoint_generation import VisitedViewpoint
from qmapnav.exploration.viewpoint_scoring import score_candidate
from qmapnav.exploration.viewpoint_scoring import select_viewpoint
from qmapnav.exploration.viewpoint_scoring import ViewpointScoringConfig
from qmapnav.mapping.grid_planning import cost_field
from qmapnav.mapping.object_map import ObjectMap
from qmapnav.mapping.occupancy_grid import OccupancyGrid2D
from qmapnav.mapping.perceived_geometry import perceived_box
from qmapnav.mapping.structural_map import StructuralMap
from qmapnav.navigation.semantic_regions import NearRegionConfig
from qmapnav.navigation.two_stage_route import PerceivedRoutePlan
from qmapnav.navigation.two_stage_route import plan_two_stage_route
from qmapnav.navigation.two_stage_route import two_stage_steps
from qmapnav.reasoning.ambiguity import AmbiguityConfig
from qmapnav.reasoning.candidate_generation import CandidateGenerationConfig
from qmapnav.reasoning.candidate_generation import (
    generate_candidates_from_maps,
)
from qmapnav.reasoning.object_reference_solver import (
    resolve_object_reference_from_maps,
)
from qmapnav.reasoning.reference_resolver import resolve_single_reference
from qmapnav.reasoning.spatial_relations import SpatialRelationConfig
from qmapnav.reasoning.support_relations import SupportRelationConfig
from qmapnav.reasoning.vertical_relations import VerticalRelationConfig


class TwoStageRouteEpisodeState(str, Enum):
    """Bounded states for one perceived two-stage instruction."""

    IDLE = 'idle'
    INITIAL_OBSERVATION = 'initial_observation'
    VIEWPOINT_ACTIVE = 'viewpoint_active'
    REOBSERVATION = 'reobservation'
    ROUTE_COMMITTED = 'route_committed'


@dataclass(frozen=True)
class StageResolution:
    """The grounding outcome for one route stage."""

    reference_id: str
    class_name: str
    instance: ObjectInstance | None
    confidence_margin: float
    status: str

    @property
    def resolved(self) -> bool:
        """Return whether a usable instance was selected."""
        return self.instance is not None

    def to_dict(self) -> dict[str, object]:
        """Return a stable trace-ready mapping."""
        return {
            'reference_id': self.reference_id,
            'class_name': self.class_name,
            'instance_id': (
                None if self.instance is None
                else str(self.instance.instance_id)
            ),
            'confidence_margin': self.confidence_margin,
            'status': self.status,
        }


@dataclass(frozen=True)
class TwoStageRouteAction:
    """One coordinator output consumed by the composition root."""

    action: str
    reason: str
    stage_resolutions: tuple[StageResolution, ...]
    plan: PerceivedRoutePlan | None = None
    selection: ViewpointSelection | None = None
    need: ExplorationNeed | None = None

    def __post_init__(self) -> None:
        if self.action not in {'route', 'explore', 'fallback', 'abort'}:
            raise ValueError('unsupported two-stage route action')
        if self.action in {'route', 'fallback'} and self.plan is None:
            raise ValueError('routing actions require a plan')
        if self.action == 'explore' and self.selection is None:
            raise ValueError('explore action requires a viewpoint selection')

    def to_dict(self) -> dict[str, object]:
        """Return a stable trace-ready mapping."""
        return {
            'event': 'two_stage_route_decision',
            'action': self.action,
            'reason': self.reason,
            'stage_resolutions': [
                item.to_dict() for item in self.stage_resolutions
            ],
            'plan': None if self.plan is None else self.plan.to_dict(),
            'selection': (
                None if self.selection is None else self.selection.to_dict()
            ),
        }


def resolve_stage_reference(
    reference,
    object_map: ObjectMap,
    structural_map: StructuralMap,
    *,
    task: TaskSpecification | None = None,
    candidate_config: CandidateGenerationConfig | None = None,
    spatial_config: SpatialRelationConfig | None = None,
    vertical_config: VerticalRelationConfig | None = None,
    support_config: SupportRelationConfig | None = None,
    ambiguity_config: AmbiguityConfig | None = None,
) -> StageResolution:
    """Ground one stage entity through the complete perceived resolver."""
    if task is not None:
        stage_task = _stage_reference_task(task, reference.entity_id)
        resolution = resolve_object_reference_from_maps(
            stage_task,
            object_map,
            structural_map,
            candidate_config=candidate_config,
            spatial_config=spatial_config,
            vertical_config=vertical_config,
            support_config=support_config,
            ambiguity_config=ambiguity_config,
        )
        instance = None
        selected = resolution.selected_target_id
        if selected is not None and selected.isdigit():
            try:
                instance = object_map.get(int(selected))
            except (KeyError, TypeError, ValueError):
                instance = None
        return StageResolution(
            reference_id=reference.entity_id,
            class_name=reference.class_name,
            instance=instance,
            confidence_margin=resolution.normalized_margin,
            status=resolution.resolution_status,
        )

    generated = generate_candidates_from_maps(
        reference, object_map, structural_map, candidate_config
    )
    assessment = resolve_single_reference(
        reference, generated, candidate_config=candidate_config
    )
    resolution = assessment.resolution
    selected = resolution.selected_candidate_ids
    instance = None
    if selected:
        try:
            instance = object_map.get(int(selected[0]))
        except (KeyError, TypeError, ValueError):
            instance = None
    if instance is None and generated.retained:
        # Bounded fallback: the strongest class match still beats silence.
        for candidate in generated.retained:
            if candidate.source_type != 'object':
                continue
            try:
                instance = object_map.get(int(candidate.candidate_id))
            except (KeyError, TypeError, ValueError):
                continue
            break
    return StageResolution(
        reference_id=reference.entity_id,
        class_name=reference.class_name,
        instance=instance,
        confidence_margin=assessment.normalized_margin,
        status=resolution.resolution_status,
    )


class TwoStageRouteEpisodeCoordinator:
    """Resolve both stages, explore once if needed, then commit a route."""

    def __init__(
        self,
        *,
        budget: ExplorationBudget | None = None,
        region_config: NearRegionConfig | None = None,
        generation_config: ViewpointGenerationConfig | None = None,
        scoring_config: ViewpointScoringConfig | None = None,
        candidate_config: CandidateGenerationConfig | None = None,
        spatial_config: SpatialRelationConfig | None = None,
        vertical_config: VerticalRelationConfig | None = None,
        support_config: SupportRelationConfig | None = None,
        ambiguity_config: AmbiguityConfig | None = None,
        resolver: Callable[..., StageResolution] = resolve_stage_reference,
    ) -> None:
        """Create an idle coordinator for one instruction episode."""
        self._budget = ExplorationBudgetTracker(
            budget or ExplorationBudget.for_task_type('instruction_following')
        )
        self._region_config = region_config or NearRegionConfig()
        self._generation_config = (
            generation_config or ViewpointGenerationConfig()
        )
        self._scoring_config = scoring_config or ViewpointScoringConfig()
        self._candidate_config = candidate_config
        self._spatial_config = spatial_config
        self._vertical_config = vertical_config
        self._support_config = support_config
        self._ambiguity_config = ambiguity_config
        self._resolver = resolver
        self._state = TwoStageRouteEpisodeState.IDLE
        self._task: TaskSpecification | None = None
        self._steps: tuple[tuple[int, str, str], ...] = ()
        self._visited: tuple[VisitedViewpoint, ...] = ()
        self._support_history = SupportSearchHistory()
        self._last_action: TwoStageRouteAction | None = None

    @property
    def state(self) -> TwoStageRouteEpisodeState:
        """Return the current bounded episode state."""
        return self._state

    @property
    def budget(self) -> ExplorationBudgetTracker:
        """Return the exploration budget tracker for this episode."""
        return self._budget

    @property
    def support_history(self) -> SupportSearchHistory:
        """Return the episode's support-surface search memory."""
        return self._support_history

    @property
    def last_action(self) -> TwoStageRouteAction | None:
        """Return the most recent decision for tracing."""
        return self._last_action

    def start(self, task: TaskSpecification) -> None:
        """Start one instruction episode from a latched parse."""
        if self._state is not TwoStageRouteEpisodeState.IDLE:
            raise RuntimeError('two-stage episode already started')
        if not isinstance(task, TaskSpecification):
            raise TypeError('task must be TaskSpecification')
        self._steps = two_stage_steps(task)
        self._task = task
        self._state = TwoStageRouteEpisodeState.INITIAL_OBSERVATION

    def evaluate(
        self,
        object_map: ObjectMap,
        structural_map: StructuralMap,
        *,
        grid: OccupancyGrid2D,
        current_pose_xy_yaw: tuple[float, float, float],
        time_remaining_sec: float,
    ) -> TwoStageRouteAction:
        """
        Ground both stages and choose to route, explore, or fall back.

        Exploration is only proposed when a stage is genuinely unresolved and
        the budget still allows it; otherwise the episode commits to the best
        route it can build so the terminal target is always attempted.
        """
        if self._state not in {
            TwoStageRouteEpisodeState.INITIAL_OBSERVATION,
            TwoStageRouteEpisodeState.REOBSERVATION,
        }:
            raise RuntimeError('coordinator is not awaiting evidence')
        if self._task is None:
            raise RuntimeError('two-stage task is unavailable')

        entities = {item.entity_id: item for item in self._task.entities}
        resolutions = tuple(
            self._resolver(
                entities[entity_id],
                object_map,
                structural_map,
                task=self._task,
                candidate_config=self._candidate_config,
                spatial_config=self._spatial_config,
                vertical_config=self._vertical_config,
                support_config=self._support_config,
                ambiguity_config=self._ambiguity_config,
            )
            for _, _, entity_id in self._steps
        )
        resolved = {
            item.reference_id: item.instance
            for item in resolutions
            if item.instance is not None
        }
        margins = {
            item.reference_id: max(0.0, min(1.0, item.confidence_margin))
            for item in resolutions
        }
        start_xy = (current_pose_xy_yaw[0], current_pose_xy_yaw[1])

        if len(resolved) == len(self._steps):
            plan = plan_two_stage_route(
                self._task,
                resolved,
                grid=grid,
                start_xy=start_xy,
                confidences=margins,
                region_config=self._region_config,
            )
            if plan.executable:
                return self._commit(
                    'route', 'both_stages_resolved', resolutions, plan
                )
            return self._explore_or_fallback(
                resolutions,
                resolved,
                margins,
                object_map=object_map,
                grid=grid,
                current_pose_xy_yaw=current_pose_xy_yaw,
                time_remaining_sec=time_remaining_sec,
                reason=f'route_{plan.route_status}',
            )

        return self._explore_or_fallback(
            resolutions,
            resolved,
            margins,
            object_map=object_map,
            grid=grid,
            current_pose_xy_yaw=current_pose_xy_yaw,
            time_remaining_sec=time_remaining_sec,
            reason='stage_unresolved',
        )

    def notify_viewpoint_arrived(
        self,
        *,
        pose_xy_yaw: tuple[float, float, float],
        distance_m: float,
        duration_sec: float,
        focus_key: str = '',
    ) -> None:
        """Consume budget and open one bounded re-observation window."""
        if self._state is not TwoStageRouteEpisodeState.VIEWPOINT_ACTIVE:
            raise RuntimeError('no exploration viewpoint is active')
        self._budget.consume(
            distance_m=distance_m, duration_sec=duration_sec
        )
        self._visited = self._visited + (
            VisitedViewpoint(tuple(pose_xy_yaw), focus_key=focus_key),
        )
        self._state = TwoStageRouteEpisodeState.REOBSERVATION

    def notify_viewpoint_failed(
        self,
        *,
        distance_m: float,
        duration_sec: float,
    ) -> None:
        """Consume a failed viewpoint attempt and open bounded fallback."""
        if self._state is not TwoStageRouteEpisodeState.VIEWPOINT_ACTIVE:
            raise RuntimeError('no exploration viewpoint is active')
        self._budget.consume(
            distance_m=distance_m, duration_sec=duration_sec
        )
        self._state = TwoStageRouteEpisodeState.REOBSERVATION

    def _explore_or_fallback(
        self,
        resolutions,
        resolved,
        margins,
        *,
        object_map,
        grid,
        current_pose_xy_yaw,
        time_remaining_sec,
        reason,
    ) -> TwoStageRouteAction:
        missing = [item for item in resolutions if item.instance is None]
        need = self._build_need(missing, reason)
        status = self._budget.status(time_remaining_sec)
        if status == 'available' and missing:
            outcome = self._generate_candidates(
                missing[0],
                object_map=object_map,
                grid=grid,
                current_pose_xy_yaw=current_pose_xy_yaw,
            )
            scored = tuple(
                score_candidate(
                    candidate,
                    grid=grid,
                    need=need,
                    config=self._scoring_config,
                    support_xy=_first_support_xy(candidate, object_map),
                )
                for candidate in outcome.candidates
            )
            selection = select_viewpoint(
                scored, need, config=self._scoring_config
            )
            if selection.selection_status == 'selected':
                self._state = TwoStageRouteEpisodeState.VIEWPOINT_ACTIVE
                action = TwoStageRouteAction(
                    'explore', need.reason, resolutions,
                    selection=selection, need=need,
                )
                self._last_action = action
                return action
        else:
            selection = None

        plan = plan_two_stage_route(
            self._task,
            resolved,
            grid=grid,
            start_xy=(current_pose_xy_yaw[0], current_pose_xy_yaw[1]),
            confidences=margins,
            region_config=self._region_config,
            allow_terminal_only=True,
        )
        if plan.executable:
            return self._commit(
                'fallback',
                f'{reason}:terminal_target_attempted',
                resolutions,
                plan,
            )
        self._state = TwoStageRouteEpisodeState.ROUTE_COMMITTED
        action = TwoStageRouteAction(
            'abort', f'{reason}:{plan.route_status}', resolutions, need=need
        )
        self._last_action = action
        return action

    def _generate_candidates(
        self,
        missing: StageResolution,
        *,
        object_map,
        grid,
        current_pose_xy_yaw,
    ) -> CandidateGenerationOutcome:
        costs = cost_field(
            grid,
            (current_pose_xy_yaw[0], current_pose_xy_yaw[1]),
            clearance=self._generation_config.robot_clearance_m,
        )
        travel_limit = min(
            self._budget.budget.max_single_viewpoint_distance_m,
            self._budget.remaining_distance_m,
        )
        outcome = CandidateGenerationOutcome()

        mode = decide_small_object_mode(missing.class_name)
        if mode.active and likely_supports(missing.class_name):
            supports = tuple(
                perceived_box(instance)
                for instance in object_map.active_instances()
                if is_support_surface(_dominant_class(instance))
            )
            ranked = rank_support_surfaces(
                missing.class_name,
                supports,
                current_pose_xy_yaw=current_pose_xy_yaw,
                history=self._support_history,
            )
            for support in ranked[:3]:
                outcome = outcome.merge(
                    generate_support_surface_viewpoints(
                        support,
                        grid=grid,
                        current_pose_xy_yaw=current_pose_xy_yaw,
                        config=self._generation_config,
                        visited=self._visited,
                        max_travel_m=travel_limit,
                        costs=costs,
                    )
                )

        for instance in object_map.active_instances(missing.class_name):
            outcome = outcome.merge(
                generate_object_annulus_viewpoints(
                    (
                        float(instance.centroid_xyz[0]),
                        float(instance.centroid_xyz[1]),
                    ),
                    grid=grid,
                    current_pose_xy_yaw=current_pose_xy_yaw,
                    prefix=f'stage_{missing.reference_id}',
                    config=self._generation_config,
                    visited=self._visited,
                    max_travel_m=travel_limit,
                    target_instance_ids=(str(instance.instance_id),),
                    costs=costs,
                )
            )

        if not outcome.candidates:
            outcome = outcome.merge(
                generate_frontier_viewpoints(
                    grid=grid,
                    current_pose_xy_yaw=current_pose_xy_yaw,
                    config=self._generation_config,
                    visited=self._visited,
                    max_travel_m=travel_limit,
                    costs=costs,
                )
            )
        return outcome

    def _build_need(
        self,
        missing: list[StageResolution],
        reason: str,
    ) -> ExplorationNeed:
        if not missing:
            return ExplorationNeed(
                need_type='weak_geometry',
                reason=f'route could not be built: {reason}',
            )
        first = missing[0]
        small = decide_small_object_mode(first.class_name)
        need_type = (
            'small_object_search'
            if small.active and likely_supports(first.class_name)
            else 'missing_target'
        )
        return ExplorationNeed(
            need_type=need_type,
            target_reference_id=first.reference_id,
            missing_classes=(first.class_name,),
            unresolved_constraints=(f'{first.reference_id}_unresolved',),
            ambiguity_score=1.0,
            urgency=0.8,
            expected_task_value=6.0,
            reason=(
                f'stage entity {first.reference_id} '
                f'({first.class_name}) is unresolved: {reason}'
            ),
        )

    def _commit(self, action, reason, resolutions, plan):
        self._state = TwoStageRouteEpisodeState.ROUTE_COMMITTED
        decision = TwoStageRouteAction(
            action, reason, resolutions, plan=plan
        )
        self._last_action = decision
        return decision


def _dominant_class(instance: ObjectInstance) -> str:
    return max(
        instance.class_scores.items(), key=lambda item: (item[1], item[0])
    )[0]


def _stage_reference_task(
    task: TaskSpecification,
    target_reference_id: str,
) -> TaskSpecification:
    """Build the relation-connected object-reference subtask for one stage."""
    entities_by_id = {item.entity_id: item for item in task.entities}
    if target_reference_id not in entities_by_id:
        raise ValueError('stage target is not present in the task')

    connected = {target_reference_id}
    included_relations = []
    changed = True
    while changed:
        changed = False
        for relation in task.relations:
            participants = {
                relation.subject_entity_id,
                *relation.anchor_entity_ids,
            }
            if not participants.intersection(connected):
                continue
            if relation not in included_relations:
                included_relations.append(relation)
            before = len(connected)
            connected.update(participants)
            changed = changed or len(connected) != before

    target = entities_by_id[target_reference_id]
    entities = [target]
    entities.extend(
        item for item in task.entities
        if item.entity_id in connected and item.entity_id != target_reference_id
    )
    return TaskSpecification(
        task_type='object_reference',
        entities=entities,
        relations=included_relations,
        ordered_route_steps=[],
        forbidden_constraints=[],
        terminal_target=None,
        parse_confidence=task.parse_confidence,
        parse_mode=task.parse_mode,
    )


def _first_support_xy(candidate, object_map):
    if candidate.source != 'support_surface':
        return None
    for identifier in candidate.target_instance_ids:
        try:
            instance = object_map.get(int(identifier))
        except (KeyError, TypeError, ValueError):
            continue
        return (
            float(instance.centroid_xyz[0]),
            float(instance.centroid_xyz[1]),
        )
    return None


__all__ = [
    'StageResolution',
    'TwoStageRouteAction',
    'TwoStageRouteEpisodeCoordinator',
    'TwoStageRouteEpisodeState',
    'resolve_stage_reference',
]
