"""Bounded object-reference object-reference episode decision coordinator."""

from dataclasses import dataclass
from enum import Enum
from math import cos
from math import sin
from typing import Callable

from qmapnav.common import ObjectInstance
from qmapnav.common import TaskSpecification
from qmapnav.mapping.object_map import ObjectMap
from qmapnav.mapping.structural_map import StructuralMap
from qmapnav.navigation import decide_targeted_viewpoint
from qmapnav.navigation import EvidenceSufficiency
from qmapnav.navigation import generate_targeted_viewpoints
from qmapnav.navigation import OneViewpointGuard
from qmapnav.navigation import TargetedViewpointCandidate
from qmapnav.navigation import TargetedViewpointConfig
from qmapnav.reasoning.object_reference_solver import (
    PerceivedObjectReferenceResolution,
)
from qmapnav.reasoning.object_reference_solver import (
    resolve_object_reference_from_maps,
)


class ObjectReferenceEpisodeState(str, Enum):
    """Small object-answer state machine separate from ROS transport."""

    IDLE = 'idle'
    INITIAL_OBSERVATION = 'initial_observation'
    VIEWPOINT_ACTIVE = 'viewpoint_active'
    REOBSERVATION = 'reobservation'
    COMMITTED = 'committed'


@dataclass(frozen=True)
class ObjectReferenceAction:
    """One coordinator output consumed by the ROS composition root."""

    action: str
    reason: str
    resolution: PerceivedObjectReferenceResolution
    selected_instance: ObjectInstance | None = None
    viewpoint: TargetedViewpointCandidate | None = None

    def __post_init__(self) -> None:
        if self.action not in {'commit', 'no_candidate', 'viewpoint'}:
            raise ValueError('unsupported object-reference action')
        if self.action == 'viewpoint' and self.viewpoint is None:
            raise ValueError('viewpoint action requires a candidate')
        if self.action == 'commit' and self.selected_instance is None:
            raise ValueError('commit action requires a selected instance')


class ObjectReferenceEpisodeCoordinator:
    """Resolve once, optionally reobserve once, then commit unconditionally."""

    def __init__(
        self,
        *,
        viewpoint_config: TargetedViewpointConfig | None = None,
        resolver: Callable[..., PerceivedObjectReferenceResolution] = (
            resolve_object_reference_from_maps
        ),
    ) -> None:
        self._viewpoint_config = viewpoint_config or TargetedViewpointConfig()
        self._resolver = resolver
        self._state = ObjectReferenceEpisodeState.IDLE
        self._task: TaskSpecification | None = None
        self._viewpoint_guard = OneViewpointGuard()
        self._initial_resolution: (
            PerceivedObjectReferenceResolution | None
        ) = None
        self._final_resolution: PerceivedObjectReferenceResolution | None = None

    @property
    def state(self) -> ObjectReferenceEpisodeState:
        """Return the current bounded episode state."""
        return self._state

    @property
    def viewpoint_attempted(self) -> bool:
        """Return whether the one special re-observation was consumed."""
        return self._viewpoint_guard.attempted

    @property
    def initial_resolution(
        self,
    ) -> PerceivedObjectReferenceResolution | None:
        """Return initial ranking evidence for before/after reports."""
        return self._initial_resolution

    @property
    def final_resolution(
        self,
    ) -> PerceivedObjectReferenceResolution | None:
        """Return the ranking that caused final commitment."""
        return self._final_resolution

    def start(self, task: TaskSpecification) -> None:
        """Start one object-reference episode from a latched parse."""
        if self._state is not ObjectReferenceEpisodeState.IDLE:
            raise RuntimeError('object-reference episode already started')
        if not isinstance(task, TaskSpecification):
            raise TypeError('task must be TaskSpecification')
        if task.task_type != 'object_reference':
            raise ValueError('coordinator requires object_reference task')
        self._task = task
        self._state = ObjectReferenceEpisodeState.INITIAL_OBSERVATION

    def evaluate_initial(
        self,
        object_map: ObjectMap,
        structural_map: StructuralMap,
        *,
        current_pose_xy_heading: tuple[float, float, float],
        time_remaining_sec: float,
        safe_pose,
    ) -> ObjectReferenceAction:
        """Resolve initial evidence and choose viewpoint or commitment."""
        if self._state is not ObjectReferenceEpisodeState.INITIAL_OBSERVATION:
            raise RuntimeError('initial evidence may be evaluated exactly once')
        resolution = self._resolve(object_map, structural_map)
        self._initial_resolution = resolution
        evidence = self._sufficiency(resolution, object_map, time_remaining_sec)
        decision = decide_targeted_viewpoint(
            evidence, self._viewpoint_config
        )
        if decision.requested:
            focus = self._focus_xy(
                resolution, object_map, current_pose_xy_heading
            )
            candidates = generate_targeted_viewpoints(
                current_pose_xy_heading,
                focus,
                anchor_missing=bool(evidence.missing_anchor_ids),
                ambiguous=(decision.reason == 'low_ranking_margin'),
                safe_pose=safe_pose,
                config=self._viewpoint_config,
            )
            selected = self._viewpoint_guard.select(candidates)
            if selected is not None:
                self._state = ObjectReferenceEpisodeState.VIEWPOINT_ACTIVE
                return ObjectReferenceAction(
                    'viewpoint', decision.reason, resolution,
                    viewpoint=selected,
                )
            return self._commit(
                resolution, object_map, 'no_safe_targeted_viewpoint'
            )
        return self._commit(resolution, object_map, decision.reason)

    def notify_viewpoint_arrived(self) -> None:
        """Open exactly one bounded re-observation window after arrival."""
        if self._state is not ObjectReferenceEpisodeState.VIEWPOINT_ACTIVE:
            raise RuntimeError('no targeted viewpoint is active')
        self._state = ObjectReferenceEpisodeState.REOBSERVATION

    def evaluate_after_reobservation(
        self,
        object_map: ObjectMap,
        structural_map: StructuralMap,
    ) -> ObjectReferenceAction:
        """Rerank once after the one viewpoint and then commit."""
        if self._state is not ObjectReferenceEpisodeState.REOBSERVATION:
            raise RuntimeError('re-observation may be evaluated exactly once')
        resolution = self._resolve(object_map, structural_map)
        return self._commit(
            resolution, object_map, 'single_reobservation_complete'
        )

    def force_commit(
        self,
        object_map: ObjectMap,
        structural_map: StructuralMap,
        reason: str,
    ) -> ObjectReferenceAction:
        """Commit best current evidence before a deadline or route failure."""
        if self._state is ObjectReferenceEpisodeState.COMMITTED:
            raise RuntimeError('object-reference answer already committed')
        resolution = self._resolve(object_map, structural_map)
        return self._commit(resolution, object_map, reason)

    def _resolve(self, object_map, structural_map):
        if self._task is None:
            raise RuntimeError('object-reference task is unavailable')
        return self._resolver(self._task, object_map, structural_map)

    def _sufficiency(self, resolution, object_map, time_remaining):
        target_generated = resolution.candidate_generation[
            resolution.target_reference_id
        ]
        target_count = sum(
            item.retained and item.source_type == 'object'
            for item in target_generated.candidates
        )
        missing = tuple(sorted(
            reference_id
            for reference_id, generated in resolution.candidate_generation.items()
            if (
                reference_id != resolution.target_reference_id
                and not generated.retained
            )
        ))
        geometry = None
        occluded = False
        instance = _selected_instance(resolution, object_map)
        if instance is not None:
            try:
                record = object_map.record(instance.instance_id)
            except KeyError:
                geometry = instance.confidence
            else:
                geometry = record.geometry_confidence
                occluded = record.status in {
                    'partially_observed', 'sparse', 'uncertain'
                }
        return EvidenceSufficiency(
            target_count,
            missing,
            resolution.normalized_margin,
            geometry,
            occluded,
            time_remaining,
            int(self._viewpoint_guard.attempted),
        )

    def _focus_xy(self, resolution, object_map, current_pose):
        instance = _selected_instance(resolution, object_map)
        if instance is not None:
            return tuple(float(item) for item in instance.centroid_xyz[:2])
        x, y, heading = current_pose
        distance = self._viewpoint_config.preferred_standoff_m
        return x + distance * cos(heading), y + distance * sin(heading)

    def _commit(self, resolution, object_map, reason):
        self._final_resolution = resolution
        self._state = ObjectReferenceEpisodeState.COMMITTED
        instance = _selected_instance(resolution, object_map)
        if instance is None:
            return ObjectReferenceAction(
                'no_candidate', reason, resolution
            )
        return ObjectReferenceAction(
            'commit', reason, resolution, selected_instance=instance
        )


def _selected_instance(resolution, object_map):
    identifier = resolution.selected_target_id
    if identifier is None:
        return None
    try:
        instance_id = int(identifier)
    except (TypeError, ValueError):
        return None
    try:
        return object_map.get(instance_id)
    except KeyError:
        return None


__all__ = [
    'ObjectReferenceAction',
    'ObjectReferenceEpisodeCoordinator',
    'ObjectReferenceEpisodeState',
]
