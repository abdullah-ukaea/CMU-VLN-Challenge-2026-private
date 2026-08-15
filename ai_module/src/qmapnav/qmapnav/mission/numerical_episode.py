"""ROS-independent numerical episode coordinator with bounded commitment."""

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from qmapnav.common import TaskSpecification
from qmapnav.counting.count_stability import CountStabilityConfig
from qmapnav.counting.count_stability import CountStabilityMachine
from qmapnav.counting.count_stability import CountStabilityState
from qmapnav.counting.numerical_result import NumericalResult
from qmapnav.counting.numerical_solver import resolve_numerical_from_maps
from qmapnav.counting.support_counting import assess_counting_supports
from qmapnav.counting.support_counting import CountingSupportAssessment
from qmapnav.counting.support_counting import strengthen_zero_with_support_evidence
from qmapnav.exploration.support_surface_search import SupportSearchHistory
from qmapnav.mapping.object_map import ObjectMap
from qmapnav.mapping.structural_map import StructuralMap


class NumericalEpisodeState(str, Enum):
    """Small bounded state space separate from ROS transport."""

    IDLE = 'idle'
    COLLECTING = 'collecting'
    COMMITTED = 'committed'


@dataclass(frozen=True)
class NumericalEpisodeAction:
    """One coordinator decision consumed by the composition root."""

    action: str
    reason: str
    result: NumericalResult
    stability: CountStabilityState
    support_assessment: CountingSupportAssessment

    def __post_init__(self) -> None:
        if self.action not in {'observe', 'commit'}:
            raise ValueError('unsupported numerical episode action')

    def to_dict(self) -> dict[str, object]:
        """Return complete JSON-safe count-decision evidence."""
        return {
            'event': 'numerical_episode_decision',
            'action': self.action,
            'reason': self.reason,
            'result': self.result.to_dict(),
            'stability': {
                'status': self.stability.status.value,
                'current_count': self.stability.current_count,
                'current_instance_ids': list(
                    self.stability.current_instance_ids
                ),
                'consecutive_stable_updates': (
                    self.stability.consecutive_stable_updates
                ),
                'independent_viewpoints': (
                    self.stability.independent_viewpoints
                ),
                'unresolved_candidate_count': (
                    self.stability.unresolved_candidate_count
                ),
                'stable': self.stability.stable,
                'should_publish': self.stability.should_publish,
            },
            'support_assessment': self.support_assessment.to_dict(),
        }


class NumericalEpisodeCoordinator:
    """Resolve map snapshots until stable, then force a deadline fallback."""

    def __init__(
        self,
        *,
        stability_config: CountStabilityConfig | None = None,
        resolver: Callable[..., NumericalResult] = resolve_numerical_from_maps,
    ) -> None:
        if not callable(resolver):
            raise TypeError('resolver must be callable')
        self._resolver = resolver
        self._stability = CountStabilityMachine(stability_config)
        self._support_history = SupportSearchHistory()
        self._state = NumericalEpisodeState.IDLE
        self._task: TaskSpecification | None = None
        self._last_action: NumericalEpisodeAction | None = None

    @property
    def state(self) -> NumericalEpisodeState:
        """Return the current numerical episode state."""
        return self._state

    @property
    def stability(self) -> CountStabilityMachine:
        """Expose count-verification evidence for tracing and tests."""
        return self._stability

    @property
    def support_history(self) -> SupportSearchHistory:
        """Expose target-specific Day 11 support-search memory."""
        return self._support_history

    @property
    def last_action(self) -> NumericalEpisodeAction | None:
        """Return the latest numerical decision."""
        return self._last_action

    def start(self, task: TaskSpecification) -> None:
        """Start one numerical episode from a latched task."""
        if self._state is not NumericalEpisodeState.IDLE:
            raise RuntimeError('numerical episode already started')
        if not isinstance(task, TaskSpecification):
            raise TypeError('task must be TaskSpecification')
        if task.task_type != 'numerical' or not task.entities:
            raise ValueError('coordinator requires a numerical task')
        self._task = task
        self._state = NumericalEpisodeState.COLLECTING

    def evaluate(
        self,
        object_map: ObjectMap,
        structural_map: StructuralMap,
        *,
        viewpoint_id: str,
        time_remaining_sec: float,
        episode_time_sec: float,
        exploration_available: bool = True,
    ) -> NumericalEpisodeAction:
        """Resolve one map snapshot and decide whether to observe or commit."""
        if self._state is not NumericalEpisodeState.COLLECTING:
            raise RuntimeError('numerical episode is not collecting evidence')
        result, support = self._resolve(object_map, structural_map)
        stability = self._stability.update(
            result,
            viewpoint_id=viewpoint_id,
            time_remaining_sec=time_remaining_sec,
            episode_time_sec=episode_time_sec,
            exploration_available=exploration_available,
        )
        action = 'commit' if stability.should_publish else 'observe'
        if action == 'commit':
            self._state = NumericalEpisodeState.COMMITTED
        self._last_action = NumericalEpisodeAction(
            action,
            stability.reason,
            stability.result,
            stability,
            support,
        )
        return self._last_action

    def force_commit(
        self,
        object_map: ObjectMap,
        structural_map: StructuralMap,
        *,
        reason: str,
    ) -> NumericalEpisodeAction:
        """Commit the best current count before the episode watchdog."""
        if self._state is NumericalEpisodeState.COMMITTED:
            if self._last_action is None:
                raise RuntimeError('committed numerical action is unavailable')
            return self._last_action
        if self._state is not NumericalEpisodeState.COLLECTING:
            raise RuntimeError('numerical episode has not started')
        result, support = self._resolve(object_map, structural_map)
        if self._stability.state.result is None:
            self._stability.update(
                result,
                viewpoint_id='deadline_snapshot',
                time_remaining_sec=0.0,
                episode_time_sec=0.0,
            )
        stability = self._stability.force_best_available(reason)
        self._state = NumericalEpisodeState.COMMITTED
        self._last_action = NumericalEpisodeAction(
            'commit', stability.reason, stability.result, stability, support
        )
        return self._last_action

    def notify_published(self) -> None:
        """Record that the official transport adapter completed once."""
        if self._state is not NumericalEpisodeState.COMMITTED:
            raise RuntimeError('numerical result has not been committed')
        self._stability.mark_published()

    def _resolve(self, object_map, structural_map):
        if self._task is None:
            raise RuntimeError('numerical task is unavailable')
        result = self._resolver(self._task, object_map, structural_map)
        support = assess_counting_supports(
            self._task.entities[0].class_name,
            object_map,
            self._support_history,
        )
        result = strengthen_zero_with_support_evidence(result, support)
        return result, support


__all__ = [
    'NumericalEpisodeAction',
    'NumericalEpisodeCoordinator',
    'NumericalEpisodeState',
]
