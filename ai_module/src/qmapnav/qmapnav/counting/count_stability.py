"""Temporal, identity-set, viewpoint, and deadline count stability."""

from dataclasses import dataclass
from dataclasses import replace
from enum import Enum
from math import isfinite

from qmapnav.counting.numerical_result import NumericalResult


class CountStabilityStatus(str, Enum):
    """Bounded numerical-answer state-machine states."""

    COLLECTING = 'collecting'
    CANDIDATE_COUNT = 'candidate_count'
    VERIFYING = 'verifying'
    STABLE = 'stable'
    TIME_BUDGET_LOW = 'time_budget_low'
    BEST_AVAILABLE_COUNT = 'best_available_count'
    PUBLISHED = 'published'


@dataclass(frozen=True)
class CountStabilityConfig:
    """Stopping requirements for a numerical episode."""

    required_consecutive_updates: int = 3
    required_independent_viewpoints: int = 2
    minimum_count_confidence: float = 0.50
    maximum_unresolved_candidates: int = 0
    final_commit_reserve_sec: float = 30.0
    maximum_verification_sec: float = 180.0

    def __post_init__(self) -> None:
        for name in (
            'required_consecutive_updates',
            'required_independent_viewpoints',
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if (
            isinstance(self.maximum_unresolved_candidates, bool)
            or self.maximum_unresolved_candidates < 0
        ):
            raise ValueError('maximum_unresolved_candidates must be non-negative')
        if not isfinite(self.minimum_count_confidence) or not (
            0.0 <= self.minimum_count_confidence <= 1.0
        ):
            raise ValueError('minimum_count_confidence must lie in [0, 1]')
        for name in ('final_commit_reserve_sec', 'maximum_verification_sec'):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')


@dataclass(frozen=True)
class CountStabilityState:
    """Immutable snapshot of numerical count verification."""

    status: CountStabilityStatus
    current_count: int | None
    current_instance_ids: tuple[int, ...]
    consecutive_stable_updates: int
    independent_viewpoint_ids: tuple[str, ...]
    unresolved_candidate_count: int
    stable: bool
    should_publish: bool
    reason: str
    result: NumericalResult | None

    @property
    def independent_viewpoints(self) -> int:
        """Return the number of distinct supporting viewpoints."""
        return len(self.independent_viewpoint_ids)


class CountStabilityMachine:
    """Reject integer-only stability and always provide a bounded fallback."""

    def __init__(self, config: CountStabilityConfig | None = None) -> None:
        self.config = config or CountStabilityConfig()
        self._state = CountStabilityState(
            CountStabilityStatus.COLLECTING,
            None,
            (),
            0,
            (),
            0,
            False,
            False,
            'awaiting_first_count',
            None,
        )
        self._verification_started_sec: float | None = None
        self._best_result: NumericalResult | None = None
        self._transition_history = [CountStabilityStatus.COLLECTING]

    @property
    def state(self) -> CountStabilityState:
        """Return the latest immutable state snapshot."""
        return self._state

    @property
    def transition_history(self) -> tuple[CountStabilityStatus, ...]:
        """Return every state entered, including bounded fallback stages."""
        return tuple(self._transition_history)

    def update(
        self,
        result: NumericalResult,
        *,
        viewpoint_id: str,
        time_remaining_sec: float,
        episode_time_sec: float,
        exploration_available: bool = True,
    ) -> CountStabilityState:
        """Fold one map snapshot into count and underlying-ID stability."""
        if self._state.status is CountStabilityStatus.PUBLISHED:
            return self._state
        if not isinstance(result, NumericalResult):
            raise TypeError('result must be NumericalResult')
        if not isinstance(viewpoint_id, str) or not viewpoint_id.strip():
            raise ValueError('viewpoint_id must be a non-empty string')
        for name, value in (
            ('time_remaining_sec', time_remaining_sec),
            ('episode_time_sec', episode_time_sec),
        ):
            if not isfinite(value) or value < 0.0:
                raise ValueError(f'{name} must be finite and non-negative')
        if not isinstance(exploration_available, bool):
            raise TypeError('exploration_available must be boolean')
        if self._verification_started_sec is None:
            self._verification_started_sec = episode_time_sec
        self._remember_best(result)
        same_set = (
            self._state.current_count == result.count
            and self._state.current_instance_ids
            == result.qualifying_instance_ids
        )
        if same_set:
            consecutive = self._state.consecutive_stable_updates + 1
            viewpoints = tuple(sorted(set(
                self._state.independent_viewpoint_ids + (viewpoint_id,)
            )))
            status = CountStabilityStatus.VERIFYING
            reason = 'count_and_persistent_id_set_repeated'
        else:
            consecutive = 1
            viewpoints = (viewpoint_id,)
            status = CountStabilityStatus.CANDIDATE_COUNT
            reason = (
                'first_candidate_count'
                if self._state.current_count is None
                else 'qualifying_id_set_changed_stability_reset'
            )
        unresolved = len(result.unresolved_instance_ids)
        stable = (
            consecutive >= self.config.required_consecutive_updates
            and len(viewpoints) >= self.config.required_independent_viewpoints
            and result.count_confidence >= self.config.minimum_count_confidence
            and unresolved <= self.config.maximum_unresolved_candidates
            and result.anchor_ambiguity.count_consistent
            and not result.hypothesis_limit_reached
        )
        if stable:
            status = CountStabilityStatus.STABLE
            reason = 'count_id_set_and_independent_viewpoints_stable'
            result = replace(result, stable=True, stability_reason=reason)
            self._remember_best(result)
        self._set_state(
            status,
            result,
            consecutive,
            viewpoints,
            unresolved,
            stable,
            stable,
            reason,
        )
        verification_age = (
            episode_time_sec - self._verification_started_sec
        )
        low_time = time_remaining_sec <= self.config.final_commit_reserve_sec
        timed_out = verification_age >= self.config.maximum_verification_sec
        if not stable and (low_time or timed_out or not exploration_available):
            trigger = (
                'time_budget_low' if low_time
                else 'verification_timeout' if timed_out
                else 'exploration_budget_exhausted'
            )
            self._transition(CountStabilityStatus.TIME_BUDGET_LOW)
            return self.force_best_available(trigger)
        return self._state

    def force_best_available(self, reason: str) -> CountStabilityState:
        """Return the strongest observed count instead of waiting forever."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('fallback reason must be non-empty')
        result = self._best_result or self._state.result
        if result is None:
            raise RuntimeError('no numerical result is available to publish')
        result = replace(
            result,
            stable=False,
            stability_reason=f'best_available:{reason}',
        )
        self._set_state(
            CountStabilityStatus.BEST_AVAILABLE_COUNT,
            result,
            self._state.consecutive_stable_updates,
            self._state.independent_viewpoint_ids,
            len(result.unresolved_instance_ids),
            False,
            True,
            f'best_available:{reason}',
        )
        return self._state

    def mark_published(self) -> CountStabilityState:
        """Close the state machine after the one official publication."""
        if self._state.result is None:
            raise RuntimeError('cannot publish without a numerical result')
        self._set_state(
            CountStabilityStatus.PUBLISHED,
            self._state.result,
            self._state.consecutive_stable_updates,
            self._state.independent_viewpoint_ids,
            self._state.unresolved_candidate_count,
            self._state.stable,
            False,
            'official_numerical_response_published',
        )
        return self._state

    def _remember_best(self, result: NumericalResult) -> None:
        if self._best_result is None or _result_key(result) < _result_key(
            self._best_result
        ):
            self._best_result = result

    def _set_state(
        self,
        status,
        result,
        consecutive,
        viewpoints,
        unresolved,
        stable,
        should_publish,
        reason,
    ):
        self._state = CountStabilityState(
            status,
            result.count,
            result.qualifying_instance_ids,
            consecutive,
            tuple(viewpoints),
            unresolved,
            stable,
            should_publish,
            reason,
            result,
        )
        self._transition(status)

    def _transition(self, status):
        if self._transition_history[-1] is not status:
            self._transition_history.append(status)


def _result_key(result):
    return (
        -int(result.stable),
        -int(result.anchor_ambiguity.count_consistent),
        len(result.unresolved_instance_ids),
        int(result.hypothesis_limit_reached),
        -result.count_confidence,
        result.qualifying_instance_ids,
    )


__all__ = [
    'CountStabilityConfig',
    'CountStabilityMachine',
    'CountStabilityState',
    'CountStabilityStatus',
]
