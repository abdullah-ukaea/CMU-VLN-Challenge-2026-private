"""Bounded exploration budgets and their consumption tracker."""

from dataclasses import dataclass
from math import isfinite


BUDGET_STATUSES = frozenset(
    {
        'available',
        'distance_exhausted',
        'time_budget_exhausted',
        'viewpoints_exhausted',
    }
)


def _require_positive(name: str, value: float) -> float:
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


@dataclass(frozen=True)
class ExplorationBudget:
    """
    Hard limits on how much motion one episode may spend on evidence.

    Instruction-following episodes deliberately receive a smaller budget than
    object-reference episodes: until the organisers confirm whether
    exploratory motion counts toward the scored trajectory, the conservative
    assumption is that every metre may be scored.
    """

    max_targeted_viewpoints: int = 3
    max_single_viewpoint_distance_m: float = 6.0
    max_total_exploration_distance_m: float = 15.0
    max_exploration_time_sec: float = 180.0
    minimum_time_remaining_sec: float = 240.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_targeted_viewpoints, bool)
            or not isinstance(self.max_targeted_viewpoints, int)
            or self.max_targeted_viewpoints <= 0
        ):
            raise ValueError('max_targeted_viewpoints must be a positive int')
        for name in (
            'max_single_viewpoint_distance_m',
            'max_total_exploration_distance_m',
            'max_exploration_time_sec',
            'minimum_time_remaining_sec',
        ):
            object.__setattr__(
                self, name, _require_positive(name, getattr(self, name))
            )
        if (
            self.max_single_viewpoint_distance_m
            > self.max_total_exploration_distance_m
        ):
            raise ValueError(
                'a single viewpoint cannot exceed the total travel budget'
            )

    @classmethod
    def for_task_type(cls, task_type: str) -> 'ExplorationBudget':
        """Return the conservative budget configured for a task family."""
        if task_type == 'instruction_following':
            return cls(
                max_targeted_viewpoints=1,
                max_single_viewpoint_distance_m=3.0,
                max_total_exploration_distance_m=4.0,
                max_exploration_time_sec=60.0,
                minimum_time_remaining_sec=300.0,
            )
        if task_type in {'numerical', 'object_reference'}:
            return cls()
        raise ValueError(f'unsupported task type: {task_type!r}')


class ExplorationBudgetTracker:
    """Track consumed viewpoints, travel, and time against a budget."""

    def __init__(self, budget: ExplorationBudget | None = None) -> None:
        """Start an unconsumed tracker for one episode."""
        self._budget = budget or ExplorationBudget()
        self._viewpoints_used = 0
        self._distance_travelled_m = 0.0
        self._time_spent_sec = 0.0

    @property
    def budget(self) -> ExplorationBudget:
        """Return the immutable budget being tracked."""
        return self._budget

    @property
    def viewpoints_used(self) -> int:
        """Return how many targeted viewpoints were consumed."""
        return self._viewpoints_used

    @property
    def distance_travelled_m(self) -> float:
        """Return exploration travel committed so far."""
        return self._distance_travelled_m

    @property
    def time_spent_sec(self) -> float:
        """Return exploration time committed so far."""
        return self._time_spent_sec

    @property
    def remaining_distance_m(self) -> float:
        """Return travel still permitted by the total distance budget."""
        return max(
            0.0,
            self._budget.max_total_exploration_distance_m
            - self._distance_travelled_m,
        )

    def status(self, time_remaining_sec: float) -> str:
        """Return why exploration is or is not currently permitted."""
        if not isfinite(time_remaining_sec) or time_remaining_sec < 0.0:
            raise ValueError('time_remaining_sec must be non-negative')
        if self._viewpoints_used >= self._budget.max_targeted_viewpoints:
            return 'viewpoints_exhausted'
        if self._time_spent_sec >= self._budget.max_exploration_time_sec:
            return 'time_budget_exhausted'
        if time_remaining_sec < self._budget.minimum_time_remaining_sec:
            return 'time_budget_exhausted'
        if self.remaining_distance_m <= 0.0:
            return 'distance_exhausted'
        return 'available'

    def allows(self, time_remaining_sec: float) -> bool:
        """Return whether any further exploration is permitted at all."""
        return self.status(time_remaining_sec) == 'available'

    def permits_travel(self, distance_m: float) -> bool:
        """Return whether one candidate's travel fits both distance limits."""
        if not isfinite(distance_m) or distance_m < 0.0:
            raise ValueError('distance_m must be finite and non-negative')
        return (
            distance_m <= self._budget.max_single_viewpoint_distance_m
            and distance_m <= self.remaining_distance_m
        )

    def consume(self, *, distance_m: float, duration_sec: float) -> None:
        """Record one executed viewpoint against the budget."""
        if not isfinite(distance_m) or distance_m < 0.0:
            raise ValueError('distance_m must be finite and non-negative')
        if not isfinite(duration_sec) or duration_sec < 0.0:
            raise ValueError('duration_sec must be finite and non-negative')
        if self._viewpoints_used >= self._budget.max_targeted_viewpoints:
            raise RuntimeError('targeted viewpoint budget already exhausted')
        self._viewpoints_used += 1
        self._distance_travelled_m += float(distance_m)
        self._time_spent_sec += float(duration_sec)

    def to_dict(self) -> dict[str, object]:
        """Return a stable trace-ready mapping of consumption."""
        return {
            'viewpoints_used': self._viewpoints_used,
            'max_targeted_viewpoints': (
                self._budget.max_targeted_viewpoints
            ),
            'distance_travelled_m': self._distance_travelled_m,
            'max_total_exploration_distance_m': (
                self._budget.max_total_exploration_distance_m
            ),
            'time_spent_sec': self._time_spent_sec,
            'max_exploration_time_sec': (
                self._budget.max_exploration_time_sec
            ),
        }


__all__ = [
    'BUDGET_STATUSES',
    'ExplorationBudget',
    'ExplorationBudgetTracker',
]
