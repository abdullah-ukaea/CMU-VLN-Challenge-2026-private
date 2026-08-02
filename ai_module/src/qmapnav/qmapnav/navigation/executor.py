"""ROS-independent sequential waypoint execution with bounded recovery."""

from collections import deque
from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from math import hypot
from math import isfinite
from time import monotonic


DEFAULT_ARRIVAL_RADIUS = 0.75
DEFAULT_PROGRESS_EPSILON = 0.15
DEFAULT_NO_PROGRESS_TIMEOUT = 12.0
DEFAULT_DIRECT_REPUBLISH_LIMIT = 1
DEFAULT_SAFE_OFFSET_LIMIT = 1


@dataclass(frozen=True)
class Waypoint2D:
    """One finite map-frame planar waypoint and heading in radians."""

    x: float
    y: float
    heading: float = 0.0

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.x, self.y, self.heading)):
            raise ValueError('waypoint values must be finite')


class WaypointExecutorState(Enum):
    """Current lifecycle state of a sequential waypoint route."""

    IDLE = 'idle'
    ACTIVE = 'active'
    RECOVERING = 'recovering'
    COMPLETE = 'complete'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


class ExecutorEventType(Enum):
    """Observable route-execution transitions emitted by the state machine."""

    ROUTE_STARTED = 'route_started'
    WAYPOINT_REACHED = 'waypoint_reached'
    WAYPOINT_ACTIVATED = 'waypoint_activated'
    PROGRESS = 'progress'
    GOAL_REPUBLISHED = 'goal_republished'
    RECOVERY_STARTED = 'recovery_started'
    RECOVERY_REACHED = 'recovery_reached'
    ROUTE_COMPLETED = 'route_completed'
    ROUTE_FAILED = 'route_failed'
    ROUTE_CANCELLED = 'route_cancelled'


@dataclass(frozen=True)
class ExecutorEvent:
    """One immutable execution transition suitable for decision tracing."""

    event_type: ExecutorEventType
    timestamp: float
    state: WaypointExecutorState
    route_index: int | None
    direct_republish_count: int
    recovery_count: int
    reason: str
    distance: float | None = None
    waypoint: Waypoint2D | None = None


SafeOffsetSelector = Callable[
    [float, float, Waypoint2D],
    Waypoint2D | None,
]


def _require_positive_finite(name: str, value: float) -> float:
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


def _require_non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a non-negative integer')
    return value


class SequentialWaypointExecutor:
    """Execute one goal at a time with a bounded no-progress watchdog."""

    def __init__(
        self,
        arrival_radius: float = DEFAULT_ARRIVAL_RADIUS,
        *,
        progress_epsilon: float = DEFAULT_PROGRESS_EPSILON,
        no_progress_timeout: float = DEFAULT_NO_PROGRESS_TIMEOUT,
        direct_republish_limit: int = DEFAULT_DIRECT_REPUBLISH_LIMIT,
        safe_offset_limit: int = DEFAULT_SAFE_OFFSET_LIMIT,
        safe_offset_selector: SafeOffsetSelector | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._arrival_radius = _require_positive_finite(
            'arrival_radius', arrival_radius
        )
        self._progress_epsilon = _require_positive_finite(
            'progress_epsilon', progress_epsilon
        )
        self._no_progress_timeout = _require_positive_finite(
            'no_progress_timeout', no_progress_timeout
        )
        self._direct_republish_limit = _require_non_negative_int(
            'direct_republish_limit', direct_republish_limit
        )
        self._safe_offset_limit = _require_non_negative_int(
            'safe_offset_limit', safe_offset_limit
        )
        if not callable(clock):
            raise TypeError('clock must be callable')
        if safe_offset_selector is not None and not callable(
            safe_offset_selector
        ):
            raise TypeError('safe_offset_selector must be callable or None')

        self._clock = clock
        self._safe_offset_selector = safe_offset_selector
        self._route: tuple[Waypoint2D, ...] = ()
        self._active_index: int | None = None
        self._state = WaypointExecutorState.IDLE
        self._recovery_waypoint: Waypoint2D | None = None
        self._latest_pose: Waypoint2D | None = None
        self._best_distance = float('inf')
        self._last_progress_time: float | None = None
        self._direct_republish_count = 0
        self._recovery_count = 0
        self._events: deque[ExecutorEvent] = deque()

    @property
    def arrival_radius(self) -> float:
        """Return the configured planar arrival radius in metres."""
        return self._arrival_radius

    @property
    def progress_epsilon(self) -> float:
        """Return the distance decrease required to count as progress."""
        return self._progress_epsilon

    @property
    def no_progress_timeout(self) -> float:
        """Return the no-progress timeout in seconds."""
        return self._no_progress_timeout

    @property
    def direct_republish_limit(self) -> int:
        """Return the direct active-goal republish budget per waypoint."""
        return self._direct_republish_limit

    @property
    def safe_offset_limit(self) -> int:
        """Return the safe-offset recovery budget per waypoint."""
        return self._safe_offset_limit

    @property
    def direct_republish_count(self) -> int:
        """Return direct republishes used for the active route waypoint."""
        return self._direct_republish_count

    @property
    def recovery_count(self) -> int:
        """Return safe-offset attempts used for the active route waypoint."""
        return self._recovery_count

    @property
    def state(self) -> WaypointExecutorState:
        """Return the executor lifecycle state."""
        return self._state

    @property
    def route(self) -> tuple[Waypoint2D, ...]:
        """Return the immutable internal copy of the submitted route."""
        return self._route

    @property
    def active_index(self) -> int | None:
        """Return the active zero-based semantic route index, if present."""
        return self._active_index

    @property
    def active_waypoint(self) -> Waypoint2D | None:
        """Return the active semantic route waypoint, if one is pending."""
        if self._active_index is None:
            return None
        return self._route[self._active_index]

    @property
    def navigation_target(self) -> Waypoint2D | None:
        """Return the currently commanded semantic or recovery target."""
        if self._state is WaypointExecutorState.RECOVERING:
            return self._recovery_waypoint
        if self._state is WaypointExecutorState.ACTIVE:
            return self.active_waypoint
        return None

    def start(
        self,
        route: Iterable[Waypoint2D],
        *,
        now: float | None = None,
    ) -> Waypoint2D:
        """Own a validated route and return only its first goal for publication."""
        if self._state in {
            WaypointExecutorState.ACTIVE,
            WaypointExecutorState.RECOVERING,
        }:
            raise RuntimeError('cannot replace an active route')

        copied_route = tuple(route)
        if not copied_route:
            raise ValueError('route must contain at least one waypoint')
        if not all(isinstance(waypoint, Waypoint2D) for waypoint in copied_route):
            raise TypeError('route must contain only Waypoint2D values')

        timestamp = self._resolve_time(now)
        self._route = copied_route
        self._active_index = 0
        self._state = WaypointExecutorState.ACTIVE
        self._recovery_waypoint = None
        self._reset_watchdog(timestamp, reset_budgets=True)
        self._emit(
            ExecutorEventType.ROUTE_STARTED,
            timestamp,
            reason='validated_route_accepted',
            waypoint=self._route[0],
        )
        self._emit(
            ExecutorEventType.WAYPOINT_ACTIVATED,
            timestamp,
            reason='first_route_waypoint',
            waypoint=self._route[0],
        )
        return self._route[0]

    def update_pose(
        self,
        x: float,
        y: float,
        heading: float | None = None,
        *,
        now: float | None = None,
    ) -> Waypoint2D | None:
        """Process one map-frame pose and return a newly activated goal."""
        values = (x, y) if heading is None else (x, y, heading)
        if not all(isfinite(value) for value in values):
            raise ValueError('robot pose must be finite')
        timestamp = self._resolve_time(now)
        previous_heading = (
            self._latest_pose.heading if self._latest_pose is not None else 0.0
        )
        self._latest_pose = Waypoint2D(
            float(x),
            float(y),
            previous_heading if heading is None else float(heading),
        )
        if self._state not in {
            WaypointExecutorState.ACTIVE,
            WaypointExecutorState.RECOVERING,
        }:
            return None

        target = self.navigation_target
        if target is None:
            raise RuntimeError('non-terminal executor state has no target')
        distance = hypot(target.x - x, target.y - y)
        if distance <= self._arrival_radius:
            return self._handle_arrival(timestamp, distance)

        if distance <= self._best_distance - self._progress_epsilon:
            self._best_distance = distance
            self._last_progress_time = timestamp
            self._emit(
                ExecutorEventType.PROGRESS,
                timestamp,
                reason='distance_decreased_by_progress_epsilon',
                distance=distance,
                waypoint=target,
            )
        return None

    def tick(self, *, now: float | None = None) -> Waypoint2D | None:
        """Run the no-progress watchdog and return a bounded recovery goal."""
        timestamp = self._resolve_time(now)
        if self._state not in {
            WaypointExecutorState.ACTIVE,
            WaypointExecutorState.RECOVERING,
        }:
            return None
        if self._last_progress_time is None:
            self._last_progress_time = timestamp
            return None
        if timestamp - self._last_progress_time < self._no_progress_timeout:
            return None

        if self._state is WaypointExecutorState.RECOVERING:
            self._fail(timestamp, 'safe_offset_made_no_progress')
            return None

        active = self.active_waypoint
        if active is None:
            raise RuntimeError('active executor state has no semantic waypoint')
        if self._direct_republish_count < self._direct_republish_limit:
            self._direct_republish_count += 1
            self._reset_watchdog(timestamp, reset_budgets=False)
            self._emit(
                ExecutorEventType.GOAL_REPUBLISHED,
                timestamp,
                reason='no_progress_timeout',
                waypoint=active,
            )
            return active

        if self._recovery_count >= self._safe_offset_limit:
            self._fail(timestamp, 'retry_budgets_exhausted')
            return None
        if self._latest_pose is None:
            self._fail(timestamp, 'safe_offset_unavailable_without_pose')
            return None
        if self._safe_offset_selector is None:
            self._fail(timestamp, 'safe_offset_selector_unavailable')
            return None

        recovery = self._safe_offset_selector(
            self._latest_pose.x,
            self._latest_pose.y,
            active,
        )
        if recovery is None:
            self._fail(timestamp, 'no_map_validated_safe_offset')
            return None
        if not isinstance(recovery, Waypoint2D):
            self._fail(timestamp, 'safe_offset_selector_returned_invalid_type')
            return None

        self._recovery_count += 1
        self._recovery_waypoint = recovery
        self._state = WaypointExecutorState.RECOVERING
        self._reset_watchdog(timestamp, reset_budgets=False)
        self._emit(
            ExecutorEventType.RECOVERY_STARTED,
            timestamp,
            reason='direct_republish_budget_exhausted',
            waypoint=recovery,
        )
        return recovery

    def cancel(self, *, now: float | None = None) -> Waypoint2D | None:
        """Cancel idempotently and return a current-pose hold goal when safe."""
        timestamp = self._resolve_time(now)
        if self._state is WaypointExecutorState.CANCELLED:
            return None
        if self._state in {
            WaypointExecutorState.COMPLETE,
            WaypointExecutorState.FAILED,
        }:
            return None

        cancelled_index = self._active_index
        hold = self._latest_pose if self._state is not WaypointExecutorState.IDLE else None
        self._active_index = None
        self._recovery_waypoint = None
        self._state = WaypointExecutorState.CANCELLED
        self._emit(
            ExecutorEventType.ROUTE_CANCELLED,
            timestamp,
            reason=(
                'current_pose_hold_requested'
                if hold is not None
                else 'cancelled_without_known_pose'
            ),
            route_index=cancelled_index,
            waypoint=hold,
        )
        return hold

    def drain_events(self) -> tuple[ExecutorEvent, ...]:
        """Return and clear accumulated observable state transitions."""
        events = tuple(self._events)
        self._events.clear()
        return events

    def _handle_arrival(
        self,
        timestamp: float,
        distance: float,
    ) -> Waypoint2D | None:
        if self._state is WaypointExecutorState.RECOVERING:
            self._state = WaypointExecutorState.ACTIVE
            self._recovery_waypoint = None
            active = self.active_waypoint
            if active is None:
                raise RuntimeError('recovery has no interrupted semantic waypoint')
            self._reset_watchdog(timestamp, reset_budgets=False)
            self._emit(
                ExecutorEventType.RECOVERY_REACHED,
                timestamp,
                reason='retrying_interrupted_waypoint',
                distance=distance,
                waypoint=active,
            )
            return active

        active = self.active_waypoint
        if active is None or self._active_index is None:
            raise RuntimeError('active executor state has no active waypoint')
        reached_index = self._active_index
        self._emit(
            ExecutorEventType.WAYPOINT_REACHED,
            timestamp,
            reason='inside_arrival_radius',
            distance=distance,
            waypoint=active,
        )
        if reached_index == len(self._route) - 1:
            self._active_index = None
            self._state = WaypointExecutorState.COMPLETE
            self._emit(
                ExecutorEventType.ROUTE_COMPLETED,
                timestamp,
                reason='final_waypoint_reached',
                distance=distance,
                route_index=reached_index,
            )
            return None

        self._active_index += 1
        next_waypoint = self._route[self._active_index]
        self._reset_watchdog(timestamp, reset_budgets=True)
        self._emit(
            ExecutorEventType.WAYPOINT_ACTIVATED,
            timestamp,
            reason='previous_waypoint_reached',
            waypoint=next_waypoint,
        )
        return next_waypoint

    def _fail(self, timestamp: float, reason: str) -> None:
        failed_index = self._active_index
        self._active_index = None
        self._recovery_waypoint = None
        self._state = WaypointExecutorState.FAILED
        self._emit(
            ExecutorEventType.ROUTE_FAILED,
            timestamp,
            reason=reason,
            route_index=failed_index,
        )

    def _reset_watchdog(
        self,
        timestamp: float,
        *,
        reset_budgets: bool,
    ) -> None:
        self._best_distance = float('inf')
        self._last_progress_time = timestamp
        if reset_budgets:
            self._direct_republish_count = 0
            self._recovery_count = 0

    def _resolve_time(self, now: float | None) -> float:
        timestamp = float(self._clock() if now is None else now)
        if not isfinite(timestamp) or timestamp < 0.0:
            raise ValueError('time must be finite and non-negative')
        return timestamp

    def _emit(
        self,
        event_type: ExecutorEventType,
        timestamp: float,
        *,
        reason: str,
        distance: float | None = None,
        waypoint: Waypoint2D | None = None,
        route_index: int | None = None,
    ) -> None:
        self._events.append(
            ExecutorEvent(
                event_type=event_type,
                timestamp=timestamp,
                state=self._state,
                route_index=(
                    self._active_index if route_index is None else route_index
                ),
                direct_republish_count=self._direct_republish_count,
                recovery_count=self._recovery_count,
                reason=reason,
                distance=distance,
                waypoint=waypoint,
            )
        )


__all__ = [
    'DEFAULT_ARRIVAL_RADIUS',
    'DEFAULT_DIRECT_REPUBLISH_LIMIT',
    'DEFAULT_NO_PROGRESS_TIMEOUT',
    'DEFAULT_PROGRESS_EPSILON',
    'DEFAULT_SAFE_OFFSET_LIMIT',
    'ExecutorEvent',
    'ExecutorEventType',
    'SafeOffsetSelector',
    'SequentialWaypointExecutor',
    'Waypoint2D',
    'WaypointExecutorState',
]
