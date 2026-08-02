"""ROS-independent sequential single-active-waypoint execution."""

from dataclasses import dataclass
from enum import Enum
from math import hypot
from math import isfinite
from typing import Iterable


DEFAULT_ARRIVAL_RADIUS = 0.75


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
    COMPLETE = 'complete'


class SequentialWaypointExecutor:
    """Advance an owned route only after pose-based arrival at its active goal."""

    def __init__(self, arrival_radius: float = DEFAULT_ARRIVAL_RADIUS) -> None:
        if not isfinite(arrival_radius) or arrival_radius <= 0.0:
            raise ValueError('arrival_radius must be finite and positive')
        self._arrival_radius = float(arrival_radius)
        self._route: tuple[Waypoint2D, ...] = ()
        self._active_index: int | None = None
        self._state = WaypointExecutorState.IDLE

    @property
    def arrival_radius(self) -> float:
        """Return the configured planar arrival radius in metres."""
        return self._arrival_radius

    @property
    def state(self) -> WaypointExecutorState:
        """Return the executor lifecycle state."""
        return self._state

    @property
    def route(self) -> tuple[Waypoint2D, ...]:
        """Return the immutable internal copy of the active or completed route."""
        return self._route

    @property
    def active_index(self) -> int | None:
        """Return the active zero-based route index, if a goal is active."""
        return self._active_index

    @property
    def active_waypoint(self) -> Waypoint2D | None:
        """Return the one active waypoint, if execution is in progress."""
        if self._active_index is None:
            return None
        return self._route[self._active_index]

    def start(self, route: Iterable[Waypoint2D]) -> Waypoint2D:
        """Own a validated route and return only its first goal for publication."""
        if self._state is WaypointExecutorState.ACTIVE:
            raise RuntimeError('cannot replace an active route')

        copied_route = tuple(route)
        if not copied_route:
            raise ValueError('route must contain at least one waypoint')
        if not all(isinstance(waypoint, Waypoint2D) for waypoint in copied_route):
            raise TypeError('route must contain only Waypoint2D values')

        self._route = copied_route
        self._active_index = 0
        self._state = WaypointExecutorState.ACTIVE
        return self._route[0]

    def update_pose(self, x: float, y: float) -> Waypoint2D | None:
        """Process one map-frame pose and return a newly activated goal, if any."""
        if not isfinite(x) or not isfinite(y):
            raise ValueError('robot pose must be finite')
        if self._state is not WaypointExecutorState.ACTIVE:
            return None

        active = self.active_waypoint
        if active is None:
            raise RuntimeError('active executor state has no active waypoint')
        distance = hypot(active.x - x, active.y - y)
        if distance > self._arrival_radius:
            return None

        if self._active_index == len(self._route) - 1:
            self._active_index = None
            self._state = WaypointExecutorState.COMPLETE
            return None

        self._active_index += 1
        return self._route[self._active_index]


__all__ = [
    'DEFAULT_ARRIVAL_RADIUS',
    'SequentialWaypointExecutor',
    'Waypoint2D',
    'WaypointExecutorState',
]
