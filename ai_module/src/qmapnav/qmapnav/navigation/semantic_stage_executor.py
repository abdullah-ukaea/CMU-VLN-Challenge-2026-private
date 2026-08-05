"""
Ordered two-stage execution gated on semantic region entry.

Reaching a commanded waypoint is not the same claim as being semantically
near the object: the executor's arrival radius is a control tolerance, while
the near region is the instruction's meaning. Stage A therefore completes only
when the robot is inside A's region, and stage B cannot start before that.
"""

from dataclasses import dataclass
from enum import Enum

from qmapnav.navigation.executor import SequentialWaypointExecutor
from qmapnav.navigation.executor import Waypoint2D
from qmapnav.navigation.semantic_regions import semantic_region_satisfied
from qmapnav.navigation.two_stage_route import PerceivedRoutePlan
from qmapnav.navigation.two_stage_route import PerceivedRouteStage


class SemanticStageState(str, Enum):
    """Bounded ordered-execution states for a two-stage route."""

    IDLE = 'idle'
    EXECUTE_STAGE_A = 'execute_stage_a'
    VERIFY_STAGE_A = 'verify_stage_a'
    EXECUTE_STAGE_B = 'execute_stage_b'
    VERIFY_STAGE_B = 'verify_stage_b'
    COMPLETE = 'complete'
    FAILED = 'failed'


@dataclass(frozen=True)
class StageCompletionEvent:
    """One semantic stage transition suitable for decision tracing."""

    stage_index: int
    target_instance_id: str
    semantic_distance_m: float
    region_satisfied: bool

    def to_dict(self) -> dict[str, object]:
        """Return the stable ``semantic_stage_complete`` trace record."""
        return {
            'event': 'semantic_stage_complete',
            'stage': self.stage_index,
            'target': self.target_instance_id,
            'semantic_distance_m': self.semantic_distance_m,
            'region_satisfied': self.region_satisfied,
        }


class SemanticStageExecutor:
    """
    Drive an ordered two-stage route through the sequential executor.

    This class owns only stage ordering and semantic verification; all
    waypoint publication, arrival gating, progress monitoring and bounded
    recovery remain the responsibility of
    :class:`~qmapnav.navigation.executor.SequentialWaypointExecutor`.
    """

    def __init__(
        self,
        plan: PerceivedRoutePlan,
        executor: SequentialWaypointExecutor,
    ) -> None:
        """Bind one planned route to one waypoint executor."""
        if not isinstance(plan, PerceivedRoutePlan):
            raise TypeError('plan must be PerceivedRoutePlan')
        if not plan.executable:
            raise ValueError('only a planned two-stage route can execute')
        if not isinstance(executor, SequentialWaypointExecutor):
            raise TypeError('executor must be SequentialWaypointExecutor')
        self._plan = plan
        self._executor = executor
        self._state = SemanticStageState.IDLE
        self._active_stage = 0
        self._events: list[StageCompletionEvent] = []
        self._reason = ''
        self._terminal_only = plan.route_status == 'terminal_only'

    @property
    def state(self) -> SemanticStageState:
        """Return the current ordered-execution state."""
        return self._state

    @property
    def failure_reason(self) -> str:
        """Return why execution failed, or an empty string if it has not."""
        return self._reason

    @property
    def active_stage_index(self) -> int:
        """Return which stage is currently being pursued."""
        return self._active_stage

    @property
    def completed_stages(self) -> tuple[int, ...]:
        """Return the stage indices already semantically verified."""
        return tuple(item.stage_index for item in self._events)

    def stage(self, index: int) -> PerceivedRouteStage:
        """Return one stage of the bound plan."""
        return self._plan.stages[index]

    def drain_events(self) -> tuple[StageCompletionEvent, ...]:
        """Return and clear buffered stage-completion events."""
        events = tuple(self._events)
        self._events = []
        return events

    def start(
        self,
        waypoints: tuple[tuple[float, float, float], ...],
        *,
        now: float | None = None,
    ) -> Waypoint2D:
        """Begin stage A and return only its first waypoint."""
        if self._state is not SemanticStageState.IDLE:
            raise RuntimeError('two-stage route already started')
        if not waypoints:
            raise ValueError('stage A requires at least one waypoint')
        self._state = SemanticStageState.EXECUTE_STAGE_A
        self._active_stage = 0
        return self._executor.start(
            tuple(Waypoint2D(*item) for item in waypoints), now=now
        )

    def begin_stage_b(
        self,
        waypoints: tuple[tuple[float, float, float], ...],
        *,
        now: float | None = None,
    ) -> Waypoint2D:
        """
        Begin stage B, which is refused until stage A is verified.

        This is the ordering guarantee: no amount of waypoint progress can
        start the terminal stage before the first is semantically satisfied.
        """
        if self._terminal_only:
            raise RuntimeError('a terminal-only route has no second stage')
        if self._state is not SemanticStageState.VERIFY_STAGE_A:
            raise RuntimeError(
                'stage B cannot begin before stage A is semantically verified'
            )
        if not waypoints:
            raise ValueError('stage B requires at least one waypoint')
        self._state = SemanticStageState.EXECUTE_STAGE_B
        self._active_stage = 1
        return self._executor.start(
            tuple(Waypoint2D(*item) for item in waypoints), now=now
        )

    def update_pose(
        self,
        x: float,
        y: float,
        heading: float | None = None,
        *,
        now: float | None = None,
    ) -> Waypoint2D | None:
        """
        Feed one pose to the executor and test the active semantic region.

        Semantic satisfaction is checked independently of waypoint arrival,
        so a stage can complete as soon as the robot is genuinely near its
        object even if the commanded waypoint has not been reached.
        """
        activated = self._executor.update_pose(x, y, heading, now=now)
        if self._state is SemanticStageState.EXECUTE_STAGE_A:
            if self._verify(0, x, y):
                # A terminal-only fallback has a single stage, which is the
                # terminal one, so verifying it finishes the route outright.
                self._state = (
                    SemanticStageState.COMPLETE
                    if self._terminal_only
                    else SemanticStageState.VERIFY_STAGE_A
                )
                self._executor.cancel(now=now)
                return None
        elif self._state is SemanticStageState.EXECUTE_STAGE_B:
            if self._verify(1, x, y):
                self._state = SemanticStageState.COMPLETE
                self._executor.cancel(now=now)
                return None
        return activated

    def fail(self, reason: str = 'stage_execution_failed') -> None:
        """Move to a clean terminal failure rather than stalling."""
        if self._state in {
            SemanticStageState.COMPLETE,
            SemanticStageState.FAILED,
        }:
            return
        self._state = SemanticStageState.FAILED
        self._reason = reason

    def _verify(self, index: int, x: float, y: float) -> bool:
        stage = self._plan.stages[index]
        if not semantic_region_satisfied((x, y), stage.target_region):
            return False
        centre = stage.target_region.polygon.centre
        distance = (
            (x - centre[0]) ** 2 + (y - centre[1]) ** 2
        ) ** 0.5
        self._events.append(
            StageCompletionEvent(
                stage_index=index,
                target_instance_id=stage.resolved_instance_id,
                semantic_distance_m=distance,
                region_satisfied=True,
            )
        )
        return True


__all__ = [
    'SemanticStageExecutor',
    'SemanticStageState',
    'StageCompletionEvent',
]
