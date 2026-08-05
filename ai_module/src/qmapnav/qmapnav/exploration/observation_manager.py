"""
Deliberate stationary observation: settle, accumulate, then select.

Small targets and sparse geometry benefit from holding still while scans
register, but an unbounded pause would be a new way for an episode to stall.
Every phase therefore advances on an injected clock and the whole sequence is
capped, so the manager always reaches a terminal state.
"""

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class ObservationState(str, Enum):
    """Phases of one bounded stationary observation."""

    IDLE = 'idle'
    SETTLING = 'settling'
    ACCUMULATING = 'accumulating'
    COMPLETE = 'complete'
    TIMED_OUT = 'timed_out'


@dataclass(frozen=True)
class ObservationConfig:
    """Timing policy for one stationary observation."""

    settle_time_sec: float = 0.75
    scan_accumulation_time_sec: float = 2.5
    panoramas_to_consider: int = 3
    max_observation_time_sec: float = 8.0

    def __post_init__(self) -> None:
        for name in (
            'settle_time_sec',
            'scan_accumulation_time_sec',
            'max_observation_time_sec',
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        if isinstance(self.panoramas_to_consider, bool) or (
            self.panoramas_to_consider < 1
        ):
            raise ValueError('panoramas_to_consider must be >= 1')
        if self.max_observation_time_sec < (
            self.settle_time_sec + self.scan_accumulation_time_sec
        ):
            raise ValueError(
                'max_observation_time_sec must cover settle plus accumulation'
            )


@dataclass(frozen=True)
class PanoramaOffer:
    """One candidate panorama observed during the accumulation window."""

    panorama_id: str
    sharpness: float
    timestamp: float

    def __post_init__(self) -> None:
        if not self.panorama_id or not self.panorama_id.strip():
            raise ValueError('panorama_id must be a non-empty string')
        for name in ('sharpness', 'timestamp'):
            value = getattr(self, name)
            if not isfinite(value):
                raise ValueError(f'{name} must be finite')
        if self.sharpness < 0.0:
            raise ValueError('sharpness must be non-negative')


@dataclass(frozen=True)
class ObservationResult:
    """What one stationary observation produced."""

    status: str
    selected_panorama_id: str | None
    panoramas_considered: int
    scan_points_added: int
    duration_sec: float

    def to_dict(self) -> dict[str, object]:
        """Return a stable trace-ready mapping."""
        return {
            'status': self.status,
            'selected_panorama_id': self.selected_panorama_id,
            'panoramas_considered': self.panoramas_considered,
            'scan_points_added': self.scan_points_added,
            'duration_sec': self.duration_sec,
        }


class ObservationManager:
    """Drive one bounded settle-accumulate-select observation."""

    def __init__(self, config: ObservationConfig | None = None) -> None:
        """Create an idle manager for a single observation."""
        self._config = config or ObservationConfig()
        self._state = ObservationState.IDLE
        self._started_at: float | None = None
        self._offers: list[PanoramaOffer] = []
        self._scan_points = 0
        self._finished_at: float | None = None

    @property
    def config(self) -> ObservationConfig:
        """Return the immutable timing policy."""
        return self._config

    @property
    def state(self) -> ObservationState:
        """Return the current bounded observation phase."""
        return self._state

    @property
    def accepting_panoramas(self) -> bool:
        """Return whether offers are currently being collected."""
        return self._state is ObservationState.ACCUMULATING

    def begin(self, now: float) -> ObservationState:
        """Start settling after the base has been commanded to stop."""
        if self._state is not ObservationState.IDLE:
            raise RuntimeError('observation already started')
        if not isfinite(now):
            raise ValueError('now must be finite')
        self._started_at = float(now)
        self._state = ObservationState.SETTLING
        return self._state

    def update(self, now: float) -> ObservationState:
        """Advance the phase clock and enforce the hard observation cap."""
        if self._state in {
            ObservationState.IDLE,
            ObservationState.COMPLETE,
            ObservationState.TIMED_OUT,
        }:
            return self._state
        if not isfinite(now):
            raise ValueError('now must be finite')
        elapsed = float(now) - self._started_at
        if elapsed >= self._config.max_observation_time_sec:
            self._finished_at = float(now)
            self._state = ObservationState.TIMED_OUT
            return self._state
        if self._state is ObservationState.SETTLING:
            if elapsed >= self._config.settle_time_sec:
                self._state = ObservationState.ACCUMULATING
            return self._state
        accumulation_deadline = (
            self._config.settle_time_sec
            + self._config.scan_accumulation_time_sec
        )
        if elapsed >= accumulation_deadline:
            self._finished_at = float(now)
            self._state = ObservationState.COMPLETE
        return self._state

    def offer_panorama(self, offer: PanoramaOffer) -> bool:
        """Offer one panorama, keeping only the configured newest window."""
        if not isinstance(offer, PanoramaOffer):
            raise TypeError('offer must be PanoramaOffer')
        if self._state is not ObservationState.ACCUMULATING:
            return False
        self._offers.append(offer)
        # Retain only the most recent window of candidates.
        self._offers = sorted(
            self._offers, key=lambda item: item.timestamp
        )[-self._config.panoramas_to_consider:]
        return True

    def note_scan_points(self, count: int) -> None:
        """Record registered points added during accumulation."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError('count must be a non-negative integer')
        self._scan_points += count

    def result(self) -> ObservationResult:
        """Return the terminal result, selecting the sharpest panorama."""
        if self._state not in {
            ObservationState.COMPLETE,
            ObservationState.TIMED_OUT,
        }:
            raise RuntimeError('observation has not reached a terminal state')
        selected = None
        if self._offers:
            # Sharpest wins; the most recent breaks a tie.
            selected = max(
                self._offers,
                key=lambda item: (item.sharpness, item.timestamp),
            ).panorama_id
        duration = 0.0
        if self._started_at is not None and self._finished_at is not None:
            duration = max(0.0, self._finished_at - self._started_at)
        return ObservationResult(
            status=self._state.value,
            selected_panorama_id=selected,
            panoramas_considered=len(self._offers),
            scan_points_added=self._scan_points,
            duration_sec=duration,
        )


__all__ = [
    'ObservationConfig',
    'ObservationManager',
    'ObservationResult',
    'ObservationState',
    'PanoramaOffer',
]
