"""Bounded non-blocking JSONL decision tracing for one process episode."""

from dataclasses import dataclass
from dataclasses import field
from json import dumps
from math import isfinite
from pathlib import Path
from queue import Empty
from queue import Full
from queue import Queue
from threading import Lock
from threading import Thread
from time import monotonic
from time import sleep
from time import time
from typing import Protocol
from uuid import uuid4


TRACE_SCHEMA_VERSION = '1.0'


@dataclass(frozen=True)
class DecisionTraceEvent:
    """One versioned, JSON-serializable observation of mission behavior."""

    event: str
    episode_elapsed_seconds: float
    mission_state: str
    raw_question: str | None = None
    normalized_question: str | None = None
    parser_mode: str | None = None
    parse_confidence: float | None = None
    known_object_count: int = 0
    known_structure_count: int = 0
    missing_entities: tuple[str, ...] = ()
    candidate_actions: tuple[str, ...] = ()
    selected_action: str | None = None
    selection_reason: str = ''
    active_route_index: int | None = None
    direct_republish_count: int = 0
    recovery_count: int = 0
    ignored_question_count: int = 0
    time_remaining_seconds: float = 0.0
    terminal_status: str | None = None
    details: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.event, str) or not self.event:
            raise ValueError('event must be a non-empty string')
        if not isinstance(self.mission_state, str) or not self.mission_state:
            raise ValueError('mission_state must be a non-empty string')
        for name, value in {
            'episode_elapsed_seconds': self.episode_elapsed_seconds,
            'time_remaining_seconds': self.time_remaining_seconds,
        }.items():
            if not isfinite(value) or value < 0.0:
                raise ValueError(f'{name} must be finite and non-negative')
        for name, value in {
            'known_object_count': self.known_object_count,
            'known_structure_count': self.known_structure_count,
            'direct_republish_count': self.direct_republish_count,
            'recovery_count': self.recovery_count,
            'ignored_question_count': self.ignored_question_count,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a non-negative integer')
        if self.active_route_index is not None:
            if (
                isinstance(self.active_route_index, bool)
                or not isinstance(self.active_route_index, int)
                or self.active_route_index < 0
            ):
                raise ValueError(
                    'active_route_index must be a non-negative integer or None'
                )
        if self.parse_confidence is not None:
            if (
                not isfinite(self.parse_confidence)
                or not 0.0 <= self.parse_confidence <= 1.0
            ):
                raise ValueError('parse_confidence must be in [0, 1] or None')
        object.__setattr__(self, 'missing_entities', tuple(self.missing_entities))
        object.__setattr__(self, 'candidate_actions', tuple(self.candidate_actions))
        object.__setattr__(self, 'details', dict(self.details))


@dataclass(frozen=True)
class TraceRecorderStats:
    """Read-only health and bound counters for a trace recorder."""

    accepted_event_count: int
    written_event_count: int
    dropped_queue_event_count: int
    dropped_file_event_count: int
    serialization_error_count: int
    write_error_count: int
    bytes_written: int
    closed: bool


class TraceRecorder(Protocol):
    """Minimal observational trace sink used by the mission composition root."""

    def record(self, event: DecisionTraceEvent) -> bool:
        """Attempt to accept one event without controlling mission behavior."""

    def close(self, timeout: float = 1.0) -> bool:
        """Perform a bounded terminal flush and stop the sink."""


class InMemoryTraceRecorder:
    """Deterministic bounded trace sink for tests and in-process evaluation."""

    def __init__(self, max_events: int = 1024) -> None:
        if isinstance(max_events, bool) or not isinstance(max_events, int):
            raise ValueError('max_events must be a positive integer')
        if max_events <= 0:
            raise ValueError('max_events must be a positive integer')
        self._max_events = max_events
        self.events: list[DecisionTraceEvent] = []
        self.dropped_event_count = 0
        self.closed = False

    def record(self, event: DecisionTraceEvent) -> bool:
        """Store an event if this bounded sink still has capacity."""
        if self.closed or len(self.events) >= self._max_events:
            self.dropped_event_count += 1
            return False
        self.events.append(event)
        return True

    def close(self, timeout: float = 1.0) -> bool:
        """Close immediately; ``timeout`` is accepted for protocol parity."""
        del timeout
        self.closed = True
        return True


_STOP = object()


class JsonlDecisionTraceRecorder:
    """Write trace events on a worker using bounded queue and file limits."""

    def __init__(
        self,
        path: str | Path,
        *,
        episode_id: str | None = None,
        max_queue_size: int = 512,
        max_file_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if isinstance(max_queue_size, bool) or not isinstance(max_queue_size, int):
            raise ValueError('max_queue_size must be a positive integer')
        if max_queue_size <= 0:
            raise ValueError('max_queue_size must be a positive integer')
        if isinstance(max_file_bytes, bool) or not isinstance(max_file_bytes, int):
            raise ValueError('max_file_bytes must be a positive integer')
        if max_file_bytes <= 0:
            raise ValueError('max_file_bytes must be a positive integer')

        self._path = Path(path)
        self._episode_id = episode_id or uuid4().hex
        if not self._episode_id:
            raise ValueError('episode_id must be non-empty')
        self._max_file_bytes = max_file_bytes
        self._queue: Queue[DecisionTraceEvent | object] = Queue(
            maxsize=max_queue_size
        )
        self._lock = Lock()
        self._accepted_event_count = 0
        self._written_event_count = 0
        self._dropped_queue_event_count = 0
        self._dropped_file_event_count = 0
        self._serialization_error_count = 0
        self._write_error_count = 0
        self._bytes_written = 0
        self._sequence = 0
        self._closed = False
        self._worker = Thread(
            target=self._run,
            name='qmapnav-trace-writer',
            daemon=True,
        )
        self._worker.start()

    @property
    def path(self) -> Path:
        """Return the configured JSONL output path."""
        return self._path

    @property
    def episode_id(self) -> str:
        """Return the stable process-episode identifier."""
        return self._episode_id

    def record(self, event: DecisionTraceEvent) -> bool:
        """Enqueue one event without waiting for file I/O or serialization."""
        if not isinstance(event, DecisionTraceEvent):
            raise TypeError('event must be a DecisionTraceEvent')
        with self._lock:
            if self._closed:
                self._dropped_queue_event_count += 1
                return False
        try:
            self._queue.put_nowait(event)
        except Full:
            with self._lock:
                self._dropped_queue_event_count += 1
            return False
        with self._lock:
            self._accepted_event_count += 1
        return True

    def stats(self) -> TraceRecorderStats:
        """Return a consistent recorder health snapshot."""
        with self._lock:
            return TraceRecorderStats(
                accepted_event_count=self._accepted_event_count,
                written_event_count=self._written_event_count,
                dropped_queue_event_count=self._dropped_queue_event_count,
                dropped_file_event_count=self._dropped_file_event_count,
                serialization_error_count=self._serialization_error_count,
                write_error_count=self._write_error_count,
                bytes_written=self._bytes_written,
                closed=self._closed,
            )

    def close(self, timeout: float = 1.0) -> bool:
        """Stop after a bounded attempt to flush already accepted events."""
        if not isfinite(timeout) or timeout < 0.0:
            raise ValueError('timeout must be finite and non-negative')
        with self._lock:
            if self._closed:
                return not self._worker.is_alive()
            self._closed = True

        deadline = monotonic() + timeout
        while True:
            try:
                self._queue.put_nowait(_STOP)
                break
            except Full:
                if monotonic() >= deadline:
                    return False
                sleep(min(0.005, max(0.0, deadline - monotonic())))
        self._worker.join(max(0.0, deadline - monotonic()))
        return not self._worker.is_alive()

    def _run(self) -> None:
        output = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            existing_size = self._path.stat().st_size if self._path.exists() else 0
            output = self._path.open('a', encoding='utf-8')
            with self._lock:
                self._bytes_written = existing_size
        except OSError:
            with self._lock:
                self._write_error_count += 1

        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except Empty:
                with self._lock:
                    if self._closed:
                        break
                continue
            try:
                if item is _STOP:
                    break
                if not isinstance(item, DecisionTraceEvent):
                    continue
                self._write_event(output, item)
            finally:
                self._queue.task_done()

        if output is not None:
            try:
                output.flush()
                output.close()
            except OSError:
                with self._lock:
                    self._write_error_count += 1

    def _write_event(
        self,
        output: object | None,
        event: DecisionTraceEvent,
    ) -> None:
        if output is None:
            with self._lock:
                self._write_error_count += 1
            return
        with self._lock:
            sequence = self._sequence
            self._sequence += 1
        record = {
            'schema_version': TRACE_SCHEMA_VERSION,
            'episode_id': self._episode_id,
            'sequence': sequence,
            'recorded_at_unix_seconds': time(),
            'event': event.event,
            'episode_elapsed_seconds': event.episode_elapsed_seconds,
            'mission_state': event.mission_state,
            'raw_question': event.raw_question,
            'normalized_question': event.normalized_question,
            'parser_mode': event.parser_mode,
            'parse_confidence': event.parse_confidence,
            'known_object_count': event.known_object_count,
            'known_structure_count': event.known_structure_count,
            'missing_entities': list(event.missing_entities),
            'candidate_actions': list(event.candidate_actions),
            'selected_action': event.selected_action,
            'selection_reason': event.selection_reason,
            'active_route_index': event.active_route_index,
            'direct_republish_count': event.direct_republish_count,
            'recovery_count': event.recovery_count,
            'ignored_question_count': event.ignored_question_count,
            'time_remaining_seconds': event.time_remaining_seconds,
            'terminal_status': event.terminal_status,
            'details': event.details,
        }
        try:
            encoded = (dumps(record, sort_keys=True) + '\n').encode('utf-8')
        except (TypeError, ValueError):
            with self._lock:
                self._serialization_error_count += 1
            return
        with self._lock:
            if self._bytes_written + len(encoded) > self._max_file_bytes:
                self._dropped_file_event_count += 1
                return
        try:
            output.write(encoded.decode('utf-8'))
            output.flush()
        except OSError:
            with self._lock:
                self._write_error_count += 1
            return
        with self._lock:
            self._bytes_written += len(encoded)
            self._written_event_count += 1


__all__ = [
    'DecisionTraceEvent',
    'InMemoryTraceRecorder',
    'JsonlDecisionTraceRecorder',
    'TRACE_SCHEMA_VERSION',
    'TraceRecorder',
    'TraceRecorderStats',
]
