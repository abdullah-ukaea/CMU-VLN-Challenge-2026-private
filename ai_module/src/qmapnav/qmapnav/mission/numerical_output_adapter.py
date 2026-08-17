"""Single-publication adapter for the official numerical response."""

from dataclasses import dataclass
from threading import Lock
from typing import Callable

from qmapnav.counting.numerical_result import NumericalResult


OFFICIAL_NUMERICAL_TOPIC = '/numerical_response'


@dataclass(frozen=True)
class NumericalOutputCommit:
    """Trace-ready record of the one official integer response."""

    count: int
    stable: bool
    reason: str


class NumericalOutputAdapter:
    """Convert one final result to Int32 and reject later replacements."""

    def __init__(self, publish: Callable[[object], None]) -> None:
        if not callable(publish):
            raise TypeError('publish must be callable')
        self._publish = publish
        self._commit: NumericalOutputCommit | None = None
        self._lock = Lock()

    @property
    def committed(self) -> bool:
        """Return whether the official response has already been published."""
        with self._lock:
            return self._commit is not None

    @property
    def commitment(self) -> NumericalOutputCommit | None:
        """Return the immutable commitment record when available."""
        with self._lock:
            return self._commit

    def commit(self, result: NumericalResult) -> NumericalOutputCommit:
        """Publish one valid count, including zero, as std_msgs/Int32."""
        if not isinstance(result, NumericalResult):
            raise TypeError('result must be NumericalResult')
        from std_msgs.msg import Int32

        with self._lock:
            if self._commit is not None:
                return self._commit
            message = Int32()
            message.data = result.count
            commit = NumericalOutputCommit(
                result.count,
                result.stable,
                result.stability_reason,
            )
            self._publish(message)
            self._commit = commit
            return commit


__all__ = [
    'OFFICIAL_NUMERICAL_TOPIC',
    'NumericalOutputAdapter',
    'NumericalOutputCommit',
]
