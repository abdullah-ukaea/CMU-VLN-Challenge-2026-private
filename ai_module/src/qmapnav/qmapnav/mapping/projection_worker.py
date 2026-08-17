"""Bounded background worker for heavy keyframe projection work."""

from collections.abc import Callable
from queue import Empty
from queue import Full
from queue import Queue
from threading import Event
from threading import Lock
from threading import Thread

from qmapnav.mapping.projection_pipeline import ProjectionFrame
from qmapnav.mapping.timed_buffers import AssociationFailure
from qmapnav.mapping.timed_buffers import TimedPanorama


class BoundedProjectionWorker:
    """Process panoramas off callbacks with deterministic newest-work retention."""

    def __init__(
        self,
        processor: Callable[[TimedPanorama], ProjectionFrame | AssociationFailure],
        result_callback: Callable[[ProjectionFrame | AssociationFailure], None],
        *,
        max_queue_size: int = 2,
    ) -> None:
        if max_queue_size <= 0:
            raise ValueError('max_queue_size must be positive')
        self._processor = processor
        self._result_callback = result_callback
        self._queue: Queue[TimedPanorama | None] = Queue(maxsize=max_queue_size)
        self._stop = Event()
        self._lock = Lock()
        self._processed = 0
        self._dropped = 0
        self._failures = 0
        self._last_error: str | None = None
        self._thread = Thread(target=self._run, name='qmapnav-projection', daemon=True)
        self._thread.start()

    def submit(self, panorama: TimedPanorama) -> None:
        """Submit newest work, dropping the oldest queued keyframe on overload."""
        if self._stop.is_set():
            return
        try:
            self._queue.put_nowait(panorama)
            return
        except Full:
            pass
        try:
            self._queue.get_nowait()
            self._queue.task_done()
            with self._lock:
                self._dropped += 1
        except Empty:
            pass
        try:
            self._queue.put_nowait(panorama)
        except Full:
            with self._lock:
                self._dropped += 1

    def stats(self) -> dict[str, int | str | None]:
        """Return a thread-safe worker diagnostic snapshot."""
        with self._lock:
            return {
                'processed': self._processed,
                'dropped': self._dropped,
                'failures': self._failures,
                'last_error': self._last_error,
                'queued': self._queue.qsize(),
            }

    def close(self, timeout_seconds: float = 2.0) -> bool:
        """Request bounded worker shutdown and report whether it stopped."""
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except Full:
                pass
        self._thread.join(timeout_seconds)
        return not self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                panorama = self._queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                if panorama is None:
                    return
                result = self._processor(panorama)
                self._result_callback(result)
                with self._lock:
                    self._processed += 1
            except Exception as error:
                with self._lock:
                    self._failures += 1
                    self._last_error = str(error)
            finally:
                self._queue.task_done()


__all__ = ['BoundedProjectionWorker']
