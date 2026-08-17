"""Tests for bounded background projection work and shutdown."""

from threading import Event

import numpy as np

from qmapnav.mapping.projection_worker import BoundedProjectionWorker
from qmapnav.mapping.timed_buffers import AssociationFailure
from qmapnav.mapping.timed_buffers import TimedPanorama


def _panorama(timestamp: int) -> TimedPanorama:
    return TimedPanorama(
        image_id=str(timestamp),
        timestamp_ns=timestamp,
        frame_id='camera',
        image_rgb=np.zeros((4, 8, 3), dtype=np.uint8),
    )


def test_worker_processes_off_caller_and_stops_cleanly() -> None:
    completed = Event()
    results = []

    def process(panorama):
        return AssociationFailure(panorama, 'test')

    def callback(result):
        results.append(result)
        completed.set()

    worker = BoundedProjectionWorker(process, callback, max_queue_size=1)
    worker.submit(_panorama(1))

    assert completed.wait(1.0)
    assert results[0].panorama.timestamp_ns == 1
    assert worker.close(1.0)
    assert worker.stats()['processed'] == 1


def test_worker_records_processing_failures_without_crashing() -> None:
    called = Event()

    def fail(_panorama):
        called.set()
        raise RuntimeError('projection failed')

    worker = BoundedProjectionWorker(fail, lambda _result: None)
    worker.submit(_panorama(1))
    assert called.wait(1.0)
    worker.close(1.0)

    assert worker.stats()['failures'] == 1
    assert worker.stats()['last_error'] == 'projection failed'


def test_worker_overload_drops_oldest_queued_image_and_keeps_newest() -> None:
    started = Event()
    release = Event()
    completed = Event()
    results = []

    def process(panorama):
        if panorama.timestamp_ns == 1:
            started.set()
            release.wait(1.0)
        return AssociationFailure(panorama, 'test')

    def callback(result):
        results.append(result.panorama.timestamp_ns)
        if len(results) == 2:
            completed.set()

    worker = BoundedProjectionWorker(process, callback, max_queue_size=1)
    worker.submit(_panorama(1))
    assert started.wait(1.0)
    worker.submit(_panorama(2))
    worker.submit(_panorama(3))
    release.set()

    assert completed.wait(1.0)
    assert worker.close(1.0)
    assert results == [1, 3]
    assert worker.stats()['dropped'] == 1
