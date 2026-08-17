"""Tests for bounded observational JSON decision traces."""

import json
from threading import Event

import pytest

from qmapnav.common.decision_trace import DecisionTraceEvent
from qmapnav.common.decision_trace import InMemoryTraceRecorder
from qmapnav.common.decision_trace import JsonlDecisionTraceRecorder
from qmapnav.common.decision_trace import TRACE_SCHEMA_VERSION


def _event(**overrides: object) -> DecisionTraceEvent:
    values = {
        'event': 'waypoint_activated',
        'episode_elapsed_seconds': 4.2,
        'mission_state': 'active',
        'raw_question': 'Find the chair.',
        'normalized_question': 'find the chair.',
        'parser_mode': 'full',
        'parse_confidence': 1.0,
        'known_object_count': 2,
        'known_structure_count': 1,
        'missing_entities': ('chair_1',),
        'candidate_actions': ('publish_waypoint',),
        'selected_action': 'publish_waypoint',
        'selection_reason': 'previous_waypoint_reached',
        'active_route_index': 1,
        'direct_republish_count': 0,
        'recovery_count': 0,
        'ignored_question_count': 2,
        'time_remaining_seconds': 595.8,
        'details': {'waypoint': {'x': 1.0, 'y': 2.0}},
    }
    values.update(overrides)
    return DecisionTraceEvent(**values)


def test_jsonl_record_contains_versioned_required_fields(tmp_path: object) -> None:
    path = tmp_path / 'trace.jsonl'
    recorder = JsonlDecisionTraceRecorder(path, episode_id='episode-1')

    assert recorder.record(_event())
    assert recorder.close(timeout=1.0)

    record = json.loads(path.read_text().strip())
    assert record['schema_version'] == TRACE_SCHEMA_VERSION == '1.0'
    assert record['episode_id'] == 'episode-1'
    assert record['event'] == 'waypoint_activated'
    assert record['mission_state'] == 'active'
    assert record['raw_question'] == 'Find the chair.'
    assert record['normalized_question'] == 'find the chair.'
    assert record['parser_mode'] == 'full'
    assert record['parse_confidence'] == 1.0
    assert record['known_object_count'] == 2
    assert record['known_structure_count'] == 1
    assert record['missing_entities'] == ['chair_1']
    assert record['candidate_actions'] == ['publish_waypoint']
    assert record['selected_action'] == 'publish_waypoint'
    assert record['active_route_index'] == 1
    assert record['time_remaining_seconds'] == 595.8
    assert record['ignored_question_count'] == 2


def test_file_size_bound_drops_events_without_blocking_control(
    tmp_path: object,
) -> None:
    recorder = JsonlDecisionTraceRecorder(
        tmp_path / 'small.jsonl',
        max_file_bytes=16,
    )

    assert recorder.record(_event())
    assert recorder.close(timeout=1.0)

    stats = recorder.stats()
    assert stats.written_event_count == 0
    assert stats.dropped_file_event_count == 1
    assert stats.bytes_written <= 16


def test_serialization_and_write_failures_are_contained(tmp_path: object) -> None:
    invalid = JsonlDecisionTraceRecorder(tmp_path / 'invalid.jsonl')
    assert invalid.record(_event(details={'not_json': object()}))
    assert invalid.close(timeout=1.0)
    assert invalid.stats().serialization_error_count == 1

    directory_path = tmp_path / 'directory'
    directory_path.mkdir()
    unwritable = JsonlDecisionTraceRecorder(directory_path)
    assert unwritable.record(_event())
    assert unwritable.close(timeout=1.0)
    assert unwritable.stats().write_error_count >= 1


def test_in_memory_recorder_is_bounded_and_close_is_idempotent() -> None:
    recorder = InMemoryTraceRecorder(max_events=1)

    assert recorder.record(_event())
    assert not recorder.record(_event(event='second'))
    assert recorder.close()
    assert recorder.close()
    assert not recorder.record(_event(event='after_close'))
    assert len(recorder.events) == 1
    assert recorder.dropped_event_count == 2


def test_async_queue_rejects_overflow_without_waiting(tmp_path: object) -> None:
    recorder = JsonlDecisionTraceRecorder(
        tmp_path / 'queue.jsonl',
        max_queue_size=1,
    )
    writer_entered = Event()
    release_writer = Event()
    original_write = recorder._write_event

    def blocked_write(output: object, event: DecisionTraceEvent) -> None:
        writer_entered.set()
        release_writer.wait(timeout=1.0)
        original_write(output, event)

    recorder._write_event = blocked_write
    assert recorder.record(_event(event='first'))
    assert writer_entered.wait(timeout=1.0)
    assert recorder.record(_event(event='queued'))
    assert not recorder.record(_event(event='overflow'))
    release_writer.set()

    assert recorder.close(timeout=1.0)
    stats = recorder.stats()
    assert stats.accepted_event_count == 2
    assert stats.dropped_queue_event_count == 1
    assert stats.written_event_count == 2


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('event', ''),
        ('episode_elapsed_seconds', -1.0),
        ('known_object_count', -1),
        ('parse_confidence', 1.1),
        ('active_route_index', -1),
        ('time_remaining_seconds', -0.1),
    ],
)
def test_trace_event_rejects_invalid_required_fields(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        _event(**{field: value})
