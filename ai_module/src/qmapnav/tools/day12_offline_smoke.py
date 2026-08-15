#!/usr/bin/env python3
"""Network-independent parser, detector, and numerical-answer smoke test."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np


def main() -> None:
    """Exercise all runtime assets and produce a synthetic count of zero."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path('/home/docker/models/yoloe-11s-seg.pt'),
    )
    parser.add_argument('--device', default='cuda:0')
    arguments = parser.parse_args()
    started = perf_counter()

    from qmapnav.counting import CountStabilityConfig
    from qmapnav.language import parse_question
    from qmapnav.mapping import ObjectMap
    from qmapnav.mapping import StructuralMap
    from qmapnav.mission.numerical_episode import NumericalEpisodeCoordinator
    from qmapnav.mission.numerical_output_adapter import NumericalOutputAdapter
    from qmapnav.perception import detector_classes_from_task_specification
    from qmapnav.perception import make_day4_baseline_worker
    from qmapnav.perception import PerceptionRequest

    question = 'How many cups are on the coffee table?'
    task = parse_question(question)
    parsed_at = perf_counter()
    worker = make_day4_baseline_worker(
        1920,
        640,
        checkpoint=arguments.checkpoint,
    )
    model_loaded_at = perf_counter()
    perception = worker.process(PerceptionRequest(
        image_id='synthetic_empty_panorama',
        timestamp_ns=0,
        panorama_rgb=np.zeros((640, 1920, 3), dtype=np.uint8),
        detector_classes=detector_classes_from_task_specification(task),
        task_type=task.task_type,
        viewpoint_id='synthetic_origin',
    ))
    detector_finished_at = perf_counter()

    coordinator = NumericalEpisodeCoordinator(
        stability_config=CountStabilityConfig(
            required_consecutive_updates=1,
            required_independent_viewpoints=1,
        )
    )
    coordinator.start(task)
    action = coordinator.force_commit(
        ObjectMap(),
        StructuralMap(),
        reason='offline_synthetic_empty_scene',
    )
    messages = []
    adapter = NumericalOutputAdapter(messages.append)
    commitment = adapter.commit(action.result)
    coordinator.notify_published()
    finished = perf_counter()
    passed = (
        task.task_type == 'numerical'
        and perception.crop_count == 8
        and commitment.count == 0
        and len(messages) == 1
        and messages[0].data == 0
    )
    report = {
        'passed': passed,
        'network_requirement': 'none',
        'question': question,
        'parse_mode': task.parse_mode,
        'detector': worker.detector_name,
        'detector_crop_count': perception.crop_count,
        'detector_raw_detections': len(perception.raw_detections),
        'detector_final_detections': len(perception.detections),
        'synthetic_expected_count': 0,
        'published_count': commitment.count,
        'official_publication_count': len(messages),
        'timing_sec': {
            'parse': parsed_at - started,
            'model_load': model_loaded_at - parsed_at,
            'detector': detector_finished_at - model_loaded_at,
            'answer': finished - detector_finished_at,
            'cold_start_to_answer': finished - started,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
