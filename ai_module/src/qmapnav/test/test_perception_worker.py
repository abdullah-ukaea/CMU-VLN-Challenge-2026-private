"""Integration tests for normalized detector output from the perception worker."""

import numpy as np

from qmapnav.language import parse_question
from qmapnav.perception import CropDetection
from qmapnav.perception import detector_classes_from_task_specification
from qmapnav.perception import DetectorClass
from qmapnav.perception import DetectorIdentity
from qmapnav.perception import eight_view_layout
from qmapnav.perception import PanoramaCameraModel
from qmapnav.perception import PerceptionRequest
from qmapnav.perception import PerceptionWorker
from qmapnav.perception import PerspectiveCropGenerator


class _CentreDetector:
    identity = DetectorIdentity('fake', 'test', 'none', '1')

    def detect(
        self,
        view,
        detector_classes,
        *,
        confidence_threshold,
    ):
        del confidence_threshold
        if view.geometry.crop_id not in (0, 1):
            return ()
        boxes = {
            0: (0.0, 250.0, 240.0, 450.0),
            1: (400.0, 250.0, 640.0, 450.0),
        }
        return (
            CropDetection(
                view.geometry.crop_id,
                detector_classes[0].canonical_name,
                detector_classes[0].prompts[0],
                0.9 - view.geometry.crop_id * 0.1,
                boxes[view.geometry.crop_id],
            ),
        )


def test_worker_emits_detector_independent_panorama_detections() -> None:
    panorama = np.zeros((640, 1920, 3), dtype=np.uint8)
    request = PerceptionRequest(
        image_id='frame_1',
        timestamp_ns=123,
        panorama_rgb=panorama,
        detector_classes=(DetectorClass('chair', ('chair',)),),
        task_type='object_reference',
    )
    worker = PerceptionWorker(
        PerspectiveCropGenerator(
            PanoramaCameraModel(1920, 640),
            eight_view_layout(),
        ),
        _CentreDetector(),
        confidence_threshold=0.2,
        cross_crop_iou_threshold=0.1,
    )

    result = worker.process(request)

    assert result.image_id == request.image_id
    assert result.timestamp_ns == request.timestamp_ns
    assert result.crop_count == 8
    assert len(result.raw_detections) == 2
    assert len(result.detections) == 1
    assert result.detections[0].crop_ids == (0, 1)
    assert np.isclose(np.linalg.norm(result.detections[0].centre_camera_ray), 1.0)


def test_real_parsed_question_builds_query_conditioned_vocabulary() -> None:
    task = parse_question(
        'Find the orange chair between the table and sink '
        'that is closest to the window.'
    )

    classes = detector_classes_from_task_specification(
        task,
        {'chair': ('seat',)},
    )

    assert tuple(item.canonical_name for item in classes) == (
        'chair',
        'table',
        'sink',
        'window',
    )
    assert classes[0].prompts == ('chair', 'seat')
