"""Frozen measured Day 4 detector baseline configuration."""

from pathlib import Path

from qmapnav.perception.crop_generator import eight_view_layout
from qmapnav.perception.crop_generator import PerspectiveCropGenerator
from qmapnav.perception.panorama_projection import PanoramaCameraModel
from qmapnav.perception.perception_worker import PerceptionWorker
from qmapnav.perception.yoloe_detector import YOLOEDetector


DAY4_BASELINE_CHECKPOINT = Path('/home/docker/models/yoloe-11s-seg.pt')
DAY4_BASELINE_CONFIDENCE = 0.20
DAY4_BASELINE_CROSS_CROP_IOU = 0.40


def make_day4_baseline_worker(
    panorama_width: int,
    panorama_height: int,
    *,
    checkpoint: str | Path = DAY4_BASELINE_CHECKPOINT,
    confidence_threshold: float = DAY4_BASELINE_CONFIDENCE,
    cross_crop_iou_threshold: float = DAY4_BASELINE_CROSS_CROP_IOU,
) -> PerceptionWorker:
    """Construct the selected eight-crop compact-YOLOE worker."""
    camera_model = PanoramaCameraModel(panorama_width, panorama_height)
    crop_generator = PerspectiveCropGenerator(
        camera_model,
        eight_view_layout(),
    )
    detector = YOLOEDetector(checkpoint)
    return PerceptionWorker(
        crop_generator,
        detector,
        confidence_threshold=confidence_threshold,
        cross_crop_iou_threshold=cross_crop_iou_threshold,
    )
