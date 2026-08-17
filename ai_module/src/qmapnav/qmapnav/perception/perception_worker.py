"""Thin query-conditioned panorama detector worker."""

from qmapnav.perception.contracts import PerceptionRequest
from qmapnav.perception.contracts import PerceptionResult
from qmapnav.perception.crop_generator import PerspectiveCropGenerator
from qmapnav.perception.cross_crop_nms import cross_crop_nms
from qmapnav.perception.cross_crop_nms import project_crop_detections
from qmapnav.perception.detector_interface import OpenVocabularyDetector


class PerceptionWorker:
    """Generate crops, run one selected detector, and merge crop duplicates."""

    def __init__(
        self,
        crop_generator: PerspectiveCropGenerator,
        detector: OpenVocabularyDetector,
        *,
        confidence_threshold: float,
        cross_crop_iou_threshold: float = 0.4,
    ) -> None:
        if not isinstance(crop_generator, PerspectiveCropGenerator):
            raise TypeError('crop_generator must be PerspectiveCropGenerator')
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError('confidence_threshold must lie in [0, 1]')
        if not 0.0 < cross_crop_iou_threshold <= 1.0:
            raise ValueError('cross_crop_iou_threshold must lie in (0, 1]')
        self._crop_generator = crop_generator
        self._detector = detector
        self._confidence_threshold = float(confidence_threshold)
        self._cross_crop_iou_threshold = float(cross_crop_iou_threshold)

    @property
    def detector_name(self) -> str:
        """Return the selected detector identity without exposing its model."""
        return self._detector.identity.candidate_name

    def process(self, request: PerceptionRequest) -> PerceptionResult:
        """Process one explicit keyframe and return normalized final detections."""
        if not isinstance(request, PerceptionRequest):
            raise TypeError('request must be PerceptionRequest')
        views = self._crop_generator.generate(
            request.panorama_rgb,
            source_image_id=request.image_id,
        )
        raw_detections = []
        for view in views:
            crop_detections = self._detector.detect(
                view,
                request.detector_classes,
                confidence_threshold=self._confidence_threshold,
            )
            raw_detections.extend(
                project_crop_detections(
                    request.image_id,
                    view,
                    crop_detections,
                    self._crop_generator.panorama_model,
                )
            )
        raw = tuple(raw_detections)
        final = cross_crop_nms(
            raw,
            iou_threshold=self._cross_crop_iou_threshold,
        )
        return PerceptionResult(
            image_id=request.image_id,
            timestamp_ns=request.timestamp_ns,
            crop_count=len(views),
            raw_detections=raw,
            detections=final,
        )
