"""Prediction-only harness for the bounded two-detector Day 4 bake-off."""

from dataclasses import dataclass

from qmapnav.perception.contracts import CropDetection
from qmapnav.perception.contracts import DetectorClass
from qmapnav.perception.contracts import PerspectiveView
from qmapnav.perception.detector_interface import OpenVocabularyDetector
from qmapnav.perception.detector_interface import validate_candidate_detections


@dataclass(frozen=True)
class DetectorBenchmarkCase:
    """One shared crop set and vocabulary presented to every candidate."""

    image_id: str
    views: tuple[PerspectiveView, ...]
    detector_classes: tuple[DetectorClass, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.image_id, str) or not self.image_id.strip():
            raise ValueError('image_id must be a non-empty string')
        views = tuple(self.views)
        if not views or not all(isinstance(item, PerspectiveView) for item in views):
            raise ValueError('views must contain PerspectiveView values')
        crop_ids = [item.geometry.crop_id for item in views]
        if len(crop_ids) != len(set(crop_ids)):
            raise ValueError('views must have unique crop IDs')
        if any(item.source_image_id != self.image_id for item in views):
            raise ValueError('every view must match the benchmark image_id')
        detector_classes = tuple(self.detector_classes)
        if not detector_classes or not all(
            isinstance(item, DetectorClass) for item in detector_classes
        ):
            raise ValueError('detector_classes must contain DetectorClass values')
        object.__setattr__(self, 'views', views)
        object.__setattr__(self, 'detector_classes', detector_classes)


@dataclass(frozen=True)
class CandidatePredictions:
    """Raw normalized predictions from one candidate on one panorama."""

    candidate_name: str
    image_id: str
    detections_by_crop: tuple[tuple[CropDetection, ...], ...]

    @property
    def detection_count(self) -> int:
        """Return total unmerged detections across all crops."""
        return sum(len(items) for items in self.detections_by_crop)


class TwoCandidateDetectorBenchmark:
    """Apply identical cases to one or two candidates, never more than two."""

    def __init__(
        self,
        candidates: tuple[OpenVocabularyDetector, ...],
    ) -> None:
        candidates = tuple(candidates)
        if not 1 <= len(candidates) <= 2:
            raise ValueError('the detector benchmark requires one or two candidates')
        names = [item.identity.candidate_name for item in candidates]
        if len(names) != len(set(names)):
            raise ValueError('detector candidate names must be unique')
        self._candidates = candidates

    @property
    def candidate_names(self) -> tuple[str, ...]:
        """Return candidates in deterministic execution order."""
        return tuple(item.identity.candidate_name for item in self._candidates)

    def run_case(
        self,
        case: DetectorBenchmarkCase,
        *,
        confidence_threshold: float,
    ) -> tuple[CandidatePredictions, ...]:
        """Collect raw normalized predictions without NMS or metric scoring."""
        if not isinstance(case, DetectorBenchmarkCase):
            raise TypeError('case must be DetectorBenchmarkCase')
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError('confidence_threshold must lie in [0, 1]')
        predictions = []
        for candidate in self._candidates:
            by_crop = []
            for view in case.views:
                detections = candidate.detect(
                    view,
                    case.detector_classes,
                    confidence_threshold=confidence_threshold,
                )
                by_crop.append(
                    validate_candidate_detections(
                        detections,
                        view,
                        case.detector_classes,
                    )
                )
            predictions.append(
                CandidatePredictions(
                    candidate_name=candidate.identity.candidate_name,
                    image_id=case.image_id,
                    detections_by_crop=tuple(by_crop),
                )
            )
        return tuple(predictions)
