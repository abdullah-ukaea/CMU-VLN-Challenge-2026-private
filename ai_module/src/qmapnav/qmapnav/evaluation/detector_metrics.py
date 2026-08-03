"""Measured Day 4 detector recall and false-positive metrics."""

from dataclasses import dataclass

from qmapnav.perception.contracts import Detection2D
from qmapnav.perception.contracts import PanoramaBox
from qmapnav.perception.cross_crop_nms import panorama_box_iou


@dataclass(frozen=True)
class VisibleInstance:
    """One manually verified visible panorama object used as ground truth."""

    instance_id: str
    class_name: str
    panorama_box: PanoramaBox
    is_target: bool
    is_anchor: bool
    is_rare: bool
    size_bin: str
    seam_case: bool = False

    def __post_init__(self) -> None:
        if not self.instance_id.strip() or not self.class_name.strip():
            raise ValueError('instance ID and class name must be non-empty')
        if not isinstance(self.panorama_box, PanoramaBox):
            raise TypeError('panorama_box must be PanoramaBox')
        if self.size_bin not in ('small', 'medium', 'large'):
            raise ValueError('size_bin must be small, medium, or large')


@dataclass(frozen=True)
class DetectionMetricCounts:
    """Additive matched/visible counts for one threshold and candidate."""

    visible_instances: int
    matched_instances: int
    target_visible: int
    target_matched: int
    anchor_visible: int
    anchor_matched: int
    rare_visible: int
    rare_matched: int
    small_visible: int
    small_matched: int
    false_positives: int
    prediction_count: int

    def __add__(self, other: object) -> 'DetectionMetricCounts':
        if not isinstance(other, DetectionMetricCounts):
            return NotImplemented
        return DetectionMetricCounts(
            **{
                field: getattr(self, field) + getattr(other, field)
                for field in self.__dataclass_fields__
            }
        )

    @property
    def target_recall(self) -> float:
        """Return target-instance recall, or zero for an empty category."""
        return _ratio(self.target_matched, self.target_visible)

    @property
    def anchor_recall(self) -> float:
        """Return anchor-instance recall, or zero for an empty category."""
        return _ratio(self.anchor_matched, self.anchor_visible)

    @property
    def rare_recall(self) -> float:
        """Return rare-instance recall, or zero for an empty category."""
        return _ratio(self.rare_matched, self.rare_visible)

    @property
    def small_recall(self) -> float:
        """Return small-instance recall, or zero for an empty category."""
        return _ratio(self.small_matched, self.small_visible)


def score_panorama_detections(
    ground_truth: tuple[VisibleInstance, ...],
    predictions: tuple[Detection2D, ...],
    *,
    iou_threshold: float = 0.25,
) -> DetectionMetricCounts:
    """Greedily match same-class predictions to manually verified instances."""
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError('iou_threshold must lie in (0, 1]')
    truth = tuple(ground_truth)
    predicted = tuple(predictions)
    if not all(isinstance(item, VisibleInstance) for item in truth):
        raise TypeError('ground_truth must contain VisibleInstance values')
    if not all(isinstance(item, Detection2D) for item in predicted):
        raise TypeError('predictions must contain Detection2D values')

    candidate_matches = []
    for truth_index, instance in enumerate(truth):
        for prediction_index, detection in enumerate(predicted):
            if instance.class_name != detection.class_name:
                continue
            overlap = panorama_box_iou(instance.panorama_box, detection.panorama_box)
            if overlap >= iou_threshold:
                candidate_matches.append(
                    (overlap, detection.confidence, truth_index, prediction_index)
                )
    matched_truth = set()
    matched_predictions = set()
    for _, _, truth_index, prediction_index in sorted(
        candidate_matches,
        reverse=True,
    ):
        if truth_index in matched_truth or prediction_index in matched_predictions:
            continue
        matched_truth.add(truth_index)
        matched_predictions.add(prediction_index)

    def category_count(attribute: str) -> tuple[int, int]:
        visible_indices = {
            index for index, item in enumerate(truth) if getattr(item, attribute)
        }
        return len(visible_indices), len(visible_indices & matched_truth)

    target_visible, target_matched = category_count('is_target')
    anchor_visible, anchor_matched = category_count('is_anchor')
    rare_visible, rare_matched = category_count('is_rare')
    small_indices = {
        index for index, item in enumerate(truth) if item.size_bin == 'small'
    }
    return DetectionMetricCounts(
        visible_instances=len(truth),
        matched_instances=len(matched_truth),
        target_visible=target_visible,
        target_matched=target_matched,
        anchor_visible=anchor_visible,
        anchor_matched=anchor_matched,
        rare_visible=rare_visible,
        rare_matched=rare_matched,
        small_visible=len(small_indices),
        small_matched=len(small_indices & matched_truth),
        false_positives=len(predicted) - len(matched_predictions),
        prediction_count=len(predicted),
    )


def empty_metric_counts() -> DetectionMetricCounts:
    """Return an additive all-zero accumulator."""
    return DetectionMetricCounts(*([0] * 12))


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator
