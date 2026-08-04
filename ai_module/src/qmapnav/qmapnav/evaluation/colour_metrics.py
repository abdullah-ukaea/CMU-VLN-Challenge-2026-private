"""Held-out colour accuracy, calibration, confusion, and coverage metrics."""

from collections import defaultdict
from dataclasses import dataclass

from qmapnav.reasoning.colour_types import ColourEstimate


@dataclass(frozen=True)
class ColourEvaluationCase:
    """One held-out expected label and observation estimate."""

    case_id: str
    expected_colour: str
    estimate: ColourEstimate


def evaluate_colour_cases(
    cases: list[ColourEvaluationCase] | tuple[ColourEvaluationCase, ...],
) -> dict[str, object]:
    """Return JSON-safe top-k, confusion, calibration, and coverage metrics."""
    total = len(cases)
    valid = [case for case in cases if case.estimate.probabilities]
    top1 = 0
    top2 = 0
    confusion = defaultdict(lambda: defaultdict(int))
    calibration = [{'count': 0, 'correct': 0, 'confidence_sum': 0.0}
                   for _ in range(5)]
    statuses = defaultdict(int)
    for case in cases:
        statuses[case.estimate.status] += 1
        if not case.estimate.probabilities:
            confusion[case.expected_colour]['<failed>'] += 1
            continue
        ranked = sorted(
            case.estimate.probabilities,
            key=lambda name: (-case.estimate.probabilities[name], name),
        )
        predicted = ranked[0]
        correct = predicted == case.expected_colour
        top1 += int(correct)
        top2 += int(case.expected_colour in ranked[:2])
        confusion[case.expected_colour][predicted] += 1
        confidence = case.estimate.probabilities[predicted]
        bin_index = min(4, int(confidence * 5.0))
        calibration[bin_index]['count'] += 1
        calibration[bin_index]['correct'] += int(correct)
        calibration[bin_index]['confidence_sum'] += confidence
    calibration_output = []
    for index, item in enumerate(calibration):
        count = item['count']
        calibration_output.append({
            'lower': index / 5.0,
            'upper': (index + 1) / 5.0,
            'count': count,
            'accuracy': item['correct'] / count if count else None,
            'mean_confidence': (
                item['confidence_sum'] / count if count else None
            ),
        })
    return {
        'case_count': total,
        'valid_count': len(valid),
        'top1_accuracy': top1 / total if total else 0.0,
        'top2_accuracy': top2 / total if total else 0.0,
        'coverage': len(valid) / total if total else 0.0,
        'status_counts': dict(sorted(statuses.items())),
        'confusion_matrix': {
            expected: dict(sorted(predicted.items()))
            for expected, predicted in sorted(confusion.items())
        },
        'calibration': calibration_output,
    }


def expected_calibration_error(metrics: dict[str, object]) -> float:
    """Compute count-weighted absolute calibration error from report bins."""
    bins = metrics['calibration']
    total = sum(item['count'] for item in bins)
    if total == 0:
        return 0.0
    return float(sum(
        item['count'] * abs(item['accuracy'] - item['mean_confidence'])
        for item in bins if item['count']
    ) / total)


__all__ = ['ColourEvaluationCase', 'evaluate_colour_cases',
           'expected_calibration_error']
