"""Precision, recall, false-support, and geometry-quality relation metrics."""

from dataclasses import dataclass

from qmapnav.reasoning.vertical_relations import RelationEvidence


@dataclass(frozen=True)
class RelationEvaluationCase:
    """One representative labelled relation hypothesis."""

    case_id: str
    expected: bool
    evidence: RelationEvidence


def evaluate_relation_cases(
    cases: list[RelationEvaluationCase] | tuple[RelationEvaluationCase, ...],
) -> dict[str, object]:
    """Aggregate relation performance and confidence by geometry quality."""
    true_positive = false_positive = true_negative = false_negative = 0
    quality = {'strong': [], 'weak': []}
    for case in cases:
        predicted = case.evidence.accepted
        true_positive += int(predicted and case.expected)
        false_positive += int(predicted and not case.expected)
        true_negative += int(not predicted and not case.expected)
        false_negative += int(not predicted and case.expected)
        key = 'strong' if case.evidence.geometry_confidence >= 0.70 else 'weak'
        quality[key].append(case.evidence.confidence)
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    negative_count = true_negative + false_positive
    return {
        'case_count': len(cases),
        'true_positive': true_positive,
        'false_positive': false_positive,
        'true_negative': true_negative,
        'false_negative': false_negative,
        'precision': (
            true_positive / precision_denominator
            if precision_denominator else 0.0
        ),
        'recall': (
            true_positive / recall_denominator if recall_denominator else 0.0
        ),
        'false_support_rate': (
            false_positive / negative_count if negative_count else 0.0
        ),
        'mean_confidence_by_geometry': {
            key: sum(values) / len(values) if values else None
            for key, values in quality.items()
        },
    }


__all__ = ['RelationEvaluationCase', 'evaluate_relation_cases']
