"""Evaluate representative held-out Day 8 support and vertical cases."""

import json

import numpy as np

from qmapnav.evaluation.relation_metrics import evaluate_relation_cases
from qmapnav.evaluation.relation_metrics import RelationEvaluationCase
from qmapnav.reasoning.support_geometry import SupportGeometry
from qmapnav.reasoning.support_relations import on_evidence
from qmapnav.reasoning.vertical_relations import above_evidence


def _geometry(
    entity_id, semantic_class, centre, dimensions, *, confidence=0.95,
    quality='active', source_type='object',
):
    centre = np.asarray(centre, dtype=np.float64)
    dimensions = np.asarray(dimensions, dtype=np.float64)
    half = dimensions[:2] / 2.0
    footprint = np.asarray([
        centre[:2] + [-half[0], -half[1]],
        centre[:2] + [half[0], -half[1]],
        centre[:2] + [half[0], half[1]],
        centre[:2] + [-half[0], half[1]],
    ])
    return SupportGeometry(
        entity_id, semantic_class, centre, dimensions, 0.0, footprint,
        centre[2] - dimensions[2] / 2.0,
        centre[2] + dimensions[2] / 2.0,
        confidence, quality, source_type,
    )


def representative_relation_report() -> dict[str, object]:
    """Score clean, floating, distant, sparse, and structural cases."""
    table = _geometry('table', 'table', (0, 0, 0.4), (1.5, 1.0, 0.8))
    book = _geometry('book', 'book', (0.1, 0, 0.87), (0.3, 0.2, 0.1))
    picture = _geometry('picture', 'picture', (0, 0.2, 2), (0.8, 0.1, 0.8))
    floating = _geometry('floating', 'cup', (0, 0, 1.4), (0.1, 0.1, 0.2))
    distant = _geometry('distant', 'cup', (4, 0, 0.9), (0.1, 0.1, 0.2))
    sparse = _geometry(
        'sparse', 'cup', (0, 0, 0.87), (0.1, 0.1, 0.1),
        confidence=0.35, quality='sparse',
    )
    shelf = _geometry(
        'shelf', 'shelf', (2, 0, 1), (1, 0.3, 0.1),
        source_type='structural',
    )
    shelf_object = _geometry(
        'shelf_object', 'book', (2, 0, 1.1), (0.2, 0.15, 0.1)
    )
    on_cases = [
        RelationEvaluationCase('book_on_table', True, on_evidence(book, table)),
        RelationEvaluationCase(
            'picture_above_not_on', False, on_evidence(picture, table)
        ),
        RelationEvaluationCase(
            'floating_not_on', False, on_evidence(floating, table)
        ),
        RelationEvaluationCase(
            'distant_not_on', False, on_evidence(distant, table)
        ),
        RelationEvaluationCase(
            'sparse_not_confident', False, on_evidence(sparse, table)
        ),
        RelationEvaluationCase(
            'structural_shelf', True, on_evidence(shelf_object, shelf)
        ),
    ]
    vertical_cases = [
        RelationEvaluationCase(
            'book_above_table', True, above_evidence(book, table)
        ),
        RelationEvaluationCase(
            'picture_above_desk', True, above_evidence(picture, table)
        ),
        RelationEvaluationCase(
            'distant_not_above', False, above_evidence(distant, table)
        ),
    ]
    return {
        'provenance': 'manually specified representative map-frame geometry',
        'on': evaluate_relation_cases(on_cases),
        'above': evaluate_relation_cases(vertical_cases),
        'cases': {
            case.case_id: {
                'expected': case.expected,
                'accepted': case.evidence.accepted,
                'confidence': case.evidence.confidence,
                'status': case.evidence.status,
                'gap_m': case.evidence.vertical_gap_m,
                'overlap': case.evidence.subject_support_overlap,
            }
            for case in (*on_cases, *vertical_cases)
        },
    }


def main() -> None:
    """Print the deterministic representative relation report."""
    print(json.dumps(
        representative_relation_report(), indent=2, sort_keys=True
    ))


if __name__ == '__main__':
    main()
