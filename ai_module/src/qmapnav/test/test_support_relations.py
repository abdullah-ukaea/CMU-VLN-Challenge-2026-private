"""Held-out geometric cases for colour and relation vertical and support relations."""

import numpy as np

from qmapnav.common import ObjectInstance
from qmapnav.mapping.structural_map import StructuralAnchor
from qmapnav.reasoning.relation_graph import RelationGraph
from qmapnav.reasoning.relation_graph import SpatialRelation
from qmapnav.reasoning.support_geometry import footprint_metrics
from qmapnav.reasoning.support_geometry import support_geometry
from qmapnav.reasoning.support_geometry import SupportGeometry
from qmapnav.reasoning.support_relations import generate_support_candidates
from qmapnav.reasoning.support_relations import on_evidence
from qmapnav.reasoning.support_relations import ranked_support_hypotheses
from qmapnav.reasoning.vertical_relations import above_evidence
from qmapnav.reasoning.vertical_relations import below_evidence
from qmapnav.reasoning.vertical_relations import RelationEvidence


def _geometry(
    entity_id: str,
    semantic_class: str,
    centre: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    *,
    confidence: float = 0.95,
    quality: str = 'active',
    source_type: str = 'object',
) -> SupportGeometry:
    centre_array = np.asarray(centre, dtype=np.float64)
    dimensions_array = np.asarray(dimensions, dtype=np.float64)
    x_value, y_value = centre_array[:2]
    half_x, half_y = dimensions_array[:2] / 2.0
    footprint = np.asarray([
        [x_value - half_x, y_value - half_y],
        [x_value + half_x, y_value - half_y],
        [x_value + half_x, y_value + half_y],
        [x_value - half_x, y_value + half_y],
    ])
    return SupportGeometry(
        entity_id,
        semantic_class,
        centre_array,
        dimensions_array,
        0.0,
        footprint,
        centre_array[2] - dimensions_array[2] / 2.0,
        centre_array[2] + dimensions_array[2] / 2.0,
        confidence,
        quality,
        source_type,
    )


def test_book_on_large_table_uses_subject_overlap_and_graph_inverses() -> None:
    table = _geometry('table_1', 'table', (0.0, 0.0, 0.4), (2.0, 1.0, 0.8))
    book = _geometry('book_1', 'book', (0.4, 0.1, 0.87), (0.3, 0.2, 0.1))

    metrics = footprint_metrics(book, table)
    evidence = on_evidence(book, table)
    graph = RelationGraph()
    graph.recompute([book, table])
    keys = {(edge.relation, edge.subject_id, edge.anchor_id)
            for edge in graph.edges}

    assert np.isclose(metrics.subject_overlap, 1.0)
    assert evidence.accepted
    assert evidence.confidence >= 0.70
    assert ('on', 'book_1', 'table_1') in keys
    assert ('above', 'book_1', 'table_1') in keys
    assert ('below', 'table_1', 'book_1') in keys
    assert ('supports', 'table_1', 'book_1') in keys


def test_picture_is_above_desk_but_not_on_and_inverse_is_exact() -> None:
    desk = _geometry('desk_1', 'desk', (0.0, 0.0, 0.4), (1.4, 0.7, 0.8))
    picture = _geometry(
        'picture_1', 'picture', (0.0, 0.2, 2.0), (0.8, 0.05, 0.8)
    )

    above = above_evidence(picture, desk)
    below = below_evidence(desk, picture)
    contact = on_evidence(picture, desk)

    assert above.accepted
    assert below.accepted == above.accepted
    assert below.confidence == above.confidence
    assert not contact.accepted
    assert contact.status == 'no_contact'


def test_distant_floating_and_sparse_objects_are_not_false_supports() -> None:
    table = _geometry('table_1', 'table', (0.0, 0.0, 0.4), (1.5, 1.0, 0.8))
    distant = _geometry('table_2', 'table', (5.0, 0.0, 0.4), (1.5, 1.0, 0.8))
    floating = _geometry('cup_1', 'cup', (0.0, 0.0, 1.5), (0.1, 0.1, 0.2))
    sparse = _geometry(
        'cup_2', 'cup', (0.1, 0.1, 0.88), (0.1, 0.1, 0.12),
        confidence=0.4, quality='sparse',
    )

    assert distant not in generate_support_candidates(floating, [table, distant])
    assert not on_evidence(floating, table).accepted
    sparse_result = on_evidence(sparse, table)
    assert not sparse_result.accepted
    assert sparse_result.confidence < 0.40


def test_structural_and_object_supports_share_interface() -> None:
    shelf = StructuralAnchor(
        'shelf_a', 'support_surface', 'shelf', np.array([0.0, 0.0, 1.0]),
        None, None, None, np.array([1.0, 0.3, 0.1]), 0.0, None, 0.95,
        1, 1, ('view_a',), ('shelf_detection',),
    )
    object_instance = ObjectInstance(
        7, {'table': 1.0}, {}, np.array([2.0, 0.0, 0.4]),
        np.array([1.0, -0.5, 0.0]), np.array([3.0, 0.5, 0.8]),
        np.array([2.0, 1.0, 0.8]), 0.0, 0.9, 1, 0.9,
    )
    shelf_geometry = support_geometry(shelf)
    table_geometry = support_geometry(object_instance)

    assert shelf_geometry.source_type == 'structural'
    assert shelf_geometry.semantic_class == 'shelf'
    assert table_geometry.source_type == 'object'
    assert table_geometry.semantic_class == 'table'


def test_ranked_multiple_supports_self_rejection_and_graph_recompute() -> None:
    book = _geometry('book', 'book', (0.0, 0.0, 0.86), (0.4, 0.3, 0.1))
    first = _geometry('table_a', 'table', (-0.08, 0.0, 0.4), (1.0, 1.0, 0.8))
    second = _geometry('table_b', 'table', (0.12, 0.0, 0.4), (1.0, 1.0, 0.8))

    hypotheses = ranked_support_hypotheses(book, [first, second])
    assert [item.anchor_id for item in hypotheses] == ['table_a', 'table_b']
    try:
        above_evidence(book, book)
    except ValueError as error:
        assert 'self-relations' in str(error)
    else:
        raise AssertionError('self-relation was accepted')

    graph = RelationGraph()
    graph.recompute([book, first])
    assert any(edge.relation == 'on' for edge in graph.edges)
    raised = _geometry('book', 'book', (0.0, 0.0, 1.4), (0.4, 0.3, 0.1))
    graph.recompute([raised, first])
    assert graph.revision == 2
    assert not any(edge.relation == 'on' for edge in graph.edges)


def test_high_confidence_vertical_contradiction_is_reported() -> None:
    evidence = RelationEvidence(
        'above', 'a', 'b', 0.9, True, 'accepted', 0.1, 1.0, 0.0, 0.9
    )
    graph = RelationGraph()
    graph.add_for_diagnostic(SpatialRelation('above', 'a', 'b', 0.9, evidence))
    inverse_name = RelationEvidence(
        'below', 'a', 'b', 0.9, True, 'accepted', 0.1, 1.0, 0.0, 0.9
    )
    graph.add_for_diagnostic(
        SpatialRelation('below', 'a', 'b', 0.9, inverse_name)
    )

    assert graph.contradictions
