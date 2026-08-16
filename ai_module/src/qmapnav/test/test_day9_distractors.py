"""Distractor-heavy complete-hypothesis and diagnostic integration tests."""

import csv
import json

from day9_helpers import candidate, geometry
from qmapnav.mapping.grid_planning import PlanningGrid
from qmapnav.reasoning.ambiguity import assess_ambiguity
from qmapnav.reasoning.corridor_evaluation import CorridorConfig
from qmapnav.reasoning.corridor_evaluation import evaluate_corridor
from qmapnav.reasoning.hypothesis_scoring import enumerate_complete_hypotheses
from qmapnav.reasoning.hypothesis_scoring import rank_complete_hypotheses
from qmapnav.reasoning.reasoning_visualisation import save_reasoning_diagnostics
from qmapnav.reasoning.resolution_contracts import CandidateHypothesis
from qmapnav.reasoning.resolution_contracts import ConstraintEvaluation
from qmapnav.reasoning.spatial_relations import evaluate_between
from qmapnav.reasoning.spatial_relations import rank_distances


def _constraint(name, score, satisfied, confidence=0.9, hard=False):
    return ConstraintEvaluation(
        name, score, hard, satisfied, confidence, {name: score}
    )


def _scene_candidates():
    chairs = (
        candidate(
            'chair_correct',
            geometry('chair_correct', 2.0, 0.0),
            colour=0.95,
        ),
        candidate(
            'chair_nearest',
            geometry('chair_nearest', 7.0, 0.0),
            colour=0.10,
        ),
        candidate(
            'chair_blue',
            geometry('chair_blue', 2.0, 1.2),
            colour=0.05,
        ),
    )
    tables = (
        candidate(
            'table_true',
            geometry('table_true', 0.0, 0.0, semantic_class='table'),
        ),
        candidate(
            'table_distractor',
            geometry('table_distractor', 10.0, 0.0,
                     semantic_class='table'),
            class_probability=0.20,
        ),
    )
    sinks = (
        candidate(
            'sink_true',
            geometry('sink_true', 4.0, 0.0, semantic_class='sink'),
        ),
        candidate(
            'sink_distractor',
            geometry('sink_distractor', 12.0, 0.0,
                     semantic_class='sink'),
            class_probability=0.20,
        ),
    )
    windows = (
        candidate(
            'window_near_wrong',
            geometry('window_near_wrong', 7.0, 0.8,
                     semantic_class='window'),
            class_probability=0.20,
        ),
        candidate(
            'window_true',
            geometry('window_true', 2.0, 2.0, semantic_class='window'),
        ),
    )
    return chairs, tables, sinks, windows


def test_complete_chair_table_sink_window_product_selects_intersection():
    chairs, tables, sinks, windows = _scene_candidates()
    distance = rank_distances(
        tuple(item.geometry for item in chairs),
        tuple(item.geometry for item in windows),
        'closest',
    )
    distance_scores = {
        (item.target_id, item.anchor_id): item.score
        for item in distance.ranked
    }

    def evaluate(roles):
        target = roles['target']
        table = roles['table']
        sink = roles['sink']
        window = roles['window']
        between = evaluate_between(
            target.geometry, table.geometry, sink.geometry
        )
        closest = distance_scores[(target.candidate_id, window.candidate_id)]
        return (
            _constraint('target_class', target.class_probability, True,
                        hard=True),
            _constraint('table_class', table.class_probability, True,
                        hard=True),
            _constraint('sink_class', sink.class_probability, True, hard=True),
            _constraint('window_class', window.class_probability, True,
                        hard=True),
            _constraint(
                'colour', target.colour_probability,
                target.colour_probability >= 0.10,
            ),
            between,
            _constraint('closest', closest, closest >= 0.25),
        )

    complete = enumerate_complete_hypotheses({
        'target': chairs,
        'table': tables,
        'sink': sinks,
        'window': windows,
    }, evaluate)
    assert len(complete) == 24
    ranked = rank_complete_hypotheses(complete, weights={
        'colour': 1.5,
        'between': 2.0,
        'closest': 1.0,
    })
    assert 'chair_correct' in ranked[0].candidate_ids
    assert 'table_true' in ranked[0].candidate_ids
    assert 'sink_true' in ranked[0].candidate_ids
    assert ranked[0].score > ranked[1].score


def test_missing_anchor_preserves_partial_target_and_unresolved_reason():
    hypothesis = (
        _constraint('class', 0.95, True, hard=True),
        _constraint('colour', 0.90, True),
        _constraint('window_anchor', 0.0, None, confidence=0.0),
    )
    from qmapnav.reasoning.hypothesis_scoring import CompleteHypothesis
    complete = CompleteHypothesis((('target', 'chair_1'),), hypothesis)
    result = assess_ambiguity(
        'chair_ref', rank_complete_hypotheses((complete,))
    )
    assert result.resolution.ranked_hypotheses
    assert 'window_anchor' in result.resolution.unresolved_constraints
    assert result.resolution.selected_candidate_ids is None


def test_diagnostics_save_score_trace_pairs_and_top_down_view(tmp_path):
    chairs, tables, _, windows = _scene_candidates()
    ranking = rank_distances(
        tuple(item.geometry for item in chairs[:2]),
        tuple(item.geometry for item in windows),
        'closest',
    )
    hypotheses = (
        # Scores deliberately resolve with a visible top-two margin.
        CandidateHypothesis(('chair_correct',), 0.90, 0.90),
        CandidateHypothesis(('chair_nearest',), 0.60, 0.85),
    )
    assessment = assess_ambiguity('chair_ref', hypotheses)
    grid = PlanningGrid(0.1, (-5.0, -5.0), 200, 200, frozenset())
    corridor = evaluate_corridor(
        tables[0].geometry,
        geometry('table_gate', 2.0, -2.0, semantic_class='table'),
        grid,
        CorridorConfig(robot_width_m=0.5),
    )
    entities = tuple(
        item.geometry for item in chairs + tables + windows
    ) + (geometry('table_gate', 2.0, -2.0, semantic_class='table'),)
    paths = save_reasoning_diagnostics(
        tmp_path,
        entities,
        assessment,
        corridors=(corridor,),
        distance_ranking=ranking,
    )
    assert paths.top_down_png.stat().st_size > 0
    with paths.candidate_scores_csv.open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]['candidate_ids'] == 'chair_correct'
    resolution = json.loads(paths.resolution_json.read_text())
    pairs = json.loads(paths.corridor_json.read_text())
    assert resolution['event'] == 'reference_resolution'
    assert resolution['normalized_margin'] > 0.0
    assert pairs['pairs_considered'] == 1
