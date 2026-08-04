"""Tests for complete scoring, ambiguity, and downstream adapters."""

from day9_helpers import candidate
from qmapnav.common import EntityReference
from qmapnav.reasoning.ambiguity import assess_ambiguity
from qmapnav.reasoning.candidate_generation import CandidateGenerationResult
from qmapnav.reasoning.hypothesis_scoring import CompleteHypothesis
from qmapnav.reasoning.hypothesis_scoring import rank_complete_hypotheses
from qmapnav.reasoning.reference_resolver import resolve_single_reference
from qmapnav.reasoning.reference_resolver import set_resolution
from qmapnav.reasoning.resolution_contracts import CandidateHypothesis
from qmapnav.reasoning.resolution_contracts import ConstraintEvaluation


def _evaluation(name, score, satisfied, *, hard=False, confidence=0.9):
    return ConstraintEvaluation(
        name, score, hard, satisfied, confidence, {name: score}
    )


def test_hard_violation_cannot_be_hidden_by_soft_scores():
    hypothesis = CompleteHypothesis(
        (('target', 'wrong'),),
        (
            _evaluation('class', 0.0, False, hard=True),
            _evaluation('colour', 1.0, True),
            _evaluation('near', 1.0, True),
            _evaluation('closest', 1.0, True),
        ),
    )
    scored = rank_complete_hypotheses((hypothesis,))[0]
    assert scored.score <= 0.0
    assert scored.evidence['hard_violation_count'] == 1.0


def test_weak_unresolved_evidence_is_recorded():
    hypothesis = CompleteHypothesis(
        (('target', 'chair'),),
        (_evaluation('geometry', 0.1, None, confidence=0.1),),
    )
    scored = rank_complete_hypotheses((hypothesis,))[0]
    assert scored.unresolved_constraints == ('geometry',)
    assert scored.confidence == 0.0


def test_small_margin_is_ambiguous_and_keeps_top_two():
    hypotheses = (
        CandidateHypothesis(('chair_1',), 0.80, 0.9),
        CandidateHypothesis(('chair_2',), 0.76, 0.9),
    )
    result = assess_ambiguity('chair', hypotheses)
    assert result.resolution.resolution_status == 'ambiguous'
    assert result.resolution.selected_candidate_ids is None
    assert len(result.resolution.ranked_hypotheses) == 2


def test_weak_absolute_winner_is_low_confidence():
    result = assess_ambiguity(
        'chair', (CandidateHypothesis(('chair_1',), 0.50, 0.9),)
    )
    assert result.resolution.resolution_status == 'low_confidence'


def test_all_hard_violations_report_conflicting_constraints():
    hypotheses = (
        CandidateHypothesis(
            ('chair_1',), -0.2, 0.9, (), ('class',), (),
            {'hard_violation_count': 1.0},
        ),
    )
    result = assess_ambiguity('chair', hypotheses)
    assert result.resolution.resolution_status == 'conflicting_constraints'


def test_unconstrained_same_class_reference_stays_underconstrained():
    reference = EntityReference('chair_ref', 'chair')
    generated = CandidateGenerationResult(
        'chair_ref', (candidate('chair_1'), candidate('chair_2')), None, True
    )
    result = resolve_single_reference(reference, generated)
    assert result.resolution.resolution_status == 'underconstrained'


def test_set_adapter_uses_only_persistent_candidate_ids():
    generated = CandidateGenerationResult(
        'chairs',
        (
            candidate('persistent_1', class_probability=0.9),
            candidate('persistent_2', class_probability=0.4),
        ),
        None,
        True,
    )
    result = set_resolution(generated)
    assert result.definite_ids == ('persistent_1',)
    assert result.probable_ids == ('persistent_2',)


def test_complete_constraints_defeat_nearest_shortcut():
    nearest_wrong = CompleteHypothesis(
        (('target', 'chair_nearest'),),
        (
            _evaluation('class', 0.95, True, hard=True),
            _evaluation('colour', 0.10, False),
            _evaluation('between', 0.15, False),
            _evaluation('closest', 1.00, True),
        ),
    )
    complete_match = CompleteHypothesis(
        (('target', 'chair_complete'),),
        (
            _evaluation('class', 0.90, True, hard=True),
            _evaluation('colour', 0.90, True),
            _evaluation('between', 0.90, True),
            _evaluation('closest', 0.75, True),
        ),
    )
    ranked = rank_complete_hypotheses((nearest_wrong, complete_match))
    assert ranked[0].candidate_ids == ('chair_complete',)
