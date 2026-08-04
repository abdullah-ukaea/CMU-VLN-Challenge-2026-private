"""Tests for footprint-aware Day 9 spatial relations."""

from day9_helpers import geometry
import pytest

from qmapnav.reasoning.spatial_relations import evaluate_between
from qmapnav.reasoning.spatial_relations import evaluate_near
from qmapnav.reasoning.spatial_relations import measure_distance
from qmapnav.reasoning.spatial_relations import rank_distances


def test_near_is_symmetric():
    first = geometry('a', 0.0, 0.0)
    second = geometry('b', 1.0, 0.0)
    forward = evaluate_near(first, second)
    reverse = evaluate_near(second, first)
    assert forward.score == pytest.approx(reverse.score)
    assert forward.evidence == reverse.evidence


def test_near_is_scale_aware():
    small_a = geometry('small_a', 0.0, 0.0, length=0.2, width=0.2)
    small_b = geometry('small_b', 1.0, 0.0, length=0.2, width=0.2)
    large_a = geometry('large_a', 0.0, 0.0, length=2.0, width=2.0)
    large_b = geometry('large_b', 1.0, 0.0, length=2.0, width=2.0)
    assert evaluate_near(large_a, large_b).evidence[
        'near_threshold_m'
    ] > evaluate_near(small_a, small_b).evidence['near_threshold_m']


def test_reliable_footprint_distance_is_preferred():
    first = geometry('sofa', 0.0, 0.0, length=4.0, width=1.0)
    second = geometry('table', 3.0, 0.0, length=1.0, width=1.0)
    measurement = measure_distance(first, second)
    assert measurement.xy_distance_m == pytest.approx(3.0)
    assert measurement.selected_distance_m == pytest.approx(0.5)
    assert measurement.used_footprints


def test_weak_geometry_falls_back_to_centre_distance():
    first = geometry('a', 0.0, 0.0, confidence=0.1)
    second = geometry('b', 1.0, 0.0)
    measurement = measure_distance(first, second)
    assert not measurement.used_footprints
    assert measurement.selected_distance_m == measurement.xy_distance_m


def test_closest_and_farthest_exhaust_all_targets_and_anchors():
    targets = (geometry('t1', 0.0, 0.0), geometry('t2', 5.0, 0.0))
    anchors = (geometry('a1', 1.0, 0.0), geometry('a2', 9.0, 0.0))
    closest = rank_distances(targets, anchors, 'closest')
    farthest = rank_distances(targets, anchors, 'farthest')
    assert len(closest.ranked) == 4
    assert closest.ranked[0].target_id == 't1'
    assert closest.ranked[0].anchor_id == 'a1'
    assert farthest.ranked[0].target_id == 't1'
    assert farthest.ranked[0].anchor_id == 'a2'


def test_distance_tie_has_zero_margin():
    targets = (geometry('left', -1.0, 0.0), geometry('right', 1.0, 0.0))
    ranking = rank_distances(targets, (geometry('anchor', 0.0, 0.0),),
                             'closest')
    assert ranking.raw_margin == pytest.approx(0.0)
    assert ranking.normalized_margin == pytest.approx(0.0)


def test_subject_beyond_endpoint_is_not_between():
    result = evaluate_between(
        geometry('subject', 3.0, 0.0),
        geometry('first', 0.0, 0.0),
        geometry('second', 2.0, 0.0),
    )
    assert result.satisfied is False
    assert result.evidence['projection_t'] > 1.0


def test_centred_subject_has_stronger_between_score():
    first = geometry('first', 0.0, 0.0)
    second = geometry('second', 4.0, 0.0)
    centred = evaluate_between(geometry('centre', 2.0, 0.0), first, second)
    offset = evaluate_between(geometry('offset', 1.0, 0.8), first, second)
    assert centred.satisfied is True
    assert centred.score > offset.score
