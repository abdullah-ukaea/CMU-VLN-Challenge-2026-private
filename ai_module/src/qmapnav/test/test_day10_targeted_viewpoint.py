"""Tests for the bounded Day 10 targeted re-observation policy."""

import pytest

from qmapnav.navigation import decide_targeted_viewpoint
from qmapnav.navigation import EvidenceSufficiency
from qmapnav.navigation import generate_targeted_viewpoints
from qmapnav.navigation import OneViewpointGuard


@pytest.mark.parametrize(
    ('evidence', 'reason'),
    (
        (
            EvidenceSufficiency(0, (), None, None, False, 100.0),
            'no_reliable_target',
        ),
        (
            EvidenceSufficiency(1, ('window_1',), 0.5, 0.8, False, 100.0),
            'required_anchor_missing',
        ),
        (
            EvidenceSufficiency(2, (), 0.02, 0.8, False, 100.0),
            'low_ranking_margin',
        ),
        (
            EvidenceSufficiency(1, (), 0.5, 0.1, False, 100.0),
            'weak_geometry',
        ),
        (
            EvidenceSufficiency(1, (), 0.5, 0.8, True, 100.0),
            'likely_occlusion',
        ),
    ),
)
def test_insufficient_evidence_requests_one_viewpoint(evidence, reason) -> None:
    decision = decide_targeted_viewpoint(evidence)

    assert decision.requested
    assert decision.reason == reason


def test_adequate_or_late_evidence_does_not_request_viewpoint() -> None:
    adequate = EvidenceSufficiency(1, (), 0.5, 0.8, False, 100.0)
    late = EvidenceSufficiency(0, (), None, None, False, 20.0)

    assert not decide_targeted_viewpoint(adequate).requested
    assert not decide_targeted_viewpoint(late).requested


def test_viewpoint_generator_is_safe_bounded_and_deterministic() -> None:
    def safe(x, y):
        return y >= 0.0

    first = generate_targeted_viewpoints(
        (0.0, 0.0, 0.0),
        (3.0, 0.0),
        anchor_missing=False,
        ambiguous=True,
        safe_pose=safe,
    )
    second = generate_targeted_viewpoints(
        (0.0, 0.0, 0.0),
        (3.0, 0.0),
        anchor_missing=False,
        ambiguous=True,
        safe_pose=safe,
    )

    assert first == second
    assert 1 <= len(first) <= 3
    assert all(item.travel_cost_m <= 4.0 for item in first)
    assert all(item.pose_xy_heading[1] >= 0.0 for item in first)


def test_guard_consumes_exactly_one_special_viewpoint_attempt() -> None:
    candidates = generate_targeted_viewpoints(
        (0.0, 0.0, 0.0),
        (3.0, 0.0),
        anchor_missing=True,
        ambiguous=False,
        safe_pose=lambda x, y: True,
    )
    guard = OneViewpointGuard()

    selected = guard.select(candidates)

    assert guard.attempted
    assert selected is guard.selected
    with pytest.raises(RuntimeError, match='already attempted'):
        guard.select(candidates)
