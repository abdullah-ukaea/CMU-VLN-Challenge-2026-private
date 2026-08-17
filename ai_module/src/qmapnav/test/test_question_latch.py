"""Tests for first-valid-question latching and duplicate suppression."""

import pytest

from qmapnav.mission import QuestionLatch
from qmapnav.mission import QuestionLatchStatus


def test_latch_ignores_empty_question_before_acceptance() -> None:
    latch = QuestionLatch()

    decision = latch.offer('  \n\t ')

    assert decision.status is QuestionLatchStatus.EMPTY
    assert decision.question is None
    assert latch.active_question is None


def test_latch_accepts_only_first_valid_question() -> None:
    latch = QuestionLatch()

    decision = latch.offer('  How many chairs are near the table?  ')

    assert decision.status is QuestionLatchStatus.ACCEPTED
    assert decision.question == 'How many chairs are near the table?'
    assert latch.active_question == decision.question


def test_latch_ignores_repeated_identical_publications() -> None:
    latch = QuestionLatch()
    question = 'Find the potted plant on the file cabinet.'
    accepted = latch.offer(question)

    decisions = [latch.offer(question) for _ in range(5)]

    assert accepted.status is QuestionLatchStatus.ACCEPTED
    assert all(
        decision.status is QuestionLatchStatus.DUPLICATE
        for decision in decisions
    )
    assert latch.active_question == question
    assert latch.duplicate_count == 5
    assert latch.conflict_count == 0


def test_latch_treats_outer_whitespace_as_same_publication() -> None:
    latch = QuestionLatch()
    latch.offer('Go near the plant.')

    decision = latch.offer('  Go near the plant.\n')

    assert decision.status is QuestionLatchStatus.DUPLICATE
    assert latch.duplicate_count == 1


def test_latch_rejects_different_question_without_replacing_active() -> None:
    latch = QuestionLatch()
    first_question = 'How many pillows are on the bed?'
    latch.offer(first_question)

    decision = latch.offer('Find the flowers near the window.')

    assert decision.status is QuestionLatchStatus.CONFLICT
    assert decision.question == first_question
    assert latch.active_question == first_question
    assert latch.conflict_count == 1


def test_latch_requires_string_input() -> None:
    with pytest.raises(TypeError, match='string'):
        QuestionLatch().offer(None)
