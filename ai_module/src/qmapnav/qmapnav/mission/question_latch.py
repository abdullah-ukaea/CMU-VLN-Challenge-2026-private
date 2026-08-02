"""First-valid-question latching for one competition process episode."""

from dataclasses import dataclass
from enum import Enum


class QuestionLatchStatus(Enum):
    """Outcome of offering one question publication to the latch."""

    ACCEPTED = 'accepted'
    EMPTY = 'empty'
    DUPLICATE = 'duplicate'
    CONFLICT = 'conflict'


@dataclass(frozen=True)
class QuestionLatchDecision:
    """Immutable result of one question-latch decision."""

    status: QuestionLatchStatus
    question: str | None


class QuestionLatch:
    """Accept only the first valid question during one process episode."""

    def __init__(self) -> None:
        self._active_question: str | None = None
        self._duplicate_count = 0
        self._conflict_count = 0

    @property
    def active_question(self) -> str | None:
        """Return the accepted question, or ``None`` before acceptance."""
        return self._active_question

    @property
    def duplicate_count(self) -> int:
        """Return the number of ignored identical republications."""
        return self._duplicate_count

    @property
    def conflict_count(self) -> int:
        """Return the number of ignored different questions after latching."""
        return self._conflict_count

    def offer(self, question: str) -> QuestionLatchDecision:
        """Offer raw text and return a decision without restarting an episode."""
        if not isinstance(question, str):
            raise TypeError('question must be a string')

        candidate = question.strip()
        if not candidate:
            return QuestionLatchDecision(QuestionLatchStatus.EMPTY, None)

        if self._active_question is None:
            self._active_question = candidate
            return QuestionLatchDecision(
                QuestionLatchStatus.ACCEPTED,
                self._active_question,
            )

        if candidate == self._active_question:
            self._duplicate_count += 1
            return QuestionLatchDecision(
                QuestionLatchStatus.DUPLICATE,
                self._active_question,
            )

        self._conflict_count += 1
        return QuestionLatchDecision(
            QuestionLatchStatus.CONFLICT,
            self._active_question,
        )


__all__ = [
    'QuestionLatch',
    'QuestionLatchDecision',
    'QuestionLatchStatus',
]
