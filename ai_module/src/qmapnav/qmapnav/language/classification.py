"""Deterministic challenge-question task classification."""

import re


NUMERICAL = 'numerical'
OBJECT_REFERENCE = 'object_reference'
INSTRUCTION_FOLLOWING = 'instruction_following'

_NUMERICAL_PATTERN = re.compile(
    r'^(?:how\s+many\b|what\s+number\s+of\b|'
    r'count\b(?:\s+the\s+number\s+of\b)?)',
    re.IGNORECASE,
)
_OBJECT_REFERENCE_PATTERN = re.compile(
    r'^(?:find\b|locate\b|identify\b|(?:the|a|an)\b)',
    re.IGNORECASE,
)
_ROUTE_ACTION_PATTERN = re.compile(
    r'\b(?:go|stop|finish|take\s+the\s+path|pass\s+(?:by|between)|'
    r'avoid(?:ing)?|stay\s+away|without\s+passing)\b',
    re.IGNORECASE,
)


def classify_task_type(question: str) -> str:
    """
    Classify a non-empty challenge question into its frozen task family.

    Classification deliberately uses command structure instead of relying only
    on released leading words. In particular, several released object-reference
    statements are noun phrases beginning with ``The`` rather than ``Find``.
    """
    if not isinstance(question, str):
        raise TypeError('question must be a string')

    normalized = ' '.join(question.split())
    if not normalized:
        raise ValueError('question must not be empty')

    if _NUMERICAL_PATTERN.search(normalized):
        return NUMERICAL
    if _ROUTE_ACTION_PATTERN.search(normalized):
        return INSTRUCTION_FOLLOWING
    if _OBJECT_REFERENCE_PATTERN.search(normalized):
        return OBJECT_REFERENCE

    raise ValueError('question does not match a supported task type')


__all__ = [
    'INSTRUCTION_FOLLOWING',
    'NUMERICAL',
    'OBJECT_REFERENCE',
    'classify_task_type',
]
