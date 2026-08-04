"""Shared released and query-facing colour vocabulary for Day 8."""

import re


COLOUR_CLASSES = (
    'black',
    'blue',
    'brown',
    'green',
    'grey',
    'orange',
    'pink',
    'purple',
    'red',
    'white',
    'yellow',
)

RELEASED_COLOUR_CLASSES = (
    'aqua',
    'black',
    'blue',
    'brown',
    'green',
    'grey',
    'maroon',
    'navy',
    'olive',
    'orange',
    'pink',
    'purple',
    'red',
    'yellow',
)

COLOUR_ALIASES = {
    'aqua': 'blue',
    'gray': 'grey',
    'light_blue': 'blue',
    'maroon': 'red',
    'navy': 'blue',
    'olive': 'green',
}


def normalize_colour_name(value: str) -> str:
    """Normalize a colour token to the query-facing canonical vocabulary."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError('colour name must be a non-empty string')
    token = re.sub(r'[^a-z0-9]+', '_', value.strip().casefold()).strip('_')
    canonical = COLOUR_ALIASES.get(token, token)
    if canonical not in COLOUR_CLASSES:
        raise ValueError(f'unsupported colour name {value!r}')
    return canonical


def normalize_released_colour_name(value: str) -> str:
    """Normalize spelling while retaining released palette distinctions."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError('released colour name must be non-empty')
    token = re.sub(r'[^a-z0-9]+', '_', value.strip().casefold()).strip('_')
    token = 'grey' if token == 'gray' else token
    if token not in RELEASED_COLOUR_CLASSES:
        raise ValueError(f'unsupported released colour name {value!r}')
    return token


__all__ = [
    'COLOUR_ALIASES',
    'COLOUR_CLASSES',
    'normalize_colour_name',
    'normalize_released_colour_name',
    'RELEASED_COLOUR_CLASSES',
]
