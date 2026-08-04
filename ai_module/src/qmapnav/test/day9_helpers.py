"""Small deterministic geometry fixtures shared by Day 9 tests."""

import numpy as np

from qmapnav.reasoning.candidate_generation import EntityCandidate
from qmapnav.reasoning.support_geometry import SupportGeometry


def geometry(
    entity_id,
    x,
    y,
    *,
    length=0.5,
    width=0.5,
    confidence=0.9,
    semantic_class='chair',
):
    """Create an axis-aligned support geometry."""
    half_length = length / 2.0
    half_width = width / 2.0
    footprint = np.array((
        (x - half_length, y - half_width),
        (x + half_length, y - half_width),
        (x + half_length, y + half_width),
        (x - half_length, y + half_width),
    ))
    return SupportGeometry(
        entity_id,
        semantic_class,
        np.array((x, y, 0.5)),
        np.array((length, width, 1.0)),
        0.0,
        footprint,
        0.0,
        1.0,
        confidence,
        'active',
        'object',
    )


def candidate(entity_id, geom=None, *, class_probability=0.9, colour=0.8):
    """Create a retained reasoning candidate."""
    geom = geom or geometry(entity_id, 0.0, 0.0)
    return EntityCandidate(
        entity_id,
        geom.semantic_class,
        geom.source_type,
        class_probability,
        colour,
        geom.confidence,
        geom,
        True,
        ('test_candidate',),
        {'class_probability': class_probability},
    )
