"""Shared persistent-map and numerical-result fixtures for Day 12 tests."""

from day11_helpers import make_candidate
from day11_helpers import make_observation

from qmapnav.counting import assess_anchor_counts
from qmapnav.counting import NumericalResult
from qmapnav.mapping import ObjectMap
from qmapnav.reasoning.colour_types import ColourEstimate


def add_object(
    object_map: ObjectMap,
    detection_id: str,
    class_name: str,
    centre: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    *,
    viewpoint: str = 'view_0',
    timestamp_ns: int = 1,
    colours: dict[str, float] | None = None,
) -> int:
    """Ingest one realistic lifted candidate through ObjectMap."""
    candidate = make_candidate(
        detection_id,
        centre,
        class_name=class_name,
        dimensions=dimensions,
        confidence=1.0,
        timestamp_ns=timestamp_ns,
    )
    instance_id = object_map.add_or_update(
        candidate,
        make_observation(candidate, viewpoint),
    )
    if colours:
        dominant = max(sorted(colours), key=colours.get)
        object_map.update_colour(
            instance_id,
            ColourEstimate(
                colours,
                dominant,
                colours[dominant],
                100,
                None,
                None,
                viewpoint,
                detection_id,
                'good',
            ),
        )
    return instance_id


def numerical_result(
    identifiers: tuple[int, ...],
    *,
    confidence: float = 0.9,
    unresolved: tuple[int, ...] = (),
) -> NumericalResult:
    """Build a minimal valid result for state-machine and protocol tests."""
    return NumericalResult(
        'chair',
        tuple(identifiers),
        (),
        (),
        tuple(unresolved),
        len(identifiers),
        confidence,
        False,
        'awaiting_stability',
        (),
        assess_anchor_counts(()),
    )


__all__ = ['add_object', 'numerical_result']
