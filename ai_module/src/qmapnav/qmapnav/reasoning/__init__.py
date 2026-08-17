"""Deterministic runtime spatial reasoning and semantic geometry."""

from qmapnav.reasoning.semantic_geometry import GateResult
from qmapnav.reasoning.semantic_geometry import make_approach_region
from qmapnav.reasoning.semantic_geometry import make_between_gate
from qmapnav.reasoning.semantic_geometry import make_near_region
from qmapnav.reasoning.semantic_geometry import object_footprint
from qmapnav.reasoning.semantic_geometry import Point2D
from qmapnav.reasoning.semantic_geometry import Polygon2D
from qmapnav.reasoning.semantic_geometry import SemanticRegion


__all__ = [
    'GateResult',
    'Point2D',
    'Polygon2D',
    'SemanticRegion',
    'make_approach_region',
    'make_between_gate',
    'make_near_region',
    'object_footprint',
]
