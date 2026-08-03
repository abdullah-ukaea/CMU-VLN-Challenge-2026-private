"""Deterministic spatial reasoning and semantic task resolution."""

from qmapnav.reasoning.oracle import CandidateDecision
from qmapnav.reasoning.oracle import EntityResolution
from qmapnav.reasoning.oracle import geometric_relation_holds
from qmapnav.reasoning.oracle import NumericalResult
from qmapnav.reasoning.oracle import ObjectReferenceResult
from qmapnav.reasoning.oracle import OracleReasoningError
from qmapnav.reasoning.oracle import resolve_task_entities
from qmapnav.reasoning.oracle import solve_numerical
from qmapnav.reasoning.oracle import solve_object_reference
from qmapnav.reasoning.route_planner import build_planning_grid
from qmapnav.reasoning.route_planner import OraclePlannerConfig
from qmapnav.reasoning.route_planner import OracleRoutePlan
from qmapnav.reasoning.route_planner import plan_semantic_route
from qmapnav.reasoning.route_planner import PlanningGrid
from qmapnav.reasoning.route_planner import RoutePlanningError
from qmapnav.reasoning.semantic_geometry import GateResult
from qmapnav.reasoning.semantic_geometry import make_approach_region
from qmapnav.reasoning.semantic_geometry import make_between_gate
from qmapnav.reasoning.semantic_geometry import make_near_region
from qmapnav.reasoning.semantic_geometry import object_footprint
from qmapnav.reasoning.semantic_geometry import Point2D
from qmapnav.reasoning.semantic_geometry import Polygon2D
from qmapnav.reasoning.semantic_geometry import SemanticRegion


__all__ = [
    'CandidateDecision',
    'EntityResolution',
    'GateResult',
    'geometric_relation_holds',
    'NumericalResult',
    'ObjectReferenceResult',
    'OraclePlannerConfig',
    'OracleReasoningError',
    'OracleRoutePlan',
    'PlanningGrid',
    'Point2D',
    'Polygon2D',
    'RoutePlanningError',
    'SemanticRegion',
    'build_planning_grid',
    'make_approach_region',
    'make_between_gate',
    'make_near_region',
    'object_footprint',
    'plan_semantic_route',
    'resolve_task_entities',
    'solve_numerical',
    'solve_object_reference',
]
