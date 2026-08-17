"""Small deterministic fixtures shared by the domain test suite."""

import numpy as np

from qmapnav.common import ObjectInstance
from qmapnav.counting import assess_anchor_counts
from qmapnav.counting import NumericalResult
from qmapnav.mapping import ObjectMap
from qmapnav.mapping.object_candidate import ConfidenceComponents
from qmapnav.mapping.object_candidate import GeometrySource
from qmapnav.mapping.object_candidate import GeometryStatus
from qmapnav.mapping.object_candidate import LiftingCounts
from qmapnav.mapping.object_candidate import ObjectCandidate3D
from qmapnav.mapping.occupancy_grid import CELL_FREE
from qmapnav.mapping.occupancy_grid import CELL_OCCUPIED
from qmapnav.mapping.occupancy_grid import OccupancyGrid2D
from qmapnav.mapping.viewpoint_observation import ViewpointObservation
from qmapnav.reasoning.candidate_generation import EntityCandidate
from qmapnav.reasoning.colour_types import ColourEstimate
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


def support_geometry(
    entity_id: str,
    semantic_class: str,
    centre: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    *,
    confidence: float = 0.95,
    quality: str = 'active',
    source_type: str = 'object',
) -> SupportGeometry:
    """Create metric geometry for relation and colour-fusion tests."""
    centre_array = np.asarray(centre, dtype=np.float64)
    dimensions_array = np.asarray(dimensions, dtype=np.float64)
    x_value, y_value = centre_array[:2]
    half_x, half_y = dimensions_array[:2] / 2.0
    footprint = np.asarray([
        [x_value - half_x, y_value - half_y],
        [x_value + half_x, y_value - half_y],
        [x_value + half_x, y_value + half_y],
        [x_value - half_x, y_value + half_y],
    ])
    return SupportGeometry(
        entity_id,
        semantic_class,
        centre_array,
        dimensions_array,
        0.0,
        footprint,
        centre_array[2] - dimensions_array[2] / 2.0,
        centre_array[2] + dimensions_array[2] / 2.0,
        confidence,
        quality,
        source_type,
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


def open_grid(
    *,
    half_extent: float = 8.0,
    resolution: float = 0.25,
    known: bool = True,
) -> OccupancyGrid2D:
    """Return a square grid centred on the origin, free or wholly unknown."""
    size = int(round(2 * half_extent / resolution))
    grid = OccupancyGrid2D(
        resolution, (-half_extent, -half_extent), size, size
    )
    if known:
        grid.fill_rectangle(
            (-half_extent, -half_extent, half_extent, half_extent), CELL_FREE
        )
    return grid


def add_wall(
    grid: OccupancyGrid2D,
    bounds: tuple[float, float, float, float],
) -> None:
    """Mark a metric rectangle as occupied."""
    grid.fill_rectangle(bounds, CELL_OCCUPIED)


def add_unknown(
    grid: OccupancyGrid2D,
    bounds: tuple[float, float, float, float],
) -> None:
    """Mark a metric rectangle as unobserved."""
    from qmapnav.mapping.occupancy_grid import CELL_UNKNOWN

    grid.fill_rectangle(bounds, CELL_UNKNOWN)


def make_instance(
    instance_id: int,
    class_name: str,
    centre: tuple[float, float, float],
    dimensions: tuple[float, float, float] = (0.6, 0.6, 0.8),
    *,
    yaw: float = 0.0,
    orientation_confidence: float = 0.9,
    confidence: float = 0.9,
    observation_count: int = 3,
    colour_scores: dict | None = None,
) -> ObjectInstance:
    """Build one fused instance with consistent AABB and OBB geometry."""
    centre_array = np.array(centre, dtype=np.float64)
    half = np.array(dimensions, dtype=np.float64) / 2.0
    return ObjectInstance(
        instance_id,
        {class_name: 1.0},
        dict(colour_scores or {}),
        centre_array,
        centre_array - half,
        centre_array + half,
        np.array(dimensions, dtype=np.float64),
        yaw,
        orientation_confidence,
        observation_count,
        confidence,
    )


def make_candidate(
    detection_id: str,
    centre: tuple[float, float, float],
    *,
    class_name: str = 'paper_cup',
    dimensions: tuple[float, float, float] = (0.09, 0.09, 0.11),
    yaw: float = 0.0,
    confidence: float = 0.6,
    timestamp_ns: int = 1,
) -> ObjectCandidate3D:
    """Build one lifted candidate suitable for real ObjectMap ingestion."""
    centre_array = np.asarray(centre, dtype=np.float64)
    dimensions_array = np.asarray(dimensions, dtype=np.float64)
    offsets = np.asarray([
        [-0.45, -0.45, -0.45],
        [-0.45, 0.45, -0.45],
        [0.45, -0.45, 0.45],
        [0.45, 0.45, 0.45],
        [0.0, 0.0, 0.0],
        [0.2, -0.2, 0.2],
        [-0.2, 0.2, -0.2],
        [0.3, 0.3, -0.3],
    ])
    points = centre_array + offsets * dimensions_array
    return ObjectCandidate3D(
        candidate_id=f'image:{detection_id}',
        detection_id=detection_id,
        class_name=class_name,
        detection_confidence=0.85,
        source=GeometrySource.ACCUMULATED,
        source_timestamp_ns=timestamp_ns,
        image_timestamp_ns=timestamp_ns,
        scan_timestamp_ns=timestamp_ns,
        pose_timestamp_ns=timestamp_ns,
        pose_mode='interpolated',
        image_scan_delta_ms=0.0,
        pose_before_delta_ms=0.0,
        pose_after_delta_ms=0.0,
        timing_warning=False,
        points_map_xyz=points,
        source_projection_indices=np.arange(8),
        point_centroid_xyz=centre_array,
        aabb_min_xyz=centre_array - dimensions_array / 2.0,
        aabb_max_xyz=centre_array + dimensions_array / 2.0,
        obb_centre_xyz=centre_array,
        obb_dimensions_xyz=dimensions_array,
        obb_yaw_rad=yaw,
        estimated_yaw_rad=yaw,
        orientation_confidence=confidence,
        geometry_confidence=confidence,
        geometry_status=GeometryStatus.GOOD,
        partial_geometry=False,
        low_orientation_fallback=False,
        counts=LiftingCounts(16, 14, 0, 12, 10, 8, 8),
        confidence_components=ConfidenceComponents(*([confidence] * 9)),
        diagnostics={},
    )


def make_observation(
    candidate: ObjectCandidate3D,
    viewpoint: str,
    *,
    robot_pose_xyz_yaw: tuple[float, float, float, float] = (
        0.0, 0.0, 0.0, 0.0
    ),
    timestamp_ns: int | None = None,
) -> ViewpointObservation:
    """Build the matching viewpoint observation for a candidate."""
    observation_timestamp = (
        candidate.source_timestamp_ns
        if timestamp_ns is None else timestamp_ns
    )
    return ViewpointObservation(
        viewpoint_id=viewpoint,
        robot_pose_xyz_yaw=np.asarray(robot_pose_xyz_yaw, dtype=np.float64),
        timestamp_ns=observation_timestamp,
        detection_id=candidate.detection_id,
        point_count=candidate.point_count,
        geometry_confidence=candidate.geometry_confidence,
        visibility='full',
        best_crop=np.full((8, 10, 3), 7, dtype=np.uint8),
        best_crop_score=0.8,
    )


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
    candidate_value = make_candidate(
        detection_id,
        centre,
        class_name=class_name,
        dimensions=dimensions,
        confidence=1.0,
        timestamp_ns=timestamp_ns,
    )
    instance_id = object_map.add_or_update(
        candidate_value, make_observation(candidate_value, viewpoint)
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


__all__ = [
    'add_object',
    'add_unknown',
    'add_wall',
    'candidate',
    'geometry',
    'make_candidate',
    'make_instance',
    'make_observation',
    'numerical_result',
    'open_grid',
    'support_geometry',
]
