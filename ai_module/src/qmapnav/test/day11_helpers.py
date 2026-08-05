"""Shared deterministic fixtures for the Day 11 exploration tests."""

import numpy as np

from qmapnav.common import ObjectInstance
from qmapnav.mapping.object_candidate import ConfidenceComponents
from qmapnav.mapping.object_candidate import GeometrySource
from qmapnav.mapping.object_candidate import GeometryStatus
from qmapnav.mapping.object_candidate import LiftingCounts
from qmapnav.mapping.object_candidate import ObjectCandidate3D
from qmapnav.mapping.occupancy_grid import CELL_FREE
from qmapnav.mapping.occupancy_grid import CELL_OCCUPIED
from qmapnav.mapping.occupancy_grid import OccupancyGrid2D
from qmapnav.mapping.viewpoint_observation import ViewpointObservation


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
) -> ViewpointObservation:
    """Build the matching viewpoint observation for a candidate."""
    return ViewpointObservation(
        viewpoint_id=viewpoint,
        robot_pose_xyz_yaw=np.asarray(robot_pose_xyz_yaw, dtype=np.float64),
        timestamp_ns=candidate.source_timestamp_ns,
        detection_id=candidate.detection_id,
        point_count=candidate.point_count,
        geometry_confidence=candidate.geometry_confidence,
        visibility='full',
        best_crop=np.full((8, 10, 3), 7, dtype=np.uint8),
        best_crop_score=0.8,
    )
