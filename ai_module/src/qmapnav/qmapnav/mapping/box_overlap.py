"""Centre, dimension, yaw, and upright 3D overlap metrics."""

from dataclasses import dataclass
from math import cos, isfinite, sin

import numpy as np

from qmapnav.mapping.bounding_boxes import canonicalize_box
from qmapnav.mapping.bounding_boxes import rectangle_yaw_difference
from qmapnav.mapping.object_candidate import ObjectCandidate3D
from qmapnav.mapping.object_candidate import readonly_array


POINT_COUNT_BINS = (
    (0, 0, '0'),
    (1, 5, '1-5'),
    (6, 10, '6-10'),
    (11, 30, '11-30'),
    (31, 100, '31-100'),
    (101, None, '>100'),
)


@dataclass(frozen=True)
class ReferenceUprightBox:
    """Canonical reference box for lifting proxy evaluation."""

    centre_xyz: np.ndarray
    dimensions_xyz: np.ndarray
    yaw_rad: float
    class_name: str = 'unknown'
    provenance: str = 'manual'

    def __post_init__(self) -> None:
        centre = readonly_array('centre_xyz', self.centre_xyz, (3,))
        dimensions = readonly_array('dimensions_xyz', self.dimensions_xyz, (3,))
        if np.any(dimensions <= 0.0):
            raise ValueError('reference dimensions must be positive')
        length, width, yaw = canonicalize_box(
            dimensions[0], dimensions[1], self.yaw_rad
        )
        canonical_dimensions = np.array([length, width, dimensions[2]])
        canonical_dimensions.setflags(write=False)
        if not self.class_name or not self.provenance:
            raise ValueError('class_name and provenance must be non-empty')
        object.__setattr__(self, 'centre_xyz', centre)
        object.__setattr__(self, 'dimensions_xyz', canonical_dimensions)
        object.__setattr__(self, 'yaw_rad', yaw)


@dataclass(frozen=True)
class GeometryEvaluation:
    """Proxy errors for one predicted candidate and one reference box."""

    centre_error_3d_m: float
    centre_error_xy_m: float
    centre_error_z_m: float
    dimension_absolute_error_xyz_m: np.ndarray
    dimension_relative_error_xyz: np.ndarray
    yaw_error_rad: float
    aabb_iou_3d: float
    oriented_iou_3d: float
    point_count: int
    point_count_bin: str

    def __post_init__(self) -> None:
        for name in (
            'dimension_absolute_error_xyz_m',
            'dimension_relative_error_xyz',
        ):
            object.__setattr__(
                self,
                name,
                readonly_array(name, getattr(self, name), (3,)),
            )
        for name in (
            'centre_error_3d_m',
            'centre_error_xy_m',
            'centre_error_z_m',
            'yaw_error_rad',
            'aabb_iou_3d',
            'oriented_iou_3d',
        ):
            if not isfinite(float(getattr(self, name))):
                raise ValueError(f'{name} must be finite')


def evaluate_candidate_geometry(
    candidate: ObjectCandidate3D,
    reference: ReferenceUprightBox,
) -> GeometryEvaluation:
    """Evaluate one candidate against a canonical upright reference."""
    centre_delta = candidate.obb_centre_xyz - reference.centre_xyz
    absolute_dimensions = np.abs(
        candidate.obb_dimensions_xyz - reference.dimensions_xyz
    )
    relative_dimensions = absolute_dimensions / reference.dimensions_xyz
    candidate_aabb = (
        candidate.aabb_min_xyz,
        candidate.aabb_max_xyz,
    )
    reference_aabb = upright_box_aabb(reference)
    return GeometryEvaluation(
        centre_error_3d_m=float(np.linalg.norm(centre_delta)),
        centre_error_xy_m=float(np.linalg.norm(centre_delta[:2])),
        centre_error_z_m=float(abs(centre_delta[2])),
        dimension_absolute_error_xyz_m=absolute_dimensions,
        dimension_relative_error_xyz=relative_dimensions,
        yaw_error_rad=rectangle_yaw_difference(
            candidate.obb_yaw_rad, reference.yaw_rad
        ),
        aabb_iou_3d=aabb_iou_3d(*candidate_aabb, *reference_aabb),
        oriented_iou_3d=upright_box_iou_3d(
            candidate.obb_centre_xyz,
            candidate.obb_dimensions_xyz,
            candidate.obb_yaw_rad,
            reference.centre_xyz,
            reference.dimensions_xyz,
            reference.yaw_rad,
        ),
        point_count=candidate.point_count,
        point_count_bin=point_count_bin(candidate.point_count),
    )


def aabb_iou_3d(
    first_min_xyz: np.ndarray,
    first_max_xyz: np.ndarray,
    second_min_xyz: np.ndarray,
    second_max_xyz: np.ndarray,
) -> float:
    """Return axis-aligned 3D intersection over union."""
    first_min = np.asarray(first_min_xyz, dtype=np.float64)
    first_max = np.asarray(first_max_xyz, dtype=np.float64)
    second_min = np.asarray(second_min_xyz, dtype=np.float64)
    second_max = np.asarray(second_max_xyz, dtype=np.float64)
    for value in (first_min, first_max, second_min, second_max):
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError('AABB corners must be finite shape (3,)')
    if np.any(first_max <= first_min) or np.any(second_max <= second_min):
        raise ValueError('AABB maximum must exceed minimum')
    overlap = np.maximum(
        0.0,
        np.minimum(first_max, second_max) - np.maximum(first_min, second_min),
    )
    intersection = float(np.prod(overlap))
    first_volume = float(np.prod(first_max - first_min))
    second_volume = float(np.prod(second_max - second_min))
    union = first_volume + second_volume - intersection
    return 0.0 if union <= 0.0 else intersection / union


def upright_box_iou_3d(
    first_centre_xyz: np.ndarray,
    first_dimensions_xyz: np.ndarray,
    first_yaw_rad: float,
    second_centre_xyz: np.ndarray,
    second_dimensions_xyz: np.ndarray,
    second_yaw_rad: float,
) -> float:
    """Return oriented IoU for two upright boxes."""
    first_centre = np.asarray(first_centre_xyz, dtype=np.float64)
    second_centre = np.asarray(second_centre_xyz, dtype=np.float64)
    first_dimensions = np.asarray(first_dimensions_xyz, dtype=np.float64)
    second_dimensions = np.asarray(second_dimensions_xyz, dtype=np.float64)
    if any(value.shape != (3,) for value in (
        first_centre, second_centre, first_dimensions, second_dimensions
    )):
        raise ValueError('upright box vectors must have shape (3,)')
    if np.any(first_dimensions <= 0.0) or np.any(second_dimensions <= 0.0):
        raise ValueError('upright box dimensions must be positive')
    first_polygon = upright_box_corners_xy(
        first_centre, first_dimensions, first_yaw_rad
    )
    second_polygon = upright_box_corners_xy(
        second_centre, second_dimensions, second_yaw_rad
    )
    intersection_polygon = _convex_polygon_intersection(
        first_polygon, second_polygon
    )
    intersection_area = _polygon_area(intersection_polygon)
    first_z = (
        first_centre[2] - first_dimensions[2] / 2.0,
        first_centre[2] + first_dimensions[2] / 2.0,
    )
    second_z = (
        second_centre[2] - second_dimensions[2] / 2.0,
        second_centre[2] + second_dimensions[2] / 2.0,
    )
    height_overlap = max(
        0.0, min(first_z[1], second_z[1]) - max(first_z[0], second_z[0])
    )
    intersection = intersection_area * height_overlap
    first_volume = float(np.prod(first_dimensions))
    second_volume = float(np.prod(second_dimensions))
    union = first_volume + second_volume - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def upright_box_corners_xy(
    centre_xyz: np.ndarray,
    dimensions_xyz: np.ndarray,
    yaw_rad: float,
) -> np.ndarray:
    """Return four counter-clockwise footprint corners."""
    centre = np.asarray(centre_xyz, dtype=np.float64)
    dimensions = np.asarray(dimensions_xyz, dtype=np.float64)
    axis = np.array([cos(yaw_rad), sin(yaw_rad)])
    perpendicular = np.array([-sin(yaw_rad), cos(yaw_rad)])
    half_length = dimensions[0] / 2.0
    half_width = dimensions[1] / 2.0
    return np.asarray(
        [
            centre[:2] - half_length * axis - half_width * perpendicular,
            centre[:2] + half_length * axis - half_width * perpendicular,
            centre[:2] + half_length * axis + half_width * perpendicular,
            centre[:2] - half_length * axis + half_width * perpendicular,
        ]
    )


def upright_box_aabb(
    box: ReferenceUprightBox,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the enclosing AABB of one upright reference box."""
    corners = upright_box_corners_xy(
        box.centre_xyz, box.dimensions_xyz, box.yaw_rad
    )
    minimum = np.array(
        [
            np.min(corners[:, 0]),
            np.min(corners[:, 1]),
            box.centre_xyz[2] - box.dimensions_xyz[2] / 2.0,
        ]
    )
    maximum = np.array(
        [
            np.max(corners[:, 0]),
            np.max(corners[:, 1]),
            box.centre_xyz[2] + box.dimensions_xyz[2] / 2.0,
        ]
    )
    return minimum, maximum


def point_count_bin(point_count: int) -> str:
    """Return the configured evidence bin label."""
    if isinstance(point_count, bool) or not isinstance(point_count, int):
        raise ValueError('point_count must be an integer')
    if point_count < 0:
        raise ValueError('point_count must be non-negative')
    for lower, upper, label in POINT_COUNT_BINS:
        if point_count >= lower and (upper is None or point_count <= upper):
            return label
    raise AssertionError('point-count bins are incomplete')


def _convex_polygon_intersection(
    subject: np.ndarray,
    clip: np.ndarray,
) -> np.ndarray:
    output = list(subject)
    for start, end in zip(clip, np.roll(clip, -1, axis=0)):
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        for current in input_points:
            current_inside = _inside_edge(current, start, end)
            previous_inside = _inside_edge(previous, start, end)
            if current_inside:
                if not previous_inside:
                    output.append(_line_intersection(previous, current, start, end))
                output.append(current)
            elif previous_inside:
                output.append(_line_intersection(previous, current, start, end))
            previous = current
    return np.asarray(output, dtype=np.float64).reshape(-1, 2)


def _inside_edge(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> bool:
    edge = end - start
    offset = point - start
    return bool(edge[0] * offset[1] - edge[1] * offset[0] >= -1e-12)


def _line_intersection(
    first: np.ndarray,
    second: np.ndarray,
    clip_start: np.ndarray,
    clip_end: np.ndarray,
) -> np.ndarray:
    direction = second - first
    clip_direction = clip_end - clip_start
    denominator = direction[0] * clip_direction[1] - direction[1] * clip_direction[0]
    if abs(denominator) <= 1e-12:
        return second
    delta = clip_start - first
    parameter = (
        delta[0] * clip_direction[1] - delta[1] * clip_direction[0]
    ) / denominator
    return first + parameter * direction


def _polygon_area(polygon: np.ndarray) -> float:
    if polygon.shape[0] < 3:
        return 0.0
    return float(
        0.5
        * abs(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
        )
    )


__all__ = [
    'aabb_iou_3d',
    'evaluate_candidate_geometry',
    'GeometryEvaluation',
    'POINT_COUNT_BINS',
    'point_count_bin',
    'ReferenceUprightBox',
    'upright_box_aabb',
    'upright_box_corners_xy',
    'upright_box_iou_3d',
]
