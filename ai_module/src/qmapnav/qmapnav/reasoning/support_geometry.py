"""Common map-frame geometry for object and structural support reasoning."""

from dataclasses import dataclass
from math import isfinite

import cv2
import numpy as np

from qmapnav.common import ObjectInstance
from qmapnav.mapping.geometry_evaluation import upright_box_corners_xy
from qmapnav.mapping.object_map import PersistentObjectRecord
from qmapnav.mapping.structural_map import StructuralAnchor


@dataclass(frozen=True)
class SupportGeometry:
    """Minimal common support interface for an object or structural anchor."""

    entity_id: str
    semantic_class: str
    centre_xyz: np.ndarray
    dimensions_xyz: np.ndarray
    yaw_rad: float
    footprint_xy: np.ndarray
    bottom_z: float
    top_z: float
    confidence: float
    quality: str
    source_type: str

    def __post_init__(self) -> None:
        centre = np.asarray(self.centre_xyz, dtype=np.float64)
        dimensions = np.asarray(self.dimensions_xyz, dtype=np.float64)
        footprint = np.asarray(self.footprint_xy, dtype=np.float64)
        if centre.shape != (3,) or dimensions.shape != (3,):
            raise ValueError('support centre and dimensions must have shape (3,)')
        if footprint.ndim != 2 or footprint.shape[1] != 2:
            raise ValueError('support footprint must have shape (N, 2)')
        if footprint.shape[0] < 3 or np.any(dimensions <= 0.0):
            raise ValueError('support geometry must have positive extent')
        values = (*centre, *dimensions, *footprint.ravel(), self.yaw_rad,
                  self.bottom_z, self.top_z, self.confidence)
        if not all(isfinite(float(value)) for value in values):
            raise ValueError('support geometry must be finite')
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError('support confidence must lie in [0, 1]')
        if self.top_z <= self.bottom_z:
            raise ValueError('support top must exceed bottom')
        for name, value in (
            ('centre_xyz', centre),
            ('dimensions_xyz', dimensions),
            ('footprint_xy', footprint),
        ):
            copied = np.ascontiguousarray(value).copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)


@dataclass(frozen=True)
class FootprintMetrics:
    """Subject-relative overlap and horizontal separation."""

    intersection_area_m2: float
    subject_overlap: float
    centre_distance_m: float
    edge_distance_m: float


def support_geometry(
    entity: ObjectInstance | PersistentObjectRecord | StructuralAnchor,
) -> SupportGeometry:
    """Adapt persistent objects and structural anchors to one interface."""
    if isinstance(entity, PersistentObjectRecord):
        instance = entity.instance
        confidence = entity.geometry_confidence
        quality = entity.status
        source_type = 'object'
        identifier = str(instance.instance_id)
        semantic_class = entity.canonical_class
        return _object_geometry(
            instance, identifier, semantic_class, confidence, quality,
            source_type,
        )
    if isinstance(entity, ObjectInstance):
        semantic_class = max(
            sorted(entity.class_scores), key=entity.class_scores.get
        )
        return _object_geometry(
            entity,
            str(entity.instance_id),
            semantic_class,
            entity.confidence,
            'active',
            'object',
        )
    if isinstance(entity, StructuralAnchor):
        if entity.extent_xyz is None:
            raise ValueError('structural support requires extent_xyz')
        dimensions = entity.extent_xyz
        yaw = float(entity.yaw_rad or 0.0)
        footprint = (
            entity.polygon_xy
            if entity.polygon_xy is not None
            else upright_box_corners_xy(entity.position_xyz, dimensions, yaw)
        )
        return SupportGeometry(
            entity.anchor_id,
            entity.semantic_class,
            entity.position_xyz,
            dimensions,
            yaw,
            footprint,
            float(entity.position_xyz[2] - dimensions[2] / 2.0),
            float(entity.position_xyz[2] + dimensions[2] / 2.0),
            entity.confidence,
            'active',
            'structural',
        )
    raise TypeError('unsupported support entity')


def footprint_metrics(
    subject: SupportGeometry,
    anchor: SupportGeometry,
) -> FootprintMetrics:
    """Measure subject support overlap, avoiding ordinary pairwise IoU."""
    subject_polygon = subject.footprint_xy.astype(np.float32)
    anchor_polygon = anchor.footprint_xy.astype(np.float32)
    subject_area = abs(float(cv2.contourArea(subject_polygon)))
    intersection, _ = cv2.intersectConvexConvex(
        subject_polygon, anchor_polygon
    )
    intersection = max(0.0, float(intersection))
    overlap = 0.0 if subject_area <= 0.0 else min(
        1.0, intersection / subject_area
    )
    centre_distance = float(np.linalg.norm(
        subject.centre_xyz[:2] - anchor.centre_xyz[:2]
    ))
    edge_distance = _polygon_edge_distance(subject_polygon, anchor_polygon)
    return FootprintMetrics(
        intersection, overlap, centre_distance, edge_distance
    )


def _object_geometry(
    instance, identifier, semantic_class, confidence, quality, source_type,
):
    footprint = upright_box_corners_xy(
        instance.centroid_xyz, instance.obb_dimensions, instance.obb_yaw
    )
    return SupportGeometry(
        identifier,
        semantic_class,
        instance.centroid_xyz,
        instance.obb_dimensions,
        instance.obb_yaw,
        footprint,
        float(instance.aabb_min_xyz[2]),
        float(instance.aabb_max_xyz[2]),
        confidence,
        quality,
        source_type,
    )


def _polygon_edge_distance(first, second):
    if cv2.intersectConvexConvex(first, second)[0] > 0.0:
        return 0.0
    distances = []
    for point in first:
        distances.append(abs(float(cv2.pointPolygonTest(
            second, tuple(map(float, point)), True
        ))))
    for point in second:
        distances.append(abs(float(cv2.pointPolygonTest(
            first, tuple(map(float, point)), True
        ))))
    return min(distances)


__all__ = ['FootprintMetrics', 'SupportGeometry', 'footprint_metrics',
           'support_geometry']
