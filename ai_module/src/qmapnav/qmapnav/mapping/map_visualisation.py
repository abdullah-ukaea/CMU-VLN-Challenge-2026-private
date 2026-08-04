"""Saved top-down diagnostics for persistent objects and structures."""

import cv2
import numpy as np

from qmapnav.mapping.geometry_evaluation import upright_box_corners_xy
from qmapnav.mapping.object_map import PersistentObjectRecord
from qmapnav.mapping.structural_map import StructuralAnchor


def draw_persistent_map_top_down(
    objects: list[PersistentObjectRecord],
    walls: list[StructuralAnchor],
    anchors: list[StructuralAnchor],
    robot_path_xy: np.ndarray | None = None,
    *,
    size_px: int = 900,
    padding_m: float = 1.0,
) -> np.ndarray:
    """Render robot path, fused OBBs, walls, normals, and anchors."""
    if size_px <= 0 or padding_m <= 0.0:
        raise ValueError('visualisation size and padding must be positive')
    path = (
        np.empty((0, 2), dtype=np.float64)
        if robot_path_xy is None else np.asarray(robot_path_xy, dtype=np.float64)
    )
    if path.ndim != 2 or path.shape[1] != 2 or not np.all(np.isfinite(path)):
        raise ValueError('robot_path_xy must have shape (N, 2) and be finite')
    geometry = [path]
    geometry.extend(
        record.instance.centroid_xyz[None, :2] for record in objects
    )
    geometry.extend(
        wall.line_segment_xy for wall in walls if wall.line_segment_xy is not None
    )
    geometry.extend(anchor.position_xyz[None, :2] for anchor in anchors)
    points = [item for item in geometry if item.size]
    if points:
        combined = np.vstack(points)
        minimum = np.min(combined, axis=0) - padding_m
        maximum = np.max(combined, axis=0) + padding_m
    else:
        minimum = np.array([-1.0, -1.0])
        maximum = np.array([1.0, 1.0])
    span = np.maximum(maximum - minimum, 1e-6)
    scale = (size_px - 40.0) / max(span)

    def pixel(xy):
        values = (np.asarray(xy) - minimum) * scale + 20.0
        values[..., 1] = size_px - values[..., 1]
        return np.rint(values).astype(np.int32)

    canvas = np.full((size_px, size_px, 3), 245, dtype=np.uint8)
    if path.shape[0] >= 2:
        cv2.polylines(canvas, [pixel(path)], False, (80, 80, 80), 2)
    for wall in walls:
        cv2.line(
            canvas,
            tuple(pixel(wall.line_segment_xy[0])),
            tuple(pixel(wall.line_segment_xy[1])),
            (230, 130, 40),
            4,
        )
        centre = pixel(wall.position_xyz[:2])
        normal_end = pixel(
            wall.position_xyz[:2] + wall.plane_parameters[:2] * 0.4
        )
        cv2.arrowedLine(
            canvas, tuple(centre), tuple(normal_end), (255, 80, 30), 2
        )
    for record in objects:
        instance = record.instance
        corners = upright_box_corners_xy(
            instance.centroid_xyz,
            instance.obb_dimensions,
            instance.obb_yaw,
        )
        cv2.polylines(canvas, [pixel(corners)], True, (40, 170, 60), 3)
        centre = tuple(pixel(instance.centroid_xyz[:2]))
        cv2.putText(
            canvas,
            f'#{instance.instance_id} {record.canonical_class}',
            centre,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 100, 30),
            1,
            cv2.LINE_AA,
        )
    for anchor in anchors:
        centre = tuple(pixel(anchor.position_xyz[:2]))
        cv2.drawMarker(
            canvas, centre, (180, 30, 180), cv2.MARKER_CROSS, 14, 3
        )
        cv2.putText(
            canvas,
            anchor.semantic_class,
            (centre[0] + 5, centre[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (120, 20, 120),
            1,
            cv2.LINE_AA,
        )
    return canvas


__all__ = ['draw_persistent_map_top_down']
