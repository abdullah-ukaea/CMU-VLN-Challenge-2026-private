"""Deterministic Day 6 selection, depth, and box diagnostics."""

import cv2
import numpy as np

from qmapnav.mapping.box_overlap import upright_box_corners_xy
from qmapnav.mapping.lidar_camera_projection import ProjectionResult
from qmapnav.mapping.object_candidate import LiftingResult
from qmapnav.perception.contracts import Detection2D


def draw_lifting_stage_overlay(
    panorama_rgb: np.ndarray,
    projection: ProjectionResult,
    detection: Detection2D,
    result: LiftingResult,
    *,
    point_radius_px: int = 2,
) -> np.ndarray:
    """Draw proposal, rejected stages, and final cluster in distinct colours."""
    canvas = _rgb_canvas(panorama_rgb)
    layers = (
        ('original_box', (130, 130, 130)),
        ('ground_removed', (255, 70, 70)),
        ('depth_removed', (255, 170, 50)),
        ('cluster_noise', (190, 80, 255)),
        ('clustered', (50, 255, 80)),
    )
    for name, colour in layers:
        indices = result.stage_indices.get(name, np.empty(0, dtype=np.int64))
        for u, v in projection.panorama_uv[indices]:
            cv2.circle(
                canvas,
                (int(round(u)) % canvas.shape[1], int(round(v))),
                point_radius_px,
                colour,
                -1,
                cv2.LINE_AA,
            )
    for x_min, x_max in detection.panorama_box.x_intervals:
        cv2.rectangle(
            canvas,
            (int(round(x_min)), int(round(detection.panorama_box.y_min))),
            (
                min(canvas.shape[1] - 1, int(round(x_max))),
                int(round(detection.panorama_box.y_max)),
            ),
            (255, 230, 40),
            2,
        )
    label = f'{detection.class_name} {result.status.value} n={result.counts.final}'
    cv2.putText(
        canvas,
        label,
        (
            int(detection.panorama_box.x_intervals[-1][0]),
            max(18, int(detection.panorama_box.y_min) - 5),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 230, 40),
        1,
        cv2.LINE_AA,
    )
    return canvas


def draw_depth_histogram(
    projection: ProjectionResult,
    result: LiftingResult,
    *,
    width: int = 640,
    height: int = 360,
    bins: int = 40,
) -> np.ndarray:
    """Render selected depth distribution and retained foreground band."""
    if width <= 0 or height <= 0 or bins <= 0:
        raise ValueError('histogram dimensions and bins must be positive')
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    selected = result.stage_indices.get('selected', np.empty(0, dtype=np.int64))
    if selected.size == 0:
        cv2.putText(
            canvas,
            'no selected depth points',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 40, 40),
            2,
        )
        return canvas
    depths = projection.euclidean_range_m[selected]
    counts, edges = np.histogram(depths, bins=bins)
    maximum = max(int(np.max(counts)), 1)
    plot_height = height - 60
    for index, count in enumerate(counts):
        x0 = int(index * width / bins)
        x1 = int((index + 1) * width / bins)
        y0 = height - 30
        y1 = y0 - int(count / maximum * plot_height)
        centre = (edges[index] + edges[index + 1]) / 2.0
        depth_band = result.diagnostics.get('depth_band_m', (None, None))
        retained = (
            depth_band[0] is not None
            and depth_band[0] <= centre <= depth_band[1]
        )
        colour = (50, 180, 70) if retained else (140, 140, 140)
        cv2.rectangle(canvas, (x0, y1), (max(x0, x1 - 1), y0), colour, -1)
    cv2.putText(
        canvas,
        f'{float(np.min(depths)):.2f}m .. {float(np.max(depths)):.2f}m',
        (15, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (30, 30, 30),
        1,
    )
    return canvas


def draw_candidate_orthographic(
    result: LiftingResult,
    sensor_position_xyz: np.ndarray,
    *,
    size_px: int = 700,
    metres_per_pixel: float = 0.01,
) -> np.ndarray:
    """Draw top-down and side projections with AABB, OBB, and yaw axis."""
    if size_px <= 0 or metres_per_pixel <= 0.0:
        raise ValueError('orthographic dimensions must be positive')
    canvas = np.zeros((size_px, size_px * 2, 3), dtype=np.uint8)
    candidate = result.candidate
    if candidate is None:
        cv2.putText(
            canvas,
            result.reason,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (220, 220, 220),
            2,
        )
        return canvas
    sensor = np.asarray(sensor_position_xyz, dtype=np.float64)
    if sensor.shape != (3,):
        raise ValueError('sensor_position_xyz must have shape (3,)')
    centre = candidate.obb_centre_xyz
    origin = np.array([size_px / 2.0, size_px / 2.0])

    def xy_pixel(points_xy: np.ndarray) -> np.ndarray:
        delta = (points_xy - centre[:2]) / metres_per_pixel
        return np.column_stack((origin[0] + delta[:, 0], origin[1] - delta[:, 1]))

    points_xy = xy_pixel(candidate.points_map_xyz[:, :2])
    for x, y in points_xy:
        if 0 <= x < size_px and 0 <= y < size_px:
            canvas[int(y), int(x)] = (80, 240, 100)
    aabb_corners = np.array(
        [
            candidate.aabb_min_xyz[:2],
            [candidate.aabb_max_xyz[0], candidate.aabb_min_xyz[1]],
            candidate.aabb_max_xyz[:2],
            [candidate.aabb_min_xyz[0], candidate.aabb_max_xyz[1]],
        ]
    )
    obb_corners = upright_box_corners_xy(
        candidate.obb_centre_xyz,
        candidate.obb_dimensions_xyz,
        candidate.obb_yaw_rad,
    )
    cv2.polylines(
        canvas,
        [np.rint(xy_pixel(aabb_corners)).astype(np.int32)],
        True,
        (255, 170, 40),
        2,
    )
    cv2.polylines(
        canvas,
        [np.rint(xy_pixel(obb_corners)).astype(np.int32)],
        True,
        (60, 120, 255),
        3,
    )
    sensor_pixel = xy_pixel(sensor[None, :2])[0]
    cv2.circle(canvas, tuple(np.rint(sensor_pixel).astype(int)), 5, (255, 255, 255), -1)

    side_origin = np.array([size_px * 1.5, size_px / 2.0])
    local_x = candidate.points_map_xyz[:, 0] - centre[0]
    local_z = candidate.points_map_xyz[:, 2] - centre[2]
    side = np.column_stack(
        (side_origin[0] + local_x / metres_per_pixel, side_origin[1] - local_z / metres_per_pixel)
    )
    for x, y in side:
        if size_px <= x < size_px * 2 and 0 <= y < size_px:
            canvas[int(y), int(x)] = (80, 240, 100)
    cv2.putText(canvas, 'XY top-down', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(
        canvas,
        'XZ side',
        (size_px + 10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )
    return canvas


def _rgb_canvas(image: np.ndarray) -> np.ndarray:
    canvas = np.asarray(image).copy()
    if canvas.ndim != 3 or canvas.shape[2] != 3:
        raise ValueError('panorama_rgb must have shape (H, W, 3)')
    return canvas


__all__ = [
    'draw_candidate_orthographic',
    'draw_depth_histogram',
    'draw_lifting_stage_overlay',
]
