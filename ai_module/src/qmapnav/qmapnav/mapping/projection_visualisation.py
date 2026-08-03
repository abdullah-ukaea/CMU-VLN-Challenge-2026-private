"""Deterministic LiDAR projection and top-down diagnostic visualisations."""

from dataclasses import dataclass

import cv2
import numpy as np

from qmapnav.mapping.lidar_camera_projection import CropProjection
from qmapnav.mapping.lidar_camera_projection import ProjectionResult
from qmapnav.mapping.projection_quality import DetectionProjection
from qmapnav.perception.contracts import Detection2D


@dataclass(frozen=True)
class ProjectionVisualisationConfig:
    """Bounded point drawing and sparse z-buffer policy."""

    max_draw_points: int = 100_000
    point_radius_px: int = 1
    z_buffer_cell_px: int = 3
    min_colour_depth_m: float = 0.3
    max_colour_depth_m: float = 15.0

    def __post_init__(self) -> None:
        for name in ('max_draw_points', 'point_radius_px', 'z_buffer_cell_px'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.min_colour_depth_m >= self.max_colour_depth_m:
            raise ValueError('colour depth limits are invalid')


def sparse_z_buffer_indices(
    uv: np.ndarray,
    depth_m: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    cell_size_px: int,
) -> np.ndarray:
    """Keep the nearest projected point in each quantized image cell."""
    pixels = np.asarray(uv, dtype=np.float64)
    depth = np.asarray(depth_m, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError('uv must have shape (N, 2)')
    if depth.shape != (pixels.shape[0],):
        raise ValueError('depth_m must have one value per pixel')
    if cell_size_px <= 0:
        raise ValueError('cell_size_px must be positive')
    if pixels.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    x = np.floor(np.mod(pixels[:, 0], image_width) / cell_size_px).astype(np.int64)
    y = np.floor(np.clip(pixels[:, 1], 0, image_height - 1) / cell_size_px).astype(np.int64)
    columns = (image_width + cell_size_px - 1) // cell_size_px
    cells = y * columns + x
    order = np.lexsort((np.arange(depth.shape[0]), depth, cells))
    ordered_cells = cells[order]
    first = np.concatenate(([True], ordered_cells[1:] != ordered_cells[:-1]))
    return np.sort(order[first])


def draw_projection_overlay(
    panorama_rgb: np.ndarray,
    projection: ProjectionResult,
    *,
    config: ProjectionVisualisationConfig | None = None,
    values: np.ndarray | None = None,
) -> np.ndarray:
    """Draw a sparse nearest-depth projection over one RGB panorama."""
    policy = config or ProjectionVisualisationConfig()
    canvas = np.asarray(panorama_rgb).copy()
    if canvas.ndim != 3 or canvas.shape[2] != 3:
        raise ValueError('panorama_rgb must have shape (H, W, 3)')
    indices = sparse_z_buffer_indices(
        projection.panorama_uv,
        projection.euclidean_range_m,
        image_width=canvas.shape[1],
        image_height=canvas.shape[0],
        cell_size_px=policy.z_buffer_cell_px,
    )
    if indices.shape[0] > policy.max_draw_points:
        sample = np.linspace(
            0,
            indices.shape[0] - 1,
            policy.max_draw_points,
            dtype=np.int64,
        )
        indices = indices[sample]
    colour_values = projection.euclidean_range_m if values is None else np.asarray(values)
    if colour_values.shape != (projection.point_count,):
        raise ValueError('values must have one entry per projected point')
    colours = _depth_colours(
        colour_values[indices],
        policy.min_colour_depth_m,
        policy.max_colour_depth_m,
    )
    for point, colour in zip(projection.panorama_uv[indices], colours):
        centre = (
            int(round(point[0])) % canvas.shape[1],
            int(np.clip(round(point[1]), 0, canvas.shape[0] - 1)),
        )
        cv2.circle(
            canvas,
            centre,
            policy.point_radius_px,
            tuple(int(value) for value in colour),
            -1,
            cv2.LINE_AA,
        )
    return canvas


def draw_detection_projection_overlay(
    panorama_rgb: np.ndarray,
    projection: ProjectionResult,
    detections: tuple[Detection2D, ...],
    summaries: tuple[DetectionProjection, ...],
    *,
    config: ProjectionVisualisationConfig | None = None,
) -> np.ndarray:
    """Draw detection boxes, in-box projected points, and support statistics."""
    if len(detections) != len(summaries):
        raise ValueError('one projection summary is required per detection')
    canvas = draw_projection_overlay(panorama_rgb, projection, config=config)
    for detection, summary in zip(detections, summaries):
        colour = (255, 220, 50)
        y_min = int(round(detection.panorama_box.y_min))
        y_max = int(round(detection.panorama_box.y_max))
        for x_min, x_max in detection.panorama_box.x_intervals:
            cv2.rectangle(
                canvas,
                (int(round(x_min)), y_min),
                (min(canvas.shape[1] - 1, int(round(x_max))), y_max),
                colour,
                2,
            )
        median = 'n/a' if summary.depth_median_m is None else f'{summary.depth_median_m:.2f}m'
        iqr = 'n/a' if summary.depth_iqr_m is None else f'{summary.depth_iqr_m:.2f}m'
        label = (
            f'{detection.class_name} {detection.confidence:.2f} '
            f'n={summary.point_count} med={median} iqr={iqr} {summary.quality}'
        )
        anchor = detection.panorama_box.x_intervals[-1][0]
        cv2.putText(
            canvas,
            label,
            (int(round(anchor)), max(18, y_min - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colour,
            1,
            cv2.LINE_AA,
        )
    return canvas


def draw_crop_projection_overlay(
    crop_rgb: np.ndarray,
    crop_projection: CropProjection,
    *,
    bbox_xyxy: tuple[float, float, float, float] | None = None,
    config: ProjectionVisualisationConfig | None = None,
) -> np.ndarray:
    """Draw projected points and an optional detector box in crop coordinates."""
    policy = config or ProjectionVisualisationConfig()
    canvas = np.asarray(crop_rgb).copy()
    colours = _depth_colours(
        crop_projection.euclidean_range_m,
        policy.min_colour_depth_m,
        policy.max_colour_depth_m,
    )
    for point, colour in zip(crop_projection.crop_uv, colours):
        cv2.circle(
            canvas,
            (int(round(point[0])), int(round(point[1]))),
            policy.point_radius_px,
            tuple(int(value) for value in colour),
            -1,
            cv2.LINE_AA,
        )
    if bbox_xyxy is not None:
        x_min, y_min, x_max, y_max = (int(round(value)) for value in bbox_xyxy)
        cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), (255, 220, 50), 2)
    cv2.putText(
        canvas,
        f'crop {crop_projection.crop_id} n={crop_projection.crop_uv.shape[0]}',
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def draw_top_down_projection(
    current_points_map_xyz: np.ndarray,
    accumulated_points_map_xyz: np.ndarray,
    sensor_position_xyz: np.ndarray,
    sensor_heading_rad: float,
    *,
    size_px: int = 800,
    metres_per_pixel: float = 0.03,
) -> np.ndarray:
    """Render a local XY map around the sensor with camera heading."""
    if size_px <= 0 or metres_per_pixel <= 0.0:
        raise ValueError('top-down dimensions must be positive')
    current = np.asarray(current_points_map_xyz, dtype=np.float64)
    accumulated = np.asarray(accumulated_points_map_xyz, dtype=np.float64)
    sensor = np.asarray(sensor_position_xyz, dtype=np.float64)
    if current.ndim != 2 or current.shape[1] != 3:
        raise ValueError('current_points_map_xyz must have shape (N, 3)')
    if accumulated.ndim != 2 or accumulated.shape[1] != 3:
        raise ValueError('accumulated_points_map_xyz must have shape (N, 3)')
    if sensor.shape != (3,):
        raise ValueError('sensor_position_xyz must have shape (3,)')
    canvas = np.zeros((size_px, size_px, 3), dtype=np.uint8)
    centre = np.array([size_px / 2.0, size_px / 2.0])

    def pixels(points: np.ndarray) -> np.ndarray:
        offsets = (points[:, :2] - sensor[:2]) / metres_per_pixel
        return np.column_stack((centre[0] + offsets[:, 0], centre[1] - offsets[:, 1]))

    for points, colour in ((accumulated, (90, 90, 90)), (current, (80, 220, 255))):
        if points.shape[0] > 150_000:
            points = points[np.linspace(0, points.shape[0] - 1, 150_000, dtype=np.int64)]
        for point in pixels(points):
            x, y = int(round(point[0])), int(round(point[1]))
            if 0 <= x < size_px and 0 <= y < size_px:
                canvas[y, x] = colour
    origin = (int(round(centre[0])), int(round(centre[1])))
    heading_end = (
        int(round(centre[0] + 50.0 * np.cos(sensor_heading_rad))),
        int(round(centre[1] - 50.0 * np.sin(sensor_heading_rad))),
    )
    cv2.circle(canvas, origin, 6, (255, 80, 80), -1)
    cv2.arrowedLine(canvas, origin, heading_end, (255, 80, 80), 3)
    return canvas


def _depth_colours(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    normalized = np.clip((np.asarray(values) - minimum) / (maximum - minimum), 0.0, 1.0)
    blue = (255.0 * (1.0 - normalized)).astype(np.uint8)
    red = (255.0 * normalized).astype(np.uint8)
    green = (255.0 * (1.0 - np.abs(2.0 * normalized - 1.0))).astype(np.uint8)
    return np.column_stack((red, green, blue))


__all__ = [
    'ProjectionVisualisationConfig',
    'draw_crop_projection_overlay',
    'draw_detection_projection_overlay',
    'draw_projection_overlay',
    'draw_top_down_projection',
    'sparse_z_buffer_indices',
]
