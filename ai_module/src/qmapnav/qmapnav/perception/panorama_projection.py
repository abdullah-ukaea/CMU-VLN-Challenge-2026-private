"""Analytic mappings between perspective pixels, panorama pixels, and rays."""

from dataclasses import dataclass
from math import cos, isfinite, pi, sin, tan

import numpy as np

from qmapnav.perception.contracts import PanoramaBox
from qmapnav.perception.contracts import PerspectiveGeometry


_TOLERANCE = 1e-9


def wrap_angle(angle: np.ndarray | float) -> np.ndarray:
    """Wrap radians to the half-open interval ``[-pi, pi)``."""
    return (np.asarray(angle, dtype=np.float64) + pi) % (2.0 * pi) - pi


@dataclass(frozen=True)
class PanoramaCameraModel:
    """
    Angular model for the challenge's cropped equirectangular panorama.

    The internal camera basis is right handed: ``+X`` forward, ``+Y`` left,
    and ``+Z`` up. Image ``v`` increases downward. ``u_yaw_sign=-1`` means
    image ``u`` increases toward camera right, which is decreasing yaw in this
    basis. The sign remains explicit because the physical camera convention
    must be checked against saved challenge images before Day 5 calibration.
    """

    width: int
    height: int
    vertical_fov_rad: float = 2.0 * pi / 3.0
    yaw_at_centre_rad: float = 0.0
    pitch_at_centre_rad: float = 0.0
    u_yaw_sign: int = -1

    def __post_init__(self) -> None:
        if isinstance(self.width, bool) or not isinstance(self.width, int):
            raise ValueError('width must be a positive integer')
        if isinstance(self.height, bool) or not isinstance(self.height, int):
            raise ValueError('height must be a positive integer')
        if self.width <= 0 or self.height <= 0:
            raise ValueError('panorama dimensions must be positive')
        if not isfinite(self.vertical_fov_rad) or not 0.0 < self.vertical_fov_rad < pi:
            raise ValueError('vertical_fov_rad must be finite and in (0, pi)')
        if not isfinite(self.yaw_at_centre_rad):
            raise ValueError('yaw_at_centre_rad must be finite')
        if not isfinite(self.pitch_at_centre_rad):
            raise ValueError('pitch_at_centre_rad must be finite')
        half_vertical = self.vertical_fov_rad / 2.0
        if not -pi / 2.0 <= self.pitch_at_centre_rad - half_vertical:
            raise ValueError('the panorama vertical span exceeds -pi/2')
        if not self.pitch_at_centre_rad + half_vertical <= pi / 2.0:
            raise ValueError('the panorama vertical span exceeds +pi/2')
        if self.u_yaw_sign not in (-1, 1):
            raise ValueError('u_yaw_sign must be -1 or +1')

    @property
    def pitch_min_rad(self) -> float:
        """Return the bottom of the panorama's angular span."""
        return self.pitch_at_centre_rad - self.vertical_fov_rad / 2.0

    @property
    def pitch_max_rad(self) -> float:
        """Return the top of the panorama's angular span."""
        return self.pitch_at_centre_rad + self.vertical_fov_rad / 2.0


def rotation_camera_from_crop(yaw_rad: float, pitch_rad: float) -> np.ndarray:
    """Build a crop-to-camera rotation from crop-centre yaw and pitch."""
    if not all(isfinite(value) for value in (yaw_rad, pitch_rad)):
        raise ValueError('yaw and pitch must be finite')
    if not -pi / 2.0 < pitch_rad < pi / 2.0:
        raise ValueError('pitch must lie in (-pi/2, pi/2)')
    forward = np.array(
        [cos(pitch_rad) * cos(yaw_rad), cos(pitch_rad) * sin(yaw_rad), sin(pitch_rad)],
        dtype=np.float64,
    )
    left = np.array([-sin(yaw_rad), cos(yaw_rad), 0.0], dtype=np.float64)
    up = np.cross(forward, left)
    return np.column_stack((forward, left, up))


def make_perspective_geometry(
    *,
    crop_id: int,
    width: int,
    height: int,
    yaw_rad: float,
    pitch_rad: float,
    horizontal_fov_rad: float,
    vertical_fov_rad: float,
) -> PerspectiveGeometry:
    """Construct validated perspective metadata and its retained rotation."""
    yaw = float(wrap_angle(yaw_rad))
    return PerspectiveGeometry(
        crop_id=crop_id,
        width=width,
        height=height,
        yaw_rad=yaw,
        pitch_rad=pitch_rad,
        horizontal_fov_rad=horizontal_fov_rad,
        vertical_fov_rad=vertical_fov_rad,
        rotation_camera_from_crop=rotation_camera_from_crop(yaw, pitch_rad),
    )


def _points(name: str, value: np.ndarray) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.shape[-1:] != (2,) or not np.all(np.isfinite(points)):
        raise ValueError(f'{name} must have shape (..., 2) with finite values')
    return points


def _rays(name: str, value: np.ndarray) -> np.ndarray:
    rays = np.asarray(value, dtype=np.float64)
    if rays.shape[-1:] != (3,) or not np.all(np.isfinite(rays)):
        raise ValueError(f'{name} must have shape (..., 3) with finite values')
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    if np.any(norms <= _TOLERANCE):
        raise ValueError(f'{name} must not contain zero-length rays')
    return rays / norms


def crop_pixels_to_camera_rays(
    pixels_uv: np.ndarray,
    geometry: PerspectiveGeometry,
) -> np.ndarray:
    """Convert continuous crop pixel coordinates to normalized camera rays."""
    pixels = _points('pixels_uv', pixels_uv)
    u = pixels[..., 0]
    v = pixels[..., 1]
    if np.any((u < 0.0) | (u > geometry.width)):
        raise ValueError('crop u coordinates must lie in [0, width]')
    if np.any((v < 0.0) | (v > geometry.height)):
        raise ValueError('crop v coordinates must lie in [0, height]')

    crop_x = np.ones_like(u)
    crop_y = -(
        2.0 * u / geometry.width - 1.0
    ) * tan(geometry.horizontal_fov_rad / 2.0)
    crop_z = (
        1.0 - 2.0 * v / geometry.height
    ) * tan(geometry.vertical_fov_rad / 2.0)
    crop_rays = np.stack((crop_x, crop_y, crop_z), axis=-1)
    crop_rays = _rays('crop rays', crop_rays)
    camera_rays = crop_rays @ geometry.rotation_camera_from_crop.T
    return _rays('camera rays', camera_rays)


def camera_rays_to_crop_pixels(
    rays_camera: np.ndarray,
    geometry: PerspectiveGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Project camera rays into one crop and return pixels plus visibility."""
    rays = _rays('rays_camera', rays_camera)
    crop_rays = rays @ geometry.rotation_camera_from_crop
    depth = crop_rays[..., 0]
    safe_depth = np.where(np.abs(depth) > _TOLERANCE, depth, 1.0)
    horizontal = crop_rays[..., 1] / safe_depth
    vertical = crop_rays[..., 2] / safe_depth
    u = geometry.width * (
        1.0 - horizontal / tan(geometry.horizontal_fov_rad / 2.0)
    ) / 2.0
    v = geometry.height * (
        1.0 - vertical / tan(geometry.vertical_fov_rad / 2.0)
    ) / 2.0
    pixels = np.stack((u, v), axis=-1)
    visible = (
        (depth > _TOLERANCE)
        & (u >= -_TOLERANCE)
        & (u <= geometry.width + _TOLERANCE)
        & (v >= -_TOLERANCE)
        & (v <= geometry.height + _TOLERANCE)
    )
    return pixels, visible


def camera_rays_to_panorama_pixels(
    rays_camera: np.ndarray,
    model: PanoramaCameraModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert normalized camera rays to panorama pixels and validity flags."""
    rays = _rays('rays_camera', rays_camera)
    yaw = np.arctan2(rays[..., 1], rays[..., 0])
    pitch = np.arcsin(np.clip(rays[..., 2], -1.0, 1.0))
    yaw_delta = wrap_angle(yaw - model.yaw_at_centre_rad)
    u = model.width * (
        0.5 + model.u_yaw_sign * yaw_delta / (2.0 * pi)
    )
    u = np.mod(u, model.width)
    v = model.height * (
        0.5 - (pitch - model.pitch_at_centre_rad) / model.vertical_fov_rad
    )
    valid = (
        (pitch >= model.pitch_min_rad - _TOLERANCE)
        & (pitch <= model.pitch_max_rad + _TOLERANCE)
    )
    return np.stack((u, v), axis=-1), valid


def panorama_pixels_to_camera_rays(
    pixels_uv: np.ndarray,
    model: PanoramaCameraModel,
) -> np.ndarray:
    """Convert continuous panorama pixel coordinates to normalized rays."""
    pixels = _points('pixels_uv', pixels_uv)
    u = np.mod(pixels[..., 0], model.width)
    v = pixels[..., 1]
    if np.any((v < 0.0) | (v > model.height)):
        raise ValueError('panorama v coordinates must lie in [0, height]')
    yaw = model.yaw_at_centre_rad + model.u_yaw_sign * (
        u / model.width - 0.5
    ) * 2.0 * pi
    pitch = model.pitch_at_centre_rad + (
        0.5 - v / model.height
    ) * model.vertical_fov_rad
    cos_pitch = np.cos(pitch)
    rays = np.stack(
        (
            cos_pitch * np.cos(yaw),
            cos_pitch * np.sin(yaw),
            np.sin(pitch),
        ),
        axis=-1,
    )
    return _rays('camera rays', rays)


def crop_pixels_to_panorama_pixels(
    pixels_uv: np.ndarray,
    geometry: PerspectiveGeometry,
    model: PanoramaCameraModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Map crop pixels into panorama coordinates through retained camera rays."""
    rays = crop_pixels_to_camera_rays(pixels_uv, geometry)
    return camera_rays_to_panorama_pixels(rays, model)


def panorama_pixels_to_crop_pixels(
    pixels_uv: np.ndarray,
    model: PanoramaCameraModel,
    geometry: PerspectiveGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Map panorama pixels into a selected perspective crop."""
    rays = panorama_pixels_to_camera_rays(pixels_uv, model)
    return camera_rays_to_crop_pixels(rays, geometry)


def _minimum_circular_intervals(
    values: np.ndarray,
    width: int,
) -> tuple[tuple[float, float], ...]:
    ordered = np.sort(np.mod(np.asarray(values, dtype=np.float64), width))
    if ordered.size == 0:
        raise ValueError('at least one horizontal sample is required')
    gaps = np.diff(np.concatenate((ordered, ordered[:1] + width)))
    largest_gap_index = int(np.argmax(gaps))
    start = float(ordered[(largest_gap_index + 1) % ordered.size])
    end = float(ordered[largest_gap_index])
    if start <= end:
        return ((start, end),)
    return ((0.0, end), (start, float(width)))


def project_crop_box_to_panorama(
    bbox_xyxy: tuple[float, float, float, float],
    geometry: PerspectiveGeometry,
    model: PanoramaCameraModel,
) -> PanoramaBox:
    """Project four corners and four edge midpoints into a wrap-aware box."""
    x_min, y_min, x_max, y_max = (float(value) for value in bbox_xyxy)
    if not all(isfinite(value) for value in (x_min, y_min, x_max, y_max)):
        raise ValueError('bbox_xyxy must contain finite values')
    if not 0.0 <= x_min < x_max <= geometry.width:
        raise ValueError('bbox x coordinates must lie inside the crop')
    if not 0.0 <= y_min < y_max <= geometry.height:
        raise ValueError('bbox y coordinates must lie inside the crop')
    x_mid = (x_min + x_max) / 2.0
    y_mid = (y_min + y_max) / 2.0
    boundary = np.array(
        [
            (x_min, y_min),
            (x_mid, y_min),
            (x_max, y_min),
            (x_max, y_mid),
            (x_max, y_max),
            (x_mid, y_max),
            (x_min, y_max),
            (x_min, y_mid),
        ],
        dtype=np.float64,
    )
    panorama_uv, valid = crop_pixels_to_panorama_pixels(boundary, geometry, model)
    if not np.all(valid):
        raise ValueError('crop box projects outside the panorama vertical span')
    panorama_uv[:, 1] = np.clip(panorama_uv[:, 1], 0.0, model.height)
    return PanoramaBox(
        panorama_width=model.width,
        panorama_height=model.height,
        x_intervals=_minimum_circular_intervals(panorama_uv[:, 0], model.width),
        y_min=float(np.min(panorama_uv[:, 1])),
        y_max=float(np.max(panorama_uv[:, 1])),
        boundary_uv=panorama_uv,
    )
