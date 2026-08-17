"""Detector-independent records for panoramic 2D perception."""

from dataclasses import dataclass, field
from math import isfinite, pi
from types import MappingProxyType
from typing import Mapping

import numpy as np


def _non_empty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')
    return value


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _finite_angle(name: str, value: float, *, maximum: float) -> float:
    value = float(value)
    if not isfinite(value) or not 0.0 < value < maximum:
        raise ValueError(f'{name} must be finite and in (0, {maximum})')
    return value


def _readonly_array(
    name: str,
    value: np.ndarray,
    *,
    shape: tuple[int | None, ...],
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape)
    ):
        raise ValueError(f'{name} must have shape {shape}')
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise ValueError(f'{name} must have a numeric or boolean dtype')
    if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must contain only finite values')
    copied = np.ascontiguousarray(array).copy()
    copied.setflags(write=False)
    return copied


def _box_xyxy(name: str, value: tuple[float, float, float, float]) -> tuple[float, ...]:
    values = tuple(float(item) for item in value)
    if len(values) != 4 or not all(isfinite(item) for item in values):
        raise ValueError(f'{name} must contain four finite values')
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError(f'{name} must satisfy x_max > x_min and y_max > y_min')
    return values


@dataclass(frozen=True)
class DetectorClass:
    """One canonical detector class and its measured text prompts."""

    canonical_name: str
    prompts: tuple[str, ...]

    def __post_init__(self) -> None:
        _non_empty('canonical_name', self.canonical_name)
        prompts = tuple(self.prompts)
        if not prompts:
            raise ValueError('prompts must contain at least one prompt')
        for prompt in prompts:
            _non_empty('prompt', prompt)
        if len({prompt.casefold() for prompt in prompts}) != len(prompts):
            raise ValueError('prompts must not contain case-insensitive duplicates')
        object.__setattr__(self, 'prompts', prompts)


@dataclass(frozen=True)
class PerceptionRequest:
    """One query-conditioned panoramic observation request."""

    image_id: str
    timestamp_ns: int
    panorama_rgb: np.ndarray
    detector_classes: tuple[DetectorClass, ...]
    task_type: str
    viewpoint_id: str | None = None
    source_encoding: str = 'rgb8'

    def __post_init__(self) -> None:
        _non_empty('image_id', self.image_id)
        if (
            isinstance(self.timestamp_ns, bool)
            or not isinstance(self.timestamp_ns, int)
            or self.timestamp_ns < 0
        ):
            raise ValueError('timestamp_ns must be a non-negative integer')
        _non_empty('task_type', self.task_type)
        _non_empty('source_encoding', self.source_encoding)
        if self.viewpoint_id is not None:
            _non_empty('viewpoint_id', self.viewpoint_id)

        panorama = _readonly_array(
            'panorama_rgb',
            self.panorama_rgb,
            shape=(None, None, 3),
        )
        if panorama.shape[0] < 2 or panorama.shape[1] < 2:
            raise ValueError('panorama_rgb dimensions must be at least 2 x 2')
        object.__setattr__(self, 'panorama_rgb', panorama)

        classes = tuple(self.detector_classes)
        if not classes or not all(isinstance(item, DetectorClass) for item in classes):
            raise ValueError('detector_classes must contain DetectorClass values')
        names = [item.canonical_name for item in classes]
        if len(names) != len(set(names)):
            raise ValueError('detector canonical names must be unique')
        object.__setattr__(self, 'detector_classes', classes)


@dataclass(frozen=True)
class PerspectiveGeometry:
    """Analytic geometry retained for one perspective crop."""

    crop_id: int
    width: int
    height: int
    yaw_rad: float
    pitch_rad: float
    horizontal_fov_rad: float
    vertical_fov_rad: float
    rotation_camera_from_crop: np.ndarray

    def __post_init__(self) -> None:
        if isinstance(self.crop_id, bool) or not isinstance(self.crop_id, int):
            raise ValueError('crop_id must be a non-negative integer')
        if self.crop_id < 0:
            raise ValueError('crop_id must be a non-negative integer')
        _positive_int('width', self.width)
        _positive_int('height', self.height)
        if not isfinite(self.yaw_rad) or not -pi <= self.yaw_rad <= pi:
            raise ValueError('yaw_rad must be finite and in [-pi, pi]')
        if not isfinite(self.pitch_rad) or not -pi / 2.0 < self.pitch_rad < pi / 2.0:
            raise ValueError('pitch_rad must be finite and in (-pi/2, pi/2)')
        _finite_angle('horizontal_fov_rad', self.horizontal_fov_rad, maximum=pi)
        _finite_angle('vertical_fov_rad', self.vertical_fov_rad, maximum=pi)

        rotation = _readonly_array(
            'rotation_camera_from_crop',
            self.rotation_camera_from_crop,
            shape=(3, 3),
            dtype=np.float64,
        )
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9):
            raise ValueError('rotation_camera_from_crop must be orthonormal')
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9):
            raise ValueError('rotation_camera_from_crop must be right-handed')
        object.__setattr__(self, 'rotation_camera_from_crop', rotation)


@dataclass(frozen=True)
class PerspectiveView:
    """A detector-friendly crop and all geometry needed to invert it."""

    source_image_id: str
    image_rgb: np.ndarray
    geometry: PerspectiveGeometry
    valid_mask: np.ndarray

    def __post_init__(self) -> None:
        _non_empty('source_image_id', self.source_image_id)
        if not isinstance(self.geometry, PerspectiveGeometry):
            raise TypeError('geometry must be PerspectiveGeometry')
        image = _readonly_array(
            'image_rgb',
            self.image_rgb,
            shape=(self.geometry.height, self.geometry.width, 3),
        )
        mask = _readonly_array(
            'valid_mask',
            self.valid_mask,
            shape=(self.geometry.height, self.geometry.width),
            dtype=np.bool_,
        )
        object.__setattr__(self, 'image_rgb', image)
        object.__setattr__(self, 'valid_mask', mask)


@dataclass(frozen=True)
class PanoramaBox:
    """A crop box projected into a horizontally wrap-aware panorama envelope."""

    panorama_width: int
    panorama_height: int
    x_intervals: tuple[tuple[float, float], ...]
    y_min: float
    y_max: float
    boundary_uv: np.ndarray

    def __post_init__(self) -> None:
        _positive_int('panorama_width', self.panorama_width)
        _positive_int('panorama_height', self.panorama_height)
        intervals = tuple(tuple(float(value) for value in item) for item in self.x_intervals)
        if not 1 <= len(intervals) <= 2:
            raise ValueError('x_intervals must contain one or two intervals')
        for interval in intervals:
            if len(interval) != 2 or not all(isfinite(value) for value in interval):
                raise ValueError('x intervals must contain two finite values')
            if not 0.0 <= interval[0] <= interval[1] <= self.panorama_width:
                raise ValueError('x intervals must lie within the panorama')
        if len(intervals) == 2 and (
            intervals[0][0] != 0.0
            or intervals[1][1] != float(self.panorama_width)
        ):
            raise ValueError('two x intervals must meet opposite panorama edges')
        if not all(isfinite(value) for value in (self.y_min, self.y_max)):
            raise ValueError('vertical bounds must be finite')
        if not 0.0 <= self.y_min <= self.y_max <= self.panorama_height:
            raise ValueError('vertical bounds must lie within the panorama')
        boundary = _readonly_array(
            'boundary_uv',
            self.boundary_uv,
            shape=(None, 2),
            dtype=np.float64,
        )
        if boundary.shape[0] < 4:
            raise ValueError('boundary_uv must contain at least four samples')
        object.__setattr__(self, 'x_intervals', intervals)
        object.__setattr__(self, 'boundary_uv', boundary)

    @property
    def crosses_seam(self) -> bool:
        """Return whether horizontal support is split across the panorama seam."""
        return len(self.x_intervals) == 2


@dataclass(frozen=True)
class CropDetection:
    """One model-normalized bounding-box detection in perspective pixels."""

    crop_id: int
    canonical_name: str
    prompt_used: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.crop_id, bool) or not isinstance(self.crop_id, int):
            raise ValueError('crop_id must be a non-negative integer')
        if self.crop_id < 0:
            raise ValueError('crop_id must be a non-negative integer')
        _non_empty('canonical_name', self.canonical_name)
        _non_empty('prompt_used', self.prompt_used)
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError('confidence must be finite and in [0, 1]')
        object.__setattr__(self, 'bbox_xyxy', _box_xyxy('bbox_xyxy', self.bbox_xyxy))
        object.__setattr__(self, 'metadata', MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class Detection2D:
    """One detector-independent panorama detection after crop projection."""

    detection_id: str
    class_name: str
    prompt_used: str
    confidence: float
    panorama_box: PanoramaBox
    crop_ids: tuple[int, ...]
    crop_boxes_xyxy: tuple[tuple[float, float, float, float], ...]
    centre_panorama_uv: tuple[float, float]
    centre_camera_ray: np.ndarray
    seam_merged: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _non_empty('detection_id', self.detection_id)
        _non_empty('class_name', self.class_name)
        _non_empty('prompt_used', self.prompt_used)
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError('confidence must be finite and in [0, 1]')
        if not isinstance(self.panorama_box, PanoramaBox):
            raise TypeError('panorama_box must be PanoramaBox')
        crop_ids = tuple(self.crop_ids)
        if not crop_ids or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in crop_ids
        ):
            raise ValueError('crop_ids must contain non-negative integers')
        if len(crop_ids) != len(set(crop_ids)):
            raise ValueError('crop_ids must be unique')
        crop_boxes = tuple(
            _box_xyxy('crop box', value) for value in self.crop_boxes_xyxy
        )
        if len(crop_boxes) != len(crop_ids):
            raise ValueError('one crop box is required for every crop ID')
        centre_uv = tuple(float(value) for value in self.centre_panorama_uv)
        if len(centre_uv) != 2 or not all(isfinite(value) for value in centre_uv):
            raise ValueError('centre_panorama_uv must contain two finite values')
        if not 0.0 <= centre_uv[0] < self.panorama_box.panorama_width:
            raise ValueError('panorama centre u must lie in [0, width)')
        if not 0.0 <= centre_uv[1] <= self.panorama_box.panorama_height:
            raise ValueError('panorama centre v must lie in [0, height]')
        centre_ray = _readonly_array(
            'centre_camera_ray',
            self.centre_camera_ray,
            shape=(3,),
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(centre_ray))
        if not np.isclose(norm, 1.0, atol=1e-9):
            raise ValueError('centre_camera_ray must be normalized')
        object.__setattr__(self, 'crop_ids', crop_ids)
        object.__setattr__(self, 'crop_boxes_xyxy', crop_boxes)
        object.__setattr__(self, 'centre_panorama_uv', centre_uv)
        object.__setattr__(self, 'centre_camera_ray', centre_ray)
        object.__setattr__(self, 'metadata', MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class PerceptionResult:
    """Final worker output for one query-conditioned panorama keyframe."""

    image_id: str
    timestamp_ns: int
    crop_count: int
    raw_detections: tuple[Detection2D, ...]
    detections: tuple[Detection2D, ...]

    def __post_init__(self) -> None:
        _non_empty('image_id', self.image_id)
        if (
            isinstance(self.timestamp_ns, bool)
            or not isinstance(self.timestamp_ns, int)
            or self.timestamp_ns < 0
        ):
            raise ValueError('timestamp_ns must be a non-negative integer')
        _positive_int('crop_count', self.crop_count)
        raw = tuple(self.raw_detections)
        final = tuple(self.detections)
        if not all(isinstance(item, Detection2D) for item in raw + final):
            raise TypeError('perception outputs must contain Detection2D values')
        object.__setattr__(self, 'raw_detections', raw)
        object.__setattr__(self, 'detections', final)
