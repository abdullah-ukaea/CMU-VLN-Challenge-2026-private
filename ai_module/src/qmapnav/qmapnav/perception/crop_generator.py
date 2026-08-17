"""Configurable inverse-mapped perspective crops for 360-degree panoramas."""

from dataclasses import dataclass
from math import isfinite, pi

import numpy as np

try:
    import cv2
except ImportError:  # Keep geometry usable without detector-only dependencies.
    cv2 = None

from qmapnav.perception.contracts import PerspectiveGeometry
from qmapnav.perception.contracts import PerspectiveView
from qmapnav.perception.panorama_projection import camera_rays_to_panorama_pixels
from qmapnav.perception.panorama_projection import crop_pixels_to_camera_rays
from qmapnav.perception.panorama_projection import make_perspective_geometry
from qmapnav.perception.panorama_projection import PanoramaCameraModel
from qmapnav.perception.panorama_projection import wrap_angle


@dataclass(frozen=True)
class PerspectiveCropLayout:
    """A deterministic set of overlapping yaw and pitch crop centres."""

    yaw_centres_rad: tuple[float, ...]
    pitch_centres_rad: tuple[float, ...]
    horizontal_fov_rad: float
    vertical_fov_rad: float
    output_width: int
    output_height: int

    def __post_init__(self) -> None:
        yaws = tuple(float(wrap_angle(value)) for value in self.yaw_centres_rad)
        pitches = tuple(float(value) for value in self.pitch_centres_rad)
        if not yaws:
            raise ValueError('yaw_centres_rad must not be empty')
        if not pitches:
            raise ValueError('pitch_centres_rad must not be empty')
        if not all(isfinite(value) for value in yaws + pitches):
            raise ValueError('crop centres must be finite')
        rounded_yaws = {round(value, 12) for value in yaws}
        if len(rounded_yaws) != len(yaws):
            raise ValueError('yaw crop centres must be unique modulo 2*pi')
        if any(not -pi / 2.0 < value < pi / 2.0 for value in pitches):
            raise ValueError('pitch crop centres must lie in (-pi/2, pi/2)')
        if not 0.0 < self.horizontal_fov_rad < pi:
            raise ValueError('horizontal_fov_rad must lie in (0, pi)')
        if not 0.0 < self.vertical_fov_rad < pi:
            raise ValueError('vertical_fov_rad must lie in (0, pi)')
        if (
            isinstance(self.output_width, bool)
            or not isinstance(self.output_width, int)
            or self.output_width <= 0
        ):
            raise ValueError('output_width must be a positive integer')
        if (
            isinstance(self.output_height, bool)
            or not isinstance(self.output_height, int)
            or self.output_height <= 0
        ):
            raise ValueError('output_height must be a positive integer')
        object.__setattr__(self, 'yaw_centres_rad', yaws)
        object.__setattr__(self, 'pitch_centres_rad', pitches)

    @property
    def crop_count(self) -> int:
        """Return total crops in pitch-major, then yaw-major order."""
        return len(self.pitch_centres_rad) * len(self.yaw_centres_rad)

    @property
    def horizontal_overlap_fraction(self) -> float | None:
        """Return overlap for an evenly spaced one-circle yaw layout."""
        if len(self.yaw_centres_rad) < 2:
            return None
        ordered = np.sort(np.mod(self.yaw_centres_rad, 2.0 * pi))
        spacings = np.diff(np.concatenate((ordered, ordered[:1] + 2.0 * pi)))
        if not np.allclose(spacings, spacings[0], atol=1e-9):
            return None
        return max(0.0, 1.0 - float(spacings[0]) / self.horizontal_fov_rad)


def eight_view_layout(
    *,
    output_width: int = 640,
    output_height: int = 640,
    horizontal_fov_rad: float = pi / 3.0,
    vertical_fov_rad: float = pi / 2.0,
    pitch_rad: float = 0.0,
) -> PerspectiveCropLayout:
    """
    Create the initial eight-yaw, one-row perception layout.

    Sixty-degree horizontal views spaced every 45 degrees produce 25 percent
    horizontal overlap. The initial 90-degree vertical view covers the useful
    central portion of the challenge's 120-degree panorama; pitch rows and FOV
    remain configurable rather than pretending this covers the full height.
    """
    return PerspectiveCropLayout(
        yaw_centres_rad=tuple(index * pi / 4.0 for index in range(8)),
        pitch_centres_rad=(pitch_rad,),
        horizontal_fov_rad=horizontal_fov_rad,
        vertical_fov_rad=vertical_fov_rad,
        output_width=output_width,
        output_height=output_height,
    )


@dataclass(frozen=True)
class _Remap:
    geometry: PerspectiveGeometry
    panorama_uv: np.ndarray
    valid_mask: np.ndarray


class PerspectiveCropGenerator:
    """Generate repeated crops while caching their image-independent ray maps."""

    def __init__(
        self,
        panorama_model: PanoramaCameraModel,
        layout: PerspectiveCropLayout,
    ) -> None:
        if not isinstance(panorama_model, PanoramaCameraModel):
            raise TypeError('panorama_model must be PanoramaCameraModel')
        if not isinstance(layout, PerspectiveCropLayout):
            raise TypeError('layout must be PerspectiveCropLayout')
        self._panorama_model = panorama_model
        self._layout = layout
        self._remaps: tuple[_Remap, ...] | None = None

    @property
    def panorama_model(self) -> PanoramaCameraModel:
        """Return the immutable panorama model."""
        return self._panorama_model

    @property
    def layout(self) -> PerspectiveCropLayout:
        """Return the immutable crop layout."""
        return self._layout

    def geometries(self) -> tuple[PerspectiveGeometry, ...]:
        """Return retained geometry in deterministic crop-ID order."""
        return tuple(remap.geometry for remap in self._get_remaps())

    def generate(
        self,
        panorama_rgb: np.ndarray,
        *,
        source_image_id: str,
    ) -> tuple[PerspectiveView, ...]:
        """Convert one RGB panorama into all configured perspective views."""
        if not isinstance(source_image_id, str) or not source_image_id.strip():
            raise ValueError('source_image_id must be a non-empty string')
        panorama = np.asarray(panorama_rgb)
        expected_shape = (
            self._panorama_model.height,
            self._panorama_model.width,
            3,
        )
        if panorama.shape != expected_shape:
            raise ValueError(f'panorama_rgb must have shape {expected_shape}')
        if not np.issubdtype(panorama.dtype, np.number):
            raise ValueError('panorama_rgb must have a numeric dtype')
        if not np.all(np.isfinite(panorama)):
            raise ValueError('panorama_rgb must contain finite values')

        views = []
        for remap in self._get_remaps():
            crop = _bilinear_sample_wrap_horizontal(
                panorama,
                remap.panorama_uv,
                remap.valid_mask,
            )
            views.append(
                PerspectiveView(
                    source_image_id=source_image_id,
                    image_rgb=crop,
                    geometry=remap.geometry,
                    valid_mask=remap.valid_mask,
                )
            )
        return tuple(views)

    def _get_remaps(self) -> tuple[_Remap, ...]:
        if self._remaps is None:
            self._remaps = self._build_remaps()
        return self._remaps

    def _build_remaps(self) -> tuple[_Remap, ...]:
        u = np.arange(self._layout.output_width, dtype=np.float64) + 0.5
        v = np.arange(self._layout.output_height, dtype=np.float64) + 0.5
        grid_u, grid_v = np.meshgrid(u, v)
        crop_pixels = np.stack((grid_u, grid_v), axis=-1)
        remaps = []
        crop_id = 0
        for pitch in self._layout.pitch_centres_rad:
            for yaw in self._layout.yaw_centres_rad:
                geometry = make_perspective_geometry(
                    crop_id=crop_id,
                    width=self._layout.output_width,
                    height=self._layout.output_height,
                    yaw_rad=yaw,
                    pitch_rad=pitch,
                    horizontal_fov_rad=self._layout.horizontal_fov_rad,
                    vertical_fov_rad=self._layout.vertical_fov_rad,
                )
                rays = crop_pixels_to_camera_rays(crop_pixels, geometry)
                panorama_uv, valid = camera_rays_to_panorama_pixels(
                    rays,
                    self._panorama_model,
                )
                panorama_uv.setflags(write=False)
                valid.setflags(write=False)
                remaps.append(_Remap(geometry, panorama_uv, valid))
                crop_id += 1
        return tuple(remaps)


def _bilinear_sample_wrap_horizontal(
    image: np.ndarray,
    pixels_uv: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Sample an image at continuous pixel-edge coordinates."""
    if cv2 is not None:
        try:
            return _opencv_bilinear_sample(image, pixels_uv, valid_mask)
        except cv2.error:
            pass
    return _numpy_bilinear_sample(image, pixels_uv, valid_mask)


def _opencv_bilinear_sample(
    image: np.ndarray,
    pixels_uv: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Use optimized remapping with explicit horizontal seam padding."""
    horizontally_wrapped = np.concatenate(
        (image[:, -1:], image, image[:, :1]),
        axis=1,
    )
    map_x = np.asarray(pixels_uv[..., 0] + 0.5, dtype=np.float32)
    map_y = np.asarray(pixels_uv[..., 1] - 0.5, dtype=np.float32)
    sampled = cv2.remap(
        horizontally_wrapped,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    sampled[~valid_mask] = 0
    return sampled


def _numpy_bilinear_sample(
    image: np.ndarray,
    pixels_uv: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Provide the dependency-free reference implementation."""
    height, width, _ = image.shape
    x = pixels_uv[..., 0] - 0.5
    y = pixels_uv[..., 1] - 0.5
    x_floor = np.floor(x).astype(np.int64)
    y_floor = np.floor(y).astype(np.int64)
    x_weight = x - x_floor
    y_weight = y - y_floor

    x0 = np.mod(x_floor, width)
    x1 = np.mod(x_floor + 1, width)
    y0 = np.clip(y_floor, 0, height - 1)
    y1 = np.clip(y_floor + 1, 0, height - 1)

    image_float = image.astype(np.float64, copy=False)
    top = (
        image_float[y0, x0] * (1.0 - x_weight[..., None])
        + image_float[y0, x1] * x_weight[..., None]
    )
    bottom = (
        image_float[y1, x0] * (1.0 - x_weight[..., None])
        + image_float[y1, x1] * x_weight[..., None]
    )
    sampled = top * (1.0 - y_weight[..., None]) + bottom * y_weight[..., None]
    sampled = np.where(valid_mask[..., None], sampled, 0.0)

    if np.issubdtype(image.dtype, np.integer):
        limits = np.iinfo(image.dtype)
        sampled = np.clip(np.rint(sampled), limits.min, limits.max)
    return sampled.astype(image.dtype, copy=False)
