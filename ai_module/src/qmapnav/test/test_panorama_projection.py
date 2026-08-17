"""Tests for panorama, perspective, and camera-ray geometry."""

from math import pi

import numpy as np
import pytest

from qmapnav.perception.panorama_projection import camera_rays_to_crop_pixels
from qmapnav.perception.panorama_projection import camera_rays_to_panorama_pixels
from qmapnav.perception.panorama_projection import crop_pixels_to_camera_rays
from qmapnav.perception.panorama_projection import crop_pixels_to_panorama_pixels
from qmapnav.perception.panorama_projection import make_perspective_geometry
from qmapnav.perception.panorama_projection import panorama_pixels_to_camera_rays
from qmapnav.perception.panorama_projection import panorama_pixels_to_crop_pixels
from qmapnav.perception.panorama_projection import PanoramaCameraModel
from qmapnav.perception.panorama_projection import project_crop_box_to_panorama
from qmapnav.perception.panorama_projection import rotation_camera_from_crop


def _geometry(yaw: float = 0.0, pitch: float = 0.0):
    return make_perspective_geometry(
        crop_id=0,
        width=640,
        height=640,
        yaw_rad=yaw,
        pitch_rad=pitch,
        horizontal_fov_rad=pi / 3.0,
        vertical_fov_rad=pi / 2.0,
    )


def test_panorama_model_uses_challenge_120_degree_vertical_span() -> None:
    model = PanoramaCameraModel(1920, 640)
    pixels = np.array(((960.0, 320.0), (960.0, 0.0), (960.0, 640.0)))

    rays = panorama_pixels_to_camera_rays(pixels, model)

    assert rays[0] == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)
    assert np.arcsin(rays[1, 2]) == pytest.approx(pi / 3.0)
    assert np.arcsin(rays[2, 2]) == pytest.approx(-pi / 3.0)


def test_panorama_horizontal_direction_and_wrap_are_explicit() -> None:
    model = PanoramaCameraModel(1920, 640, u_yaw_sign=-1)
    rays = np.array(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (-1.0, 1e-12, 0.0),
            (-1.0, -1e-12, 0.0),
        )
    )

    pixels, valid = camera_rays_to_panorama_pixels(rays, model)

    assert np.all(valid)
    assert pixels[0] == pytest.approx((960.0, 320.0))
    assert pixels[1, 0] == pytest.approx(480.0)
    assert pixels[2, 0] == pytest.approx(1440.0)
    assert min(pixels[3, 0], 1920.0 - pixels[3, 0]) < 1e-8
    assert min(pixels[4, 0], 1920.0 - pixels[4, 0]) < 1e-8


def test_crop_centre_ray_matches_configured_yaw_and_pitch() -> None:
    yaw = pi / 4.0
    pitch = pi / 12.0
    geometry = _geometry(yaw=yaw, pitch=pitch)

    ray = crop_pixels_to_camera_rays(np.array((320.0, 320.0)), geometry)
    expected = (
        np.cos(pitch) * np.cos(yaw),
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch),
    )

    assert ray == pytest.approx(expected, abs=1e-12)


def test_crop_pixel_ray_round_trip_is_subpixel_accurate() -> None:
    geometry = _geometry(yaw=-2.6, pitch=0.2)
    pixels = np.array(
        (
            (0.0, 0.0),
            (640.0, 0.0),
            (640.0, 640.0),
            (0.0, 640.0),
            (123.25, 456.75),
            (320.0, 320.0),
        )
    )

    rays = crop_pixels_to_camera_rays(pixels, geometry)
    reconstructed, visible = camera_rays_to_crop_pixels(rays, geometry)

    assert np.all(visible)
    assert reconstructed == pytest.approx(pixels, abs=1e-9)


def test_panorama_pixel_ray_round_trip_handles_horizontal_modulo() -> None:
    model = PanoramaCameraModel(1920, 640)
    pixels = np.array(
        (
            (0.0, 320.0),
            (1.0, 0.0),
            (480.0, 100.0),
            (960.0, 320.0),
            (1919.0, 639.0),
            (1920.0, 320.0),
        )
    )

    rays = panorama_pixels_to_camera_rays(pixels, model)
    reconstructed, valid = camera_rays_to_panorama_pixels(rays, model)

    expected = pixels.copy()
    expected[:, 0] %= model.width
    assert np.all(valid)
    assert reconstructed == pytest.approx(expected, abs=1e-9)


def test_crop_and_panorama_round_trip_preserves_shared_coordinates() -> None:
    model = PanoramaCameraModel(1920, 640)
    geometry = _geometry(yaw=pi / 4.0)
    crop_pixels = np.array(((20.0, 30.0), (320.0, 320.0), (620.0, 610.0)))

    panorama_pixels, valid_panorama = crop_pixels_to_panorama_pixels(
        crop_pixels,
        geometry,
        model,
    )
    reconstructed, valid_crop = panorama_pixels_to_crop_pixels(
        panorama_pixels,
        model,
        geometry,
    )

    assert np.all(valid_panorama)
    assert np.all(valid_crop)
    assert reconstructed == pytest.approx(crop_pixels, abs=1e-9)


def test_projected_box_at_panorama_boundary_uses_two_intervals() -> None:
    model = PanoramaCameraModel(1920, 640)
    geometry = _geometry(yaw=pi)

    projected = project_crop_box_to_panorama(
        (260.0, 200.0, 380.0, 440.0),
        geometry,
        model,
    )

    assert projected.crosses_seam
    assert projected.x_intervals[0][0] == 0.0
    assert projected.x_intervals[1][1] == 1920.0
    covered_width = sum(end - start for start, end in projected.x_intervals)
    assert covered_width < 400.0
    assert projected.boundary_uv.shape == (8, 2)


def test_rotation_is_right_handed_for_arbitrary_crop() -> None:
    rotation = rotation_camera_from_crop(-1.2, 0.4)

    assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_invalid_vertical_mapping_and_zero_ray_fail_clearly() -> None:
    model = PanoramaCameraModel(1920, 640)

    with pytest.raises(ValueError, match='v coordinates'):
        panorama_pixels_to_camera_rays(np.array((10.0, -0.1)), model)
    with pytest.raises(ValueError, match='zero-length'):
        camera_rays_to_panorama_pixels(np.zeros(3), model)
