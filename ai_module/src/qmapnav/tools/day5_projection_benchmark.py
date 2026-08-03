"""Benchmark saved-case projection, crop mapping, drawing, and replay I/O."""

import argparse
import json
from pathlib import Path
from resource import getrusage
from resource import RUSAGE_SELF
from time import perf_counter

import numpy as np
from qmapnav.mapping import load_projection_regression_case
from qmapnav.mapping import replay_projection_regression_case
from qmapnav.mapping.lidar_camera_projection import project_map_points
from qmapnav.mapping.lidar_camera_projection import project_result_into_crops
from qmapnav.mapping.lidar_camera_projection import ProjectionConfig
from qmapnav.mapping.projection_visualisation import draw_projection_overlay
from qmapnav.perception import eight_view_layout
from qmapnav.perception import PanoramaCameraModel
from qmapnav.perception import PerspectiveCropGenerator


def main() -> None:
    """Run deterministic repeated timings and save percentile measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_directory', type=Path)
    parser.add_argument('output_json', type=Path)
    parser.add_argument('--repeat', type=int, default=100)
    arguments = parser.parse_args()
    if arguments.repeat <= 0:
        raise ValueError('--repeat must be positive')
    panorama, baseline, manifest = load_projection_regression_case(
        arguments.case_directory
    )
    arrays = np.load(arguments.case_directory / 'inputs.npz')
    model = PanoramaCameraModel(
        width=int(manifest['image']['width']),
        height=int(manifest['image']['height']),
        vertical_fov_rad=float(manifest['image']['vertical_fov_rad']),
        u_yaw_sign=int(manifest['image']['u_yaw_sign']),
    )
    config = ProjectionConfig(**manifest['projection_config'])
    crop_generator = PerspectiveCropGenerator(model, eight_view_layout())
    crop_generator.geometries()
    projection_ms = []
    crop_ms = []
    draw_ms = []
    result = baseline
    for _ in range(arguments.repeat):
        started = perf_counter()
        result = project_map_points(
            points_map_xyz=arrays['points_map_xyz'],
            transform_camera_internal_from_map=(
                baseline.transform_camera_internal_from_map
            ),
            panorama_model=model,
            image_id=baseline.image_id,
            image_timestamp_ns=baseline.image_timestamp_ns,
            scan_timestamp_ns=baseline.scan_timestamp_ns,
            intensity=(
                arrays['intensity']
                if bool(manifest['scan']['has_intensity'])
                else None
            ),
            config=config,
        )
        projection_ms.append((perf_counter() - started) * 1000.0)
        started = perf_counter()
        project_result_into_crops(result, crop_generator.geometries())
        crop_ms.append((perf_counter() - started) * 1000.0)
        started = perf_counter()
        draw_projection_overlay(panorama, result)
        draw_ms.append((perf_counter() - started) * 1000.0)
    replay_ms = []
    for _ in range(min(10, arguments.repeat)):
        started = perf_counter()
        replay_projection_regression_case(arguments.case_directory)
        replay_ms.append((perf_counter() - started) * 1000.0)
    payload = {
        'case_directory': str(arguments.case_directory),
        'repeat': arguments.repeat,
        'input_point_count': int(arrays['points_map_xyz'].shape[0]),
        'projected_point_count': result.point_count,
        'projection_ms': _stats(projection_ms),
        'eight_crop_mapping_ms': _stats(crop_ms),
        'depth_overlay_ms': _stats(draw_ms),
        'full_saved_replay_io_ms': _stats(replay_ms),
        'peak_rss_kib_process': int(getrusage(RUSAGE_SELF).ru_maxrss),
    }
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        'median': float(np.median(array)),
        'p95': float(np.percentile(array, 95)),
        'maximum': float(np.max(array)),
    }


if __name__ == '__main__':
    main()
