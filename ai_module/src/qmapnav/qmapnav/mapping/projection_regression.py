"""Save and replay source-complete Day 5 projection regression cases."""

from dataclasses import asdict
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

import cv2
import numpy as np

from qmapnav.mapping.lidar_camera_projection import project_map_points
from qmapnav.mapping.lidar_camera_projection import ProjectionConfig
from qmapnav.mapping.lidar_camera_projection import ProjectionResult
from qmapnav.mapping.projection_pipeline import ProjectionFrame
from qmapnav.mapping.transforms import camera_internal_from_map
from qmapnav.mapping.transforms import transform_from_pose
from qmapnav.perception.panorama_projection import PanoramaCameraModel


DAY5_REGRESSION_CATEGORIES = (
    'nearby_furniture',
    'walls',
    'tabletop_objects',
    'panorama_seams',
    'sparse_detections',
)


@dataclass(frozen=True)
class ProjectionRegressionMetrics:
    """Pixel discrepancy and point-count checks from one replay."""

    sample_count: int
    missing_sample_count: int
    median_pixel_error: float
    maximum_pixel_error: float
    projected_point_count: int
    baseline_projected_point_count: int
    passed: bool


def save_projection_regression_case(
    directory: str | Path,
    *,
    category: str,
    scene_id: str,
    pose_id: str,
    frame: ProjectionFrame,
    transform_sensor_from_camera_optical: np.ndarray,
    panorama_model: PanoramaCameraModel,
    projection_config: ProjectionConfig,
    overlay_rgb: np.ndarray,
    notes: str,
    max_baseline_samples: int = 2048,
) -> Path:
    """Save raw inputs, source metadata, overlay, and sampled expected pixels."""
    if category not in DAY5_REGRESSION_CATEGORIES:
        raise ValueError(f'unknown Day 5 regression category {category!r}')
    if not scene_id or not pose_id or not notes:
        raise ValueError('scene_id, pose_id, and notes must be non-empty')
    if max_baseline_samples <= 0:
        raise ValueError('max_baseline_samples must be positive')
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    current = frame.current
    if current.point_count:
        sample_positions = np.linspace(
            0,
            current.point_count - 1,
            min(max_baseline_samples, current.point_count),
            dtype=np.int64,
        )
        sample_source_indices = current.source_point_indices[sample_positions]
        sample_uv = current.panorama_uv[sample_positions]
    else:
        sample_source_indices = np.empty((0,), dtype=np.int64)
        sample_uv = np.empty((0, 2), dtype=np.float64)
    scan = frame.association.scan
    np.savez_compressed(
        target / 'inputs.npz',
        points_map_xyz=scan.points_xyz,
        intensity=(
            scan.intensity
            if scan.intensity is not None
            else np.empty((0,), dtype=np.float64)
        ),
        sample_source_indices=sample_source_indices,
        sample_panorama_uv=sample_uv,
    )
    panorama_bgr = np.ascontiguousarray(frame.panorama.image_rgb[..., ::-1])
    overlay_bgr = np.ascontiguousarray(np.asarray(overlay_rgb)[..., ::-1])
    if not cv2.imwrite(str(target / 'panorama.png'), panorama_bgr):
        raise RuntimeError('failed to save regression panorama')
    if not cv2.imwrite(str(target / 'overlay.png'), overlay_bgr):
        raise RuntimeError('failed to save regression overlay')
    pose = frame.association.pose
    manifest = {
        'schema_version': 1,
        'category': category,
        'scene_id': scene_id,
        'pose_id': pose_id,
        'notes': notes,
        'image': {
            'image_id': frame.panorama.image_id,
            'timestamp_ns': frame.panorama.timestamp_ns,
            'frame_id': frame.panorama.frame_id,
            'width': panorama_model.width,
            'height': panorama_model.height,
            'vertical_fov_rad': panorama_model.vertical_fov_rad,
            'u_yaw_sign': panorama_model.u_yaw_sign,
        },
        'scan': {
            'timestamp_ns': scan.timestamp_ns,
            'frame_id': scan.frame_id,
            'point_count': int(scan.points_xyz.shape[0]),
            'has_intensity': scan.intensity is not None,
        },
        'pose': {
            'timestamp_ns': pose.timestamp_ns,
            'parent_frame_id': pose.parent_frame_id,
            'child_frame_id': pose.child_frame_id,
            'position_xyz': pose.position_xyz.tolist(),
            'orientation_xyzw': pose.orientation_xyzw.tolist(),
            'association_mode': frame.association.pose_mode,
            'before_delta_ns': frame.association.pose_before_delta_ns,
            'after_delta_ns': frame.association.pose_after_delta_ns,
        },
        'transform_sensor_from_camera_optical': np.asarray(
            transform_sensor_from_camera_optical,
            dtype=np.float64,
        ).tolist(),
        'projection_config': asdict(projection_config),
        'baseline': {
            'projected_point_count': current.point_count,
            'valid_fraction': current.diagnostics.valid_fraction,
            'sample_count': int(sample_source_indices.shape[0]),
        },
    }
    (target / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    checksum_files = ('inputs.npz', 'panorama.png', 'overlay.png', 'manifest.json')
    checksum_lines = [
        f'{_sha256_file(target / filename)}  {filename}'
        for filename in checksum_files
    ]
    (target / 'checksums.sha256').write_text(
        '\n'.join(checksum_lines) + '\n',
        encoding='utf-8',
    )
    return target


def load_projection_regression_case(
    directory: str | Path,
    *,
    transform_sensor_from_camera_override: np.ndarray | None = None,
    transform_camera_from_map_override: np.ndarray | None = None,
    u_yaw_sign_override: int | None = None,
) -> tuple[np.ndarray, ProjectionResult, dict[str, object]]:
    """Load a saved panorama and reproject its raw registered scan."""
    target = Path(directory)
    manifest = json.loads(
        (target / 'manifest.json').read_text(encoding='utf-8')
    )
    arrays = np.load(target / 'inputs.npz')
    image = manifest['image']
    pose = manifest['pose']
    scan = manifest['scan']
    extrinsic = (
        np.asarray(transform_sensor_from_camera_override, dtype=np.float64)
        if transform_sensor_from_camera_override is not None
        else np.asarray(
            manifest['transform_sensor_from_camera_optical'],
            dtype=np.float64,
        )
    )
    transform_camera_from_map = (
        np.asarray(transform_camera_from_map_override, dtype=np.float64)
        if transform_camera_from_map_override is not None
        else camera_internal_from_map(
            transform_from_pose(
                np.asarray(pose['position_xyz'], dtype=np.float64),
                np.asarray(pose['orientation_xyzw'], dtype=np.float64),
            ),
            extrinsic,
        )
    )
    model = PanoramaCameraModel(
        width=int(image['width']),
        height=int(image['height']),
        vertical_fov_rad=float(image['vertical_fov_rad']),
        u_yaw_sign=(
            int(u_yaw_sign_override)
            if u_yaw_sign_override is not None
            else int(image['u_yaw_sign'])
        ),
    )
    intensity = arrays['intensity']
    result = project_map_points(
        points_map_xyz=arrays['points_map_xyz'],
        transform_camera_internal_from_map=transform_camera_from_map,
        panorama_model=model,
        image_id=str(image['image_id']),
        image_timestamp_ns=int(image['timestamp_ns']),
        scan_timestamp_ns=int(scan['timestamp_ns']),
        intensity=intensity if bool(scan['has_intensity']) else None,
        pose_mode=str(pose['association_mode']),
        pose_before_delta_ns=pose['before_delta_ns'],
        pose_after_delta_ns=pose['after_delta_ns'],
        config=ProjectionConfig(**manifest['projection_config']),
    )
    panorama_bgr = cv2.imread(str(target / 'panorama.png'), cv2.IMREAD_COLOR)
    if panorama_bgr is None:
        raise ValueError('failed to load saved regression panorama')
    panorama_rgb = np.ascontiguousarray(panorama_bgr[..., ::-1])
    return panorama_rgb, result, manifest


def replay_projection_regression_case(
    directory: str | Path,
    *,
    pixel_tolerance: float = 1e-6,
    transform_sensor_from_camera_override: np.ndarray | None = None,
    transform_camera_from_map_override: np.ndarray | None = None,
    u_yaw_sign_override: int | None = None,
) -> ProjectionRegressionMetrics:
    """Replay one saved case and compare source-indexed panorama pixels."""
    if not isfinite(pixel_tolerance) or pixel_tolerance < 0.0:
        raise ValueError('pixel_tolerance must be finite and non-negative')
    target = Path(directory)
    _, result, manifest = load_projection_regression_case(
        target,
        transform_sensor_from_camera_override=(
            transform_sensor_from_camera_override
        ),
        transform_camera_from_map_override=transform_camera_from_map_override,
        u_yaw_sign_override=u_yaw_sign_override,
    )
    arrays = np.load(target / 'inputs.npz')
    sample_indices = arrays['sample_source_indices'].astype(np.int64)
    expected_uv = arrays['sample_panorama_uv']
    source_to_projection = {
        int(source): position
        for position, source in enumerate(result.source_point_indices)
    }
    present = np.array(
        [index in source_to_projection for index in sample_indices],
        dtype=np.bool_,
    )
    missing_count = int(np.count_nonzero(~present))
    if np.any(present):
        actual_uv = np.asarray(
            [
                result.panorama_uv[source_to_projection[int(index)]]
                for index in sample_indices[present]
            ]
        )
        delta = np.abs(actual_uv - expected_uv[present])
        width = int(manifest['image']['width'])
        delta[:, 0] = np.minimum(delta[:, 0], width - delta[:, 0])
        errors = np.linalg.norm(delta, axis=1)
        median_error = float(np.median(errors))
        maximum_error = float(np.max(errors))
    else:
        median_error = float('inf') if sample_indices.size else 0.0
        maximum_error = float('inf') if sample_indices.size else 0.0
    baseline_count = int(manifest['baseline']['projected_point_count'])
    passed = (
        missing_count == 0
        and maximum_error <= pixel_tolerance
        and result.point_count == baseline_count
    )
    return ProjectionRegressionMetrics(
        sample_count=int(sample_indices.shape[0]),
        missing_sample_count=missing_count,
        median_pixel_error=median_error,
        maximum_pixel_error=maximum_error,
        projected_point_count=result.point_count,
        baseline_projected_point_count=baseline_count,
        passed=passed,
    )


def verify_projection_regression_checksums(directory: str | Path) -> bool:
    """Verify every SHA-256 entry saved with a regression case."""
    target = Path(directory)
    checksum_path = target / 'checksums.sha256'
    for line in checksum_path.read_text(encoding='utf-8').splitlines():
        expected, filename = line.split('  ', maxsplit=1)
        if _sha256_file(target / filename) != expected:
            return False
    return True


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    'DAY5_REGRESSION_CATEGORIES',
    'load_projection_regression_case',
    'ProjectionRegressionMetrics',
    'replay_projection_regression_case',
    'save_projection_regression_case',
    'verify_projection_regression_checksums',
]
