"""Save and replay source-complete Day 6 lifting regression cases."""

from dataclasses import asdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np

from qmapnav.mapping.bounding_boxes import BoxEstimationConfig
from qmapnav.mapping.cluster_selection import ClusterSelectionConfig
from qmapnav.mapping.depth_filter import DepthFilterConfig
from qmapnav.mapping.ground_filter import GroundPlane
from qmapnav.mapping.lidar_camera_projection import ProjectionDiagnostics
from qmapnav.mapping.lidar_camera_projection import ProjectionResult
from qmapnav.mapping.object_candidate import GeometrySource
from qmapnav.mapping.object_lifting import ObjectLifter
from qmapnav.mapping.object_lifting import ObjectLiftingConfig
from qmapnav.mapping.orientation_confidence import OrientationConfidenceConfig
from qmapnav.mapping.point_selection import PointSelectionConfig
from qmapnav.perception.contracts import Detection2D
from qmapnav.perception.contracts import PanoramaBox


DAY6_REGRESSION_CATEGORIES = (
    'large_box_like',
    'narrow_object',
    'floor_standing',
    'small_tabletop',
    'wall_adjacent',
)


@dataclass(frozen=True)
class LiftingRegressionMetrics:
    """Saved baseline discrepancy for one deterministic lifting replay."""

    status_matches: bool
    point_indices_match: bool
    centre_error_m: float | None
    dimension_error_m: float | None
    yaw_error_rad: float | None
    checksum_valid: bool
    passed: bool


def save_lifting_regression_case(
    directory: str | Path,
    *,
    category: str,
    scene_id: str,
    pose_id: str,
    panorama_rgb: np.ndarray,
    detection: Detection2D,
    projection: ProjectionResult,
    ground_plane: GroundPlane | None,
    source: GeometrySource,
    use_mask: bool,
    config: ObjectLiftingConfig,
    result,
    stage_overlay_rgb: np.ndarray,
    depth_histogram_rgb: np.ndarray,
    geometry_overlay_rgb: np.ndarray,
    notes: str,
) -> Path:
    """Save one source-complete deterministic candidate-lifting case."""
    if category not in DAY6_REGRESSION_CATEGORIES:
        raise ValueError(f'unknown Day 6 regression category {category!r}')
    if not scene_id or not pose_id or not notes:
        raise ValueError('scene_id, pose_id, and notes must be non-empty')
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    candidate = result.candidate
    np.savez_compressed(
        target / 'inputs.npz',
        points_map_xyz=projection.points_map_xyz,
        points_camera_xyz=projection.points_camera_xyz,
        panorama_uv=projection.panorama_uv,
        euclidean_range_m=projection.euclidean_range_m,
        forward_depth_m=projection.forward_depth_m,
        source_projection_indices=projection.source_point_indices,
        source_valid_mask=projection.source_valid_mask,
        transform_camera_internal_from_map=(
            projection.transform_camera_internal_from_map
        ),
        intensity=(
            projection.intensity
            if projection.intensity is not None
            else np.empty(0, dtype=np.float64)
        ),
        detection_boundary_uv=detection.panorama_box.boundary_uv,
        baseline_final_indices=(
            candidate.source_projection_indices
            if candidate is not None
            else np.empty(0, dtype=np.int64)
        ),
    )
    images = {
        'panorama.png': panorama_rgb,
        'stages.png': stage_overlay_rgb,
        'depth_histogram.png': depth_histogram_rgb,
        'geometry.png': geometry_overlay_rgb,
    }
    for filename, image in images.items():
        if not cv2.imwrite(
            str(target / filename),
            np.ascontiguousarray(np.asarray(image)[..., ::-1]),
        ):
            raise RuntimeError(f'failed to save {filename}')
    manifest = {
        'schema_version': 1,
        'category': category,
        'scene_id': scene_id,
        'pose_id': pose_id,
        'notes': notes,
        'source': source.value,
        'use_mask': bool(use_mask),
        'detection': {
            'detection_id': detection.detection_id,
            'class_name': detection.class_name,
            'prompt_used': detection.prompt_used,
            'confidence': detection.confidence,
            'x_intervals': detection.panorama_box.x_intervals,
            'y_min': detection.panorama_box.y_min,
            'y_max': detection.panorama_box.y_max,
            'panorama_width': detection.panorama_box.panorama_width,
            'panorama_height': detection.panorama_box.panorama_height,
            'crop_ids': detection.crop_ids,
            'crop_boxes_xyxy': detection.crop_boxes_xyxy,
            'centre_panorama_uv': detection.centre_panorama_uv,
            'centre_camera_ray': detection.centre_camera_ray.tolist(),
            'seam_merged': detection.seam_merged,
            'metadata': _jsonable(dict(detection.metadata)),
        },
        'projection': {
            'image_id': projection.image_id,
            'image_timestamp_ns': projection.image_timestamp_ns,
            'scan_timestamp_ns': projection.scan_timestamp_ns,
            'has_intensity': projection.intensity is not None,
            'diagnostics': asdict(projection.diagnostics),
        },
        'ground_plane': (
            {
                **asdict(ground_plane),
                'normal_xyz': ground_plane.normal_xyz.tolist(),
            }
            if ground_plane is not None
            else None
        ),
        'config': asdict(config),
        'baseline': {
            'status': result.status.value,
            'reason': result.reason,
            'point_count': result.counts.final,
            'centre_xyz': (
                candidate.obb_centre_xyz.tolist() if candidate is not None else None
            ),
            'dimensions_xyz': (
                candidate.obb_dimensions_xyz.tolist() if candidate is not None else None
            ),
            'yaw_rad': candidate.obb_yaw_rad if candidate is not None else None,
            'orientation_confidence': (
                candidate.orientation_confidence if candidate is not None else None
            ),
            'geometry_confidence': (
                candidate.geometry_confidence if candidate is not None else None
            ),
        },
    }
    (target / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    checksum_files = ('inputs.npz', *images, 'manifest.json')
    (target / 'checksums.sha256').write_text(
        ''.join(
            f'{_sha256_file(target / filename)}  {filename}\n'
            for filename in checksum_files
        ),
        encoding='utf-8',
    )
    return target


def replay_lifting_regression_case(
    directory: str | Path,
    *,
    centre_tolerance_m: float = 1e-8,
    dimension_tolerance_m: float = 1e-8,
    yaw_tolerance_rad: float = 1e-8,
    ground_plane_override: GroundPlane | None | bool = False,
) -> LiftingRegressionMetrics:
    """Replay one case and compare status, indices, centre, dimensions and yaw."""
    target = Path(directory)
    manifest = json.loads((target / 'manifest.json').read_text(encoding='utf-8'))
    arrays = np.load(target / 'inputs.npz')
    detection_data = manifest['detection']
    box = PanoramaBox(
        panorama_width=int(detection_data['panorama_width']),
        panorama_height=int(detection_data['panorama_height']),
        x_intervals=tuple(tuple(value) for value in detection_data['x_intervals']),
        y_min=float(detection_data['y_min']),
        y_max=float(detection_data['y_max']),
        boundary_uv=arrays['detection_boundary_uv'],
    )
    detection = Detection2D(
        detection_id=detection_data['detection_id'],
        class_name=detection_data['class_name'],
        prompt_used=detection_data['prompt_used'],
        confidence=float(detection_data['confidence']),
        panorama_box=box,
        crop_ids=tuple(detection_data['crop_ids']),
        crop_boxes_xyxy=tuple(tuple(value) for value in detection_data['crop_boxes_xyxy']),
        centre_panorama_uv=tuple(detection_data['centre_panorama_uv']),
        centre_camera_ray=np.asarray(detection_data['centre_camera_ray']),
        seam_merged=bool(detection_data['seam_merged']),
        metadata=detection_data['metadata'],
    )
    projection_data = manifest['projection']
    projection = ProjectionResult(
        image_id=projection_data['image_id'],
        image_timestamp_ns=int(projection_data['image_timestamp_ns']),
        scan_timestamp_ns=int(projection_data['scan_timestamp_ns']),
        transform_camera_internal_from_map=arrays['transform_camera_internal_from_map'],
        source_valid_mask=arrays['source_valid_mask'],
        source_point_indices=arrays['source_projection_indices'],
        points_map_xyz=arrays['points_map_xyz'],
        points_camera_xyz=arrays['points_camera_xyz'],
        panorama_uv=arrays['panorama_uv'],
        euclidean_range_m=arrays['euclidean_range_m'],
        forward_depth_m=arrays['forward_depth_m'],
        intensity=(arrays['intensity'] if projection_data['has_intensity'] else None),
        diagnostics=ProjectionDiagnostics(**projection_data['diagnostics']),
    )
    saved_ground = _ground_from_manifest(manifest['ground_plane'])
    ground = saved_ground if ground_plane_override is False else ground_plane_override
    config = _config_from_dict(manifest['config'])
    result = ObjectLifter(config).lift(
        detection,
        projection,
        source=GeometrySource(manifest['source']),
        ground_plane=ground,
        use_mask=bool(manifest['use_mask']),
    )
    baseline = manifest['baseline']
    status_matches = result.status.value == baseline['status']
    actual_indices = (
        result.candidate.source_projection_indices
        if result.candidate is not None
        else np.empty(0, dtype=np.int64)
    )
    point_indices_match = np.array_equal(
        actual_indices, arrays['baseline_final_indices']
    )
    if result.candidate is None or baseline['centre_xyz'] is None:
        centre_error = dimension_error = yaw_error = None
        numeric_pass = result.candidate is None and baseline['centre_xyz'] is None
    else:
        centre_error = float(np.linalg.norm(
            result.candidate.obb_centre_xyz - np.asarray(baseline['centre_xyz'])
        ))
        dimension_error = float(np.max(np.abs(
            result.candidate.obb_dimensions_xyz - np.asarray(baseline['dimensions_xyz'])
        )))
        yaw_error = float(abs(result.candidate.obb_yaw_rad - baseline['yaw_rad']))
        numeric_pass = (
            centre_error <= centre_tolerance_m
            and dimension_error <= dimension_tolerance_m
            and yaw_error <= yaw_tolerance_rad
        )
    checksum_valid = verify_lifting_regression_checksums(target)
    return LiftingRegressionMetrics(
        status_matches=status_matches,
        point_indices_match=point_indices_match,
        centre_error_m=centre_error,
        dimension_error_m=dimension_error,
        yaw_error_rad=yaw_error,
        checksum_valid=checksum_valid,
        passed=(status_matches and point_indices_match and numeric_pass and checksum_valid),
    )


def verify_lifting_regression_checksums(directory: str | Path) -> bool:
    """Verify every SHA-256 entry in a saved Day 6 case."""
    target = Path(directory)
    for line in (target / 'checksums.sha256').read_text(encoding='utf-8').splitlines():
        expected, filename = line.split('  ', maxsplit=1)
        if _sha256_file(target / filename) != expected:
            return False
    return True


def _config_from_dict(values: dict[str, object]) -> ObjectLiftingConfig:
    values = dict(values)
    values['selection'] = PointSelectionConfig(**values['selection'])
    values['depth'] = DepthFilterConfig(**values['depth'])
    values['clustering'] = ClusterSelectionConfig(**values['clustering'])
    values['boxes'] = BoxEstimationConfig(**values['boxes'])
    values['orientation'] = OrientationConfidenceConfig(**values['orientation'])
    return ObjectLiftingConfig(**values)


def _ground_from_manifest(values: dict[str, object] | None) -> GroundPlane | None:
    if values is None:
        return None
    return GroundPlane(**values)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    'DAY6_REGRESSION_CATEGORIES',
    'LiftingRegressionMetrics',
    'replay_lifting_regression_case',
    'save_lifting_regression_case',
    'verify_lifting_regression_checksums',
]
