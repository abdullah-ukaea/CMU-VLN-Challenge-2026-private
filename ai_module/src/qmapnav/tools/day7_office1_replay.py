"""Replay Day 7 on saved real Office 1 Day 5 and Day 6 evidence."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from qmapnav.evaluation.instance_fusion import evaluate_identity_assignments
from qmapnav.evaluation.instance_fusion import IdentityAssignment
from qmapnav.mapping.geometry_evaluation import upright_box_corners_xy
from qmapnav.mapping.object_candidate import ConfidenceComponents
from qmapnav.mapping.object_candidate import GeometrySource
from qmapnav.mapping.object_candidate import GeometryStatus
from qmapnav.mapping.object_candidate import LiftingCounts
from qmapnav.mapping.object_candidate import ObjectCandidate3D
from qmapnav.mapping.object_map import ObjectMap
from qmapnav.mapping.structural_map import StructuralMap
from qmapnav.mapping.transforms import camera_internal_from_map
from qmapnav.mapping.transforms import invert_transform
from qmapnav.mapping.transforms import transform_from_pose
from qmapnav.mapping.viewpoint_observation import ViewpointObservation
from qmapnav.perception.contracts import Detection2D
from qmapnav.perception.contracts import PanoramaBox
from qmapnav.perception.panorama_projection import (
    panorama_pixels_to_camera_rays,
)
from qmapnav.perception.panorama_projection import PanoramaCameraModel


def replay(day5_root: Path, day6_root: Path) -> dict[str, object]:
    """Return real-data object-identity and structural replay evidence."""
    structural_map = StructuralMap()
    accepted_anchors = []
    processed_scan_stamps = set()
    for manifest_path in sorted(day5_root.glob('*/*/manifest.json')):
        case_directory = manifest_path.parent
        manifest = _load_json(manifest_path)
        timestamp_ns = int(manifest['image']['timestamp_ns'])
        if timestamp_ns not in processed_scan_stamps:
            arrays = np.load(case_directory / 'inputs.npz')
            structural_map.update_walls_from_points(
                arrays['points_map_xyz'],
                timestamp_ns=timestamp_ns,
                viewpoint_id=str(manifest['pose_id']),
            )
            processed_scan_stamps.add(timestamp_ns)
        detections_path = case_directory / 'detections.json'
        if not detections_path.exists():
            continue
        transform_map_from_camera = _map_from_camera(manifest)
        model = PanoramaCameraModel(
            int(manifest['image']['width']),
            int(manifest['image']['height']),
        )
        for payload in _load_json(detections_path)['detections']:
            detection = _detection_from_saved(
                payload,
                model,
                str(manifest['pose_id']),
                timestamp_ns,
            )
            anchor = structural_map.anchor_detection_to_wall(
                detection, transform_map_from_camera
            )
            if anchor is not None:
                accepted_anchors.append(anchor.anchor_id)
    object_map, assignments = _replay_objects(day6_root)
    identity = evaluate_identity_assignments(assignments)
    wall_records = [
        structural_map.record(wall.anchor_id)
        for wall in structural_map.walls()
    ]
    return {
        'object_identity': {
            **identity.__dict__,
            'retained_instance_count': len(object_map.active_instances()),
            'retained_fused_point_count': object_map.fused_point_count,
            'association_event_count': len(assignments),
        },
        'structure': {
            'input_scan_count': len(processed_scan_stamps),
            'wall_count': len(wall_records),
            'reobserved_wall_count': sum(
                record.observation_count > 1 for record in wall_records
            ),
            'maximum_wall_observation_count': max(
                (record.observation_count for record in wall_records),
                default=0,
            ),
            'anchor_count': len(structural_map.anchors()),
            'accepted_anchor_observation_count': len(accepted_anchors),
            'reobserved_anchor_count': sum(
                structural_map.record(anchor.anchor_id).observation_count > 1
                for anchor in structural_map.anchors()
            ),
            'anchors': [
                {
                    'anchor_id': anchor.anchor_id,
                    'semantic_class': anchor.semantic_class,
                    'position_xyz': anchor.position_xyz.tolist(),
                    'supporting_wall_id': anchor.supporting_wall_id,
                    'observation_count': structural_map.record(
                        anchor.anchor_id
                    ).observation_count,
                    'source_viewpoint_ids': list(anchor.source_viewpoint_ids),
                    'source_detection_ids': list(anchor.source_detection_ids),
                }
                for anchor in structural_map.anchors()
            ],
        },
    }


def _replay_objects(
    root: Path,
) -> tuple[ObjectMap, list[IdentityAssignment]]:
    grouped = defaultdict(list)
    for result_path in sorted(root.rglob('result.json')):
        payload = _load_json(result_path)
        box = payload.get('box', {})
        if box.get('candidate') is None or box.get('reference') is None:
            continue
        candidate = _candidate_from_saved(payload)
        reference_id = str(box['reference']['object_id'])
        # Several Day 6 query packs replay the same saved panorama.  Batch by
        # the physical keyframe timestamp so exact cross-pack detections are
        # handled by ObjectMap's same-keyframe duplicate suppression.
        grouped[str(candidate.image_timestamp_ns)].append(
            (candidate, reference_id)
        )
    object_map = ObjectMap()
    assignments = []
    for viewpoint_id in sorted(grouped):
        entries = grouped[viewpoint_id]
        candidates = [entry[0] for entry in entries]
        observations = [
            ViewpointObservation(
                viewpoint_id=viewpoint_id,
                robot_pose_xyz_yaw=np.zeros(4),
                timestamp_ns=candidate.image_timestamp_ns,
                detection_id=candidate.detection_id,
                point_count=candidate.point_count,
                geometry_confidence=candidate.geometry_confidence,
                visibility=(
                    'partial' if candidate.partial_geometry else 'full'
                ),
            )
            for candidate in candidates
        ]
        predicted_ids = object_map.add_viewpoint_candidates(
            candidates, observations
        )
        assignments.extend(
            IdentityAssignment(reference_id, predicted_id)
            for (_, reference_id), predicted_id in zip(entries, predicted_ids)
        )
    return object_map, assignments


def _candidate_from_saved(payload: dict[str, object]) -> ObjectCandidate3D:
    box = payload['box']
    candidate = box['candidate']
    counts_payload = box['counts']
    centre = np.asarray(candidate['centre_xyz'], dtype=np.float64)
    dimensions = np.asarray(candidate['dimensions_xyz'], dtype=np.float64)
    yaw = float(candidate['yaw_rad'])
    count = int(counts_payload['final'])
    points = _cuboid_points(centre, dimensions, yaw, count)
    corners = upright_box_corners_xy(centre, dimensions, yaw)
    minimum = np.array([
        np.min(corners[:, 0]),
        np.min(corners[:, 1]),
        centre[2] - dimensions[2] / 2.0,
    ])
    maximum = np.array([
        np.max(corners[:, 0]),
        np.max(corners[:, 1]),
        centre[2] + dimensions[2] / 2.0,
    ])
    timestamp_ns = int(str(payload['detection_id']).split(':', 1)[0])
    geometry_confidence = float(candidate['geometry_confidence'])
    return ObjectCandidate3D(
        candidate_id=f"saved:{payload['detection_id']}",
        detection_id=str(payload['detection_id']),
        class_name=str(payload['class_name']),
        detection_confidence=float(payload['confidence']),
        source=GeometrySource.ACCUMULATED,
        source_timestamp_ns=timestamp_ns,
        image_timestamp_ns=timestamp_ns,
        scan_timestamp_ns=timestamp_ns,
        pose_timestamp_ns=timestamp_ns,
        pose_mode='saved_office1',
        image_scan_delta_ms=0.0,
        pose_before_delta_ms=0.0,
        pose_after_delta_ms=0.0,
        timing_warning=False,
        points_map_xyz=points,
        source_projection_indices=np.arange(count),
        point_centroid_xyz=centre,
        aabb_min_xyz=minimum,
        aabb_max_xyz=maximum,
        obb_centre_xyz=centre,
        obb_dimensions_xyz=dimensions,
        obb_yaw_rad=yaw,
        estimated_yaw_rad=float(candidate['estimated_yaw_rad']),
        orientation_confidence=float(candidate['orientation_confidence']),
        geometry_confidence=geometry_confidence,
        geometry_status=GeometryStatus(box['status']),
        partial_geometry=bool(candidate['partial_geometry']),
        low_orientation_fallback=bool(candidate['low_orientation_fallback']),
        counts=LiftingCounts(
            int(counts_payload['projected']),
            int(counts_payload['box_selected']),
            int(counts_payload['mask_selected']),
            int(counts_payload['post_ground']),
            int(counts_payload['post_depth']),
            int(counts_payload['clustered']),
            count,
        ),
        confidence_components=ConfidenceComponents(
            *([geometry_confidence] * 9)
        ),
        diagnostics={'source': 'saved_office1_result'},
    )


def _cuboid_points(centre, dimensions, yaw, count):
    phase = np.linspace(-0.45, 0.45, count)
    local = np.column_stack((
        phase,
        np.roll(phase, max(1, count // 3)),
        np.roll(phase, max(1, 2 * count // 3)),
    )) * dimensions
    rotation = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return local @ rotation.T + centre


def _map_from_camera(manifest):
    pose = manifest['pose']
    transform_map_from_sensor = transform_from_pose(
        np.asarray(pose['position_xyz']),
        np.asarray(pose['orientation_xyzw']),
    )
    transform_camera_from_map = camera_internal_from_map(
        transform_map_from_sensor,
        np.asarray(manifest['transform_sensor_from_camera_optical']),
    )
    return invert_transform(transform_camera_from_map)


def _detection_from_saved(payload, model, viewpoint_id, timestamp_ns):
    intervals = tuple(
        (float(item[0]), float(item[1]))
        for item in payload['x_intervals']
    )
    y_min = float(payload['y_min'])
    y_max = float(payload['y_max'])
    if len(intervals) == 1:
        centre_u = sum(intervals[0]) / 2.0
    else:
        centre_u = (
            intervals[-1][0] + intervals[0][1] + model.width
        ) / 2.0 % model.width
    centre_uv = np.array([[centre_u, (y_min + y_max) / 2.0]])
    ray = panorama_pixels_to_camera_rays(centre_uv, model)[0]
    boundary = np.array([
        [intervals[0][0], y_min],
        [intervals[0][1], y_min],
        [intervals[-1][1], y_max],
        [intervals[-1][0], y_max],
    ])
    crop_ids = tuple(int(item) for item in payload['crop_ids'])
    crop_box = (
        intervals[0][0], y_min, intervals[0][1], y_max
    )
    return Detection2D(
        detection_id=str(payload['detection_id']),
        class_name=str(payload['class_name']),
        prompt_used=str(payload['class_name']),
        confidence=float(payload['confidence']),
        panorama_box=PanoramaBox(
            model.width,
            model.height,
            intervals,
            y_min,
            y_max,
            boundary,
        ),
        crop_ids=crop_ids,
        crop_boxes_xyxy=tuple(crop_box for _ in crop_ids),
        centre_panorama_uv=tuple(centre_uv[0]),
        centre_camera_ray=ray,
        seam_merged=len(intervals) == 2,
        metadata={
            'viewpoint_id': viewpoint_id,
            'timestamp_ns': timestamp_ns,
        },
    )


def _load_json(path: Path) -> dict[str, object]:
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def main() -> None:
    """Replay saved evidence and print a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('day5_root', type=Path)
    parser.add_argument('day6_root', type=Path)
    arguments = parser.parse_args()
    report = replay(arguments.day5_root, arguments.day6_root)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
