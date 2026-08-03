"""Extract three-pose Day 5 regression cases from an official-topic bag."""

import argparse
from dataclasses import asdict
import json
from math import atan2
from pathlib import Path

import cv2
import numpy as np

from qmapnav.mapping import AssociationConfig
from qmapnav.mapping import AssociationFailure
from qmapnav.mapping import Day5ProjectionPipeline
from qmapnav.mapping import DenseRegisteredScanAccumulator
from qmapnav.mapping import DenseScanAccumulatorConfig
from qmapnav.mapping import ProjectionConfig
from qmapnav.mapping import ProjectionSynchronizer
from qmapnav.mapping import TimedPanorama
from qmapnav.mapping import TimedPose
from qmapnav.mapping import TimedRegisteredScan
from qmapnav.mapping.point_cloud import decode_scan_arrays
from qmapnav.mapping.projection_regression import save_projection_regression_case
from qmapnav.mapping.projection_visualisation import draw_projection_overlay
from qmapnav.mapping.projection_visualisation import draw_top_down_projection
from qmapnav.mapping.transforms import make_transform
from qmapnav.mapping.transforms import quaternion_xyzw_to_rotation
from qmapnav.mission.node import _decode_image_rgb
from qmapnav.perception import PanoramaCameraModel
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message


TOPIC_TYPES = {
    '/camera/image': 'sensor_msgs/msg/Image',
    '/registered_scan': 'sensor_msgs/msg/PointCloud2',
    '/state_estimation': 'nav_msgs/msg/Odometry',
}


def main() -> None:
    """Select start/farthest/end keyframes and save five named cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag_directory', type=Path)
    parser.add_argument('output_directory', type=Path)
    arguments = parser.parse_args()
    selected = _select_keyframes(arguments.bag_directory)
    payload = _extract(arguments.bag_directory, arguments.output_directory, selected)
    (arguments.output_directory / 'multi_pose_summary.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _select_keyframes(bag_directory: Path) -> dict[str, int]:
    reader = _reader(bag_directory)
    pose_type = get_message(TOPIC_TYPES['/state_estimation'])
    image_type = get_message(TOPIC_TYPES['/camera/image'])
    poses = []
    image_stamps = []
    while reader.has_next():
        topic, serialized, _ = reader.read_next()
        if topic == '/state_estimation':
            message = deserialize_message(serialized, pose_type)
            poses.append(
                (
                    _stamp_ns(message.header.stamp),
                    message.pose.pose.position.x,
                    message.pose.pose.position.y,
                )
            )
        elif topic == '/camera/image':
            message = deserialize_message(serialized, image_type)
            image_stamps.append(_stamp_ns(message.header.stamp))
    pose_array = np.asarray(poses, dtype=np.float64)
    images = np.asarray(image_stamps, dtype=np.int64)
    if pose_array.shape[0] < 3 or images.shape[0] < 3:
        raise ValueError('bag does not contain enough pose/image samples')
    distance = np.linalg.norm(pose_array[:, 1:3] - pose_array[0, 1:3], axis=1)
    farthest_index = int(np.argmax(distance))
    pose_targets = {
        'pose_a': int(pose_array[min(1000, pose_array.shape[0] - 1), 0]),
        'pose_b': int(pose_array[farthest_index, 0]),
        'pose_c': int(pose_array[max(0, pose_array.shape[0] - 1000), 0]),
    }
    return {
        pose_id: int(images[np.argmin(np.abs(images - timestamp))])
        for pose_id, timestamp in pose_targets.items()
    }


def _extract(
    bag_directory: Path,
    output_directory: Path,
    selected: dict[str, int],
) -> dict[str, object]:
    pipeline = _pipeline()
    reader = _reader(bag_directory)
    types = {topic: get_message(name) for topic, name in TOPIC_TYPES.items()}
    timestamp_to_pose_id = {timestamp: pose_id for pose_id, timestamp in selected.items()}
    latest_pose = None
    saved = {}
    category_plan = {
        'pose_a': ('walls', 'tabletop_objects'),
        'pose_b': ('nearby_furniture', 'sparse_detections'),
        'pose_c': ('panorama_seams',),
    }
    while reader.has_next() and len(saved) < len(selected):
        topic, serialized, recording_ns = reader.read_next()
        if topic not in types:
            continue
        message = deserialize_message(serialized, types[topic])
        source_ns = _stamp_ns(message.header.stamp)
        if topic == '/state_estimation':
            pose = message.pose.pose
            latest_pose = TimedPose(
                timestamp_ns=source_ns,
                parent_frame_id=message.header.frame_id,
                child_frame_id=message.child_frame_id,
                position_xyz=np.array(
                    [pose.position.x, pose.position.y, pose.position.z]
                ),
                orientation_xyzw=np.array(
                    [
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ]
                ),
                receipt_timestamp_ns=recording_ns,
            )
            pipeline.add_pose(latest_pose)
        elif topic == '/registered_scan':
            scan = decode_scan_arrays(message, expected_frame='map')
            timed_scan = TimedRegisteredScan(
                timestamp_ns=source_ns,
                frame_id=message.header.frame_id,
                points_xyz=scan.xyz,
                intensity=scan.intensity,
                receipt_timestamp_ns=recording_ns,
            )
            pipeline.add_scan(
                timed_scan,
                sensor_origin_xyz=(
                    latest_pose.position_xyz if latest_pose is not None else None
                ),
            )
        elif topic == '/camera/image' and source_ns in timestamp_to_pose_id:
            pose_id = timestamp_to_pose_id[source_ns]
            panorama = TimedPanorama(
                image_id=str(source_ns),
                timestamp_ns=source_ns,
                frame_id=message.header.frame_id,
                image_rgb=_decode_image_rgb(message),
                receipt_timestamp_ns=recording_ns,
            )
            frame = pipeline.process(panorama)
            if isinstance(frame, AssociationFailure):
                raise RuntimeError(
                    f'{pose_id} association failed: {frame.reason}'
                )
            overlay = draw_projection_overlay(panorama.image_rgb, frame.current)
            accumulated_overlay = draw_projection_overlay(
                panorama.image_rgb,
                frame.accumulated,
            )
            orientation = frame.association.pose.orientation_xyzw
            heading = atan2(
                2.0
                * (
                    orientation[3] * orientation[2]
                    + orientation[0] * orientation[1]
                ),
                1.0
                - 2.0
                * (
                    orientation[1] * orientation[1]
                    + orientation[2] * orientation[2]
                ),
            )
            top_down = draw_top_down_projection(
                frame.association.scan.points_xyz,
                frame.accumulated_snapshot.points_xyz,
                frame.association.pose.position_xyz,
                heading,
            )
            for category in category_plan[pose_id]:
                case_directory = save_projection_regression_case(
                    output_directory / category / str(source_ns),
                    category=category,
                    scene_id='office_1',
                    pose_id=pose_id,
                    frame=frame,
                    transform_sensor_from_camera_optical=(
                        pipeline.transform_sensor_from_camera_optical
                    ),
                    panorama_model=pipeline.panorama_model,
                    projection_config=pipeline.projection_config,
                    overlay_rgb=overlay,
                    notes=_notes(category, pose_id),
                )
                _save_rgb(
                    case_directory / 'accumulated_overlay.png',
                    accumulated_overlay,
                )
                _save_rgb(case_directory / 'top_down.png', top_down)
            saved[pose_id] = {
                'image_timestamp_ns': source_ns,
                'position_xyz': frame.association.pose.position_xyz.tolist(),
                'yaw_rad': heading,
                'current_projection': asdict(frame.current.diagnostics),
                'accumulated_projection': asdict(frame.accumulated.diagnostics),
                'dense_accumulator': asdict(pipeline.dense_accumulator.stats()),
                'categories': list(category_plan[pose_id]),
            }
    missing = sorted(set(selected) - set(saved))
    if missing:
        raise RuntimeError(f'failed to extract selected keyframes: {missing}')
    output_directory.mkdir(parents=True, exist_ok=True)
    return {
        'bag_directory': str(bag_directory),
        'selected_source_timestamps_ns': selected,
        'poses': saved,
    }


def _pipeline() -> Day5ProjectionPipeline:
    extrinsic = make_transform(
        quaternion_xyzw_to_rotation(np.array([-0.5, 0.5, -0.5, 0.5])),
        np.array([0.0, 0.0, 0.1]),
    )
    return Day5ProjectionPipeline(
        synchronizer=ProjectionSynchronizer(
            AssociationConfig(
                max_pose_delta_ns=50_000_000,
                max_scan_delta_ns=150_000_000,
            )
        ),
        dense_accumulator=DenseRegisteredScanAccumulator(
            DenseScanAccumulatorConfig(
                voxel_size_m=0.04,
                max_age_seconds=15.0,
                max_radius_m=12.0,
                max_points=1_000_000,
            )
        ),
        transform_sensor_from_camera_optical=extrinsic,
        panorama_model=PanoramaCameraModel(1920, 640),
        projection_config=ProjectionConfig(),
    )


def _reader(bag_directory: Path) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(
            uri=str(bag_directory),
            storage_id='mcap',
        ),
        rosbag2_py.ConverterOptions('', ''),
    )
    return reader


def _stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _notes(category: str, pose_id: str) -> str:
    return (
        f'Real Office 1 {category.replace("_", " ")} alignment at {pose_id}; '
        'raw map scan, source-time pose, extrinsic, panorama, and baseline pixels saved.'
    )


def _save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    if not cv2.imwrite(str(path), np.ascontiguousarray(image_rgb[..., ::-1])):
        raise RuntimeError(f'failed to save {path}')


if __name__ == '__main__':
    main()
