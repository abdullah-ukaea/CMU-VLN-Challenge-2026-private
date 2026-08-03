"""Audit Day 5 source frames, rates, timing deltas, and static extrinsics."""

import argparse
import json
from math import atan2
from pathlib import Path

import numpy as np
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message


TOPICS = (
    '/camera/image',
    '/registered_scan',
    '/state_estimation',
    '/tf_static',
)


def main() -> None:
    """Read one MCAP bag and write a compact quantitative Day 5 audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag_directory', type=Path)
    parser.add_argument('output_json', type=Path)
    arguments = parser.parse_args()
    payload = audit_bag(arguments.bag_directory)
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(payload['association'], indent=2, sort_keys=True))


def audit_bag(bag_directory: Path) -> dict[str, object]:
    """Return frame, source-time, association, and motion evidence."""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(
            uri=str(bag_directory),
            storage_id='mcap',
        ),
        rosbag2_py.ConverterOptions('', ''),
    )
    topic_types = {
        topic.name: topic.type for topic in reader.get_all_topics_and_types()
    }
    missing = [topic for topic in TOPICS if topic not in topic_types]
    if missing:
        raise ValueError(f'bag is missing Day 5 topics: {missing}')
    message_types = {
        topic: get_message(topic_types[topic]) for topic in TOPICS
    }
    source_stamps = {topic: [] for topic in TOPICS[:-1]}
    recording_stamps = {topic: [] for topic in TOPICS[:-1]}
    frame_ids = {topic: set() for topic in TOPICS[:-1]}
    scan_fields = None
    poses = []
    static_transforms = []
    while reader.has_next():
        topic, serialized, recording_ns = reader.read_next()
        if topic not in message_types:
            continue
        message = deserialize_message(serialized, message_types[topic])
        if topic == '/tf_static':
            for transform in message.transforms:
                static_transforms.append(
                    {
                        'parent_frame_id': transform.header.frame_id,
                        'child_frame_id': transform.child_frame_id,
                        'translation_xyz': [
                            transform.transform.translation.x,
                            transform.transform.translation.y,
                            transform.transform.translation.z,
                        ],
                        'orientation_xyzw': [
                            transform.transform.rotation.x,
                            transform.transform.rotation.y,
                            transform.transform.rotation.z,
                            transform.transform.rotation.w,
                        ],
                    }
                )
            continue
        stamp_ns = _stamp_ns(message.header.stamp)
        source_stamps[topic].append(stamp_ns)
        recording_stamps[topic].append(recording_ns)
        frame_ids[topic].add(message.header.frame_id)
        if topic == '/registered_scan' and scan_fields is None:
            scan_fields = [
                {
                    'name': field.name,
                    'offset': field.offset,
                    'datatype': field.datatype,
                    'count': field.count,
                }
                for field in message.fields
            ]
        if topic == '/state_estimation':
            pose = message.pose.pose
            orientation = pose.orientation
            poses.append(
                [
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                    atan2(
                        2.0
                        * (
                            orientation.w * orientation.z
                            + orientation.x * orientation.y
                        ),
                        1.0
                        - 2.0
                        * (
                            orientation.y * orientation.y
                            + orientation.z * orientation.z
                        ),
                    ),
                    message.child_frame_id,
                ]
            )
    streams = {
        topic: _stream_stats(
            np.asarray(source_stamps[topic], dtype=np.int64),
            np.asarray(recording_stamps[topic], dtype=np.int64),
            sorted(frame_ids[topic]),
        )
        for topic in TOPICS[:-1]
    }
    camera = np.asarray(source_stamps['/camera/image'], dtype=np.int64)
    scans = np.asarray(source_stamps['/registered_scan'], dtype=np.int64)
    pose_stamps = np.asarray(source_stamps['/state_estimation'], dtype=np.int64)
    pose_array = np.asarray([item[:4] for item in poses], dtype=np.float64)
    child_frames = sorted({str(item[4]) for item in poses})
    return {
        'bag_directory': str(bag_directory),
        'streams': streams,
        'frames': {
            'pose_child_frame_ids': child_frames,
            'registered_scan_fields': scan_fields,
            'static_transforms': _unique_dicts(static_transforms),
        },
        'association': {
            'camera_to_nearest_scan_ms': _nearest_delta_stats(camera, scans),
            'camera_to_nearest_pose_ms': _nearest_delta_stats(camera, pose_stamps),
        },
        'motion': _motion_stats(pose_array),
    }


def _stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _stream_stats(
    source_ns: np.ndarray,
    recording_ns: np.ndarray,
    frames: list[str],
) -> dict[str, object]:
    if source_ns.size == 0:
        return {'count': 0, 'frame_ids': frames}
    intervals_ms = np.diff(source_ns) / 1_000_000.0
    duration_seconds = (source_ns[-1] - source_ns[0]) / 1_000_000_000.0
    recording_delay_ms = (recording_ns - source_ns) / 1_000_000.0
    return {
        'count': int(source_ns.size),
        'frame_ids': frames,
        'source_rate_hz': (
            float((source_ns.size - 1) / duration_seconds)
            if duration_seconds > 0.0
            else None
        ),
        'source_interval_ms': _distribution(intervals_ms),
        'recording_minus_source_ms': _distribution(recording_delay_ms),
    }


def _nearest_delta_stats(query: np.ndarray, samples: np.ndarray) -> dict[str, float]:
    if query.size == 0 or samples.size == 0:
        raise ValueError('cannot associate empty source-time streams')
    positions = np.searchsorted(samples, query)
    before = np.clip(positions - 1, 0, samples.size - 1)
    after = np.clip(positions, 0, samples.size - 1)
    before_delta = np.abs(samples[before] - query)
    after_delta = np.abs(samples[after] - query)
    nearest_ms = np.minimum(before_delta, after_delta) / 1_000_000.0
    signed_indices = np.where(before_delta <= after_delta, before, after)
    signed_ms = (samples[signed_indices] - query) / 1_000_000.0
    result = _distribution(nearest_ms)
    result['signed_median'] = float(np.median(signed_ms))
    result['within_50_ms_fraction'] = float(np.mean(nearest_ms <= 50.0))
    result['within_150_ms_fraction'] = float(np.mean(nearest_ms <= 150.0))
    return result


def _distribution(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {'minimum': None, 'median': None, 'p95': None, 'maximum': None}
    return {
        'minimum': float(np.min(values)),
        'median': float(np.median(values)),
        'p95': float(np.percentile(values, 95)),
        'maximum': float(np.max(values)),
    }


def _motion_stats(poses: np.ndarray) -> dict[str, object]:
    if poses.size == 0:
        return {'pose_count': 0}
    distance_from_start = np.linalg.norm(poses[:, :2] - poses[0, :2], axis=1)
    selected = []
    for fraction in (0.05, 0.50, 0.95):
        index = min(poses.shape[0] - 1, int(fraction * poses.shape[0]))
        selected.append(
            {
                'source_fraction': fraction,
                'position_xyz': poses[index, :3].tolist(),
                'yaw_rad': float(poses[index, 3]),
            }
        )
    return {
        'pose_count': int(poses.shape[0]),
        'x_range_m': [float(np.min(poses[:, 0])), float(np.max(poses[:, 0]))],
        'y_range_m': [float(np.min(poses[:, 1])), float(np.max(poses[:, 1]))],
        'yaw_range_rad': [
            float(np.min(poses[:, 3])),
            float(np.max(poses[:, 3])),
        ],
        'maximum_distance_from_start_m': float(np.max(distance_from_start)),
        'representative_samples': selected,
    }


def _unique_dicts(values: list[dict[str, object]]) -> list[dict[str, object]]:
    unique = {}
    for value in values:
        unique[json.dumps(value, sort_keys=True)] = value
    return list(unique.values())


if __name__ == '__main__':
    main()
