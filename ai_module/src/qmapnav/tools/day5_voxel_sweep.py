"""Measure three dense voxel sizes on the saved moving Office 1 stream."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from resource import getrusage
from resource import RUSAGE_SELF
from time import perf_counter

import numpy as np
from qmapnav.mapping import DenseRegisteredScanAccumulator
from qmapnav.mapping import DenseScanAccumulatorConfig
from qmapnav.mapping.point_cloud import decode_scan_arrays
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message


VOXEL_SIZES_M = (0.03, 0.04, 0.05)


def main() -> None:
    """Accumulate through pose B and save point/memory/runtime measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag_directory', type=Path)
    parser.add_argument('multi_pose_summary', type=Path)
    parser.add_argument('output_json', type=Path)
    arguments = parser.parse_args()
    summary = json.loads(arguments.multi_pose_summary.read_text(encoding='utf-8'))
    target_ns = int(summary['poses']['pose_b']['image_timestamp_ns'])
    payload = sweep_voxels(arguments.bag_directory, target_ns)
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def sweep_voxels(bag_directory: Path, target_ns: int) -> dict[str, object]:
    """Return dense-map statistics at one source timestamp for three voxels."""
    accumulators = {
        size: DenseRegisteredScanAccumulator(
            DenseScanAccumulatorConfig(
                voxel_size_m=size,
                max_age_seconds=15.0,
                max_radius_m=12.0,
                max_points=1_000_000,
            )
        )
        for size in VOXEL_SIZES_M
    }
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_directory), storage_id='mcap'),
        rosbag2_py.ConverterOptions('', ''),
    )
    pose_type = get_message('nav_msgs/msg/Odometry')
    scan_type = get_message('sensor_msgs/msg/PointCloud2')
    image_type = get_message('sensor_msgs/msg/Image')
    origin = None
    insert_seconds = {size: 0.0 for size in VOXEL_SIZES_M}
    while reader.has_next():
        topic, serialized, _ = reader.read_next()
        if topic == '/state_estimation':
            message = deserialize_message(serialized, pose_type)
            position = message.pose.pose.position
            origin = np.array([position.x, position.y, position.z])
        elif topic == '/registered_scan':
            message = deserialize_message(serialized, scan_type)
            source_ns = _stamp_ns(message.header.stamp)
            if source_ns > target_ns:
                continue
            scan = decode_scan_arrays(message, expected_frame='map')
            for size, accumulator in accumulators.items():
                started = perf_counter()
                accumulator.add_scan(
                    scan.xyz,
                    frame_id=message.header.frame_id,
                    timestamp_ns=source_ns,
                    sensor_origin_xyz=origin,
                )
                insert_seconds[size] += perf_counter() - started
        elif topic == '/camera/image':
            message = deserialize_message(serialized, image_type)
            if _stamp_ns(message.header.stamp) >= target_ns:
                break
    results = {}
    for size, accumulator in accumulators.items():
        started = perf_counter()
        snapshot = accumulator.snapshot()
        snapshot_seconds = perf_counter() - started
        ages = snapshot.age_seconds(target_ns)
        stats = accumulator.stats()
        results[f'{size:.2f}'] = {
            'stats': asdict(stats),
            'voxel_count': int(snapshot.points_xyz.shape[0]),
            'reduction_ratio': (
                float(snapshot.points_xyz.shape[0] / stats.raw_point_count)
                if stats.raw_point_count
                else 0.0
            ),
            'insert_seconds_total': insert_seconds[size],
            'snapshot_seconds': snapshot_seconds,
            'voxel_age_seconds': {
                'median': float(np.median(ages)) if ages.size else None,
                'p95': float(np.percentile(ages, 95)) if ages.size else None,
                'maximum': float(np.max(ages)) if ages.size else None,
            },
        }
    return {
        'bag_directory': str(bag_directory),
        'target_timestamp_ns': target_ns,
        'peak_rss_kib_process': int(getrusage(RUSAGE_SELF).ru_maxrss),
        'results': results,
    }


def _stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


if __name__ == '__main__':
    main()
