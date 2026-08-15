#!/usr/bin/env python3
"""Audit official ROS interfaces, frames, and bounded timeout parameters."""

import argparse
import json
from math import isfinite
from pathlib import Path

from qmapnav.evaluation import InMemoryTraceRecorder
from qmapnav.mission.marker_adapter import OFFICIAL_MARKER_TOPIC
from qmapnav.mission.numerical_output_adapter import OFFICIAL_NUMERICAL_TOPIC


WAYPOINT_TOPIC = '/way_point_with_heading'
EXPECTED_PUBLISHERS = {
    OFFICIAL_NUMERICAL_TOPIC: 'Int32',
    OFFICIAL_MARKER_TOPIC: 'Marker',
    WAYPOINT_TOPIC: 'Pose2D',
}
EXPECTED_SUBSCRIPTIONS = {
    '/camera/image': 'Image',
    '/challenge_question': 'String',
    '/registered_scan': 'PointCloud2',
    '/state_estimation': 'Odometry',
}
POSITIVE_TIMEOUTS = (
    'episode_time_limit',
    'no_progress_timeout',
    'numerical_final_commit_reserve_sec',
    'numerical_maximum_verification_sec',
    'object_reference_final_commit_reserve_sec',
    'projection_buffer_seconds',
    'projection_shutdown_timeout',
    'targeted_viewpoint_minimum_time_remaining',
    'trace_flush_timeout',
    'watchdog_period',
)


def audit_node(node) -> dict[str, object]:
    """Return a fail-closed audit of one constructed composition root."""
    publishers = {
        item.topic_name: item.msg_type.__name__
        for item in (
            node._numerical_publisher,
            node._official_marker_publisher,
            node._waypoint_publisher,
        )
    }
    subscriptions = {
        item.topic_name: item.msg_type.__name__
        for item in (
            node._image_subscription,
            node._question_subscription,
            node._scan_subscription,
            node._pose_subscription,
        )
    }
    parameters = {
        name: node.get_parameter(name).value for name in POSITIVE_TIMEOUTS
    }
    timeout_errors = {
        name: value for name, value in parameters.items()
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not isfinite(float(value))
            or value <= 0
        )
    }
    frames = {
        'scan_frame': node.get_parameter('scan_frame').value,
        'pose_parent_frame': node.get_parameter('pose_parent_frame').value,
        'pose_child_frame': node.get_parameter('pose_child_frame').value,
    }
    errors = []
    if publishers != EXPECTED_PUBLISHERS:
        errors.append('official_publisher_contract_mismatch')
    if subscriptions != EXPECTED_SUBSCRIPTIONS:
        errors.append('official_subscription_contract_mismatch')
    if timeout_errors:
        errors.append('non_positive_or_unbounded_timeout')
    if frames['scan_frame'] != 'map' or frames['pose_parent_frame'] != 'map':
        errors.append('official_map_frame_mismatch')
    return {
        'passed': not errors,
        'errors': errors,
        'official_publishers': publishers,
        'official_subscriptions': subscriptions,
        'frames': frames,
        'bounded_timeouts_sec': parameters,
        'timeout_errors': timeout_errors,
        'detector_execution': (
            'bounded worker queue; episode watchdog and final-response reserve '
            'remain independent of detector completion'
        ),
    }


def main() -> None:
    """Construct the ROS node and persist the protocol audit."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    import rclpy
    from qmapnav.mission.node import QMapNavNode

    rclpy.init()
    node = QMapNavNode(trace_recorder=InMemoryTraceRecorder())
    try:
        report = audit_node(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
