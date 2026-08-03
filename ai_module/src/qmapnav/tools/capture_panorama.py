"""Capture one live challenge panorama and its source metadata."""

import argparse
import json
from pathlib import Path

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import Image


def capture_panorama(
    output_path: Path,
    *,
    topic: str = '/camera/image',
    timeout_sec: float = 30.0,
) -> dict[str, object]:
    """Wait for one ROS image, save it losslessly, and return metadata."""
    if timeout_sec <= 0.0:
        raise ValueError('timeout_sec must be positive')
    rclpy.init()
    node = rclpy.create_node('qmapnav_day4_panorama_capture')
    try:
        received, message = wait_for_message(
            Image,
            node,
            topic,
            time_to_wait=timeout_sec,
        )
        if not received or message is None:
            raise TimeoutError(f'no image received on {topic} within {timeout_sec}s')
        bgr = CvBridge().imgmsg_to_cv2(message, desired_encoding='bgr8')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), bgr):
            raise RuntimeError(f'failed to write image: {output_path}')
        metadata = {
            'topic': topic,
            'header_timestamp_ns': (
                message.header.stamp.sec * 1_000_000_000
                + message.header.stamp.nanosec
            ),
            'frame_id': message.header.frame_id,
            'source_encoding': message.encoding,
            'width': message.width,
            'height': message.height,
            'step': message.step,
            'output_path': str(output_path),
        }
        metadata_path = output_path.with_suffix(output_path.suffix + '.json')
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        return metadata
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    """Capture one live panorama from command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_path', type=Path)
    parser.add_argument('--topic', default='/camera/image')
    parser.add_argument('--timeout-sec', default=30.0, type=float)
    arguments = parser.parse_args()
    metadata = capture_panorama(
        arguments.output_path,
        topic=arguments.topic,
        timeout_sec=arguments.timeout_sec,
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == '__main__':
    main()
