"""Extract one unmodified RGB panorama from an MCAP rosbag for benchmarking."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image


_ENCODING_CHANNELS = {
    'bgr8': (3, False),
    'bgra8': (4, False),
    'rgb8': (3, True),
    'rgba8': (4, True),
}


def extract_panorama(
    bag_path: Path,
    output_path: Path,
    *,
    topic: str = '/camera/image',
    message_index: int = 0,
) -> dict[str, object]:
    """Extract one selected image and return its source metadata."""
    if message_index < 0:
        raise ValueError('message_index must be non-negative')
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id='mcap'),
        rosbag2_py.ConverterOptions('', ''),
    )
    observed = 0
    while reader.has_next():
        current_topic, serialized, timestamp_ns = reader.read_next()
        if current_topic != topic:
            continue
        if observed != message_index:
            observed += 1
            continue
        message = deserialize_message(serialized, Image)
        image_bgr = _image_message_to_bgr(message)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), image_bgr):
            raise RuntimeError(f'failed to write panorama to {output_path}')
        return {
            'topic': topic,
            'message_index': message_index,
            'bag_timestamp_ns': int(timestamp_ns),
            'header_timestamp_ns': (
                int(message.header.stamp.sec) * 1_000_000_000
                + int(message.header.stamp.nanosec)
            ),
            'frame_id': message.header.frame_id,
            'encoding': message.encoding,
            'width': int(message.width),
            'height': int(message.height),
            'step': int(message.step),
            'output_path': str(output_path),
        }
    raise ValueError(
        f'topic {topic!r} contains fewer than {message_index + 1} messages'
    )


def _image_message_to_bgr(message: Image) -> np.ndarray:
    encoding = message.encoding.casefold()
    if encoding not in _ENCODING_CHANNELS:
        raise ValueError(f'unsupported image encoding: {message.encoding!r}')
    channels, is_rgb = _ENCODING_CHANNELS[encoding]
    minimum_step = int(message.width) * channels
    if message.step < minimum_step:
        raise ValueError('image row step is smaller than encoded pixel data')
    raw = np.frombuffer(message.data, dtype=np.uint8)
    expected_size = int(message.height) * int(message.step)
    if raw.size != expected_size:
        raise ValueError('image data size does not match height and row step')
    rows = raw.reshape(int(message.height), int(message.step))
    image = rows[:, :minimum_step].reshape(
        int(message.height),
        int(message.width),
        channels,
    )
    if channels == 4:
        conversion = cv2.COLOR_RGBA2BGR if is_rgb else cv2.COLOR_BGRA2BGR
        return cv2.cvtColor(image, conversion)
    if is_rgb:
        return np.ascontiguousarray(image[..., ::-1])
    return np.ascontiguousarray(image)


def main() -> None:
    """Extract a selected panorama and print deterministic source metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag_path', type=Path)
    parser.add_argument('output_path', type=Path)
    parser.add_argument('--topic', default='/camera/image')
    parser.add_argument('--message-index', type=int, default=0)
    arguments = parser.parse_args()
    metadata = extract_panorama(
        arguments.bag_path,
        arguments.output_path,
        topic=arguments.topic,
        message_index=arguments.message_index,
    )
    for key, value in metadata.items():
        print(f'{key}={value}')


if __name__ == '__main__':
    main()
