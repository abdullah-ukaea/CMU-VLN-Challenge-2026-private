"""Tests for the ROS PointCloud2 registered-scan adapter."""

import numpy as np
from qmapnav.mapping.point_cloud import decode_xyz_points
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


def test_point_cloud_decoder_extracts_only_xyz() -> None:
    message = point_cloud2.create_cloud_xyz32(
        Header(frame_id='map'),
        [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
    )

    points = decode_xyz_points(message)

    assert points.shape == (2, 3)
    assert points.dtype == np.float64
    np.testing.assert_allclose(
        points,
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )


def test_point_cloud_decoder_handles_empty_cloud() -> None:
    message = point_cloud2.create_cloud_xyz32(Header(frame_id='map'), [])

    assert decode_xyz_points(message).shape == (0, 3)
