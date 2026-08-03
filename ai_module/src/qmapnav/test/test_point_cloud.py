"""Tests for the ROS PointCloud2 registered-scan adapter."""

import numpy as np
from qmapnav.mapping.point_cloud import decode_scan_arrays
from qmapnav.mapping.point_cloud import decode_xyz_points
from sensor_msgs.msg import PointField
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


def test_point_cloud_decoder_preserves_intensity_and_filters_invalid_xyz() -> None:
    fields = [
        PointField(name=name, offset=index * 4, datatype=PointField.FLOAT32, count=1)
        for index, name in enumerate(('x', 'y', 'z', 'intensity'))
    ]
    message = point_cloud2.create_cloud(
        Header(frame_id='map'),
        fields,
        [(1.0, 2.0, 3.0, 4.0), (float('nan'), 0.0, 0.0, 5.0)],
    )

    decoded = decode_scan_arrays(message, expected_frame='map')

    np.testing.assert_allclose(decoded.xyz, [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(decoded.intensity, [4.0])
    assert decoded.input_point_count == 2
    assert decoded.invalid_xyz_count == 1


def test_point_cloud_decoder_rejects_wrong_frame_and_missing_fields() -> None:
    import pytest

    message = point_cloud2.create_cloud_xyz32(
        Header(frame_id='sensor'),
        [(1.0, 2.0, 3.0)],
    )
    with pytest.raises(ValueError, match='unexpected point cloud frame'):
        decode_scan_arrays(message, expected_frame='map')

    message.fields = message.fields[:2]
    with pytest.raises(ValueError, match='missing required fields'):
        decode_scan_arrays(message)


def test_point_cloud_decoder_uses_named_reordered_fields_and_extra_fields() -> None:
    fields = [
        PointField(name='intensity', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='x', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='label', offset=12, datatype=PointField.UINT32, count=1),
        PointField(name='y', offset=16, datatype=PointField.FLOAT32, count=1),
    ]
    message = point_cloud2.create_cloud(
        Header(frame_id='map'),
        fields,
        [(7.0, 3.0, 1.0, 42, 2.0)],
    )

    decoded = decode_scan_arrays(message, expected_frame='map')

    np.testing.assert_allclose(decoded.xyz, [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(decoded.intensity, [7.0])


def test_point_cloud_decoder_handles_large_and_all_invalid_clouds() -> None:
    count = 50_000
    values = np.arange(count, dtype=np.float32)
    message = point_cloud2.create_cloud_xyz32(
        Header(frame_id='map'),
        np.column_stack((values, values + 1.0, values + 2.0)),
    )

    decoded = decode_scan_arrays(message, expected_frame='map')

    assert decoded.xyz.shape == (count, 3)
    assert decoded.input_point_count == count
    assert decoded.invalid_xyz_count == 0

    invalid = point_cloud2.create_cloud_xyz32(
        Header(frame_id='map'),
        [(float('nan'), 0.0, 0.0), (0.0, float('inf'), 0.0)],
    )
    filtered = decode_scan_arrays(invalid, expected_frame='map')
    assert filtered.xyz.shape == (0, 3)
    assert filtered.invalid_xyz_count == 2
