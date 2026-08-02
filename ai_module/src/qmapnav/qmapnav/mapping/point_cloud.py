"""ROS PointCloud2 decoding adapter for the pure scan accumulator."""

import numpy as np
from sensor_msgs_py import point_cloud2


def decode_xyz_points(message: object) -> np.ndarray:
    """Decode only XYZ fields from a sensor_msgs PointCloud2 message."""
    points = point_cloud2.read_points_numpy(
        message,
        field_names=['x', 'y', 'z'],
        skip_nans=False,
    )
    array = np.asarray(points)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(array, dtype=np.float64).reshape((-1, 3))


__all__ = ['decode_xyz_points']
