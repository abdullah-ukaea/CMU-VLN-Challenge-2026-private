"""ROS PointCloud2 decoding adapter for scan accumulation and projection."""

from dataclasses import dataclass

import numpy as np
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2


@dataclass(frozen=True)
class ScanArrays:
    """Finite decoded scan channels and conversion diagnostics."""

    xyz: np.ndarray
    intensity: np.ndarray | None
    input_point_count: int
    invalid_xyz_count: int


def stamp_to_nanoseconds(stamp: object) -> int:
    """Convert a ROS builtin time message into exact integer nanoseconds."""
    seconds = int(getattr(stamp, 'sec'))
    nanoseconds = int(getattr(stamp, 'nanosec'))
    if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
        raise ValueError('ROS timestamp is invalid')
    return seconds * 1_000_000_000 + nanoseconds


def decode_scan_arrays(
    message: object,
    *,
    expected_frame: str | None = None,
) -> ScanArrays:
    """Decode finite XYZ and optional intensity after validating field layout."""
    header = getattr(message, 'header', None)
    if header is None:
        raise ValueError('PointCloud2 message is missing a header')
    if expected_frame is not None and header.frame_id != expected_frame:
        raise ValueError(
            f'unexpected point cloud frame {header.frame_id!r}; '
            f'expected {expected_frame!r}'
        )
    fields = {field.name: field for field in getattr(message, 'fields', ())}
    missing = sorted({'x', 'y', 'z'} - set(fields))
    if missing:
        raise ValueError(f'PointCloud2 is missing required fields: {missing}')
    numeric_types = {
        PointField.INT8,
        PointField.UINT8,
        PointField.INT16,
        PointField.UINT16,
        PointField.INT32,
        PointField.UINT32,
        PointField.FLOAT32,
        PointField.FLOAT64,
    }
    for name in ('x', 'y', 'z'):
        if fields[name].count != 1 or fields[name].datatype not in numeric_types:
            raise ValueError(f'PointCloud2 field {name!r} is not scalar numeric')
    has_intensity = (
        'intensity' in fields
        and fields['intensity'].count == 1
        and fields['intensity'].datatype in numeric_types
    )
    requested = ['x', 'y', 'z'] + (['intensity'] if has_intensity else [])
    decoded = point_cloud2.read_points_numpy(
        message,
        field_names=requested,
        skip_nans=False,
    )
    array = np.asarray(decoded)
    column_count = len(requested)
    if array.size == 0:
        array = np.empty((0, column_count), dtype=np.float64)
    elif array.dtype.names:
        array = np.column_stack([array[name] for name in requested])
    else:
        array = array.reshape((-1, column_count))
    array = np.asarray(array, dtype=np.float64)
    input_count = int(array.shape[0])
    finite = np.all(np.isfinite(array[:, :3]), axis=1)
    xyz = np.ascontiguousarray(array[finite, :3])
    xyz.setflags(write=False)
    intensity = None
    if has_intensity:
        intensity = np.ascontiguousarray(array[finite, 3])
        intensity.setflags(write=False)
    return ScanArrays(
        xyz=xyz,
        intensity=intensity,
        input_point_count=input_count,
        invalid_xyz_count=input_count - int(np.count_nonzero(finite)),
    )


def decode_xyz_points(message: object) -> np.ndarray:
    """Decode only XYZ fields from a sensor_msgs PointCloud2 message."""
    return decode_scan_arrays(message).xyz.copy()


__all__ = [
    'ScanArrays',
    'decode_scan_arrays',
    'decode_xyz_points',
    'stamp_to_nanoseconds',
]
