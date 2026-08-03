"""Timestamped translation interpolation and shortest-arc quaternion SLERP."""

import numpy as np


def normalize_quaternion_xyzw(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Return a normalized finite XYZW quaternion with deterministic sign."""
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError('quaternion_xyzw must contain four finite values')
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError('quaternion_xyzw must have non-zero norm')
    normalized = quaternion / norm
    if normalized[3] < 0.0:
        normalized = -normalized
    return normalized


def slerp_quaternion_xyzw(
    before_xyzw: np.ndarray,
    after_xyzw: np.ndarray,
    ratio: float,
) -> np.ndarray:
    """Interpolate unit quaternions along the shortest arc."""
    if not np.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError('ratio must be finite and in [0, 1]')
    before = normalize_quaternion_xyzw(before_xyzw)
    after = normalize_quaternion_xyzw(after_xyzw)
    dot = float(np.dot(before, after))
    if dot < 0.0:
        after = -after
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion_xyzw(before + ratio * (after - before))
    theta = float(np.arccos(dot))
    sin_theta = float(np.sin(theta))
    interpolated = (
        np.sin((1.0 - ratio) * theta) / sin_theta * before
        + np.sin(ratio * theta) / sin_theta * after
    )
    return normalize_quaternion_xyzw(interpolated)


__all__ = ['normalize_quaternion_xyzw', 'slerp_quaternion_xyzw']
