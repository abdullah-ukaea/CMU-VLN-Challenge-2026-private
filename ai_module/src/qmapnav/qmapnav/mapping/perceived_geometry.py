"""
Present a perceived object through the frozen oracle geometry shape.

``reasoning.semantic_geometry`` builds footprints and annuli from anything
exposing ``object_id``, ``class_name``, ``centre_xyz``, ``dimensions_xyz`` and
``yaw``. A fused :class:`~qmapnav.common.ObjectInstance` carries the same
information plus an explicit ``orientation_confidence``, so it is adapted here
rather than duplicating the polygon logic for perceived maps.

When orientation evidence is weak the adapter falls back to the axis-aligned
box and reports ``yaw = 0``, matching the Day 6 conservative marker policy:
never invent a precise orientation from weak evidence, and prefer slightly
over-covering the object.
"""

from dataclasses import dataclass
from math import isfinite

from qmapnav.common import ObjectInstance


DEFAULT_ORIENTATION_CONFIDENCE_THRESHOLD = 0.35


@dataclass(frozen=True)
class PerceivedBox:
    """A perceived object expressed in the oracle geometry field shape."""

    object_id: str
    class_name: str
    centre_xyz: tuple[float, float, float]
    dimensions_xyz: tuple[float, float, float]
    yaw: float
    orientation_confidence: float
    used_axis_aligned_fallback: bool

    def __post_init__(self) -> None:
        if not self.object_id or not self.object_id.strip():
            raise ValueError('object_id must be a non-empty string')
        if not self.class_name or not self.class_name.strip():
            raise ValueError('class_name must be a non-empty string')
        for name in ('centre_xyz', 'dimensions_xyz'):
            values = tuple(getattr(self, name))
            if len(values) != 3 or not all(
                isfinite(value) for value in values
            ):
                raise ValueError(f'{name} must hold three finite values')
            object.__setattr__(
                self, name, tuple(float(value) for value in values)
            )
        if any(value <= 0.0 for value in self.dimensions_xyz):
            raise ValueError('dimensions_xyz must be strictly positive')
        if not isfinite(self.yaw):
            raise ValueError('yaw must be finite')

    @property
    def footprint_radius(self) -> float:
        """Return half the diagonal of the floor-plane footprint."""
        length, width, _ = self.dimensions_xyz
        return 0.5 * (length ** 2 + width ** 2) ** 0.5


def perceived_box(
    instance: ObjectInstance,
    *,
    class_name: str | None = None,
    orientation_confidence_threshold: float = (
        DEFAULT_ORIENTATION_CONFIDENCE_THRESHOLD
    ),
    inflation_m: float = 0.0,
) -> PerceivedBox:
    """
    Adapt one fused instance, degrading to its AABB when yaw is unreliable.

    ``inflation_m`` grows the footprint symmetrically and is applied on top of
    the fallback, so a low-confidence orientation yields a conservative box
    rather than a confidently wrong one.
    """
    if not isinstance(instance, ObjectInstance):
        raise TypeError('instance must be ObjectInstance')
    if not isfinite(inflation_m) or inflation_m < 0.0:
        raise ValueError('inflation_m must be finite and non-negative')
    if not isfinite(orientation_confidence_threshold) or not (
        0.0 <= orientation_confidence_threshold <= 1.0
    ):
        raise ValueError('orientation threshold must lie in [0, 1]')

    resolved_class = class_name or max(
        instance.class_scores.items(), key=lambda item: (item[1], item[0])
    )[0]
    trusted = (
        instance.orientation_confidence >= orientation_confidence_threshold
    )
    if trusted:
        dimensions = tuple(float(value) for value in instance.obb_dimensions)
        yaw = float(instance.obb_yaw)
        centre = tuple(float(value) for value in instance.centroid_xyz)
    else:
        extent = instance.aabb_max_xyz - instance.aabb_min_xyz
        centre_array = (instance.aabb_max_xyz + instance.aabb_min_xyz) / 2.0
        dimensions = tuple(
            max(float(value), 1e-3) for value in extent
        )
        yaw = 0.0
        centre = tuple(float(value) for value in centre_array)
    inflated = (
        dimensions[0] + 2.0 * inflation_m,
        dimensions[1] + 2.0 * inflation_m,
        dimensions[2],
    )
    return PerceivedBox(
        object_id=str(instance.instance_id),
        class_name=resolved_class,
        centre_xyz=centre,
        dimensions_xyz=inflated,
        yaw=yaw,
        orientation_confidence=float(instance.orientation_confidence),
        used_axis_aligned_fallback=not trusted,
    )


__all__ = [
    'DEFAULT_ORIENTATION_CONFIDENCE_THRESHOLD',
    'PerceivedBox',
    'perceived_box',
]
