"""Pure-Python floor-plane geometry for oracle semantic reasoning."""

from dataclasses import dataclass
from math import atan2, cos, hypot, isfinite, sin

from qmapnav.evaluation.ground_truth import OracleObject


Point2D = tuple[float, float]


def _finite_point(name: str, point: Point2D) -> Point2D:
    values = tuple(point)
    if len(values) != 2 or not all(isfinite(value) for value in values):
        raise ValueError(f'{name} must contain two finite coordinates')
    return float(values[0]), float(values[1])


def _polygon_area(vertices: tuple[Point2D, ...]) -> float:
    return 0.5 * sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(
            vertices,
            vertices[1:] + vertices[:1],
        )
    )


def _point_on_segment(
    point: Point2D,
    start: Point2D,
    end: Point2D,
    tolerance: float = 1e-9,
) -> bool:
    px, py = point
    ax, ay = start
    bx, by = end
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > tolerance:
        return False
    dot = (px - ax) * (bx - ax) + (py - ay) * (by - ay)
    squared_length = (bx - ax) ** 2 + (by - ay) ** 2
    return -tolerance <= dot <= squared_length + tolerance


@dataclass(frozen=True)
class Polygon2D:
    """A validated simple polygon represented by ordered floor-plane vertices."""

    vertices: tuple[Point2D, ...]

    def __post_init__(self) -> None:
        vertices = tuple(
            _finite_point('polygon vertex', vertex) for vertex in self.vertices
        )
        if len(vertices) < 3:
            raise ValueError('a polygon requires at least three vertices')
        if abs(_polygon_area(vertices)) <= 1e-12:
            raise ValueError('polygon area must be non-zero')
        object.__setattr__(self, 'vertices', vertices)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return ``min_x, min_y, max_x, max_y``."""
        xs, ys = zip(*self.vertices)
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def centre(self) -> Point2D:
        """Compute the arithmetic mean of the polygon vertices."""
        count = len(self.vertices)
        return (
            sum(point[0] for point in self.vertices) / count,
            sum(point[1] for point in self.vertices) / count,
        )

    def contains(self, point: Point2D) -> bool:
        """Return whether a point lies inside or on the polygon boundary."""
        point = _finite_point('point', point)
        edges = tuple(zip(self.vertices, self.vertices[1:] + self.vertices[:1]))
        if any(_point_on_segment(point, start, end) for start, end in edges):
            return True

        x, y = point
        inside = False
        for (x1, y1), (x2, y2) in edges:
            if (y1 > y) == (y2 > y):
                continue
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
        return inside


@dataclass(frozen=True)
class SemanticRegion:
    """A semantic floor region with optional excluded interior polygons."""

    region_id: str
    region_type: str
    polygon: Polygon2D
    source_object_ids: tuple[str, ...]
    required: bool
    exclusions: tuple[Polygon2D, ...] = ()

    def __post_init__(self) -> None:
        if not self.region_id or not self.region_id.strip():
            raise ValueError('region_id must be a non-empty string')
        if not self.region_type or not self.region_type.strip():
            raise ValueError('region_type must be a non-empty string')
        if not isinstance(self.polygon, Polygon2D):
            raise TypeError('polygon must be a Polygon2D')
        object_ids = tuple(self.source_object_ids)
        if not object_ids or any(not item.strip() for item in object_ids):
            raise ValueError('source_object_ids must contain non-empty IDs')
        if len(object_ids) != len(set(object_ids)):
            raise ValueError('source_object_ids must not contain duplicates')
        object.__setattr__(self, 'source_object_ids', object_ids)
        exclusions = tuple(self.exclusions)
        if not all(isinstance(item, Polygon2D) for item in exclusions):
            raise TypeError('exclusions must contain only Polygon2D values')
        object.__setattr__(self, 'exclusions', exclusions)

    def contains(self, point: Point2D) -> bool:
        """Return whether a point is in the usable portion of this region."""
        return self.polygon.contains(point) and not any(
            exclusion.contains(point) for exclusion in self.exclusions
        )


@dataclass(frozen=True)
class GateResult:
    """Result of attempting to construct a traversable between-object gate."""

    valid: bool
    region: SemanticRegion | None
    gap_width: float
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.gap_width) or self.gap_width < 0.0:
            raise ValueError('gap_width must be finite and non-negative')
        if self.valid != (self.region is not None):
            raise ValueError('valid must agree with whether a region exists')
        if self.valid and self.reason is not None:
            raise ValueError('a valid gate must not have a failure reason')
        if not self.valid and not self.reason:
            raise ValueError('an invalid gate requires a reason')


def _oriented_rectangle(
    centre: Point2D,
    length: float,
    width: float,
    yaw: float,
) -> Polygon2D:
    if not all(isfinite(value) for value in (length, width, yaw)):
        raise ValueError('rectangle dimensions and yaw must be finite')
    if length <= 0.0 or width <= 0.0:
        raise ValueError('rectangle dimensions must be positive')
    centre_x, centre_y = _finite_point('rectangle centre', centre)
    axis_x = (cos(yaw), sin(yaw))
    axis_y = (-sin(yaw), cos(yaw))
    half_length = length / 2.0
    half_width = width / 2.0
    return Polygon2D(
        tuple(
            (
                centre_x + sign_x * half_length * axis_x[0]
                + sign_y * half_width * axis_y[0],
                centre_y + sign_x * half_length * axis_x[1]
                + sign_y * half_width * axis_y[1],
            )
            for sign_x, sign_y in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        )
    )


def object_footprint(obj: OracleObject, inflation: float = 0.0) -> Polygon2D:
    """Project an oracle object's oriented 3D box onto the XY floor plane."""
    if not isfinite(inflation) or inflation < 0.0:
        raise ValueError('inflation must be finite and non-negative')
    length, width, _ = obj.dimensions_xyz
    return _oriented_rectangle(
        obj.centre_xyz[:2],
        length + 2.0 * inflation,
        width + 2.0 * inflation,
        obj.yaw,
    )


def make_near_region(
    obj: OracleObject,
    min_distance: float = 0.6,
    max_distance: float = 1.8,
    *,
    required: bool = True,
    region_id: str | None = None,
) -> SemanticRegion:
    """Create an oriented rectangular annulus around an object footprint."""
    if not all(isfinite(value) for value in (min_distance, max_distance)):
        raise ValueError('near distances must be finite')
    if min_distance < 0.0 or max_distance <= min_distance:
        raise ValueError('near distances require 0 <= min < max')
    return SemanticRegion(
        region_id=region_id or f'near_{obj.object_id}',
        region_type='near' if required else 'forbidden_near',
        polygon=object_footprint(obj, max_distance),
        exclusions=(object_footprint(obj, min_distance),),
        source_object_ids=(obj.object_id,),
        required=required,
    )


def make_approach_region(
    obj: OracleObject,
    minimum_clearance: float = 0.45,
    maximum_distance: float = 1.25,
    *,
    region_id: str | None = None,
) -> SemanticRegion:
    """Create collision-safe candidate goal positions around an object."""
    if maximum_distance <= minimum_clearance:
        raise ValueError('maximum_distance must exceed minimum_clearance')
    region = make_near_region(
        obj,
        min_distance=minimum_clearance,
        max_distance=maximum_distance,
        region_id=region_id or f'approach_{obj.object_id}',
    )
    return SemanticRegion(
        region_id=region.region_id,
        region_type='approach',
        polygon=region.polygon,
        exclusions=region.exclusions,
        source_object_ids=region.source_object_ids,
        required=True,
    )


def _support_radius(obj: OracleObject, direction: Point2D) -> float:
    dx, dy = direction
    local_x = (cos(obj.yaw), sin(obj.yaw))
    local_y = (-sin(obj.yaw), cos(obj.yaw))
    length, width, _ = obj.dimensions_xyz
    return (
        abs(dx * local_x[0] + dy * local_x[1]) * length / 2.0
        + abs(dx * local_y[0] + dy * local_y[1]) * width / 2.0
    )


def make_between_gate(
    object_a: OracleObject,
    object_b: OracleObject,
    *,
    robot_diameter: float = 0.55,
    clearance: float = 0.10,
    gate_depth: float = 0.35,
    required: bool = True,
    region_id: str | None = None,
    allow_inferred_clearance: bool = False,
) -> GateResult:
    """Construct a thin semantic gate spanning the free gap between two OBBs."""
    if object_a.object_id == object_b.object_id:
        raise ValueError('between gate requires two distinct objects')
    values = (robot_diameter, clearance, gate_depth)
    if not all(isfinite(value) and value > 0.0 for value in values):
        raise ValueError('gate dimensions must be finite and positive')

    ax, ay = object_a.centre_xyz[:2]
    bx, by = object_b.centre_xyz[:2]
    centre_distance = hypot(bx - ax, by - ay)
    if centre_distance <= 1e-9:
        return GateResult(False, None, 0.0, 'object centres coincide')
    direction = ((bx - ax) / centre_distance, (by - ay) / centre_distance)
    radius_a = _support_radius(object_a, direction)
    radius_b = _support_radius(object_b, direction)
    raw_gap = max(0.0, centre_distance - radius_a - radius_b)
    required_width = robot_diameter + 2.0 * clearance
    if raw_gap < required_width and not allow_inferred_clearance:
        return GateResult(
            False,
            None,
            raw_gap,
            f'gap {raw_gap:.3f} m is narrower than {required_width:.3f} m',
        )

    boundary_a = (ax + direction[0] * radius_a, ay + direction[1] * radius_a)
    boundary_b = (bx - direction[0] * radius_b, by - direction[1] * radius_b)
    midpoint = (
        (boundary_a[0] + boundary_b[0]) / 2.0,
        (boundary_a[1] + boundary_b[1]) / 2.0,
    )
    connector_yaw = atan2(direction[1], direction[0])
    gate = SemanticRegion(
        region_id=(
            region_id
            or f'between_{object_a.object_id}_{object_b.object_id}'
        ),
        region_type='between_gate' if required else 'forbidden_between',
        polygon=_oriented_rectangle(
            midpoint,
            max(raw_gap, required_width),
            gate_depth,
            connector_yaw,
        ),
        source_object_ids=(object_a.object_id, object_b.object_id),
        required=required,
    )
    return GateResult(True, gate, raw_gap)


__all__ = [
    'GateResult',
    'Point2D',
    'Polygon2D',
    'SemanticRegion',
    'make_approach_region',
    'make_between_gate',
    'make_near_region',
    'object_footprint',
]
