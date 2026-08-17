"""ROS-independent proxy metrics for oracle answers and semantic routes."""

from dataclasses import dataclass
from math import ceil, hypot, isfinite

from qmapnav.reasoning.semantic_geometry import Point2D
from qmapnav.reasoning.semantic_geometry import Polygon2D
from qmapnav.reasoning.semantic_geometry import SemanticRegion


RelationKey = tuple[str, str, tuple[str, ...]]


def _finite_non_negative(name: str, value: float) -> float:
    if not isfinite(value) or value < 0.0:
        raise ValueError(f'{name} must be finite and non-negative')
    return float(value)


@dataclass(frozen=True)
class ObjectSelectionMetric:
    """Exact object-ID selection result, or an explicit unavailable label."""

    predicted_object_id: str | None
    expected_object_id: str | None
    label_available: bool
    correct: bool | None


@dataclass(frozen=True)
class CountAccuracyMetric:
    """Exact and absolute numerical-answer errors when a label is available."""

    predicted_count: int
    expected_count: int | None
    label_available: bool
    exact_match: bool | None
    absolute_error: int | None

    def __post_init__(self) -> None:
        if self.predicted_count < 0:
            raise ValueError('predicted_count must be non-negative')


@dataclass(frozen=True)
class RelationMetric:
    """Binary confusion counts and derived metrics for one relation."""

    relation: str
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float
    recall: float
    f1: float

    @property
    def support(self) -> int:
        """Return the number of expected positive observations."""
        return self.true_positive + self.false_negative

    @property
    def sample_count(self) -> int:
        """Return the total number of evaluated observations."""
        return (
            self.true_positive
            + self.false_positive
            + self.false_negative
            + self.true_negative
        )


@dataclass(frozen=True)
class RequiredRegionMetric:
    """Whether and where an ordered trajectory first intersects one region."""

    region_id: str
    satisfied: bool
    first_hit_index: float | None


@dataclass(frozen=True)
class ForbiddenRegionMetric:
    """Forbidden-region intersection and approximate travelled length inside."""

    region_id: str
    violated: bool
    approximate_length_inside: float

    def __post_init__(self) -> None:
        _finite_non_negative(
            'approximate_length_inside',
            self.approximate_length_inside,
        )


@dataclass(frozen=True)
class SemanticRouteMetric:
    """Complete six-point semantic instruction proxy and raw diagnostics."""

    required_regions: tuple[RequiredRegionMetric, ...]
    forbidden_regions: tuple[ForbiddenRegionMetric, ...]
    required_intersection_fraction: float
    ordered_hit_indices: tuple[float | None, ...]
    ordered_constraints_completed: int
    order_correct: bool
    forbidden_violation_count: int
    terminal_goal_distance: float
    terminal_goal_reached: bool
    proxy_score: float
    success: bool

    def __post_init__(self) -> None:
        if not 0.0 <= self.required_intersection_fraction <= 1.0:
            raise ValueError('required_intersection_fraction must be in [0, 1]')
        _finite_non_negative('terminal_goal_distance', self.terminal_goal_distance)
        if not 0.0 <= self.proxy_score <= 6.0:
            raise ValueError('instruction proxy_score must be in [0, 6]')


@dataclass(frozen=True)
class TimingMetric:
    """Offline stage timings and optional live execution completion time."""

    parser_seconds: float
    reasoning_seconds: float
    planning_seconds: float
    execution_seconds: float | None
    total_seconds: float

    def __post_init__(self) -> None:
        for name in (
            'parser_seconds',
            'reasoning_seconds',
            'planning_seconds',
            'total_seconds',
        ):
            _finite_non_negative(name, getattr(self, name))
        if self.execution_seconds is not None:
            _finite_non_negative('execution_seconds', self.execution_seconds)


def object_selection_metric(
    predicted_object_id: str | None,
    expected_object_id: str | None,
) -> ObjectSelectionMetric:
    """Measure exact object selection without inventing a missing label."""
    available = expected_object_id is not None
    return ObjectSelectionMetric(
        predicted_object_id=predicted_object_id,
        expected_object_id=expected_object_id,
        label_available=available,
        correct=(predicted_object_id == expected_object_id if available else None),
    )


def count_accuracy_metric(
    predicted_count: int,
    expected_count: int | None,
) -> CountAccuracyMetric:
    """Measure exact count and absolute error without fabricating an answer."""
    if predicted_count < 0:
        raise ValueError('predicted_count must be non-negative')
    available = expected_count is not None
    return CountAccuracyMetric(
        predicted_count=predicted_count,
        expected_count=expected_count,
        label_available=available,
        exact_match=(predicted_count == expected_count if available else None),
        absolute_error=(
            abs(predicted_count - expected_count) if available else None
        ),
    )


def relation_metrics(
    expected: set[RelationKey],
    predicted: set[RelationKey],
    *,
    relation: str,
    negatives_evaluated: int = 0,
) -> RelationMetric:
    """Calculate precision, recall, and F1 for one relation edge set."""
    expected_edges = {item for item in expected if item[0] == relation}
    predicted_edges = {item for item in predicted if item[0] == relation}
    true_positive = len(expected_edges & predicted_edges)
    false_positive = len(predicted_edges - expected_edges)
    false_negative = len(expected_edges - predicted_edges)
    true_negative = negatives_evaluated - false_positive
    if true_negative < 0:
        raise ValueError('negatives_evaluated is smaller than false positives')
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = (
        true_positive / precision_denominator if precision_denominator else 0.0
    )
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return RelationMetric(
        relation=relation,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _validate_trajectory(trajectory: tuple[Point2D, ...]) -> tuple[Point2D, ...]:
    points = tuple(tuple(float(value) for value in point) for point in trajectory)
    if not points:
        raise ValueError('trajectory must contain at least one point')
    if any(
        len(point) != 2 or not all(isfinite(value) for value in point)
        for point in points
    ):
        raise ValueError('trajectory points must contain two finite coordinates')
    return points


def _sample_trajectory(
    trajectory: tuple[Point2D, ...],
    resolution: float,
) -> tuple[tuple[float, Point2D], ...]:
    if not isfinite(resolution) or resolution <= 0.0:
        raise ValueError('sampling resolution must be finite and positive')
    samples: list[tuple[float, Point2D]] = [(0.0, trajectory[0])]
    for index, (start, end) in enumerate(zip(trajectory, trajectory[1:])):
        distance = hypot(end[0] - start[0], end[1] - start[1])
        divisions = max(1, ceil(distance / resolution))
        for offset in range(1, divisions + 1):
            fraction = offset / divisions
            samples.append(
                (
                    index + fraction,
                    (
                        start[0] + fraction * (end[0] - start[0]),
                        start[1] + fraction * (end[1] - start[1]),
                    ),
                )
            )
    return tuple(samples)


def required_region_metrics(
    trajectory: tuple[Point2D, ...],
    regions: tuple[SemanticRegion, ...],
    *,
    sampling_resolution: float = 0.05,
) -> tuple[RequiredRegionMetric, ...]:
    """Measure each required region's first trajectory intersection."""
    points = _validate_trajectory(trajectory)
    samples = _sample_trajectory(points, sampling_resolution)
    results = []
    for region in regions:
        first_hit = next(
            (position for position, point in samples if region.contains(point)),
            None,
        )
        results.append(
            RequiredRegionMetric(
                region_id=region.region_id,
                satisfied=first_hit is not None,
                first_hit_index=first_hit,
            )
        )
    return tuple(results)


def forbidden_region_metrics(
    trajectory: tuple[Point2D, ...],
    regions: tuple[SemanticRegion, ...],
    *,
    sampling_resolution: float = 0.05,
) -> tuple[ForbiddenRegionMetric, ...]:
    """Measure forbidden intersections and approximate travel inside them."""
    points = _validate_trajectory(trajectory)
    results = []
    for region in regions:
        length_inside = 0.0
        violated = region.contains(points[0])
        for start, end in zip(points, points[1:]):
            length = hypot(end[0] - start[0], end[1] - start[1])
            divisions = max(1, ceil(length / sampling_resolution))
            for offset in range(divisions):
                fraction = (offset + 0.5) / divisions
                midpoint = (
                    start[0] + fraction * (end[0] - start[0]),
                    start[1] + fraction * (end[1] - start[1]),
                )
                if region.contains(midpoint):
                    violated = True
                    length_inside += length / divisions
        results.append(
            ForbiddenRegionMetric(
                region_id=region.region_id,
                violated=violated,
                approximate_length_inside=length_inside,
            )
        )
    return tuple(results)


def _point_segment_distance(
    point: Point2D,
    start: Point2D,
    end: Point2D,
) -> float:
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    squared_length = delta_x ** 2 + delta_y ** 2
    if squared_length <= 1e-18:
        return hypot(point[0] - start[0], point[1] - start[1])
    projection = (
        (point[0] - start[0]) * delta_x
        + (point[1] - start[1]) * delta_y
    ) / squared_length
    projection = min(1.0, max(0.0, projection))
    closest = (
        start[0] + projection * delta_x,
        start[1] + projection * delta_y,
    )
    return hypot(point[0] - closest[0], point[1] - closest[1])


def _polygon_boundary_distance(point: Point2D, polygon: Polygon2D) -> float:
    edges = zip(polygon.vertices, polygon.vertices[1:] + polygon.vertices[:1])
    return min(_point_segment_distance(point, start, end) for start, end in edges)


def terminal_goal_distance(point: Point2D, region: SemanticRegion) -> float:
    """Measure distance from a final point to a region's usable area."""
    if region.contains(point):
        return 0.0
    if region.polygon.contains(point):
        containing_exclusions = [
            exclusion for exclusion in region.exclusions if exclusion.contains(point)
        ]
        if containing_exclusions:
            return min(
                _polygon_boundary_distance(point, exclusion)
                for exclusion in containing_exclusions
            )
    return _polygon_boundary_distance(point, region.polygon)


def semantic_route_metric(
    trajectory: tuple[Point2D, ...],
    required_regions: tuple[SemanticRegion, ...],
    forbidden_regions: tuple[SemanticRegion, ...],
    *,
    sampling_resolution: float = 0.05,
    terminal_tolerance: float = 1e-6,
) -> SemanticRouteMetric:
    """Evaluate ordered, forbidden, and terminal semantics on a trajectory."""
    points = _validate_trajectory(trajectory)
    if not required_regions:
        raise ValueError('at least one required region is needed')
    required = required_region_metrics(
        points,
        required_regions,
        sampling_resolution=sampling_resolution,
    )
    forbidden = forbidden_region_metrics(
        points,
        forbidden_regions,
        sampling_resolution=sampling_resolution,
    )
    samples = _sample_trajectory(points, sampling_resolution)
    ordered_hits: list[float | None] = []
    previous = float('-inf')
    for region in required_regions:
        hit = next(
            (
                position
                for position, point in samples
                if position >= previous and region.contains(point)
            ),
            None,
        )
        ordered_hits.append(hit)
        if hit is None:
            ordered_hits.extend(
                None for _ in range(len(required_regions) - len(ordered_hits))
            )
            break
        previous = hit
    ordered_completed = sum(hit is not None for hit in ordered_hits)
    order_correct = ordered_completed == len(required)
    required_fraction = sum(item.satisfied for item in required) / len(required)
    violations = sum(item.violated for item in forbidden)
    terminal_distance = terminal_goal_distance(points[-1], required_regions[-1])
    terminal_reached = terminal_distance <= terminal_tolerance
    score = (
        3.0 * required_fraction
        + float(order_correct)
        + float(violations == 0)
        + float(terminal_reached)
    )
    return SemanticRouteMetric(
        required_regions=required,
        forbidden_regions=forbidden,
        required_intersection_fraction=required_fraction,
        ordered_hit_indices=tuple(ordered_hits),
        ordered_constraints_completed=ordered_completed,
        order_correct=order_correct,
        forbidden_violation_count=violations,
        terminal_goal_distance=terminal_distance,
        terminal_goal_reached=terminal_reached,
        proxy_score=score,
        success=(
            required_fraction == 1.0
            and order_correct
            and violations == 0
            and terminal_reached
        ),
    )


__all__ = [
    'CountAccuracyMetric',
    'ForbiddenRegionMetric',
    'ObjectSelectionMetric',
    'RelationKey',
    'RelationMetric',
    'RequiredRegionMetric',
    'SemanticRouteMetric',
    'TimingMetric',
    'count_accuracy_metric',
    'forbidden_region_metrics',
    'object_selection_metric',
    'relation_metrics',
    'required_region_metrics',
    'semantic_route_metric',
    'terminal_goal_distance',
]
