"""
Score viewpoints by expected information gain against travel cost.

    S(v) = w_t*V_t + w_a*V_a + w_u*U + w_d*D + w_s*S_s - w_c*C - w_r*R

Every term is computed independently and stored on the candidate, so a trace
shows which evidence justified the move. Query relevance is deliberately
weighted above raw frontier gain: a viewpoint that reveals a large empty room
is worth less than one likely to expose the specific missing target.
"""

from dataclasses import dataclass
from math import atan2
from math import cos
from math import hypot
from math import isfinite
from math import pi
from math import sin

from qmapnav.exploration.exploration_need import ExplorationNeed
from qmapnav.exploration.viewpoint_candidate import SELECTION_STATUSES
from qmapnav.exploration.viewpoint_candidate import ViewpointCandidate
from qmapnav.exploration.viewpoint_candidate import ViewpointScoreTerms
from qmapnav.exploration.viewpoint_candidate import ViewpointSelection
from qmapnav.mapping.occupancy_grid import OccupancyGrid2D


@dataclass(frozen=True)
class ViewpointScoringConfig:
    """Weights and normalizers for the six-term utility."""

    target_weight: float = 1.0
    anchor_weight: float = 0.8
    unexplored_weight: float = 0.45
    ambiguity_weight: float = 0.9
    support_weight: float = 0.85
    travel_weight: float = 0.6
    risk_weight: float = 1.0

    travel_normalization_m: float = 6.0
    preferred_observation_distance_m: float = 1.8
    maximum_useful_distance_m: float = 6.0
    unexplored_normalization_cells: int = 120
    minimum_selectable_score: float = 0.15
    horizontal_fov_rad: float = pi / 2.0
    sensor_range_m: float = 6.0

    def __post_init__(self) -> None:
        weights = (
            'target_weight',
            'anchor_weight',
            'unexplored_weight',
            'ambiguity_weight',
            'support_weight',
            'travel_weight',
            'risk_weight',
            'travel_normalization_m',
            'preferred_observation_distance_m',
            'maximum_useful_distance_m',
            'horizontal_fov_rad',
            'sensor_range_m',
        )
        for name in weights:
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        if isinstance(self.unexplored_normalization_cells, bool) or (
            self.unexplored_normalization_cells < 1
        ):
            raise ValueError('unexplored_normalization_cells must be >= 1')
        if not isfinite(self.minimum_selectable_score):
            raise ValueError('minimum_selectable_score must be finite')
        if (
            self.preferred_observation_distance_m
            >= self.maximum_useful_distance_m
        ):
            raise ValueError(
                'preferred observation distance must be below the maximum'
            )


def distance_quality(
    distance_m: float,
    config: ViewpointScoringConfig,
) -> float:
    """
    Return how well an observation distance suits reliable perception.

    Peaks at the preferred stand-off and decays toward zero at the maximum
    useful range, so both too-far and pressed-against-the-object poses lose.
    """
    if not isfinite(distance_m) or distance_m < 0.0:
        return 0.0
    preferred = config.preferred_observation_distance_m
    if distance_m >= config.maximum_useful_distance_m:
        return 0.0
    if distance_m <= preferred:
        # Very close poses lose the surface from the field of view.
        return max(0.0, min(1.0, distance_m / preferred))
    span = config.maximum_useful_distance_m - preferred
    return max(0.0, 1.0 - (distance_m - preferred) / span)


def target_visibility_score(
    candidate: ViewpointCandidate,
    *,
    grid: OccupancyGrid2D,
    need: ExplorationNeed,
    target_xy: tuple[float, float] | None,
    config: ViewpointScoringConfig,
    previous_bearings: tuple[float, ...] = (),
) -> float:
    """
    Score expected visibility of the missing target.

    Branches on what is actually known: a weak existing candidate is scored
    geometrically, an unseen target with a support hypothesis defers to the
    support term, and a wholly unknown target falls back to frontier gain.
    """
    if target_xy is None:
        return 0.0
    pose = candidate.pose_xy_yaw
    if not grid.line_of_sight((pose[0], pose[1]), target_xy):
        return 0.0
    distance = hypot(target_xy[0] - pose[0], target_xy[1] - pose[1])
    quality = distance_quality(distance, config)
    if quality <= 0.0:
        return 0.0
    bearing = atan2(target_xy[1] - pose[1], target_xy[0] - pose[0])
    centring = max(
        0.0, 1.0 - abs(_wrap(bearing - pose[2])) / (config.horizontal_fov_rad)
    )
    baseline = 1.0
    if previous_bearings:
        separations = [
            abs(_wrap(bearing - previous)) for previous in previous_bearings
        ]
        baseline = min(1.0, min(separations) / (pi / 4.0))
    weight_ambiguous = 0.35 if need.need_type == 'ambiguous_target' else 0.2
    return _unit(
        (1.0 - weight_ambiguous) * (0.6 * quality + 0.4 * centring)
        + weight_ambiguous * baseline
    )


def anchor_visibility_score(
    candidate: ViewpointCandidate,
    *,
    grid: OccupancyGrid2D,
    anchor_targets: tuple[tuple[float, float], ...],
    config: ViewpointScoringConfig,
) -> float:
    """Score expected visibility of any still-missing anchor region."""
    if not anchor_targets:
        return 0.0
    pose = candidate.pose_xy_yaw
    best = 0.0
    for anchor_xy in anchor_targets:
        if not grid.line_of_sight((pose[0], pose[1]), anchor_xy):
            continue
        distance = hypot(anchor_xy[0] - pose[0], anchor_xy[1] - pose[1])
        bearing = atan2(anchor_xy[1] - pose[1], anchor_xy[0] - pose[0])
        centring = max(
            0.0,
            1.0 - abs(_wrap(bearing - pose[2])) / config.horizontal_fov_rad,
        )
        best = max(
            best, 0.65 * distance_quality(distance, config) + 0.35 * centring
        )
    return _unit(best)


def unexplored_gain_score(
    candidate: ViewpointCandidate,
    *,
    grid: OccupancyGrid2D,
    config: ViewpointScoringConfig,
) -> float:
    """Score how much unknown area a bounded sensor cone would reveal."""
    revealed = grid.count_visible_unknown(
        candidate.pose_xy_yaw,
        horizontal_fov=config.horizontal_fov_rad,
        max_range=config.sensor_range_m,
    )
    return _unit(revealed / config.unexplored_normalization_cells)


def ambiguity_reduction_score(
    candidate: ViewpointCandidate,
    *,
    grid: OccupancyGrid2D,
    hypothesis_xy: tuple[tuple[float, float], ...],
    config: ViewpointScoringConfig,
) -> float:
    """
    Score how well a pose separates the top competing hypotheses.

    Larger angular separation makes the two candidates easier to detect and
    associate independently, so bearing spread is the primary signal.
    """
    if len(hypothesis_xy) < 2:
        return 0.0
    pose = candidate.pose_xy_yaw
    first, second = hypothesis_xy[0], hypothesis_xy[1]
    visible = sum(
        grid.line_of_sight((pose[0], pose[1]), item)
        for item in (first, second)
    )
    bearing_first = atan2(first[1] - pose[1], first[0] - pose[0])
    bearing_second = atan2(second[1] - pose[1], second[0] - pose[0])
    separation = abs(_wrap(bearing_first - bearing_second))
    spread = min(1.0, separation / (pi / 3.0))
    distance_first = hypot(first[0] - pose[0], first[1] - pose[1])
    distance_second = hypot(second[0] - pose[0], second[1] - pose[1])
    contrast = min(
        1.0, abs(distance_first - distance_second) / max(
            1e-6, config.preferred_observation_distance_m
        )
    )
    coverage = visible / 2.0
    return _unit(0.6 * spread + 0.2 * contrast + 0.2 * coverage)


def support_visibility_score(
    candidate: ViewpointCandidate,
    *,
    grid: OccupancyGrid2D,
    support_xy: tuple[float, float] | None,
    support_yaw: float = 0.0,
    config: ViewpointScoringConfig,
) -> float:
    """
    Score how much of a support surface a pose would expose.

    A frontal or oblique view of an unobstructed surface at a sensible
    distance outranks a rear or blocked view, because the robot base cannot
    raise the camera to see over the near edge.
    """
    if support_xy is None:
        return 0.0
    pose = candidate.pose_xy_yaw
    if not grid.line_of_sight((pose[0], pose[1]), support_xy):
        return 0.0
    distance = hypot(support_xy[0] - pose[0], support_xy[1] - pose[1])
    quality = distance_quality(distance, config)
    if quality <= 0.0:
        return 0.0
    approach = atan2(pose[1] - support_xy[1], pose[0] - support_xy[0])
    # Prefer viewing along a face normal rather than a corner diagonal.
    relative = abs(_wrap(approach - support_yaw)) % (pi / 2.0)
    face_alignment = 1.0 - abs(relative - pi / 4.0) / (pi / 4.0)
    frontality = 0.6 + 0.4 * (1.0 - face_alignment)
    return _unit(0.7 * quality + 0.3 * frontality)


def travel_cost_score(
    candidate: ViewpointCandidate,
    config: ViewpointScoringConfig,
) -> float:
    """Normalize planned path length into a bounded penalty."""
    return _unit(candidate.travel_cost_m / config.travel_normalization_m)


def score_candidate(
    candidate: ViewpointCandidate,
    *,
    grid: OccupancyGrid2D,
    need: ExplorationNeed,
    config: ViewpointScoringConfig | None = None,
    target_xy: tuple[float, float] | None = None,
    anchor_targets: tuple[tuple[float, float], ...] = (),
    hypothesis_xy: tuple[tuple[float, float], ...] = (),
    support_xy: tuple[float, float] | None = None,
    support_yaw: float = 0.0,
    previous_bearings: tuple[float, ...] = (),
    traversal_risk: float = 0.0,
) -> ViewpointCandidate:
    """Return the candidate with every term and its final score attached."""
    policy = config or ViewpointScoringConfig()
    terms = ViewpointScoreTerms(
        target_visibility=target_visibility_score(
            candidate,
            grid=grid,
            need=need,
            target_xy=target_xy,
            config=policy,
            previous_bearings=previous_bearings,
        ),
        anchor_visibility=anchor_visibility_score(
            candidate,
            grid=grid,
            anchor_targets=anchor_targets,
            config=policy,
        ),
        unexplored_gain=unexplored_gain_score(
            candidate, grid=grid, config=policy
        ),
        ambiguity_reduction=ambiguity_reduction_score(
            candidate,
            grid=grid,
            hypothesis_xy=hypothesis_xy,
            config=policy,
        ),
        support_visibility=support_visibility_score(
            candidate,
            grid=grid,
            support_xy=support_xy,
            support_yaw=support_yaw,
            config=policy,
        ),
        travel_cost=travel_cost_score(candidate, policy),
        traversal_risk=_unit(traversal_risk),
    )
    gain = (
        policy.target_weight * terms.target_visibility
        + policy.anchor_weight * terms.anchor_visibility
        + policy.unexplored_weight * terms.unexplored_gain
        + policy.ambiguity_weight * terms.ambiguity_reduction
        + policy.support_weight * terms.support_visibility
    )
    score = (
        gain
        - policy.travel_weight * terms.travel_cost
        - policy.risk_weight * terms.traversal_risk
    )
    return ViewpointCandidate(
        viewpoint_id=candidate.viewpoint_id,
        pose_xy_yaw=candidate.pose_xy_yaw,
        source=candidate.source,
        target_instance_ids=candidate.target_instance_ids,
        target_regions=candidate.target_regions,
        score_terms=terms,
        expected_information_gain=gain,
        travel_cost_m=candidate.travel_cost_m,
        score=score,
        reasons=candidate.reasons,
    )


def select_viewpoint(
    candidates: tuple[ViewpointCandidate, ...],
    need: ExplorationNeed,
    *,
    config: ViewpointScoringConfig | None = None,
    budget_status: str = 'available',
    confidence_margin: float = 0.0,
) -> ViewpointSelection:
    """
    Rank scored candidates and explain the outcome, including refusals.

    A selection is always produced: refusing to move is a decision that must
    be as traceable as moving.
    """
    policy = config or ViewpointScoringConfig()
    ranked = tuple(
        sorted(
            candidates,
            key=lambda item: (-item.score, item.viewpoint_id),
        )
    )
    if budget_status != 'available':
        status = (
            budget_status
            if budget_status in SELECTION_STATUSES
            else 'budget_exhausted'
        )
        return ViewpointSelection(
            ranked_candidates=ranked,
            selected_viewpoint_id=None,
            selection_status=status,
            unresolved_need=need,
            confidence_margin=confidence_margin,
        )
    if not ranked:
        return ViewpointSelection(
            ranked_candidates=(),
            selected_viewpoint_id=None,
            selection_status='no_reachable_viewpoint',
            unresolved_need=need,
            confidence_margin=confidence_margin,
        )
    best = ranked[0]
    if best.score < policy.minimum_selectable_score:
        return ViewpointSelection(
            ranked_candidates=ranked,
            selected_viewpoint_id=None,
            selection_status='gain_too_low',
            unresolved_need=need,
            confidence_margin=confidence_margin,
        )
    return ViewpointSelection(
        ranked_candidates=ranked,
        selected_viewpoint_id=best.viewpoint_id,
        selection_status='selected',
        unresolved_need=need,
        expected_gain=best.expected_information_gain,
        confidence_margin=confidence_margin,
    )


def _wrap(angle: float) -> float:
    return atan2(sin(angle), cos(angle))


def _unit(value: float) -> float:
    if not isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


__all__ = [
    'ViewpointScoringConfig',
    'ambiguity_reduction_score',
    'anchor_visibility_score',
    'distance_quality',
    'score_candidate',
    'select_viewpoint',
    'support_visibility_score',
    'target_visibility_score',
    'travel_cost_score',
    'unexplored_gain_score',
]
