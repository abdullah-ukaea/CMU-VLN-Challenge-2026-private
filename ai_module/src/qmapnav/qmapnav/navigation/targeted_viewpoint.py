"""One-shot heuristic re-observation for Day 10 object references."""

from dataclasses import dataclass
from math import atan2
from math import cos
from math import isfinite
from math import sin
from typing import Callable

import numpy as np


SafePosePredicate = Callable[[float, float], bool]


@dataclass(frozen=True)
class TargetedViewpointConfig:
    """Bounded trigger, sampling, and utility parameters."""

    minimum_confidence_margin: float = 0.12
    minimum_geometry_confidence: float = 0.35
    minimum_time_remaining_sec: float = 45.0
    preferred_standoff_m: float = 2.0
    minimum_standoff_m: float = 1.5
    maximum_standoff_m: float = 2.5
    lateral_offset_m: float = 0.8
    maximum_travel_m: float = 4.0
    target_weight: float = 1.0
    anchor_weight: float = 0.8
    geometry_weight: float = 0.8
    disambiguation_weight: float = 0.9
    travel_weight: float = 0.25
    risk_weight: float = 1.0

    def __post_init__(self) -> None:
        bounded = (
            self.minimum_confidence_margin,
            self.minimum_geometry_confidence,
        )
        if any(not isfinite(item) or not 0.0 <= item <= 1.0
               for item in bounded):
            raise ValueError('confidence thresholds must lie in [0, 1]')
        positive = (
            self.minimum_time_remaining_sec,
            self.preferred_standoff_m,
            self.minimum_standoff_m,
            self.maximum_standoff_m,
            self.lateral_offset_m,
            self.maximum_travel_m,
            self.target_weight,
            self.anchor_weight,
            self.geometry_weight,
            self.disambiguation_weight,
            self.travel_weight,
            self.risk_weight,
        )
        if any(not isfinite(item) or item <= 0.0 for item in positive):
            raise ValueError('viewpoint distances and weights must be positive')
        if not (
            self.minimum_standoff_m
            <= self.preferred_standoff_m
            <= self.maximum_standoff_m
        ):
            raise ValueError('preferred standoff must lie inside its bounds')


@dataclass(frozen=True)
class EvidenceSufficiency:
    """Compact initial evidence used for the one-shot trigger."""

    reliable_target_candidates: int
    missing_anchor_ids: tuple[str, ...]
    confidence_margin: float | None
    geometry_confidence: float | None
    likely_occluded: bool
    time_remaining_sec: float
    viewpoint_attempts: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.reliable_target_candidates, bool)
            or self.reliable_target_candidates < 0
        ):
            raise ValueError('target candidate count must be non-negative')
        if isinstance(self.viewpoint_attempts, bool) or not (
            0 <= self.viewpoint_attempts <= 1
        ):
            raise ValueError('viewpoint attempts must be zero or one')
        if not isfinite(self.time_remaining_sec) or self.time_remaining_sec < 0:
            raise ValueError('time_remaining_sec must be non-negative')
        for name in ('confidence_margin', 'geometry_confidence'):
            value = getattr(self, name)
            if value is not None and (
                not isfinite(value) or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f'{name} must lie in [0, 1]')
        if not isinstance(self.likely_occluded, bool):
            raise TypeError('likely_occluded must be bool')


@dataclass(frozen=True)
class ViewpointTriggerDecision:
    """Deterministic request/no-request with one primary reason."""

    requested: bool
    reason: str


@dataclass(frozen=True)
class TargetedViewpointCandidate:
    """One safe map-frame viewpoint with transparent utility terms."""

    pose_xy_heading: tuple[float, float, float]
    target_visibility: float
    anchor_visibility: float
    geometry_gain: float
    disambiguation: float
    travel_cost_m: float
    risk: float
    utility: float
    reason: str


def decide_targeted_viewpoint(
    evidence: EvidenceSufficiency,
    config: TargetedViewpointConfig | None = None,
) -> ViewpointTriggerDecision:
    """Request at most one view only when initial evidence is inadequate."""
    if not isinstance(evidence, EvidenceSufficiency):
        raise TypeError('evidence must be EvidenceSufficiency')
    policy = config or TargetedViewpointConfig()
    if evidence.viewpoint_attempts >= 1:
        return ViewpointTriggerDecision(False, 'viewpoint_limit_reached')
    if evidence.time_remaining_sec < policy.minimum_time_remaining_sec:
        return ViewpointTriggerDecision(False, 'insufficient_time_reserve')
    if evidence.reliable_target_candidates == 0:
        return ViewpointTriggerDecision(True, 'no_reliable_target')
    if evidence.missing_anchor_ids:
        return ViewpointTriggerDecision(True, 'required_anchor_missing')
    if (
        evidence.confidence_margin is not None
        and evidence.confidence_margin < policy.minimum_confidence_margin
    ):
        return ViewpointTriggerDecision(True, 'low_ranking_margin')
    if (
        evidence.geometry_confidence is not None
        and evidence.geometry_confidence
        < policy.minimum_geometry_confidence
    ):
        return ViewpointTriggerDecision(True, 'weak_geometry')
    if evidence.likely_occluded:
        return ViewpointTriggerDecision(True, 'likely_occlusion')
    return ViewpointTriggerDecision(False, 'initial_evidence_sufficient')


def generate_targeted_viewpoints(
    current_pose_xy_heading: tuple[float, float, float],
    focus_xy: tuple[float, float],
    *,
    anchor_missing: bool,
    ambiguous: bool,
    safe_pose: SafePosePredicate,
    config: TargetedViewpointConfig | None = None,
) -> tuple[TargetedViewpointCandidate, ...]:
    """Sample a small closer/lateral set and reject unsafe travel."""
    policy = config or TargetedViewpointConfig()
    current = np.asarray(current_pose_xy_heading[:2], dtype=np.float64)
    focus = np.asarray(focus_xy, dtype=np.float64)
    if current.shape != (2,) or focus.shape != (2,):
        raise ValueError('current pose and focus must contain finite XY')
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(focus)):
        raise ValueError('current pose and focus must be finite')
    if not callable(safe_pose):
        raise TypeError('safe_pose must be callable')
    connector = current - focus
    distance = float(np.linalg.norm(connector))
    if distance < 1.0e-6:
        connector = np.array((1.0, 0.0), dtype=np.float64)
        distance = 1.0
    radial = connector / distance
    lateral = np.array((-radial[1], radial[0]), dtype=np.float64)
    standoff = min(
        policy.maximum_standoff_m,
        max(policy.minimum_standoff_m, policy.preferred_standoff_m),
    )
    sampled = (
        ('closer', focus + radial * standoff),
        ('lateral_left', focus + radial * standoff
         + lateral * policy.lateral_offset_m),
        ('lateral_right', focus + radial * standoff
         - lateral * policy.lateral_offset_m),
    )
    output = []
    for reason, point in sampled:
        travel = float(np.linalg.norm(point - current))
        if travel > policy.maximum_travel_m:
            continue
        if not safe_pose(float(point[0]), float(point[1])):
            continue
        heading = atan2(focus[1] - point[1], focus[0] - point[0])
        bearing_change = abs(_angle_delta(
            heading, float(current_pose_xy_heading[2])
        )) / np.pi
        target_visibility = min(1.0, 0.55 + 0.25 * bearing_change)
        anchor_visibility = min(
            1.0, 0.45 + (0.35 if anchor_missing else 0.10)
        )
        geometry_gain = 0.80 if reason == 'closer' else 0.65
        disambiguation = (
            0.85 if ambiguous and reason.startswith('lateral') else 0.35
        )
        risk = 0.0
        utility = (
            policy.target_weight * target_visibility
            + policy.anchor_weight * anchor_visibility
            + policy.geometry_weight * geometry_gain
            + policy.disambiguation_weight * disambiguation
            - policy.travel_weight * travel
            - policy.risk_weight * risk
        )
        output.append(TargetedViewpointCandidate(
            (float(point[0]), float(point[1]), float(heading)),
            target_visibility,
            anchor_visibility,
            geometry_gain,
            disambiguation,
            travel,
            risk,
            utility,
            reason,
        ))
    return tuple(sorted(
        output,
        key=lambda item: (-item.utility, item.reason, item.pose_xy_heading),
    ))


class OneViewpointGuard:
    """Allow a single special viewpoint decision per episode."""

    def __init__(self) -> None:
        self._attempted = False
        self._selected: TargetedViewpointCandidate | None = None

    @property
    def attempted(self) -> bool:
        """Return whether the one allowed request was consumed."""
        return self._attempted

    @property
    def selected(self) -> TargetedViewpointCandidate | None:
        """Return the immutable selected viewpoint, if one was safe."""
        return self._selected

    def select(
        self,
        candidates: tuple[TargetedViewpointCandidate, ...],
    ) -> TargetedViewpointCandidate | None:
        """Consume the one attempt and select the highest utility candidate."""
        if self._attempted:
            raise RuntimeError('targeted viewpoint already attempted')
        self._attempted = True
        self._selected = candidates[0] if candidates else None
        return self._selected


def _angle_delta(first: float, second: float) -> float:
    return atan2(sin(first - second), cos(first - second))


__all__ = [
    'EvidenceSufficiency',
    'OneViewpointGuard',
    'SafePosePredicate',
    'TargetedViewpointCandidate',
    'TargetedViewpointConfig',
    'ViewpointTriggerDecision',
    'decide_targeted_viewpoint',
    'generate_targeted_viewpoints',
]
