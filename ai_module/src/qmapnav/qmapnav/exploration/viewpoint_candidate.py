"""Viewpoint candidate, score decomposition, and ranked selection result."""

from dataclasses import dataclass
from dataclasses import field
from math import isfinite

from qmapnav.exploration.exploration_need import ExplorationNeed


VIEWPOINT_SOURCES = frozenset(
    {
        'frontier',
        'object_annulus',
        'occluder_offset',
        'route_adjacent',
        'support_surface',
    }
)

SELECTION_STATUSES = frozenset(
    {
        'budget_exhausted',
        'gain_too_low',
        'no_reachable_viewpoint',
        'route_first_required',
        'selected',
        'time_budget_exhausted',
    }
)


def _require_unit(name: str, value: float) -> float:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f'{name} must be finite and in [0, 1]')
    return float(value)


def _require_non_negative(name: str, value: float) -> float:
    if not isfinite(value) or value < 0.0:
        raise ValueError(f'{name} must be finite and non-negative')
    return float(value)


@dataclass(frozen=True)
class ViewpointScoreTerms:
    """
    The six requested scoring terms plus an explicit risk penalty.

    Every term is retained separately so a trace reader can see why one
    viewpoint outranked another instead of only seeing a final scalar.
    """

    target_visibility: float = 0.0
    anchor_visibility: float = 0.0
    unexplored_gain: float = 0.0
    ambiguity_reduction: float = 0.0
    support_visibility: float = 0.0
    travel_cost: float = 0.0
    traversal_risk: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            'target_visibility',
            'anchor_visibility',
            'unexplored_gain',
            'ambiguity_reduction',
            'support_visibility',
            'travel_cost',
            'traversal_risk',
        ):
            object.__setattr__(
                self, name, _require_unit(name, getattr(self, name))
            )

    def to_dict(self) -> dict[str, float]:
        """Return a stable trace-ready mapping of every term."""
        return {
            'target_visibility': self.target_visibility,
            'anchor_visibility': self.anchor_visibility,
            'unexplored_gain': self.unexplored_gain,
            'ambiguity_reduction': self.ambiguity_reduction,
            'support_visibility': self.support_visibility,
            'travel_cost': self.travel_cost,
            'traversal_risk': self.traversal_risk,
        }


@dataclass(frozen=True)
class ViewpointCandidate:
    """One safe map-frame pose that may reveal a specific missing evidence."""

    viewpoint_id: str
    pose_xy_yaw: tuple[float, float, float]
    source: str
    target_instance_ids: tuple[str, ...] = ()
    target_regions: tuple[str, ...] = ()
    score_terms: ViewpointScoreTerms = field(
        default_factory=ViewpointScoreTerms
    )
    expected_information_gain: float = 0.0
    travel_cost_m: float = 0.0
    score: float = 0.0
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.viewpoint_id, str) or not (
            self.viewpoint_id.strip()
        ):
            raise ValueError('viewpoint_id must be a non-empty string')
        if self.source not in VIEWPOINT_SOURCES:
            expected = ', '.join(sorted(VIEWPOINT_SOURCES))
            raise ValueError(f'source must be one of: {expected}')
        pose = tuple(self.pose_xy_yaw)
        if len(pose) != 3 or not all(isfinite(value) for value in pose):
            raise ValueError('pose_xy_yaw must hold three finite values')
        object.__setattr__(
            self, 'pose_xy_yaw', tuple(float(value) for value in pose)
        )
        if not isinstance(self.score_terms, ViewpointScoreTerms):
            raise TypeError('score_terms must be ViewpointScoreTerms')
        object.__setattr__(
            self,
            'target_instance_ids',
            tuple(self.target_instance_ids),
        )
        object.__setattr__(
            self, 'target_regions', tuple(self.target_regions)
        )
        object.__setattr__(self, 'reasons', tuple(self.reasons))
        object.__setattr__(
            self,
            'expected_information_gain',
            _require_non_negative(
                'expected_information_gain', self.expected_information_gain
            ),
        )
        object.__setattr__(
            self,
            'travel_cost_m',
            _require_non_negative('travel_cost_m', self.travel_cost_m),
        )
        if not isfinite(self.score):
            raise ValueError('score must be finite')
        object.__setattr__(self, 'score', float(self.score))

    @property
    def traversal_risk(self) -> float:
        """Return the risk penalty carried by this candidate's terms."""
        return self.score_terms.traversal_risk

    def to_dict(self) -> dict[str, object]:
        """Return a stable trace-ready mapping."""
        return {
            'viewpoint_id': self.viewpoint_id,
            'pose_xy_yaw': list(self.pose_xy_yaw),
            'source': self.source,
            'target_instance_ids': list(self.target_instance_ids),
            'target_regions': list(self.target_regions),
            'score_terms': self.score_terms.to_dict(),
            'expected_information_gain': self.expected_information_gain,
            'travel_cost_m': self.travel_cost_m,
            'score': self.score,
            'reasons': list(self.reasons),
        }


@dataclass(frozen=True)
class ViewpointSelection:
    """
    Ranked candidates plus the explicit reason a pose was or was not used.

    A selection is returned even when nothing is chosen so that a refusal to
    move is as traceable as a decision to move.
    """

    ranked_candidates: tuple[ViewpointCandidate, ...]
    selected_viewpoint_id: str | None
    selection_status: str
    unresolved_need: ExplorationNeed
    expected_gain: float = 0.0
    confidence_margin: float = 0.0

    def __post_init__(self) -> None:
        if self.selection_status not in SELECTION_STATUSES:
            expected = ', '.join(sorted(SELECTION_STATUSES))
            raise ValueError(f'selection_status must be one of: {expected}')
        if not isinstance(self.unresolved_need, ExplorationNeed):
            raise TypeError('unresolved_need must be ExplorationNeed')
        candidates = tuple(self.ranked_candidates)
        if not all(
            isinstance(item, ViewpointCandidate) for item in candidates
        ):
            raise TypeError('ranked_candidates must hold ViewpointCandidate')
        identifiers = [item.viewpoint_id for item in candidates]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError('viewpoint_id values must be unique')
        object.__setattr__(self, 'ranked_candidates', candidates)
        selected = self.selected_viewpoint_id
        if (selected is not None) != (self.selection_status == 'selected'):
            raise ValueError(
                'selected_viewpoint_id must be set exactly when status is '
                'selected'
            )
        if selected is not None and selected not in identifiers:
            raise ValueError('selected_viewpoint_id must name a candidate')
        object.__setattr__(
            self,
            'expected_gain',
            _require_non_negative('expected_gain', self.expected_gain),
        )
        if not isfinite(self.confidence_margin):
            raise ValueError('confidence_margin must be finite')
        object.__setattr__(
            self, 'confidence_margin', float(self.confidence_margin)
        )

    @property
    def selected(self) -> ViewpointCandidate | None:
        """Return the selected candidate, or ``None`` when none was safe."""
        for candidate in self.ranked_candidates:
            if candidate.viewpoint_id == self.selected_viewpoint_id:
                return candidate
        return None

    def to_dict(self) -> dict[str, object]:
        """Return a stable trace-ready mapping."""
        return {
            'ranked_candidates': [
                item.to_dict() for item in self.ranked_candidates
            ],
            'selected_viewpoint_id': self.selected_viewpoint_id,
            'selection_status': self.selection_status,
            'expected_gain': self.expected_gain,
            'confidence_margin': self.confidence_margin,
            'unresolved_need': self.unresolved_need.to_dict(),
        }


__all__ = [
    'SELECTION_STATUSES',
    'VIEWPOINT_SOURCES',
    'ViewpointCandidate',
    'ViewpointScoreTerms',
    'ViewpointSelection',
]
