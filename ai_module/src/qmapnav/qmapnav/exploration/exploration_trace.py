"""Serializable exploration and observation decision-trace events."""

from dataclasses import dataclass
from dataclasses import field
from math import isfinite

from qmapnav.exploration.viewpoint_candidate import ViewpointSelection


EXPLORATION_TRACE_SCHEMA_VERSION = '1.0'


def _require_non_negative(name: str, value: float) -> float:
    if not isfinite(value) or value < 0.0:
        raise ValueError(f'{name} must be finite and non-negative')
    return float(value)


def _require_non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a non-negative integer')
    return value


@dataclass(frozen=True)
class ViewpointSelectionEvent:
    """Why one viewpoint was preferred over every other candidate."""

    selection: ViewpointSelection
    candidates_considered: int
    rejected_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.selection, ViewpointSelection):
            raise TypeError('selection must be ViewpointSelection')
        _require_non_negative_int(
            'candidates_considered', self.candidates_considered
        )
        rejected = dict(self.rejected_counts)
        for key, value in rejected.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError('rejected_counts keys must be non-empty')
            _require_non_negative_int(f'rejected_counts[{key}]', value)
        object.__setattr__(self, 'rejected_counts', rejected)

    def to_dict(self) -> dict[str, object]:
        """Return the stable ``viewpoint_selection`` trace record."""
        need = self.selection.unresolved_need
        selected = self.selection.selected
        record: dict[str, object] = {
            'event': 'viewpoint_selection',
            'schema_version': EXPLORATION_TRACE_SCHEMA_VERSION,
            'need_type': need.need_type,
            'target_reference_id': need.target_reference_id,
            'missing_classes': list(need.missing_classes),
            'candidates_considered': self.candidates_considered,
            'rejected_counts': dict(sorted(self.rejected_counts.items())),
            'selection_status': self.selection.selection_status,
            'selected_viewpoint': self.selection.selected_viewpoint_id,
            'expected_gain': self.selection.expected_gain,
            'confidence_margin': self.selection.confidence_margin,
            'ranked_candidates': [
                item.to_dict() for item in self.selection.ranked_candidates
            ],
        }
        if selected is not None:
            record['score_terms'] = selected.score_terms.to_dict()
            record['final_score'] = selected.score
            record['reason'] = '; '.join(selected.reasons)
        else:
            record['score_terms'] = None
            record['final_score'] = None
            record['reason'] = need.reason
        return record


@dataclass(frozen=True)
class ViewpointOutcomeEvent:
    """What one executed viewpoint actually recovered."""

    viewpoint_id: str
    new_object_ids: tuple[str, ...] = ()
    updated_object_ids: tuple[str, ...] = ()
    target_found: bool = False
    scan_points_added: int = 0
    observation_duration_sec: float = 0.0
    travel_distance_m: float = 0.0
    margin_before: float | None = None
    margin_after: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.viewpoint_id, str) or not (
            self.viewpoint_id.strip()
        ):
            raise ValueError('viewpoint_id must be a non-empty string')
        if not isinstance(self.target_found, bool):
            raise TypeError('target_found must be bool')
        object.__setattr__(
            self, 'new_object_ids', tuple(self.new_object_ids)
        )
        object.__setattr__(
            self, 'updated_object_ids', tuple(self.updated_object_ids)
        )
        _require_non_negative_int('scan_points_added', self.scan_points_added)
        object.__setattr__(
            self,
            'observation_duration_sec',
            _require_non_negative(
                'observation_duration_sec', self.observation_duration_sec
            ),
        )
        object.__setattr__(
            self,
            'travel_distance_m',
            _require_non_negative(
                'travel_distance_m', self.travel_distance_m
            ),
        )
        for name in ('margin_before', 'margin_after'):
            value = getattr(self, name)
            if value is not None and not isfinite(value):
                raise ValueError(f'{name} must be finite or None')

    @property
    def margin_improvement(self) -> float | None:
        """Return the ranking-margin gain, when both margins are known."""
        if self.margin_before is None or self.margin_after is None:
            return None
        return self.margin_after - self.margin_before

    @property
    def useful(self) -> bool:
        """Return whether this viewpoint materially advanced the query."""
        improvement = self.margin_improvement
        return bool(
            self.target_found
            or self.new_object_ids
            or (improvement is not None and improvement > 0.0)
        )

    def to_dict(self) -> dict[str, object]:
        """Return the stable ``viewpoint_outcome`` trace record."""
        return {
            'event': 'viewpoint_outcome',
            'schema_version': EXPLORATION_TRACE_SCHEMA_VERSION,
            'viewpoint': self.viewpoint_id,
            'new_objects': list(self.new_object_ids),
            'updated_objects': list(self.updated_object_ids),
            'target_found': self.target_found,
            'scan_points_added': self.scan_points_added,
            'observation_duration_sec': self.observation_duration_sec,
            'travel_distance_m': self.travel_distance_m,
            'margin_before': self.margin_before,
            'margin_after': self.margin_after,
            'margin_improvement': self.margin_improvement,
            'useful': self.useful,
        }


__all__ = [
    'EXPLORATION_TRACE_SCHEMA_VERSION',
    'ViewpointOutcomeEvent',
    'ViewpointSelectionEvent',
]
