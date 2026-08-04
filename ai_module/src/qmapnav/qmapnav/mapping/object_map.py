"""Bounded episode-level object identity and evidence fusion."""

from dataclasses import dataclass
from dataclasses import field
from math import atan2
from math import cos
from math import exp
from math import isfinite
from math import sin
from threading import RLock
from types import MappingProxyType
from typing import Mapping

import numpy as np

from qmapnav.common import ObjectInstance
from qmapnav.mapping.bounding_boxes import estimate_upright_obb
from qmapnav.mapping.bounding_boxes import robust_aabb
from qmapnav.mapping.geometry_evaluation import aabb_iou_3d
from qmapnav.mapping.object_association import AssociationConfig
from qmapnav.mapping.object_association import AssociationDecision
from qmapnav.mapping.object_association import AssociationScore
from qmapnav.mapping.object_association import canonicalize_class_name
from qmapnav.mapping.object_association import class_compatibility
from qmapnav.mapping.object_association import score_candidate_instance
from qmapnav.mapping.object_candidate import GeometryStatus
from qmapnav.mapping.object_candidate import ObjectCandidate3D
from qmapnav.mapping.object_candidate import readonly_array
from qmapnav.mapping.viewpoint_observation import ViewpointObservation
from qmapnav.reasoning.colour_types import ColourEstimate


INSTANCE_STATUSES = frozenset({
    'active',
    'partially_observed',
    'possible_duplicate',
    'sparse',
    'uncertain',
})


@dataclass(frozen=True)
class ObjectMapConfig:
    """Association, fusion, history, and hard memory bounds."""

    association: AssociationConfig = field(default_factory=AssociationConfig)
    same_keyframe_distance_m: float = 0.30
    same_keyframe_overlap_threshold: float = 0.60
    fused_voxel_size_m: float = 0.03
    max_fused_points_per_instance: int = 50_000
    max_total_fused_points: int = 500_000
    max_observation_history: int = 100
    max_colour_history: int = 32
    max_colour_evidence: float = 12.0
    max_instances: int = 512
    minimum_refit_points: int = 8
    minimum_new_refit_points: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.association, AssociationConfig):
            raise TypeError('association must be AssociationConfig')
        for name in (
            'same_keyframe_distance_m',
            'same_keyframe_overlap_threshold',
            'fused_voxel_size_m',
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        if self.same_keyframe_overlap_threshold > 1.0:
            raise ValueError('same-keyframe overlap must not exceed one')
        for name in (
            'max_fused_points_per_instance',
            'max_total_fused_points',
            'max_observation_history',
            'max_colour_history',
            'max_instances',
            'minimum_refit_points',
            'minimum_new_refit_points',
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if not isfinite(self.max_colour_evidence) or (
            self.max_colour_evidence <= 0.0
        ):
            raise ValueError('max_colour_evidence must be finite and positive')
        if self.max_total_fused_points < self.max_fused_points_per_instance:
            raise ValueError('total point cap must cover one per-instance cap')


@dataclass(frozen=True)
class PersistentObjectRecord:
    """Rich read-only persistent state around the frozen shared contract."""

    instance: ObjectInstance
    canonical_class: str
    geometry_confidence: float
    first_seen_ns: int
    last_seen_ns: int
    source_viewpoint_ids: tuple[str, ...]
    source_detection_ids: tuple[str, ...]
    best_crop: np.ndarray | None
    best_crop_score: float
    status: str
    fused_points_xyz: np.ndarray
    observations: tuple[ViewpointObservation, ...]
    best_view_candidate_id: str
    colour_evidence: Mapping[str, float] = field(default_factory=dict)
    colour_confidence: float = 0.0
    best_colour_estimate: ColourEstimate | None = None
    colour_estimates: tuple[ColourEstimate, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in INSTANCE_STATUSES:
            raise ValueError('unsupported persistent object status')
        if not isfinite(self.geometry_confidence) or not (
            0.0 <= self.geometry_confidence <= 1.0
        ):
            raise ValueError('geometry_confidence must lie in [0, 1]')
        object.__setattr__(
            self,
            'fused_points_xyz',
            readonly_array('fused_points_xyz', self.fused_points_xyz, (None, 3)),
        )
        if self.best_crop is not None:
            crop = np.ascontiguousarray(self.best_crop).copy()
            crop.setflags(write=False)
            object.__setattr__(self, 'best_crop', crop)
        if not isfinite(self.colour_confidence) or not (
            0.0 <= self.colour_confidence <= 1.0
        ):
            raise ValueError('colour_confidence must lie in [0, 1]')
        object.__setattr__(
            self,
            'colour_evidence',
            MappingProxyType(dict(self.colour_evidence)),
        )


@dataclass(frozen=True)
class ObjectAssociationEvent:
    """Trace-ready outcome of one ObjectMap input candidate."""

    candidate_id: str
    candidate_class: str
    matched_instance_id: int
    components: Mapping[str, float | None]
    final_score: float
    decision: str
    reason: str
    observation_count_after: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, 'components', MappingProxyType(dict(self.components))
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe association trace payload."""
        return {
            'event': 'object_association',
            'candidate_id': self.candidate_id,
            'candidate_class': self.candidate_class,
            'matched_instance_id': self.matched_instance_id,
            'scores': dict(self.components),
            'final_score': self.final_score,
            'decision': self.decision,
            'reason': self.reason,
            'observation_count_after': self.observation_count_after,
        }


@dataclass
class _InstanceState:
    instance: ObjectInstance
    class_evidence: dict[str, float]
    centre_weight: float
    geometry_confidence: float
    first_seen_ns: int
    last_seen_ns: int
    source_viewpoint_ids: list[str]
    source_detection_ids: list[str]
    best_crop: np.ndarray | None
    best_crop_score: float
    status: str
    fused_points_xyz: np.ndarray
    observations: list[ViewpointObservation]
    best_view_candidate: ObjectCandidate3D
    colour_evidence: dict[str, float]
    colour_confidence: float
    best_colour_estimate: ColourEstimate | None
    colour_estimates: list[ColourEstimate]
    colour_view_weights: dict[str, float]


class ObjectMap:
    """Assign stable IDs and conservatively fuse one challenge episode."""

    def __init__(self, config: ObjectMapConfig | None = None) -> None:
        self.config = config or ObjectMapConfig()
        self._states: dict[int, _InstanceState] = {}
        self._next_instance_id = 0
        self._lock = RLock()
        self._last_events: tuple[ObjectAssociationEvent, ...] = ()

    @property
    def next_instance_id(self) -> int:
        """Return the deterministic ID that the next new instance receives."""
        with self._lock:
            return self._next_instance_id

    @property
    def last_events(self) -> tuple[ObjectAssociationEvent, ...]:
        """Return trace records from the most recent update call."""
        with self._lock:
            return self._last_events

    @property
    def fused_point_count(self) -> int:
        """Return total retained ObjectMap point evidence."""
        with self._lock:
            return sum(
                state.fused_points_xyz.shape[0]
                for state in self._states.values()
            )

    def add_or_update(
        self,
        candidate: ObjectCandidate3D,
        observation: ViewpointObservation,
    ) -> int:
        """Return the persistent episode-level instance ID."""
        self._validate_pair(candidate, observation)
        with self._lock:
            scores = self._scores(candidate)
            best = scores[0] if scores else None
            if best is not None and best.decision is AssociationDecision.MERGE:
                instance_id = best.instance_id
                self._merge(instance_id, candidate, observation)
                event = self._event(
                    candidate, instance_id, best, 'accepted_best_score'
                )
            else:
                status = (
                    'possible_duplicate'
                    if best is not None
                    and best.decision is AssociationDecision.UNCERTAIN
                    else self._candidate_status(candidate, observation)
                )
                instance_id = self._create(candidate, observation, status)
                reason = (
                    'uncertain_match_preserved_separately'
                    if status == 'possible_duplicate'
                    else 'no_accepted_existing_instance'
                )
                event = self._event(candidate, instance_id, best, reason)
            self._last_events = (event,)
            self._enforce_total_bound()
            return instance_id

    def add_viewpoint_candidates(
        self,
        candidates: list[ObjectCandidate3D],
        observations: list[ViewpointObservation],
    ) -> list[int]:
        """Perform duplicate suppression and one-to-one viewpoint matching."""
        if len(candidates) != len(observations):
            raise ValueError('candidates and observations must have equal length')
        for candidate, observation in zip(candidates, observations):
            self._validate_pair(candidate, observation)
        if not candidates:
            with self._lock:
                self._last_events = ()
            return []
        with self._lock:
            representatives, aliases = self._same_keyframe_groups(candidates)
            score_table = []
            for representative in representatives:
                for score in self._scores(candidates[representative]):
                    if score.decision is AssociationDecision.MERGE:
                        score_table.append((score.final_score, representative, score))
            assigned_candidates = set()
            assigned_instances = set()
            result_by_representative = {}
            score_by_representative = {}
            for _, candidate_index, score in sorted(
                score_table,
                key=lambda item: (-item[0], item[1], item[2].instance_id),
            ):
                if (
                    candidate_index in assigned_candidates
                    or score.instance_id in assigned_instances
                ):
                    continue
                assigned_candidates.add(candidate_index)
                assigned_instances.add(score.instance_id)
                result_by_representative[candidate_index] = score.instance_id
                score_by_representative[candidate_index] = score
                self._merge(
                    score.instance_id,
                    candidates[candidate_index],
                    observations[candidate_index],
                )
            for candidate_index in representatives:
                if candidate_index in result_by_representative:
                    continue
                candidate = candidates[candidate_index]
                scores = self._scores(candidate)
                best = scores[0] if scores else None
                status = (
                    'possible_duplicate'
                    if best is not None
                    and best.decision is AssociationDecision.UNCERTAIN
                    else self._candidate_status(
                        candidate, observations[candidate_index]
                    )
                )
                result_by_representative[candidate_index] = self._create(
                    candidate, observations[candidate_index], status
                )
                score_by_representative[candidate_index] = best
            result = [0] * len(candidates)
            events = []
            for index, candidate in enumerate(candidates):
                representative = aliases.get(index, index)
                instance_id = result_by_representative[representative]
                result[index] = instance_id
                score = score_by_representative.get(representative)
                if index != representative:
                    reason = 'same_keyframe_overlap_duplicate'
                    decision = 'same_keyframe_duplicate'
                    components = {}
                    final_score = 1.0
                else:
                    reason = (
                        'accepted_one_to_one_match'
                        if score is not None
                        and score.decision is AssociationDecision.MERGE
                        else 'new_or_uncertain_instance'
                    )
                    decision = (
                        score.decision.value
                        if score is not None
                        else AssociationDecision.CREATE_NEW.value
                    )
                    components = score.components if score is not None else {}
                    final_score = score.final_score if score is not None else 0.0
                events.append(ObjectAssociationEvent(
                    candidate.candidate_id,
                    canonicalize_class_name(candidate.class_name),
                    instance_id,
                    components,
                    final_score,
                    decision,
                    reason,
                    self._states[instance_id].instance.observation_count,
                ))
            self._last_events = tuple(events)
            self._enforce_total_bound()
            return result

    def get(self, instance_id: int) -> ObjectInstance:
        """Return a defensive frozen-contract snapshot by ID."""
        with self._lock:
            try:
                return _copy_instance(self._states[instance_id].instance)
            except KeyError as error:
                raise KeyError(f'unknown object instance {instance_id}') from error

    def record(self, instance_id: int) -> PersistentObjectRecord:
        """Return rich metadata and bounded geometry/colour evidence by ID."""
        with self._lock:
            try:
                state = self._states[instance_id]
            except KeyError as error:
                raise KeyError(f'unknown object instance {instance_id}') from error
            return self._record(state)

    def update_colour(
        self,
        instance_id: int,
        estimate: ColourEstimate,
        *,
        crop_quality: float = 1.0,
        mask_quality: float = 1.0,
        geometry_support: float = 1.0,
    ) -> float:
        """Fuse one colour observation and return its bounded evidence weight."""
        if not isinstance(estimate, ColourEstimate):
            raise TypeError('estimate must be ColourEstimate')
        qualities = (crop_quality, mask_quality, geometry_support)
        if any(not isfinite(value) or not 0.0 <= value <= 1.0
               for value in qualities):
            raise ValueError('colour quality terms must lie in [0, 1]')
        with self._lock:
            try:
                state = self._states[instance_id]
            except KeyError as error:
                raise KeyError(f'unknown object instance {instance_id}') from error
            state.colour_estimates.append(estimate)
            if len(state.colour_estimates) > self.config.max_colour_history:
                del state.colour_estimates[:-self.config.max_colour_history]
            if not estimate.probabilities:
                return 0.0
            pixel_quality = min(1.0, estimate.valid_pixel_count / 250.0)
            exposure_quality = 0.55 if estimate.status in {
                'overexposed', 'underexposed', 'low_saturation'
            } else 1.0
            weight = (
                estimate.confidence
                * (0.25 + 0.75 * pixel_quality)
                * crop_quality
                * mask_quality
                * geometry_support
                * exposure_quality
            )
            view_key = estimate.source_viewpoint_id or '<unknown>'
            previous_view_weight = state.colour_view_weights.get(view_key, 0.0)
            remaining_view_capacity = max(0.0, 1.5 - previous_view_weight)
            weight = min(weight, remaining_view_capacity)
            if weight <= 0.0:
                return 0.0
            state.colour_view_weights[view_key] = previous_view_weight + weight
            while (
                len(state.colour_view_weights)
                > self.config.max_colour_history
            ):
                oldest_view = next(iter(state.colour_view_weights))
                state.colour_view_weights.pop(oldest_view)
            for name, probability in estimate.probabilities.items():
                state.colour_evidence[name] = (
                    state.colour_evidence.get(name, 0.0)
                    + weight * probability
                )
            total = sum(state.colour_evidence.values())
            if total > self.config.max_colour_evidence:
                scale = self.config.max_colour_evidence / total
                state.colour_evidence = {
                    name: value * scale
                    for name, value in state.colour_evidence.items()
                }
                total = self.config.max_colour_evidence
            probabilities = {
                name: value / total
                for name, value in sorted(state.colour_evidence.items())
            }
            state.colour_confidence = min(
                1.0,
                max(probabilities.values())
                * (1.0 - exp(-total / 2.0)),
            )
            if (
                state.best_colour_estimate is None
                or estimate.confidence > state.best_colour_estimate.confidence
            ):
                state.best_colour_estimate = estimate
            previous = state.instance
            state.instance = ObjectInstance(
                previous.instance_id,
                previous.class_scores,
                probabilities,
                previous.centroid_xyz,
                previous.aabb_min_xyz,
                previous.aabb_max_xyz,
                previous.obb_dimensions,
                previous.obb_yaw,
                previous.orientation_confidence,
                previous.observation_count,
                previous.confidence,
            )
            return weight

    def active_instances(
        self,
        class_name: str | None = None,
    ) -> list[ObjectInstance]:
        """Return deterministic snapshots, optionally by canonical class."""
        canonical = (
            canonicalize_class_name(class_name)
            if class_name is not None
            else None
        )
        with self._lock:
            result = []
            for instance_id in sorted(self._states):
                state = self._states[instance_id]
                if canonical is not None and _canonical_class(
                    state.instance
                ) != canonical:
                    continue
                result.append(_copy_instance(state.instance))
            return result

    def serialize(self, *, include_points: bool = False) -> list[dict[str, object]]:
        """Serialize state without raw images and without points by default."""
        with self._lock:
            output = []
            for instance_id in sorted(self._states):
                state = self._states[instance_id]
                item = {
                    'instance_id': instance_id,
                    'canonical_class': _canonical_class(state.instance),
                    'class_scores': dict(state.instance.class_scores),
                    'centroid_xyz': state.instance.centroid_xyz.tolist(),
                    'aabb_min_xyz': state.instance.aabb_min_xyz.tolist(),
                    'aabb_max_xyz': state.instance.aabb_max_xyz.tolist(),
                    'obb_dimensions': state.instance.obb_dimensions.tolist(),
                    'obb_yaw': state.instance.obb_yaw,
                    'orientation_confidence': (
                        state.instance.orientation_confidence
                    ),
                    'geometry_confidence': state.geometry_confidence,
                    'observation_count': state.instance.observation_count,
                    'first_seen_ns': state.first_seen_ns,
                    'last_seen_ns': state.last_seen_ns,
                    'source_viewpoint_ids': list(state.source_viewpoint_ids),
                    'source_detection_ids': list(state.source_detection_ids),
                    'best_crop_score': state.best_crop_score,
                    'best_crop_shape': (
                        list(state.best_crop.shape)
                        if state.best_crop is not None else None
                    ),
                    'status': state.status,
                    'colour_scores': dict(state.instance.colour_scores),
                    'colour_confidence': state.colour_confidence,
                    'colour_evidence': dict(state.colour_evidence),
                    'best_colour_source': (
                        {
                            'viewpoint_id': (
                                state.best_colour_estimate.source_viewpoint_id
                            ),
                            'detection_id': (
                                state.best_colour_estimate.source_detection_id
                            ),
                            'status': state.best_colour_estimate.status,
                        }
                        if state.best_colour_estimate is not None else None
                    ),
                    'fused_point_count': int(state.fused_points_xyz.shape[0]),
                    'best_view_candidate_id': (
                        state.best_view_candidate.candidate_id
                    ),
                }
                if include_points:
                    item['fused_points_xyz'] = state.fused_points_xyz.tolist()
                output.append(item)
            return output

    def reset_episode(self) -> None:
        """Clear all identities, evidence, traces, and allocator state."""
        with self._lock:
            self._states.clear()
            self._next_instance_id = 0
            self._last_events = ()

    def _validate_pair(
        self,
        candidate: ObjectCandidate3D,
        observation: ViewpointObservation,
    ) -> None:
        if not isinstance(candidate, ObjectCandidate3D):
            raise TypeError('candidate must be ObjectCandidate3D')
        if not isinstance(observation, ViewpointObservation):
            raise TypeError('observation must be ViewpointObservation')
        if candidate.detection_id != observation.detection_id:
            raise ValueError('candidate and observation detection IDs differ')
        if candidate.point_count != observation.point_count:
            raise ValueError('candidate and observation point counts differ')

    def _scores(self, candidate: ObjectCandidate3D) -> list[AssociationScore]:
        scores = [
            score_candidate_instance(
                candidate, state.instance, self.config.association
            )
            for state in self._states.values()
        ]
        return sorted(
            scores,
            key=lambda score: (-score.final_score, score.instance_id),
        )

    def _same_keyframe_groups(
        self,
        candidates: list[ObjectCandidate3D],
    ) -> tuple[list[int], dict[int, int]]:
        representatives = []
        aliases = {}
        for index, candidate in enumerate(candidates):
            duplicate_of = None
            for representative in representatives:
                existing = candidates[representative]
                if class_compatibility(
                    candidate.class_name, existing.class_name
                ) <= 0.0:
                    continue
                distance = float(np.linalg.norm(
                    candidate.obb_centre_xyz - existing.obb_centre_xyz
                ))
                overlap = aabb_iou_3d(
                    candidate.aabb_min_xyz,
                    candidate.aabb_max_xyz,
                    existing.aabb_min_xyz,
                    existing.aabb_max_xyz,
                )
                if (
                    overlap >= self.config.same_keyframe_overlap_threshold
                    or (
                        distance <= self.config.same_keyframe_distance_m
                        and (
                            overlap >= 0.25
                            or candidate.partial_geometry
                            or existing.partial_geometry
                            or min(
                                candidate.geometry_confidence,
                                existing.geometry_confidence,
                            ) <= 0.45
                        )
                    )
                ):
                    duplicate_of = representative
                    break
            if duplicate_of is None:
                representatives.append(index)
            else:
                aliases[index] = duplicate_of
        return representatives, aliases

    def _create(
        self,
        candidate: ObjectCandidate3D,
        observation: ViewpointObservation,
        status: str,
    ) -> int:
        self._make_room_for_instance()
        instance_id = self._next_instance_id
        self._next_instance_id += 1
        class_name = canonicalize_class_name(candidate.class_name)
        evidence = max(candidate.detection_confidence, 1e-6)
        points = _voxel_points(
            candidate.points_map_xyz,
            self.config.fused_voxel_size_m,
            self.config.max_fused_points_per_instance,
        )
        instance = ObjectInstance(
            instance_id=instance_id,
            class_scores={class_name: 1.0},
            colour_scores={},
            centroid_xyz=candidate.obb_centre_xyz,
            aabb_min_xyz=candidate.aabb_min_xyz,
            aabb_max_xyz=candidate.aabb_max_xyz,
            obb_dimensions=candidate.obb_dimensions_xyz,
            obb_yaw=candidate.obb_yaw_rad,
            orientation_confidence=candidate.orientation_confidence,
            observation_count=1,
            confidence=candidate.geometry_confidence,
        )
        crop = None if observation.best_crop is None else observation.best_crop.copy()
        self._states[instance_id] = _InstanceState(
            instance=instance,
            class_evidence={class_name: evidence},
            centre_weight=max(candidate.geometry_confidence, 0.05),
            geometry_confidence=candidate.geometry_confidence,
            first_seen_ns=observation.timestamp_ns,
            last_seen_ns=observation.timestamp_ns,
            source_viewpoint_ids=[observation.viewpoint_id],
            source_detection_ids=[observation.detection_id],
            best_crop=crop,
            best_crop_score=observation.best_crop_score,
            status=status,
            fused_points_xyz=points,
            observations=[observation],
            best_view_candidate=candidate,
            colour_evidence={},
            colour_confidence=0.0,
            best_colour_estimate=None,
            colour_estimates=[],
            colour_view_weights={},
        )
        return instance_id

    def _merge(
        self,
        instance_id: int,
        candidate: ObjectCandidate3D,
        observation: ViewpointObservation,
    ) -> None:
        state = self._states[instance_id]
        previous = state.instance
        weight = max(candidate.geometry_confidence, 0.05)
        total_weight = state.centre_weight + weight
        centre = (
            previous.centroid_xyz * state.centre_weight
            + candidate.obb_centre_xyz * weight
        ) / total_weight
        state.centre_weight = total_weight
        class_name = canonicalize_class_name(candidate.class_name)
        state.class_evidence[class_name] = (
            state.class_evidence.get(class_name, 0.0)
            + max(candidate.detection_confidence, 1e-6)
        )
        maximum_evidence = max(state.class_evidence.values())
        class_scores = {
            name: min(1.0, value / maximum_evidence)
            for name, value in sorted(state.class_evidence.items())
        }
        new_points = _novel_points(
            state.fused_points_xyz,
            candidate.points_map_xyz,
            self.config.fused_voxel_size_m,
        )
        state.fused_points_xyz = _voxel_points(
            np.vstack((state.fused_points_xyz, new_points)),
            self.config.fused_voxel_size_m,
            self.config.max_fused_points_per_instance,
        )
        aabb_min = previous.aabb_min_xyz
        aabb_max = previous.aabb_max_xyz
        dimensions = previous.obb_dimensions
        yaw = previous.obb_yaw
        orientation_confidence = previous.orientation_confidence
        if (
            state.fused_points_xyz.shape[0] >= self.config.minimum_refit_points
            and new_points.shape[0] >= self.config.minimum_new_refit_points
        ):
            try:
                aabb = robust_aabb(state.fused_points_xyz)
                obb = estimate_upright_obb(state.fused_points_xyz)
                aabb_min = aabb.minimum_xyz
                aabb_max = aabb.maximum_xyz
                dimensions = obb.dimensions_xyz
                if (
                    candidate.orientation_confidence
                    >= self.config.association.yaw_confidence_threshold
                ):
                    yaw = obb.yaw_rad
                    orientation_confidence = _bounded_average(
                        orientation_confidence,
                        candidate.orientation_confidence,
                    )
            except ValueError:
                pass
        if (
            candidate.orientation_confidence
            >= self.config.association.yaw_confidence_threshold
            and previous.orientation_confidence
            >= self.config.association.yaw_confidence_threshold
        ):
            yaw = _fuse_rectangle_yaw(
                previous.obb_yaw,
                candidate.obb_yaw_rad,
                previous.orientation_confidence,
                candidate.orientation_confidence,
            )
        disagreement = float(np.linalg.norm(
            candidate.obb_centre_xyz - previous.centroid_xyz
        ))
        scale = max(float(np.linalg.norm(previous.obb_dimensions)), 0.1)
        agreement = exp(-disagreement / scale)
        fused_confidence = _bounded_average(
            state.geometry_confidence,
            candidate.geometry_confidence,
        ) * (0.85 + 0.15 * agreement)
        state.geometry_confidence = min(0.98, max(0.0, fused_confidence))
        state.instance = ObjectInstance(
            instance_id=instance_id,
            class_scores=class_scores,
            colour_scores=previous.colour_scores,
            centroid_xyz=centre,
            aabb_min_xyz=aabb_min,
            aabb_max_xyz=aabb_max,
            obb_dimensions=dimensions,
            obb_yaw=yaw,
            orientation_confidence=orientation_confidence,
            observation_count=previous.observation_count + 1,
            confidence=state.geometry_confidence,
        )
        state.first_seen_ns = min(state.first_seen_ns, observation.timestamp_ns)
        state.last_seen_ns = max(state.last_seen_ns, observation.timestamp_ns)
        _append_unique(state.source_viewpoint_ids, observation.viewpoint_id)
        _append_unique(state.source_detection_ids, observation.detection_id)
        state.observations.append(observation)
        if len(state.observations) > self.config.max_observation_history:
            del state.observations[:-self.config.max_observation_history]
        if observation.best_crop_score > state.best_crop_score:
            state.best_crop = (
                None
                if observation.best_crop is None
                else observation.best_crop.copy()
            )
            state.best_crop_score = observation.best_crop_score
        if (
            candidate.geometry_confidence
            > state.best_view_candidate.geometry_confidence
        ):
            state.best_view_candidate = candidate
        state.status = self._candidate_status(candidate, observation)

    def _event(
        self,
        candidate: ObjectCandidate3D,
        instance_id: int,
        score: AssociationScore | None,
        reason: str,
    ) -> ObjectAssociationEvent:
        return ObjectAssociationEvent(
            candidate_id=candidate.candidate_id,
            candidate_class=canonicalize_class_name(candidate.class_name),
            matched_instance_id=instance_id,
            components=score.components if score is not None else {},
            final_score=score.final_score if score is not None else 0.0,
            decision=(
                score.decision.value
                if score is not None else AssociationDecision.CREATE_NEW.value
            ),
            reason=reason,
            observation_count_after=(
                self._states[instance_id].instance.observation_count
            ),
        )

    @staticmethod
    def _candidate_status(
        candidate: ObjectCandidate3D,
        observation: ViewpointObservation,
    ) -> str:
        if observation.visibility == 'sparse' or (
            candidate.geometry_status is GeometryStatus.SPARSE
        ):
            return 'sparse'
        if observation.visibility == 'partial' or candidate.partial_geometry:
            return 'partially_observed'
        if candidate.geometry_confidence < 0.4:
            return 'uncertain'
        return 'active'

    def _record(self, state: _InstanceState) -> PersistentObjectRecord:
        return PersistentObjectRecord(
            instance=_copy_instance(state.instance),
            canonical_class=_canonical_class(state.instance),
            geometry_confidence=state.geometry_confidence,
            first_seen_ns=state.first_seen_ns,
            last_seen_ns=state.last_seen_ns,
            source_viewpoint_ids=tuple(state.source_viewpoint_ids),
            source_detection_ids=tuple(state.source_detection_ids),
            best_crop=state.best_crop,
            best_crop_score=state.best_crop_score,
            status=state.status,
            fused_points_xyz=state.fused_points_xyz,
            observations=tuple(state.observations),
            best_view_candidate_id=state.best_view_candidate.candidate_id,
            colour_evidence=state.colour_evidence,
            colour_confidence=state.colour_confidence,
            best_colour_estimate=state.best_colour_estimate,
            colour_estimates=tuple(state.colour_estimates),
        )

    def _make_room_for_instance(self) -> None:
        if len(self._states) < self.config.max_instances:
            return
        victim = min(
            self._states,
            key=lambda instance_id: (
                self._states[instance_id].status != 'possible_duplicate',
                self._states[instance_id].geometry_confidence,
                self._states[instance_id].last_seen_ns,
                instance_id,
            ),
        )
        del self._states[victim]

    def _enforce_total_bound(self) -> None:
        excess = self.fused_point_count - self.config.max_total_fused_points
        while excess > 0 and self._states:
            instance_id = max(
                self._states,
                key=lambda value: (
                    self._states[value].fused_points_xyz.shape[0], -value
                ),
            )
            state = self._states[instance_id]
            current = state.fused_points_xyz.shape[0]
            remove = min(excess, max(0, current - 1))
            if remove <= 0:
                break
            state.fused_points_xyz = _evenly_sample(
                state.fused_points_xyz, current - remove
            )
            excess -= remove


def _copy_instance(instance: ObjectInstance) -> ObjectInstance:
    return ObjectInstance(
        instance.instance_id,
        instance.class_scores,
        instance.colour_scores,
        instance.centroid_xyz,
        instance.aabb_min_xyz,
        instance.aabb_max_xyz,
        instance.obb_dimensions,
        instance.obb_yaw,
        instance.orientation_confidence,
        instance.observation_count,
        instance.confidence,
    )


def _canonical_class(instance: ObjectInstance) -> str:
    return max(
        sorted(instance.class_scores),
        key=lambda name: instance.class_scores[name],
    )


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _bounded_average(first: float, second: float) -> float:
    return min(1.0, max(0.0, 0.65 * first + 0.35 * second))


def _fuse_rectangle_yaw(
    first: float,
    second: float,
    first_weight: float,
    second_weight: float,
) -> float:
    x_value = first_weight * cos(2.0 * first) + second_weight * cos(
        2.0 * second
    )
    y_value = first_weight * sin(2.0 * first) + second_weight * sin(
        2.0 * second
    )
    return float(atan2(y_value, x_value) / 2.0)


def _novel_points(
    existing: np.ndarray,
    incoming: np.ndarray,
    voxel_size_m: float,
) -> np.ndarray:
    if existing.shape[0] == 0:
        return incoming
    existing_keys = set(map(tuple, np.floor(
        existing / voxel_size_m
    ).astype(np.int64)))
    incoming_keys = np.floor(incoming / voxel_size_m).astype(np.int64)
    keep = np.asarray([
        tuple(key) not in existing_keys for key in incoming_keys
    ], dtype=np.bool_)
    return incoming[keep]


def _voxel_points(
    points_xyz: np.ndarray,
    voxel_size_m: float,
    maximum_points: int,
) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    keys = np.floor(points / voxel_size_m).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    sums = np.zeros((unique_keys.shape[0], 3), dtype=np.float64)
    counts = np.zeros(unique_keys.shape[0], dtype=np.int64)
    np.add.at(sums, inverse, points)
    np.add.at(counts, inverse, 1)
    centroids = sums / counts[:, None]
    return _evenly_sample(centroids, maximum_points)


def _evenly_sample(points: np.ndarray, maximum_points: int) -> np.ndarray:
    if points.shape[0] <= maximum_points:
        return np.ascontiguousarray(points).copy()
    indices = np.linspace(
        0, points.shape[0] - 1, maximum_points, dtype=np.int64
    )
    return np.ascontiguousarray(points[indices]).copy()


__all__ = [
    'INSTANCE_STATUSES',
    'ObjectAssociationEvent',
    'ObjectMap',
    'ObjectMapConfig',
    'PersistentObjectRecord',
]
