"""Episode-local walls and ray-anchored architectural landmarks."""

from dataclasses import dataclass
from math import isfinite
from threading import RLock
from types import MappingProxyType
from typing import Mapping

import numpy as np

from qmapnav.mapping.object_association import canonicalize_class_name
from qmapnav.mapping.object_candidate import readonly_array
from qmapnav.mapping.ray_wall_intersection import intersect_ray_with_wall
from qmapnav.mapping.ray_wall_intersection import transform_camera_ray_to_map
from qmapnav.mapping.wall_extraction import extract_wall_candidates
from qmapnav.mapping.wall_extraction import merge_wall_candidates
from qmapnav.mapping.wall_extraction import WallCandidate
from qmapnav.mapping.wall_extraction import WallExtractionConfig
from qmapnav.perception.contracts import Detection2D


STRUCTURAL_CLASSES = frozenset({
    'clock', 'decal', 'door', 'doorway', 'picture', 'screen', 'tv', 'window'
})


@dataclass(frozen=True)
class StructuralAnchor:
    """Persistent map-frame architectural feature or supporting wall."""

    anchor_id: str
    anchor_type: str
    semantic_class: str
    position_xyz: np.ndarray
    line_segment_xy: np.ndarray | None
    polygon_xy: np.ndarray | None
    plane_parameters: np.ndarray | None
    extent_xyz: np.ndarray | None
    yaw_rad: float | None
    supporting_wall_id: str | None
    confidence: float
    first_seen_ns: int
    last_seen_ns: int
    source_viewpoint_ids: tuple[str, ...]
    source_detection_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ('anchor_id', 'anchor_type', 'semantic_class'):
            if not isinstance(getattr(self, name), str) or not getattr(
                self, name
            ).strip():
                raise ValueError(f'{name} must be a non-empty string')
        object.__setattr__(
            self,
            'position_xyz',
            readonly_array('position_xyz', self.position_xyz, (3,)),
        )
        optional_arrays = {
            'line_segment_xy': (2, 2),
            'polygon_xy': (None, 2),
            'plane_parameters': (4,),
            'extent_xyz': (3,),
        }
        for name, shape in optional_arrays.items():
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self, name, readonly_array(name, value, shape)
                )
        if self.extent_xyz is not None and np.any(self.extent_xyz <= 0.0):
            raise ValueError('extent_xyz must be positive')
        if self.yaw_rad is not None and not isfinite(self.yaw_rad):
            raise ValueError('yaw_rad must be finite when provided')
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError('confidence must lie in [0, 1]')
        if self.first_seen_ns < 0 or self.last_seen_ns < self.first_seen_ns:
            raise ValueError('structural timestamps are invalid')
        object.__setattr__(
            self, 'source_viewpoint_ids', tuple(self.source_viewpoint_ids)
        )
        object.__setattr__(
            self, 'source_detection_ids', tuple(self.source_detection_ids)
        )


@dataclass(frozen=True)
class StructuralMapConfig:
    """Wall association and structural-anchor fusion policy."""

    wall_extraction: WallExtractionConfig = WallExtractionConfig()
    ray_parallel_epsilon: float = 1.0e-6
    max_wall_extent_margin_m: float = 0.25
    anchor_merge_distance_m: float = 0.55
    ambiguous_wall_distance_margin_m: float = 0.10
    max_walls: int = 128
    max_anchors: int = 512

    def __post_init__(self) -> None:
        if not isinstance(self.wall_extraction, WallExtractionConfig):
            raise TypeError('wall_extraction must be WallExtractionConfig')
        for name in (
            'ray_parallel_epsilon',
            'max_wall_extent_margin_m',
            'anchor_merge_distance_m',
            'ambiguous_wall_distance_margin_m',
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        if self.max_walls <= 0 or self.max_anchors <= 0:
            raise ValueError('structural map bounds must be positive')


@dataclass(frozen=True)
class StructuralRecord:
    """Anchor plus fit diagnostics not present in the shared geometry."""

    anchor: StructuralAnchor
    supporting_point_count: int = 0
    fit_residual_m: float | None = None
    vertical_extent_m: tuple[float, float] | None = None
    observation_count: int = 1


@dataclass(frozen=True)
class StructuralAssociationEvent:
    """Trace-ready record of a wall or structural-anchor choice."""

    detection_id: str
    semantic_class: str
    anchor_id: str | None
    supporting_wall_id: str | None
    decision: str
    reason: str
    candidate_wall_distances_m: Mapping[str, float]
    ray_origin_xyz: tuple[float, float, float] | None = None
    intersection_xyz: tuple[float, float, float] | None = None
    lidar_depth_m: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            'candidate_wall_distances_m',
            MappingProxyType(dict(self.candidate_wall_distances_m)),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe structural decision payload."""
        return {
            'event': 'structural_anchor_association',
            'detection_id': self.detection_id,
            'semantic_class': self.semantic_class,
            'anchor_id': self.anchor_id,
            'supporting_wall_id': self.supporting_wall_id,
            'decision': self.decision,
            'reason': self.reason,
            'candidate_wall_distances_m': dict(
                self.candidate_wall_distances_m
            ),
            'ray_origin_xyz': self.ray_origin_xyz,
            'intersection_xyz': self.intersection_xyz,
            'lidar_depth_m': self.lidar_depth_m,
        }


class StructuralMap:
    """Maintain bounded walls and architectural anchors for one episode."""

    def __init__(self, config: StructuralMapConfig | None = None) -> None:
        self.config = config or StructuralMapConfig()
        self._walls: dict[str, StructuralRecord] = {}
        self._anchors: dict[str, StructuralRecord] = {}
        self._next_wall_id = 0
        self._next_anchor_id = 0
        self._last_events: tuple[StructuralAssociationEvent, ...] = ()
        self._lock = RLock()

    @property
    def last_events(self) -> tuple[StructuralAssociationEvent, ...]:
        """Return structural decisions from the most recent operation."""
        with self._lock:
            return self._last_events

    def update_walls_from_points(
        self,
        points_map_xyz: np.ndarray,
        *,
        timestamp_ns: int,
        viewpoint_id: str = 'scan_map',
    ) -> tuple[str, ...]:
        """Extract and add stable wall candidates from accumulated geometry."""
        candidates = extract_wall_candidates(
            points_map_xyz,
            timestamp_ns=timestamp_ns,
            config=self.config.wall_extraction,
        )
        return tuple(
            self.add_or_update_wall(
                wall_candidate_to_anchor(candidate, viewpoint_id=viewpoint_id),
                supporting_point_count=candidate.supporting_point_count,
                fit_residual_m=candidate.fit_residual_m,
                vertical_extent_m=candidate.vertical_extent_m,
            )
            for candidate in candidates
        )

    def add_or_update_wall(
        self,
        wall_candidate: StructuralAnchor,
        *,
        supporting_point_count: int = 0,
        fit_residual_m: float | None = None,
        vertical_extent_m: tuple[float, float] | None = None,
    ) -> str:
        """Return a persistent wall ID after conservative collinear fusion."""
        if not isinstance(wall_candidate, StructuralAnchor):
            raise TypeError('wall_candidate must be StructuralAnchor')
        if wall_candidate.anchor_type != 'wall':
            raise ValueError('wall candidate must have anchor_type wall')
        with self._lock:
            for wall_id in sorted(self._walls):
                existing = self._walls[wall_id]
                merged = _try_merge_walls(
                    existing,
                    wall_candidate,
                    supporting_point_count,
                    fit_residual_m,
                    vertical_extent_m,
                    self.config.wall_extraction,
                )
                if merged is not None:
                    self._walls[wall_id] = merged
                    return wall_id
            if len(self._walls) >= self.config.max_walls:
                victim = min(
                    self._walls,
                    key=lambda value: (
                        self._walls[value].anchor.confidence,
                        self._walls[value].anchor.last_seen_ns,
                        value,
                    ),
                )
                del self._walls[victim]
            wall_id = f'wall_{self._next_wall_id:04d}'
            self._next_wall_id += 1
            anchor = _copy_anchor(wall_candidate, anchor_id=wall_id)
            self._walls[wall_id] = StructuralRecord(
                anchor,
                supporting_point_count,
                fit_residual_m,
                vertical_extent_m,
                1,
            )
            return wall_id

    def anchor_detection_to_wall(
        self,
        detection: Detection2D,
        camera_pose: np.ndarray,
    ) -> StructuralAnchor | None:
        """Anchor one structural detection using ``T_map_from_camera``."""
        if not isinstance(detection, Detection2D):
            raise TypeError('detection must be Detection2D')
        semantic_class = canonicalize_class_name(detection.class_name)
        if semantic_class not in STRUCTURAL_CLASSES:
            self._last_events = (StructuralAssociationEvent(
                detection.detection_id,
                semantic_class,
                None,
                None,
                'reject',
                'class_is_not_structural',
                {},
            ),)
            return None
        origin, direction = transform_camera_ray_to_map(
            detection.centre_camera_ray, camera_pose
        )
        lidar_depth = _optional_positive_depth(
            detection.metadata.get('lidar_depth_m')
        )
        with self._lock:
            hits = []
            distances = {}
            for wall_id, record in self._walls.items():
                wall = record.anchor
                intersection = intersect_ray_with_wall(
                    origin,
                    direction,
                    wall.plane_parameters,
                    wall.line_segment_xy,
                    parallel_epsilon=self.config.ray_parallel_epsilon,
                    extent_margin_m=self.config.max_wall_extent_margin_m,
                )
                if intersection is None:
                    continue
                if not _height_is_plausible(
                    semantic_class,
                    intersection.position_xyz[2],
                    record.vertical_extent_m,
                    self.config.max_wall_extent_margin_m,
                ):
                    continue
                distances[wall_id] = intersection.distance_m
                hits.append((intersection.distance_m, wall_id, intersection))
            if lidar_depth is None:
                hits.sort(key=lambda item: (item[0], item[1]))
            else:
                hits.sort(key=lambda item: (
                    abs(item[0] - lidar_depth), item[0], item[1]
                ))
            if not hits:
                self._last_events = (StructuralAssociationEvent(
                    detection.detection_id,
                    semantic_class,
                    None,
                    None,
                    'reject',
                    'no_plausible_forward_wall_intersection',
                    distances,
                ),)
                return None
            first_rank = (
                hits[0][0] if lidar_depth is None
                else abs(hits[0][0] - lidar_depth)
            )
            second_rank = (
                hits[1][0] if lidar_depth is None and len(hits) > 1
                else abs(hits[1][0] - lidar_depth)
                if len(hits) > 1 else float('inf')
            )
            if second_rank - first_rank < (
                self.config.ambiguous_wall_distance_margin_m
            ):
                self._last_events = (StructuralAssociationEvent(
                    detection.detection_id,
                    semantic_class,
                    None,
                    None,
                    'reject',
                    'ambiguous_near_equal_wall_intersections',
                    distances,
                ),)
                return None
            _, wall_id, intersection = hits[0]
            wall = self._walls[wall_id].anchor
            confidence = float(
                detection.confidence
                * wall.confidence
                * np.exp(-intersection.extent_overrun_m / 0.25)
            )
            if lidar_depth is not None:
                depth_scale = max(0.25, 0.20 * lidar_depth)
                confidence *= float(np.exp(
                    -abs(intersection.distance_m - lidar_depth) / depth_scale
                ))
            viewpoint_id = str(
                detection.metadata.get('viewpoint_id', 'unknown_viewpoint')
            )
            timestamp_ns = int(detection.metadata.get('timestamp_ns', 0))
            extent = _default_extent(semantic_class)
            existing_id = self._matching_anchor(
                semantic_class, wall_id, intersection.position_xyz
            )
            if existing_id is None:
                self._make_room_for_anchor()
                anchor_id = f'anchor_{self._next_anchor_id:04d}'
                self._next_anchor_id += 1
                anchor = StructuralAnchor(
                    anchor_id=anchor_id,
                    anchor_type=_anchor_type(semantic_class),
                    semantic_class=semantic_class,
                    position_xyz=intersection.position_xyz,
                    line_segment_xy=None,
                    polygon_xy=None,
                    plane_parameters=wall.plane_parameters,
                    extent_xyz=extent,
                    yaw_rad=wall.yaw_rad,
                    supporting_wall_id=wall_id,
                    confidence=confidence,
                    first_seen_ns=timestamp_ns,
                    last_seen_ns=timestamp_ns,
                    source_viewpoint_ids=(viewpoint_id,),
                    source_detection_ids=(detection.detection_id,),
                )
                self._anchors[anchor_id] = StructuralRecord(anchor)
                decision = 'create_new'
                reason = (
                    'lidar_depth_consistent_forward_wall'
                    if lidar_depth is not None
                    else 'nearest_plausible_forward_wall'
                )
            else:
                anchor_id = existing_id
                anchor = self._fuse_anchor(
                    anchor_id,
                    intersection.position_xyz,
                    confidence,
                    timestamp_ns,
                    viewpoint_id,
                    detection.detection_id,
                )
                decision = 'merge'
                reason = 'same_class_same_wall_within_merge_gate'
            self._last_events = (StructuralAssociationEvent(
                detection.detection_id,
                semantic_class,
                anchor_id,
                wall_id,
                decision,
                reason,
                distances,
                tuple(map(float, origin)),
                tuple(map(float, intersection.position_xyz)),
                lidar_depth,
            ),)
            return anchor

    def get(self, anchor_id: str) -> StructuralAnchor:
        """Return one defensive wall or architectural anchor snapshot."""
        with self._lock:
            record = self._walls.get(anchor_id) or self._anchors.get(anchor_id)
            if record is None:
                raise KeyError(f'unknown structural anchor {anchor_id!r}')
            return _copy_anchor(record.anchor)

    def record(self, anchor_id: str) -> StructuralRecord:
        """Return one structural record with fit and count diagnostics."""
        with self._lock:
            record = self._walls.get(anchor_id) or self._anchors.get(anchor_id)
            if record is None:
                raise KeyError(f'unknown structural anchor {anchor_id!r}')
            return StructuralRecord(
                _copy_anchor(record.anchor),
                record.supporting_point_count,
                record.fit_residual_m,
                record.vertical_extent_m,
                record.observation_count,
            )

    def get_by_class(self, semantic_class: str) -> list[StructuralAnchor]:
        """Return deterministic wall and anchor snapshots by class."""
        canonical = canonicalize_class_name(semantic_class)
        with self._lock:
            records = list(self._walls.values()) + list(self._anchors.values())
            return [
                _copy_anchor(record.anchor)
                for record in records
                if record.anchor.semantic_class == canonical
            ]

    def walls(self) -> list[StructuralAnchor]:
        """Return deterministic map wall snapshots."""
        with self._lock:
            return [
                _copy_anchor(self._walls[key].anchor)
                for key in sorted(self._walls)
            ]

    def anchors(self) -> list[StructuralAnchor]:
        """Return deterministic non-wall structural snapshots."""
        with self._lock:
            return [
                _copy_anchor(self._anchors[key].anchor)
                for key in sorted(self._anchors)
            ]

    def serialize(self) -> list[dict[str, object]]:
        """Return bounded JSON-safe structural state."""
        output = []
        with self._lock:
            records = {
                **self._walls,
                **self._anchors,
            }
            for anchor_id in sorted(records):
                record = records[anchor_id]
                anchor = record.anchor
                output.append({
                    'anchor_id': anchor.anchor_id,
                    'anchor_type': anchor.anchor_type,
                    'semantic_class': anchor.semantic_class,
                    'position_xyz': anchor.position_xyz.tolist(),
                    'line_segment_xy': (
                        None if anchor.line_segment_xy is None
                        else anchor.line_segment_xy.tolist()
                    ),
                    'plane_parameters': (
                        None if anchor.plane_parameters is None
                        else anchor.plane_parameters.tolist()
                    ),
                    'extent_xyz': (
                        None if anchor.extent_xyz is None
                        else anchor.extent_xyz.tolist()
                    ),
                    'yaw_rad': anchor.yaw_rad,
                    'supporting_wall_id': anchor.supporting_wall_id,
                    'confidence': anchor.confidence,
                    'first_seen_ns': anchor.first_seen_ns,
                    'last_seen_ns': anchor.last_seen_ns,
                    'source_viewpoint_ids': list(
                        anchor.source_viewpoint_ids
                    ),
                    'source_detection_ids': list(
                        anchor.source_detection_ids
                    ),
                    'supporting_point_count': record.supporting_point_count,
                    'fit_residual_m': record.fit_residual_m,
                    'vertical_extent_m': record.vertical_extent_m,
                    'observation_count': record.observation_count,
                })
        return output

    def reset_episode(self) -> None:
        """Clear structural state and reset deterministic ID allocation."""
        with self._lock:
            self._walls.clear()
            self._anchors.clear()
            self._next_wall_id = 0
            self._next_anchor_id = 0
            self._last_events = ()

    def _matching_anchor(self, semantic_class, wall_id, position):
        matches = []
        for anchor_id, record in self._anchors.items():
            anchor = record.anchor
            if (
                anchor.semantic_class == semantic_class
                and anchor.supporting_wall_id == wall_id
            ):
                distance = float(np.linalg.norm(
                    anchor.position_xyz - position
                ))
                if distance <= self.config.anchor_merge_distance_m:
                    matches.append((distance, anchor_id))
        return min(matches)[1] if matches else None

    def _fuse_anchor(
        self,
        anchor_id,
        position,
        confidence,
        timestamp_ns,
        viewpoint_id,
        detection_id,
    ):
        record = self._anchors[anchor_id]
        anchor = record.anchor
        weight = max(confidence, 0.05)
        prior_weight = max(anchor.confidence, 0.05) * record.observation_count
        unconstrained_position = (
            anchor.position_xyz * prior_weight + position * weight
        ) / (prior_weight + weight)
        wall = self._walls[anchor.supporting_wall_id].anchor
        fused_position = _project_to_wall_coordinates(
            unconstrained_position, wall
        )
        viewpoints = list(anchor.source_viewpoint_ids)
        detections = list(anchor.source_detection_ids)
        if viewpoint_id not in viewpoints:
            viewpoints.append(viewpoint_id)
        if detection_id not in detections:
            detections.append(detection_id)
        fused_confidence = min(
            0.98,
            (anchor.confidence * prior_weight + confidence * weight)
            / (prior_weight + weight),
        )
        fused = StructuralAnchor(
            anchor.anchor_id,
            anchor.anchor_type,
            anchor.semantic_class,
            fused_position,
            anchor.line_segment_xy,
            anchor.polygon_xy,
            anchor.plane_parameters,
            anchor.extent_xyz,
            anchor.yaw_rad,
            anchor.supporting_wall_id,
            fused_confidence,
            min(anchor.first_seen_ns, timestamp_ns),
            max(anchor.last_seen_ns, timestamp_ns),
            tuple(viewpoints),
            tuple(detections),
        )
        self._anchors[anchor_id] = StructuralRecord(
            fused,
            observation_count=record.observation_count + 1,
        )
        return _copy_anchor(fused)

    def _make_room_for_anchor(self):
        if len(self._anchors) < self.config.max_anchors:
            return
        victim = min(
            self._anchors,
            key=lambda value: (
                self._anchors[value].anchor.confidence,
                self._anchors[value].anchor.last_seen_ns,
                value,
            ),
        )
        del self._anchors[victim]


def wall_candidate_to_anchor(
    candidate: WallCandidate,
    *,
    viewpoint_id: str,
) -> StructuralAnchor:
    """Adapt one extracted wall candidate to the structural contract."""
    centre_xy = np.mean(candidate.line_segment_xy, axis=0)
    height = candidate.vertical_extent_m[1] - candidate.vertical_extent_m[0]
    return StructuralAnchor(
        anchor_id='wall_candidate',
        anchor_type='wall',
        semantic_class='wall',
        position_xyz=np.array([
            centre_xy[0],
            centre_xy[1],
            sum(candidate.vertical_extent_m) / 2.0,
        ]),
        line_segment_xy=candidate.line_segment_xy,
        polygon_xy=None,
        plane_parameters=candidate.plane_parameters,
        extent_xyz=np.array([
            candidate.length_m,
            max(0.05, 2.0 * candidate.fit_residual_m),
            height,
        ]),
        yaw_rad=candidate.yaw_rad,
        supporting_wall_id=None,
        confidence=candidate.confidence,
        first_seen_ns=candidate.timestamp_ns,
        last_seen_ns=candidate.timestamp_ns,
        source_viewpoint_ids=(viewpoint_id,),
        source_detection_ids=(),
    )


def _try_merge_walls(
    existing,
    candidate,
    supporting_point_count,
    fit_residual_m,
    vertical_extent_m,
    config,
):
    first = _anchor_to_wall_candidate(existing)
    second_record = StructuralRecord(
        candidate,
        supporting_point_count,
        fit_residual_m,
        vertical_extent_m,
    )
    second = _anchor_to_wall_candidate(second_record)
    merged = merge_wall_candidates([first, second], config=config)
    if len(merged) != 1:
        return None
    wall = wall_candidate_to_anchor(
        merged[0],
        viewpoint_id=(
            candidate.source_viewpoint_ids[0]
            if candidate.source_viewpoint_ids else 'scan_map'
        ),
    )
    viewpoints = tuple(dict.fromkeys(
        existing.anchor.source_viewpoint_ids
        + candidate.source_viewpoint_ids
    ))
    wall = _copy_anchor(
        wall,
        anchor_id=existing.anchor.anchor_id,
        first_seen_ns=min(
            existing.anchor.first_seen_ns, candidate.first_seen_ns
        ),
        last_seen_ns=max(existing.anchor.last_seen_ns, candidate.last_seen_ns),
        source_viewpoint_ids=viewpoints,
    )
    return StructuralRecord(
        wall,
        merged[0].supporting_point_count,
        merged[0].fit_residual_m,
        merged[0].vertical_extent_m,
        existing.observation_count + 1,
    )


def _anchor_to_wall_candidate(record):
    anchor = record.anchor
    vertical = record.vertical_extent_m
    if vertical is None:
        vertical = (
            anchor.position_xyz[2] - anchor.extent_xyz[2] / 2.0,
            anchor.position_xyz[2] + anchor.extent_xyz[2] / 2.0,
        )
    return WallCandidate(
        anchor.line_segment_xy,
        anchor.plane_parameters,
        vertical,
        anchor.yaw_rad,
        max(1, record.supporting_point_count),
        0.0 if record.fit_residual_m is None else record.fit_residual_m,
        anchor.confidence,
        anchor.last_seen_ns,
    )


def _copy_anchor(anchor, **overrides):
    values = {
        name: getattr(anchor, name)
        for name in anchor.__dataclass_fields__
    }
    values.update(overrides)
    return StructuralAnchor(**values)


def _height_is_plausible(class_name, height, wall_extent, margin):
    if wall_extent is not None and not (
        wall_extent[0] - margin <= height <= wall_extent[1] + margin
    ):
        return False
    ranges = {
        'clock': (0.7, 3.5),
        'decal': (0.3, 3.5),
        'door': (0.0, 2.5),
        'doorway': (0.0, 2.5),
        'picture': (0.4, 3.5),
        'screen': (0.4, 3.5),
        'tv': (0.4, 3.5),
        'window': (0.3, 3.5),
    }
    lower, upper = ranges[class_name]
    return lower <= height <= upper


def _default_extent(class_name):
    values = {
        'clock': (0.30, 0.05, 0.30),
        'decal': (0.40, 0.05, 0.30),
        'door': (0.90, 0.05, 2.00),
        'doorway': (0.90, 0.05, 2.00),
        'picture': (0.60, 0.05, 0.40),
        'screen': (0.70, 0.05, 0.45),
        'tv': (0.80, 0.08, 0.50),
        'window': (1.00, 0.05, 1.00),
    }
    return np.asarray(values[class_name], dtype=np.float64)


def _anchor_type(class_name):
    if class_name in {'door', 'doorway'}:
        return 'opening'
    if class_name == 'window':
        return 'window'
    return 'wall_mounted_object'


def _optional_positive_depth(value):
    if value is None:
        return None
    try:
        depth = float(value)
    except (TypeError, ValueError):
        return None
    return depth if isfinite(depth) and depth > 0.0 else None


def _project_to_wall_coordinates(position, wall):
    """Project a fused XYZ position into its supporting wall coordinates."""
    if wall.line_segment_xy is None:
        return np.asarray(position, dtype=np.float64)
    start, end = wall.line_segment_xy
    tangent = end - start
    length = float(np.linalg.norm(tangent))
    if length <= 1.0e-9:
        return np.asarray(position, dtype=np.float64)
    tangent /= length
    along = float(np.dot(np.asarray(position)[:2] - start, tangent))
    xy = start + np.clip(along, 0.0, length) * tangent
    return np.array([xy[0], xy[1], position[2]], dtype=np.float64)


__all__ = [
    'STRUCTURAL_CLASSES',
    'StructuralAnchor',
    'StructuralAssociationEvent',
    'StructuralMap',
    'StructuralMapConfig',
    'StructuralRecord',
    'wall_candidate_to_anchor',
]
