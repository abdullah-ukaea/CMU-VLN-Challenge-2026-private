"""Regression tests for persistent object identity and bounded fusion."""

import numpy as np

from qmapnav.mapping.object_association import score_candidate_instance
from qmapnav.mapping.object_candidate import ConfidenceComponents
from qmapnav.mapping.object_candidate import GeometrySource
from qmapnav.mapping.object_candidate import GeometryStatus
from qmapnav.mapping.object_candidate import LiftingCounts
from qmapnav.mapping.object_candidate import ObjectCandidate3D
from qmapnav.mapping.object_map import ObjectMap
from qmapnav.mapping.object_map import ObjectMapConfig
from qmapnav.mapping.viewpoint_observation import ViewpointObservation
from qmapnav.reasoning.colour_types import ColourEstimate


def _candidate(
    detection_id: str,
    centre: tuple[float, float, float],
    *,
    class_name: str = 'chair',
    dimensions: tuple[float, float, float] = (0.6, 0.5, 1.0),
    partial: bool = False,
    yaw: float = 0.1,
    confidence: float = 0.8,
) -> ObjectCandidate3D:
    centre_array = np.asarray(centre, dtype=np.float64)
    dimensions_array = np.asarray(dimensions, dtype=np.float64)
    offsets = np.asarray([
        [-0.45, -0.45, -0.45],
        [-0.45, 0.45, -0.45],
        [0.45, -0.45, 0.45],
        [0.45, 0.45, 0.45],
        [0.0, 0.0, 0.0],
        [0.2, -0.2, 0.2],
        [-0.2, 0.2, -0.2],
        [0.3, 0.3, -0.3],
    ])
    points = centre_array + offsets * dimensions_array
    minimum = centre_array - dimensions_array / 2.0
    maximum = centre_array + dimensions_array / 2.0
    counts = LiftingCounts(16, 14, 0, 12, 10, 8, 8)
    return ObjectCandidate3D(
        candidate_id=f'image:{detection_id}',
        detection_id=detection_id,
        class_name=class_name,
        detection_confidence=0.85,
        source=GeometrySource.ACCUMULATED,
        source_timestamp_ns=1,
        image_timestamp_ns=1,
        scan_timestamp_ns=1,
        pose_timestamp_ns=1,
        pose_mode='interpolated',
        image_scan_delta_ms=0.0,
        pose_before_delta_ms=0.0,
        pose_after_delta_ms=0.0,
        timing_warning=False,
        points_map_xyz=points,
        source_projection_indices=np.arange(8),
        point_centroid_xyz=centre_array,
        aabb_min_xyz=minimum,
        aabb_max_xyz=maximum,
        obb_centre_xyz=centre_array,
        obb_dimensions_xyz=dimensions_array,
        obb_yaw_rad=yaw,
        estimated_yaw_rad=yaw,
        orientation_confidence=confidence,
        geometry_confidence=confidence,
        geometry_status=(
            GeometryStatus.BACKGROUND_CONTAMINATED
            if partial else GeometryStatus.GOOD
        ),
        partial_geometry=partial,
        low_orientation_fallback=False,
        counts=counts,
        confidence_components=ConfidenceComponents(*([confidence] * 9)),
        diagnostics={},
    )


def _observation(
    candidate: ObjectCandidate3D,
    viewpoint: str,
    timestamp_ns: int,
) -> ViewpointObservation:
    return ViewpointObservation(
        viewpoint_id=viewpoint,
        robot_pose_xyz_yaw=np.array([0.0, 0.0, 0.0, 0.0]),
        timestamp_ns=timestamp_ns,
        detection_id=candidate.detection_id,
        point_count=candidate.point_count,
        geometry_confidence=candidate.geometry_confidence,
        visibility='partial' if candidate.partial_geometry else 'full',
        best_crop=np.full((8, 10, 3), timestamp_ns, dtype=np.uint8),
        best_crop_score=min(1.0, 0.5 + timestamp_ns / 100.0),
    )


def _colour(
    probabilities: dict[str, float],
    confidence: float,
    viewpoint: str,
    *,
    status: str = 'good',
) -> ColourEstimate:
    dominant = (
        max(sorted(probabilities), key=probabilities.get)
        if probabilities else None
    )
    return ColourEstimate(
        probabilities,
        dominant,
        confidence,
        250,
        np.array([10.0, 0.8, 0.7]),
        np.array([50.0, 20.0, 30.0]),
        viewpoint,
        f'detection_{viewpoint}',
        status,
    )


def test_lifecycle_query_serialization_and_deterministic_reset() -> None:
    object_map = ObjectMap()
    candidate = _candidate('chair_a', (1.0, 2.0, 0.5))

    instance_id = object_map.add_or_update(
        candidate, _observation(candidate, 'view_a', 1)
    )

    assert instance_id == 0
    assert object_map.get(0).observation_count == 1
    assert len(object_map.active_instances('chair')) == 1
    serialized = object_map.serialize()
    assert serialized[0]['best_crop_shape'] == [8, 10, 3]
    assert 'best_crop' not in serialized[0]
    assert 'fused_points_xyz' not in serialized[0]

    object_map.reset_episode()
    assert object_map.active_instances() == []
    assert object_map.next_instance_id == 0
    assert object_map.add_or_update(
        candidate, _observation(candidate, 'view_a', 1)
    ) == 0


def test_colour_fusion_preserves_strong_evidence_and_invalid_updates() -> None:
    object_map = ObjectMap(ObjectMapConfig(max_colour_history=3))
    candidate = _candidate('chair_a', (1.0, 2.0, 0.5))
    instance_id = object_map.add_or_update(
        candidate, _observation(candidate, 'view_a', 1)
    )
    strong = _colour({'blue': 0.9, 'purple': 0.1}, 0.95, 'view_a')
    poor = _colour({'blue': 0.1, 'purple': 0.9}, 0.15, 'view_b')
    invalid = _colour({}, 0.0, 'view_c', status='too_few_pixels')

    strong_weight = object_map.update_colour(instance_id, strong)
    for _ in range(20):
        object_map.update_colour(
            instance_id,
            poor,
            crop_quality=0.2,
            mask_quality=0.3,
            geometry_support=0.2,
        )
    before_invalid = dict(object_map.get(instance_id).colour_scores)
    assert object_map.update_colour(instance_id, invalid) == 0.0

    record = object_map.record(instance_id)
    assert strong_weight > 0.8
    assert record.instance.colour_scores['blue'] > 0.8
    assert dict(record.instance.colour_scores) == before_invalid
    assert record.best_colour_estimate is strong
    assert len(record.colour_estimates) == 3
    assert record.colour_confidence > 0.0
    serialized = object_map.serialize()[0]
    assert serialized['best_colour_source']['viewpoint_id'] == 'view_a'
    assert serialized['colour_scores']['blue'] > 0.8


def test_colour_evidence_is_bounded_and_same_view_is_capped() -> None:
    object_map = ObjectMap(ObjectMapConfig(max_colour_evidence=2.0))
    candidate = _candidate('chair_a', (1.0, 2.0, 0.5))
    instance_id = object_map.add_or_update(
        candidate, _observation(candidate, 'view_a', 1)
    )
    estimate = _colour({'red': 0.8, 'brown': 0.2}, 1.0, 'view_a')

    weights = [
        object_map.update_colour(instance_id, estimate) for _ in range(10)
    ]

    assert sum(object_map.record(instance_id).colour_evidence.values()) <= 2.0
    assert sum(weights) <= 1.5 + 1.0e-9


def test_same_object_from_three_views_keeps_one_id_and_fuses_history() -> None:
    object_map = ObjectMap()
    ids = []
    for index, centre in enumerate((
        (1.0, 2.0, 0.5),
        (1.03, 1.98, 0.51),
        (0.98, 2.02, 0.49),
    )):
        candidate = _candidate(f'chair_{index}', centre)
        ids.append(object_map.add_or_update(
            candidate,
            _observation(candidate, f'view_{index}', index + 1),
        ))

    assert ids == [0, 0, 0]
    record = object_map.record(0)
    assert record.instance.observation_count == 3
    assert record.source_viewpoint_ids == ('view_0', 'view_1', 'view_2')
    assert record.best_crop_score == 0.53
    assert object_map.fused_point_count <= 24


def test_panorama_overlap_duplicates_share_one_new_instance() -> None:
    object_map = ObjectMap()
    first = _candidate('crop_1', (1.0, 0.0, 0.5))
    second = _candidate('crop_2', (1.02, 0.01, 0.5))

    ids = object_map.add_viewpoint_candidates(
        [first, second],
        [_observation(first, 'keyframe', 1),
         _observation(second, 'keyframe', 1)],
    )

    assert ids == [0, 0]
    assert len(object_map.active_instances()) == 1
    assert object_map.get(0).observation_count == 1
    assert object_map.last_events[1].decision == 'same_keyframe_duplicate'


def test_low_confidence_same_keyframe_fragments_share_one_instance() -> None:
    object_map = ObjectMap()
    first = _candidate(
        'fragment_1', (1.0, 0.0, 0.5),
        dimensions=(0.4, 0.08, 0.5), confidence=0.40,
    )
    second = _candidate(
        'fragment_2', (1.25, 0.0, 0.5),
        dimensions=(0.1, 0.08, 0.4), confidence=0.55,
    )

    ids = object_map.add_viewpoint_candidates(
        [first, second],
        [_observation(first, 'keyframe', 1),
         _observation(second, 'keyframe', 1)],
    )

    assert ids == [0, 0]
    assert object_map.last_events[1].decision == 'same_keyframe_duplicate'


def test_sequential_neighbouring_chairs_do_not_false_merge() -> None:
    object_map = ObjectMap()
    first = _candidate('chair_left', (0.0, 0.0, 0.5))
    second = _candidate('chair_right', (0.72, 0.0, 0.5))

    first_id = object_map.add_or_update(
        first, _observation(first, 'view_a', 1)
    )
    second_id = object_map.add_or_update(
        second, _observation(second, 'view_b', 2)
    )

    assert first_id == 0
    assert second_id == 1


def test_neighbouring_same_class_objects_remain_separate_on_revisit() -> None:
    object_map = ObjectMap()
    first = _candidate('left_a', (0.0, 0.0, 0.5))
    second = _candidate('right_a', (0.72, 0.0, 0.5))
    initial = object_map.add_viewpoint_candidates(
        [first, second],
        [_observation(first, 'view_a', 1),
         _observation(second, 'view_a', 1)],
    )
    first_return = _candidate('left_b', (0.02, -0.01, 0.5))
    second_return = _candidate('right_b', (0.74, 0.01, 0.5))
    revisited = object_map.add_viewpoint_candidates(
        [first_return, second_return],
        [_observation(first_return, 'view_b', 2),
         _observation(second_return, 'view_b', 2)],
    )

    assert initial == [0, 1]
    assert revisited == [0, 1]
    assert [item.observation_count for item in object_map.active_instances()] == [2, 2]


def test_partial_observation_and_compatible_class_do_not_duplicate() -> None:
    object_map = ObjectMap()
    full = _candidate('full', (1.0, 0.0, 0.5), class_name='chair')
    partial = _candidate(
        'partial',
        (1.01, 0.0, 0.5),
        class_name='stool',
        dimensions=(0.35, 0.25, 0.55),
        partial=True,
    )

    assert object_map.add_or_update(
        full, _observation(full, 'view_a', 1)
    ) == 0
    assert object_map.add_or_update(
        partial, _observation(partial, 'view_b', 2)
    ) == 0

    record = object_map.record(0)
    assert record.instance.observation_count == 2
    assert set(record.instance.class_scores) == {'chair', 'stool'}
    assert record.status == 'partially_observed'
    assert record.geometry_confidence < 1.0


def test_yaw_only_affects_association_when_geometry_is_reliable() -> None:
    object_map = ObjectMap()
    first = _candidate('first', (0.0, 0.0, 0.5), yaw=0.0)
    object_map.add_or_update(first, _observation(first, 'view_a', 1))
    reliable = _candidate('reliable', (0.02, 0.0, 0.5), yaw=1.5)
    weak = _candidate(
        'weak', (0.02, 0.0, 0.5), yaw=1.5, confidence=0.30,
    )

    reliable_score = score_candidate_instance(reliable, object_map.get(0))
    weak_score = score_candidate_instance(weak, object_map.get(0))

    assert reliable_score.components['yaw'] < 0.01
    assert weak_score.components['yaw'] is None
    assert weak_score.final_score > reliable_score.final_score


def test_fused_point_memory_is_hard_bounded() -> None:
    object_map = ObjectMap(ObjectMapConfig(
        max_fused_points_per_instance=12,
        max_total_fused_points=12,
    ))
    for index in range(20):
        candidate = _candidate(
            f'chair_{index}', (index * 0.001, 0.0, 0.5)
        )
        object_map.add_or_update(
            candidate, _observation(candidate, f'view_{index}', index + 1)
        )

    assert object_map.fused_point_count <= 12
    assert object_map.get(0).observation_count == 20
