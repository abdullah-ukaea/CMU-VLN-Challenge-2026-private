"""Tests for support-surface search, negative evidence, and small-object mode."""

from math import hypot

from fixtures import add_wall
from fixtures import make_instance
from fixtures import open_grid
import pytest

from qmapnav.exploration import classify_negative_evidence
from qmapnav.exploration import decide_small_object_mode
from qmapnav.exploration import generate_support_surface_viewpoints
from qmapnav.exploration import likely_supports
from qmapnav.exploration import rank_support_surfaces
from qmapnav.exploration import SmallObjectTriggerConfig
from qmapnav.exploration import SupportSearchHistory
from qmapnav.mapping.perceived_geometry import perceived_box


def _support(instance_id, class_name, centre, dimensions=(1.2, 0.8, 0.75)):
    return perceived_box(
        make_instance(instance_id, class_name, centre, dimensions)
    )


def test_likely_supports_are_ordered_and_bounded() -> None:
    assert likely_supports('paper_cup')[0] == 'table'
    assert 'shelf' in likely_supports('book')
    # Unknown classes get no invented prior.
    assert likely_supports('helicopter') == ()


def test_small_object_mode_activates_for_a_known_small_class() -> None:
    mode = decide_small_object_mode('paper_cup')
    assert mode.active is True
    assert mode.target_classes == ('paper_cup',)
    assert mode.detector_threshold_override is not None


def test_small_object_mode_ignores_ordinary_furniture() -> None:
    mode = decide_small_object_mode(
        'chair', detected=True, detection_confidence=0.9,
        box_area_ratio=0.2, projected_points=900,
    )
    assert mode.active is False


def test_small_object_mode_activates_on_measured_sparsity() -> None:
    tiny_box = decide_small_object_mode(
        'chair', detected=True, detection_confidence=0.9,
        box_area_ratio=0.001, projected_points=900,
    )
    assert tiny_box.active is True
    sparse = decide_small_object_mode(
        'chair', detected=True, detection_confidence=0.9,
        box_area_ratio=0.2, projected_points=5,
    )
    assert sparse.active is True
    weak = decide_small_object_mode(
        'chair', detected=True, detection_confidence=0.1,
        box_area_ratio=0.2, projected_points=900,
    )
    assert weak.active is True


def test_small_object_mode_uses_shorter_standoff_for_shelves() -> None:
    shelf = decide_small_object_mode('book', support_class='shelf')
    table = decide_small_object_mode('book', support_class='table')
    assert shelf.preferred_distance_m < table.preferred_distance_m
    assert 1.0 <= shelf.preferred_distance_m <= 1.8
    assert 1.2 <= table.preferred_distance_m <= 2.0


def test_small_object_trigger_config_validates_thresholds() -> None:
    with pytest.raises(ValueError, match='max_box_area_ratio'):
        SmallObjectTriggerConfig(max_box_area_ratio=0.0)
    with pytest.raises(ValueError, match='min_projected_points'):
        SmallObjectTriggerConfig(min_projected_points=-1)


def test_support_viewpoints_face_the_surface_at_close_standoff() -> None:
    grid = open_grid(half_extent=8.0)
    support = _support(3, 'table', (2.0, 0.0, 0.4))
    outcome = generate_support_surface_viewpoints(
        support,
        grid=grid,
        current_pose_xy_yaw=(-2.0, 0.0, 0.0),
    )
    assert outcome.candidates
    for candidate in outcome.candidates:
        x, y, _ = candidate.pose_xy_yaw
        distance = hypot(x - 2.0, y - 0.0)
        # Close enough for a small target, far enough to see the whole top.
        assert 1.0 < distance < 3.0
        assert candidate.source == 'support_surface'
        assert candidate.target_instance_ids == ('3',)


def test_support_viewpoints_reject_a_blocked_side() -> None:
    grid = open_grid(half_extent=8.0)
    support = _support(4, 'table', (0.0, 0.0, 0.4))
    # Wall off everything beyond +x so that side is unusable.
    add_wall(grid, (1.4, -6.0, 8.0, 6.0))
    outcome = generate_support_surface_viewpoints(
        support,
        grid=grid,
        current_pose_xy_yaw=(-3.0, 0.0, 0.0),
    )
    assert outcome.rejected_counts
    for candidate in outcome.candidates:
        assert candidate.pose_xy_yaw[0] < 1.4


def test_supports_rank_by_semantic_affinity_then_distance() -> None:
    near_sofa = _support(1, 'sofa', (1.0, 0.0, 0.4))
    far_table = _support(2, 'table', (6.0, 0.0, 0.4))
    near_desk = _support(3, 'desk', (2.0, 0.0, 0.4))
    ranked = rank_support_surfaces(
        'paper_cup',
        (near_sofa, far_table, near_desk),
        current_pose_xy_yaw=(0.0, 0.0, 0.0),
    )
    # table outranks desk by the lookup order, both outrank the unrelated sofa.
    assert [item.class_name for item in ranked] == ['table', 'desk', 'sofa']


def test_strongly_negative_supports_leave_the_search_order() -> None:
    table = _support(1, 'table', (1.0, 0.0, 0.4))
    desk = _support(2, 'desk', (2.0, 0.0, 0.4))
    history = SupportSearchHistory()
    history.note_observation(
        '1',
        target_class='paper_cup',
        viewpoint_id='vp_0',
        found=False,
        visible_surface_fraction=0.95,
        distance_m=1.4,
    )
    ranked = rank_support_surfaces(
        'paper_cup', (table, desk),
        current_pose_xy_yaw=(0.0, 0.0, 0.0),
        history=history,
    )
    assert [item.object_id for item in ranked] == ['2']


def test_weak_negative_keeps_a_support_in_the_search_order() -> None:
    table = _support(1, 'table', (1.0, 0.0, 0.4))
    history = SupportSearchHistory()
    record = history.note_observation(
        '1',
        target_class='paper_cup',
        viewpoint_id='vp_0',
        found=False,
        visible_surface_fraction=0.2,
        distance_m=5.0,
        occluded=True,
    )
    assert record.last_result == 'weak'
    assert record.exhausted is False
    ranked = rank_support_surfaces(
        'paper_cup', (table,),
        current_pose_xy_yaw=(0.0, 0.0, 0.0),
        history=history,
    )
    assert len(ranked) == 1


def test_negative_evidence_is_graded_by_view_quality() -> None:
    assert classify_negative_evidence(
        visible_surface_fraction=0.95, distance_m=1.2
    ) == 'strong'
    assert classify_negative_evidence(
        visible_surface_fraction=0.95, distance_m=5.0
    ) == 'moderate'
    assert classify_negative_evidence(
        visible_surface_fraction=0.6, distance_m=1.2
    ) == 'moderate'
    assert classify_negative_evidence(
        visible_surface_fraction=0.2, distance_m=1.2
    ) == 'weak'
    # Occlusion can never produce certainty, however close the robot is.
    assert classify_negative_evidence(
        visible_surface_fraction=0.99, distance_m=1.0, occluded=True
    ) == 'weak'


def test_negative_evidence_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match='visible_surface_fraction'):
        classify_negative_evidence(
            visible_surface_fraction=1.5, distance_m=1.0
        )
    with pytest.raises(ValueError, match='distance_m'):
        classify_negative_evidence(
            visible_surface_fraction=0.5, distance_m=-1.0
        )


def test_history_never_downgrades_a_strong_negative() -> None:
    history = SupportSearchHistory()
    history.note_observation(
        '1', target_class='cup', viewpoint_id='vp_0', found=False,
        visible_surface_fraction=0.9, distance_m=1.2,
    )
    record = history.note_observation(
        '1', target_class='cup', viewpoint_id='vp_1', found=False,
        visible_surface_fraction=0.1, distance_m=8.0,
    )
    assert record.last_result == 'strong'
    assert record.viewpoints_tried == ('vp_0', 'vp_1')


def test_finding_the_target_records_no_negative_evidence() -> None:
    history = SupportSearchHistory()
    record = history.note_observation(
        '1', target_class='cup', viewpoint_id='vp_0', found=True,
        visible_surface_fraction=0.9, distance_m=1.2,
    )
    assert record.last_result == 'none'
    assert record.exhausted is False
    assert history.to_dict()['1']['last_result'] == 'none'
