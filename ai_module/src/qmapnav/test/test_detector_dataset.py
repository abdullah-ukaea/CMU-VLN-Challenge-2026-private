"""Tests for benchmark manifest validation and seam-roll annotations."""

from pathlib import Path

from qmapnav.evaluation import load_detector_dataset
from qmapnav.evaluation import roll_visible_instance


_MANIFEST = Path(__file__).parents[1] / 'benchmark' / 'day4_detector_manifest.json'


def test_released_scene_manifest_has_required_category_coverage() -> None:
    dataset = load_detector_dataset(_MANIFEST)

    instances = tuple(item for case in dataset.cases for item in case.instances)
    assert len(dataset.cases) == 6
    assert len({case.scene for case in dataset.cases}) == 6
    assert len(instances) >= 50
    assert any(item.is_target for item in instances)
    assert any(item.is_anchor for item in instances)
    assert any(item.is_rare for item in instances)
    assert any(item.size_bin == 'small' for item in instances)
    assert any(item.seam_case for item in instances)


def test_roll_moves_real_seam_box_to_one_contiguous_interval() -> None:
    dataset = load_detector_dataset(_MANIFEST)
    seam = next(
        item
        for case in dataset.cases
        for item in case.instances
        if item.seam_case
    )

    rolled = roll_visible_instance(seam, dataset.roll_shift_pixels)

    assert not rolled.panorama_box.crosses_seam
    assert rolled.panorama_box.x_intervals == ((584.0, 666.0),)
