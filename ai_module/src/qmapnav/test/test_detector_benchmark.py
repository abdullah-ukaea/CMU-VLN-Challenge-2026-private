"""Tests for the bounded detector interface and prediction-only bake-off."""

from dataclasses import dataclass, field

import numpy as np
import pytest

from qmapnav.evaluation import DetectorBenchmarkCase
from qmapnav.evaluation import TwoCandidateDetectorBenchmark
from qmapnav.perception import CropDetection
from qmapnav.perception import DetectorClass
from qmapnav.perception import DetectorIdentity
from qmapnav.perception import eight_view_layout
from qmapnav.perception import PanoramaCameraModel
from qmapnav.perception import PerspectiveCropGenerator
from qmapnav.perception.detector_interface import flatten_detector_prompts


@dataclass
class _FakeDetector:
    name: str
    output_class: str = 'chair'
    calls: list[tuple[int, tuple[str, ...], float]] = field(default_factory=list)

    @property
    def identity(self) -> DetectorIdentity:
        return DetectorIdentity(self.name, 'fake', 'none', 'test')

    def detect(
        self,
        view,
        detector_classes,
        *,
        confidence_threshold,
    ):
        self.calls.append(
            (
                view.geometry.crop_id,
                tuple(item.canonical_name for item in detector_classes),
                confidence_threshold,
            )
        )
        return (
            CropDetection(
                crop_id=view.geometry.crop_id,
                canonical_name=self.output_class,
                prompt_used=self.output_class,
                confidence=0.8,
                bbox_xyxy=(1.0, 2.0, 5.0, 6.0),
            ),
        )


def _case() -> DetectorBenchmarkCase:
    model = PanoramaCameraModel(64, 24)
    layout = eight_view_layout(output_width=8, output_height=8)
    panorama = np.zeros((24, 64, 3), dtype=np.uint8)
    views = PerspectiveCropGenerator(model, layout).generate(
        panorama,
        source_image_id='frame_1',
    )
    return DetectorBenchmarkCase(
        image_id='frame_1',
        views=views,
        detector_classes=(
            DetectorClass('chair', ('chair', 'seat')),
            DetectorClass('table', ('table',)),
        ),
    )


def test_two_candidates_receive_identical_views_vocabulary_and_threshold() -> None:
    first = _FakeDetector('candidate_a')
    second = _FakeDetector('candidate_b')
    benchmark = TwoCandidateDetectorBenchmark((first, second))

    results = benchmark.run_case(_case(), confidence_threshold=0.2)

    assert benchmark.candidate_names == ('candidate_a', 'candidate_b')
    assert [item.candidate_name for item in results] == [
        'candidate_a',
        'candidate_b',
    ]
    assert [item.detection_count for item in results] == [8, 8]
    assert first.calls == second.calls
    assert first.calls == [
        (crop_id, ('chair', 'table'), 0.2) for crop_id in range(8)
    ]


def test_benchmark_refuses_zero_or_more_than_two_candidates() -> None:
    with pytest.raises(ValueError, match='one or two'):
        TwoCandidateDetectorBenchmark(())
    with pytest.raises(ValueError, match='one or two'):
        TwoCandidateDetectorBenchmark(
            (_FakeDetector('a'), _FakeDetector('b'), _FakeDetector('c'))
        )


def test_benchmark_refuses_duplicate_candidate_names() -> None:
    with pytest.raises(ValueError, match='unique'):
        TwoCandidateDetectorBenchmark(
            (_FakeDetector('same'), _FakeDetector('same'))
        )


def test_benchmark_rejects_unrequested_adapter_class() -> None:
    benchmark = TwoCandidateDetectorBenchmark(
        (_FakeDetector('bad', output_class='window'),)
    )

    with pytest.raises(ValueError, match='not requested'):
        benchmark.run_case(_case(), confidence_threshold=0.2)


def test_prompt_flattening_preserves_alias_order_and_canonical_mapping() -> None:
    prompts, mapping = flatten_detector_prompts(
        (
            DetectorClass('sofa', ('sofa', 'couch')),
            DetectorClass('trash_can', ('trash can', 'garbage bin')),
        )
    )

    assert prompts == ('sofa', 'couch', 'trash can', 'garbage bin')
    assert mapping == {
        'sofa': 'sofa',
        'couch': 'sofa',
        'trash can': 'trash_can',
        'garbage bin': 'trash_can',
    }


def test_prompt_cannot_map_to_multiple_canonical_classes() -> None:
    with pytest.raises(ValueError, match='multiple canonical'):
        flatten_detector_prompts(
            (
                DetectorClass('display', ('screen',)),
                DetectorClass('monitor', ('screen',)),
            )
        )
