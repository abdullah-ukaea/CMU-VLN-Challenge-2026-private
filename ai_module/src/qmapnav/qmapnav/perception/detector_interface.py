"""Common open-vocabulary detector boundary for the two-candidate bake-off."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
from typing import Protocol

import numpy as np

from qmapnav.perception.contracts import CropDetection
from qmapnav.perception.contracts import DetectorClass
from qmapnav.perception.contracts import PerspectiveView


@dataclass(frozen=True)
class DetectorIdentity:
    """Pinned identity reported by one detector adapter."""

    candidate_name: str
    framework: str
    checkpoint: str
    version: str
    device: str = 'unknown'
    precision: str = 'unknown'
    input_size: str = 'dynamic'

    def __post_init__(self) -> None:
        for name, value in (
            ('candidate_name', self.candidate_name),
            ('framework', self.framework),
            ('checkpoint', self.checkpoint),
            ('version', self.version),
            ('device', self.device),
            ('precision', self.precision),
            ('input_size', self.input_size),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{name} must be a non-empty string')


class OpenVocabularyDetector(Protocol):
    """Detector adapter that returns only model-independent crop boxes."""

    @property
    def identity(self) -> DetectorIdentity:
        """Return the candidate and model identity."""
        ...

    @property
    def last_timing_ms(self) -> Mapping[str, float]:
        """Return the latest preprocessing, inference, and postprocess timing."""
        ...

    def detect(
        self,
        view: PerspectiveView,
        detector_classes: tuple[DetectorClass, ...],
        *,
        confidence_threshold: float,
    ) -> tuple[CropDetection, ...]:
        """Detect requested classes in one perspective view."""
        ...


def flatten_detector_prompts(
    detector_classes: tuple[DetectorClass, ...],
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Flatten aliases and return a casefolded prompt-to-canonical lookup."""
    classes = tuple(detector_classes)
    if not classes or not all(isinstance(item, DetectorClass) for item in classes):
        raise ValueError('detector_classes must contain DetectorClass values')
    prompts = []
    prompt_to_canonical = {}
    for detector_class in classes:
        for prompt in detector_class.prompts:
            normalized = _normalize_prompt(prompt)
            existing = prompt_to_canonical.get(normalized)
            if existing is not None and existing != detector_class.canonical_name:
                raise ValueError(f'prompt {prompt!r} maps to multiple canonical classes')
            if existing is None:
                prompts.append(prompt)
                prompt_to_canonical[normalized] = detector_class.canonical_name
    return tuple(prompts), prompt_to_canonical


def canonical_for_text_label(
    label: str,
    prompt_to_canonical: dict[str, str],
) -> tuple[str, str] | None:
    """Resolve a detector phrase to its canonical class and measured prompt."""
    normalized_label = _normalize_prompt(label)
    exact = prompt_to_canonical.get(normalized_label)
    if exact is not None:
        return exact, normalized_label
    matching = [
        prompt
        for prompt in prompt_to_canonical
        if prompt in normalized_label or normalized_label in prompt
    ]
    if not matching:
        return None
    prompt = max(matching, key=len)
    return prompt_to_canonical[prompt], prompt


def validate_candidate_detections(
    detections: tuple[CropDetection, ...],
    view: PerspectiveView,
    requested_classes: tuple[DetectorClass, ...],
) -> tuple[CropDetection, ...]:
    """Validate adapter results against the view and requested vocabulary."""
    detections = tuple(detections)
    requested_names = {item.canonical_name for item in requested_classes}
    for detection in detections:
        if not isinstance(detection, CropDetection):
            raise TypeError('detector output must contain only CropDetection values')
        if detection.crop_id != view.geometry.crop_id:
            raise ValueError('detector returned a mismatched crop_id')
        if detection.canonical_name not in requested_names:
            raise ValueError('detector returned a class that was not requested')
        x_min, y_min, x_max, y_max = detection.bbox_xyxy
        if not 0.0 <= x_min < x_max <= view.geometry.width:
            raise ValueError('detector x coordinates lie outside the crop')
        if not 0.0 <= y_min < y_max <= view.geometry.height:
            raise ValueError('detector y coordinates lie outside the crop')
    return detections


def _normalize_prompt(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError('detector labels must be non-empty strings')
    return ' '.join(value.casefold().strip().rstrip('.').split())


def immutable_timing(values: Mapping[str, float]) -> Mapping[str, float]:
    """Validate and freeze one detector adapter timing sample."""
    normalized = {str(name): float(value) for name, value in values.items()}
    if set(normalized) != {'preprocess', 'inference', 'postprocess'}:
        raise ValueError('timing requires preprocess, inference, and postprocess')
    if any(value < 0.0 or not np.isfinite(value) for value in normalized.values()):
        raise ValueError('detector timings must be finite and non-negative')
    return MappingProxyType(normalized)
