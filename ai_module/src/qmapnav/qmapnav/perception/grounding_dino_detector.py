"""Lazy Transformers GroundingDINO-Tiny adapter for Day 4 candidate B."""

from importlib import import_module
from pathlib import Path
from time import perf_counter

import numpy as np

from qmapnav.perception.contracts import CropDetection
from qmapnav.perception.contracts import DetectorClass
from qmapnav.perception.contracts import PerspectiveView
from qmapnav.perception.detector_interface import canonical_for_text_label
from qmapnav.perception.detector_interface import DetectorIdentity
from qmapnav.perception.detector_interface import flatten_detector_prompts
from qmapnav.perception.detector_interface import immutable_timing
from qmapnav.perception.detector_interface import validate_candidate_detections
from qmapnav.perception.yoloe_detector import _clip_box
from qmapnav.perception.yoloe_detector import DetectorDependencyError


class GroundingDinoTinyDetector:
    """GroundingDINO-Tiny candidate using the official Transformers model."""

    def __init__(
        self,
        model_name_or_path: str | Path = 'IDEA-Research/grounding-dino-tiny',
        *,
        revision: str = 'a2bb814dd30d776dcf7e30523b00659f4f141c71',
        device: str = 'cuda:0',
        local_files_only: bool = True,
        text_threshold: float = 0.25,
    ) -> None:
        try:
            torch = import_module('torch')
            transformers = import_module('transformers')
            auto_processor = getattr(transformers, 'AutoProcessor')
            auto_model = getattr(
                transformers,
                'AutoModelForZeroShotObjectDetection',
            )
        except (ImportError, AttributeError) as error:
            raise DetectorDependencyError(
                'GroundingDINO-Tiny requires PyTorch and Transformers'
            ) from error
        if not 0.0 <= text_threshold <= 1.0:
            raise ValueError('text_threshold must lie in [0, 1]')
        self._torch = torch
        self._device = device
        self._text_threshold = float(text_threshold)
        self._model_name_or_path = str(model_name_or_path)
        load_arguments = {
            'revision': revision,
            'local_files_only': local_files_only,
        }
        self._processor = auto_processor.from_pretrained(
            self._model_name_or_path,
            **load_arguments,
        )
        self._model = auto_model.from_pretrained(
            self._model_name_or_path,
            **load_arguments,
        ).to(device)
        self._model.eval()
        self._last_timing_ms = immutable_timing(
            {'preprocess': 0.0, 'inference': 0.0, 'postprocess': 0.0}
        )
        self._identity = DetectorIdentity(
            candidate_name='grounding_dino_tiny',
            framework='transformers',
            checkpoint=f'{self._model_name_or_path}@{revision}',
            version=str(getattr(transformers, '__version__', 'unknown')),
            device=device,
            precision='fp32',
            input_size='processor-resized',
        )

    @property
    def identity(self) -> DetectorIdentity:
        """Return the loaded GroundingDINO identity."""
        return self._identity

    @property
    def last_timing_ms(self):
        """Return the latest measured per-crop timing breakdown."""
        return self._last_timing_ms

    def detect(
        self,
        view: PerspectiveView,
        detector_classes: tuple[DetectorClass, ...],
        *,
        confidence_threshold: float,
    ) -> tuple[CropDetection, ...]:
        """Run GroundingDINO-Tiny on one RGB crop and normalize its boxes."""
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError('confidence_threshold must lie in [0, 1]')
        prompts, prompt_to_canonical = flatten_detector_prompts(detector_classes)
        text_labels = [list(prompts)]
        self._synchronize()
        preprocess_start = perf_counter()
        inputs = self._processor(
            images=np.asarray(view.image_rgb),
            text=text_labels,
            return_tensors='pt',
        ).to(self._device)
        self._synchronize()
        inference_start = perf_counter()
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        self._synchronize()
        postprocess_start = perf_counter()
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=confidence_threshold,
            text_threshold=self._text_threshold,
            target_sizes=[(view.geometry.height, view.geometry.width)],
        )
        self._synchronize()
        postprocess_end = perf_counter()
        self._last_timing_ms = immutable_timing(
            {
                'preprocess': (inference_start - preprocess_start) * 1000.0,
                'inference': (postprocess_start - inference_start) * 1000.0,
                'postprocess': (postprocess_end - postprocess_start) * 1000.0,
            }
        )
        if len(results) != 1:
            raise RuntimeError('GroundingDINO must return exactly one crop result')
        result = results[0]
        labels = result.get('text_labels', result.get('labels', ()))
        detections = []
        for box, score, label in zip(result['boxes'], result['scores'], labels):
            if not isinstance(label, str) or not label.strip():
                continue
            resolved = canonical_for_text_label(label, prompt_to_canonical)
            if resolved is None:
                continue
            canonical_name, normalized_prompt = resolved
            clipped = _clip_box(
                box.detach().cpu().numpy(),
                view.geometry.width,
                view.geometry.height,
            )
            if clipped is None:
                continue
            detections.append(
                CropDetection(
                    crop_id=view.geometry.crop_id,
                    canonical_name=canonical_name,
                    prompt_used=normalized_prompt,
                    confidence=float(score.detach().cpu().item()),
                    bbox_xyxy=clipped,
                    metadata={'candidate': self.identity.candidate_name},
                )
            )
        return validate_candidate_detections(
            tuple(detections),
            view,
            detector_classes,
        )

    def _synchronize(self) -> None:
        if self._device.startswith('cuda') and self._torch.cuda.is_available():
            self._torch.cuda.synchronize()
