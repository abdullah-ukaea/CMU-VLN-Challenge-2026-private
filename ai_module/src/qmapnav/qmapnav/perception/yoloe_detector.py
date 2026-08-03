"""Lazy Ultralytics YOLOE adapter for Day 4 candidate A."""

from importlib import import_module
from pathlib import Path

import numpy as np

from qmapnav.perception.contracts import CropDetection
from qmapnav.perception.contracts import DetectorClass
from qmapnav.perception.contracts import PerspectiveView
from qmapnav.perception.detector_interface import canonical_for_text_label
from qmapnav.perception.detector_interface import DetectorIdentity
from qmapnav.perception.detector_interface import flatten_detector_prompts
from qmapnav.perception.detector_interface import validate_candidate_detections


class DetectorDependencyError(RuntimeError):
    """Raised when an optional detector runtime is unavailable."""


class YOLOEDetector:
    """Compact text-prompted YOLOE candidate returning crop-local boxes."""

    def __init__(
        self,
        checkpoint: str | Path = 'yoloe-11s-seg.pt',
        *,
        device: str = 'cuda:0',
        image_size: int = 640,
        half_precision: bool = True,
    ) -> None:
        try:
            ultralytics = import_module('ultralytics')
            yoloe_class = getattr(ultralytics, 'YOLOE')
        except (ImportError, AttributeError) as error:
            raise DetectorDependencyError(
                'YOLOE requires an Ultralytics release that exports YOLOE'
            ) from error
        if image_size <= 0:
            raise ValueError('image_size must be positive')
        self._checkpoint = str(checkpoint)
        self._device = device
        self._image_size = int(image_size)
        self._half_precision = bool(half_precision)
        self._model = yoloe_class(self._checkpoint)
        self._active_prompts: tuple[str, ...] = ()
        self._identity = DetectorIdentity(
            candidate_name='compact_yoloe',
            framework='ultralytics',
            checkpoint=self._checkpoint,
            version=str(getattr(ultralytics, '__version__', 'unknown')),
        )

    @property
    def identity(self) -> DetectorIdentity:
        """Return the loaded YOLOE identity."""
        return self._identity

    def detect(
        self,
        view: PerspectiveView,
        detector_classes: tuple[DetectorClass, ...],
        *,
        confidence_threshold: float,
    ) -> tuple[CropDetection, ...]:
        """Run YOLOE on one RGB crop and normalize its boxes."""
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError('confidence_threshold must lie in [0, 1]')
        prompts, prompt_to_canonical = flatten_detector_prompts(detector_classes)
        if prompts != self._active_prompts:
            try:
                self._model.set_classes(list(prompts))
            except TypeError as error:
                if 'embeddings' not in str(error):
                    raise
                embeddings = self._model.get_text_pe(list(prompts))
                self._model.set_classes(list(prompts), embeddings)
            self._active_prompts = prompts

        bgr = np.ascontiguousarray(view.image_rgb[..., ::-1])
        results = self._model.predict(
            source=bgr,
            conf=confidence_threshold,
            imgsz=self._image_size,
            device=self._device,
            half=self._half_precision,
            agnostic_nms=False,
            verbose=False,
        )
        if len(results) != 1:
            raise RuntimeError('YOLOE must return exactly one result per crop')
        result = results[0]
        if result.boxes is None:
            return ()
        names = result.names
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        detections = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            label = names[class_id] if isinstance(names, dict) else names[class_id]
            resolved = canonical_for_text_label(str(label), prompt_to_canonical)
            if resolved is None:
                continue
            canonical_name, normalized_prompt = resolved
            clipped = _clip_box(box, view.geometry.width, view.geometry.height)
            if clipped is None:
                continue
            detections.append(
                CropDetection(
                    crop_id=view.geometry.crop_id,
                    canonical_name=canonical_name,
                    prompt_used=normalized_prompt,
                    confidence=float(score),
                    bbox_xyxy=clipped,
                    metadata={'candidate': self.identity.candidate_name},
                )
            )
        return validate_candidate_detections(
            tuple(detections),
            view,
            detector_classes,
        )


def _clip_box(
    box: np.ndarray,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    values = np.asarray(box, dtype=np.float64).reshape(-1)
    if values.size != 4 or not np.all(np.isfinite(values)):
        return None
    x_min = float(np.clip(values[0], 0.0, width))
    y_min = float(np.clip(values[1], 0.0, height))
    x_max = float(np.clip(values[2], 0.0, width))
    y_max = float(np.clip(values[3], 0.0, height))
    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max
