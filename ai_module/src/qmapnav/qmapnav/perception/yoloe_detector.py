"""Lazy Ultralytics YOLOE adapter for perception candidate A."""

from contextlib import chdir
from importlib import import_module
from pathlib import Path

import numpy as np

from qmapnav.perception.contracts import CropDetection
from qmapnav.perception.contracts import DetectorClass
from qmapnav.perception.contracts import PerspectiveView
from qmapnav.perception.detector_interface import canonical_for_text_label
from qmapnav.perception.detector_interface import DetectorIdentity
from qmapnav.perception.detector_interface import flatten_detector_prompts
from qmapnav.perception.detector_interface import immutable_timing
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
        self._asset_directory = (
            Path(self._checkpoint).expanduser().resolve().parent
        )
        self._device = device
        self._image_size = int(image_size)
        self._half_precision = bool(half_precision)
        self._model = yoloe_class(self._checkpoint)
        self._active_prompts: tuple[str, ...] = ()
        self._last_timing_ms = immutable_timing(
            {'preprocess': 0.0, 'inference': 0.0, 'postprocess': 0.0}
        )
        self._identity = DetectorIdentity(
            candidate_name='compact_yoloe',
            framework='ultralytics',
            checkpoint=self._checkpoint,
            version=str(getattr(ultralytics, '__version__', 'unknown')),
            device=device,
            precision='fp16' if half_precision else 'fp32',
            input_size=f'{image_size}x{image_size}',
        )

    @property
    def identity(self) -> DetectorIdentity:
        """Return the loaded YOLOE identity."""
        return self._identity

    @property
    def last_timing_ms(self):
        """Return the latest Ultralytics per-crop timing breakdown."""
        return self._last_timing_ms

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
            with chdir(self._asset_directory):
                try:
                    self._model.set_classes(list(prompts))
                except TypeError as error:
                    if 'embeddings' not in str(error):
                        raise
                    embeddings = self._get_text_embeddings(prompts)
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
        speed = result.speed or {}
        self._last_timing_ms = immutable_timing(
            {
                'preprocess': speed.get('preprocess', 0.0),
                'inference': speed.get('inference', 0.0),
                'postprocess': speed.get('postprocess', 0.0),
            }
        )
        if result.boxes is None:
            return ()
        names = result.names
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        mask_polygons = (
            tuple(result.masks.xy)
            if result.masks is not None
            else tuple(None for _ in range(len(boxes)))
        )
        if len(mask_polygons) != len(boxes):
            mask_polygons = tuple(None for _ in range(len(boxes)))
        detections = []
        for box, score, class_id, mask_polygon in zip(
            boxes,
            scores,
            class_ids,
            mask_polygons,
        ):
            label = names[class_id] if isinstance(names, dict) else names[class_id]
            resolved = canonical_for_text_label(str(label), prompt_to_canonical)
            if resolved is None:
                continue
            canonical_name, normalized_prompt = resolved
            clipped = _clip_box(box, view.geometry.width, view.geometry.height)
            if clipped is None:
                continue
            metadata = {'candidate': self.identity.candidate_name}
            polygon = _mask_polygon_tuple(mask_polygon)
            if polygon:
                metadata['mask_polygon_crop_xy'] = polygon
            detections.append(
                CropDetection(
                    crop_id=view.geometry.crop_id,
                    canonical_name=canonical_name,
                    prompt_used=normalized_prompt,
                    confidence=float(score),
                    bbox_xyxy=clipped,
                    metadata=metadata,
                )
            )
        return validate_candidate_detections(
            tuple(detections),
            view,
            detector_classes,
        )

    def _get_text_embeddings(self, prompts: tuple[str, ...]):
        """Project MobileCLIP features using the detector head's active dtype."""
        model = self._model.model
        try:
            raw_features = model.get_text_pe(
                list(prompts),
                without_reprta=True,
            )
            head = model.model[-1]
            head_dtype = next(head.parameters()).dtype
            torch = import_module('torch')
            with torch.inference_mode():
                return head.get_tpe(raw_features.to(dtype=head_dtype))
        except (AttributeError, TypeError):
            return self._model.get_text_pe(list(prompts))


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


def _mask_polygon_tuple(
    polygon: np.ndarray | None,
) -> tuple[tuple[float, float], ...]:
    """Convert one optional Ultralytics polygon into immutable coordinates."""
    if polygon is None:
        return ()
    points = np.asarray(polygon, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[0] < 3
        or points.shape[1] != 2
        or not np.all(np.isfinite(points))
    ):
        return ()
    return tuple((float(point[0]), float(point[1])) for point in points)
