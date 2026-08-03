"""Validated loader for the manually verified Day 4 panorama benchmark."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from qmapnav.evaluation.detector_metrics import VisibleInstance
from qmapnav.perception.contracts import DetectorClass
from qmapnav.perception.contracts import PanoramaBox


@dataclass(frozen=True)
class DetectorDatasetCase:
    """One source panorama and its visible-instance annotations."""

    image_id: str
    scene: str
    image_path: Path
    detector_classes: tuple[DetectorClass, ...]
    instances: tuple[VisibleInstance, ...]


@dataclass(frozen=True)
class DetectorDataset:
    """Versioned detector benchmark manifest and seam-roll policy."""

    schema_version: int
    width: int
    height: int
    matching_iou_threshold: float
    roll_shift_pixels: int
    cases: tuple[DetectorDatasetCase, ...]


def load_detector_dataset(
    manifest_path: Path,
    *,
    width: int = 1920,
    height: int = 640,
) -> DetectorDataset:
    """Load and validate classes, paths, boxes, flags, and category coverage."""
    payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    schema_version = payload.get('schema_version')
    if schema_version != 1:
        raise ValueError('unsupported detector manifest schema version')
    policy = payload.get('annotation_policy', {})
    matching_iou = float(policy.get('matching_iou_threshold', 0.0))
    roll_shift = int(policy.get('roll_shift_pixels', 0))
    if not 0.0 < matching_iou <= 1.0:
        raise ValueError('matching IoU must lie in (0, 1]')
    if not 0 < roll_shift < width:
        raise ValueError('roll shift must lie inside the panorama width')

    prompt_map = payload.get('class_prompts')
    if not isinstance(prompt_map, dict) or not prompt_map:
        raise ValueError('class_prompts must be a non-empty object')
    detector_class_map = {
        name: DetectorClass(name, tuple(prompts))
        for name, prompts in prompt_map.items()
    }
    cases = []
    image_ids = set()
    for case_payload in payload.get('panoramas', ()):
        image_id = str(case_payload.get('image_id', '')).strip()
        scene = str(case_payload.get('scene', '')).strip()
        image_path = Path(case_payload.get('image_path', ''))
        if not image_id or image_id in image_ids or not scene:
            raise ValueError('panorama image IDs must be unique and non-empty')
        image_ids.add(image_id)
        instances = tuple(
            _parse_instance(item, image_id, width, height)
            for item in case_payload.get('annotations', ())
        )
        if not instances:
            raise ValueError(f'{image_id} has no visible-instance annotations')
        class_names = sorted({item.class_name for item in instances})
        unknown = set(class_names) - detector_class_map.keys()
        if unknown:
            raise ValueError(f'{image_id} uses unknown classes: {sorted(unknown)}')
        cases.append(
            DetectorDatasetCase(
                image_id,
                scene,
                image_path,
                tuple(detector_class_map[name] for name in class_names),
                instances,
            )
        )
    if len(cases) < 5 or len({case.scene for case in cases}) < 5:
        raise ValueError('the detector dataset must cover at least five scenes')
    all_instances = tuple(item for case in cases for item in case.instances)
    for attribute in ('is_target', 'is_anchor', 'is_rare'):
        if not any(getattr(item, attribute) for item in all_instances):
            raise ValueError(f'the detector dataset has no {attribute} instances')
    if not any(item.size_bin == 'small' for item in all_instances):
        raise ValueError('the detector dataset has no small instances')
    if not any(item.seam_case for item in all_instances):
        raise ValueError('the detector dataset has no real seam instance')
    return DetectorDataset(
        schema_version,
        width,
        height,
        matching_iou,
        roll_shift,
        tuple(cases),
    )


def roll_visible_instance(
    instance: VisibleInstance,
    shift_pixels: int,
) -> VisibleInstance:
    """Shift one annotation under a horizontal panorama roll."""
    box = instance.panorama_box
    intervals = _shift_intervals(
        box.x_intervals,
        shift_pixels,
        box.panorama_width,
    )
    boundary = box.boundary_uv.copy()
    boundary[:, 0] = np.mod(boundary[:, 0] + shift_pixels, box.panorama_width)
    shifted_box = PanoramaBox(
        box.panorama_width,
        box.panorama_height,
        intervals,
        box.y_min,
        box.y_max,
        boundary,
    )
    return VisibleInstance(
        instance_id=f'{instance.instance_id}_rolled',
        class_name=instance.class_name,
        panorama_box=shifted_box,
        is_target=instance.is_target,
        is_anchor=instance.is_anchor,
        is_rare=instance.is_rare,
        size_bin=instance.size_bin,
        seam_case=shifted_box.crosses_seam,
    )


def _parse_instance(
    payload: dict[str, object],
    image_id: str,
    width: int,
    height: int,
) -> VisibleInstance:
    intervals = tuple(
        (float(interval[0]), float(interval[1]))
        for interval in payload.get('x', ())
    )
    y_values = tuple(float(value) for value in payload.get('y', ()))
    if len(y_values) != 2:
        raise ValueError(f'{image_id} annotation has invalid y bounds')
    boundary = np.array(
        [
            (x, y)
            for start, end in intervals
            for x, y in (
                (start, y_values[0]),
                (end, y_values[0]),
                (end, y_values[1]),
                (start, y_values[1]),
            )
        ],
        dtype=np.float64,
    )
    box = PanoramaBox(
        width,
        height,
        intervals,
        y_values[0],
        y_values[1],
        boundary,
    )
    shorter_side = min(
        sum(end - start for start, end in intervals),
        y_values[1] - y_values[0],
    )
    return VisibleInstance(
        instance_id=f"{image_id}:{payload.get('id', '')}",
        class_name=str(payload.get('class', '')),
        panorama_box=box,
        is_target=bool(payload.get('target', False)),
        is_anchor=bool(payload.get('anchor', False)),
        is_rare=bool(payload.get('rare', False)),
        size_bin='small' if shorter_side < 45.0 else 'medium',
        seam_case=bool(payload.get('seam', False)),
    )


def _shift_intervals(
    intervals: tuple[tuple[float, float], ...],
    shift: int,
    width: int,
) -> tuple[tuple[float, float], ...]:
    pieces = []
    for start, end in intervals:
        shifted_start = (start + shift) % width
        length = end - start
        shifted_end = shifted_start + length
        if shifted_end <= width:
            pieces.append((shifted_start, shifted_end))
        else:
            pieces.extend(((0.0, shifted_end - width), (shifted_start, float(width))))
    pieces.sort()
    merged = []
    for start, end in pieces:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)
