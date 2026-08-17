"""Side-view and top-down diagnostic plots for support hypotheses."""

import json
from pathlib import Path

import cv2
import numpy as np

from qmapnav.reasoning.support_geometry import SupportGeometry
from qmapnav.reasoning.vertical_relations import RelationEvidence


def save_relation_diagnostic(
    output_directory: Path,
    case_id: str,
    subject: SupportGeometry,
    support: SupportGeometry,
    evidence: RelationEvidence,
) -> tuple[Path, Path, Path]:
    """Save side geometry, top footprint overlap, and evidence JSON."""
    output_directory.mkdir(parents=True, exist_ok=True)
    side = np.full((400, 600, 3), 245, dtype=np.uint8)
    z_min = min(subject.bottom_z, support.bottom_z) - 0.1
    z_max = max(subject.top_z, support.top_z) + 0.1
    x_min = min(
        subject.centre_xyz[0] - subject.dimensions_xyz[0] / 2.0,
        support.centre_xyz[0] - support.dimensions_xyz[0] / 2.0,
    ) - 0.1
    x_max = max(
        subject.centre_xyz[0] + subject.dimensions_xyz[0] / 2.0,
        support.centre_xyz[0] + support.dimensions_xyz[0] / 2.0,
    ) + 0.1
    _draw_side_box(
        side, support, x_min, x_max, z_min, z_max, (180, 100, 20)
    )
    _draw_side_box(
        side, subject, x_min, x_max, z_min, z_max, (30, 80, 220)
    )
    cv2.putText(
        side, f'{evidence.relation}: {evidence.confidence:.2f}',
        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1,
        cv2.LINE_AA,
    )
    side_path = output_directory / f'{case_id}_relation_side.png'
    cv2.imwrite(str(side_path), side)
    top = np.full((500, 500, 3), 245, dtype=np.uint8)
    all_points = np.vstack((subject.footprint_xy, support.footprint_xy))
    minimum = np.min(all_points, axis=0)
    maximum = np.max(all_points, axis=0)
    span = np.maximum(maximum - minimum, 0.1)
    scale = 420.0 / float(np.max(span))
    support_px = _polygon_pixels(support.footprint_xy, minimum, scale)
    subject_px = _polygon_pixels(subject.footprint_xy, minimum, scale)
    cv2.fillPoly(top, [support_px], (220, 185, 120))
    cv2.polylines(top, [support_px], True, (130, 70, 10), 2)
    cv2.fillPoly(top, [subject_px], (120, 160, 245))
    cv2.polylines(top, [subject_px], True, (10, 50, 180), 2)
    top_path = output_directory / f'{case_id}_relation_top.png'
    cv2.imwrite(str(top_path), top)
    payload = {
        'relation': evidence.relation,
        'subject_id': evidence.subject_id,
        'anchor_id': evidence.anchor_id,
        'status': evidence.status,
        'confidence': evidence.confidence,
        'vertical_gap_m': evidence.vertical_gap_m,
        'subject_support_overlap': evidence.subject_support_overlap,
        'horizontal_distance_m': evidence.horizontal_distance_m,
        'geometry_confidence': evidence.geometry_confidence,
    }
    json_path = output_directory / f'{case_id}_relation.json'
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8'
    )
    return side_path, top_path, json_path


def _draw_side_box(image, geometry, x_min, x_max, z_min, z_max, colour):
    x_value = geometry.centre_xyz[0]
    half = geometry.dimensions_xyz[0] / 2.0
    left = int(40 + (x_value - half - x_min) / (x_max - x_min) * 520)
    right = int(40 + (x_value + half - x_min) / (x_max - x_min) * 520)
    bottom = int(370 - (geometry.bottom_z - z_min) /
                 (z_max - z_min) * 340)
    top = int(370 - (geometry.top_z - z_min) / (z_max - z_min) * 340)
    cv2.rectangle(image, (left, top), (right, bottom), colour, 2)


def _polygon_pixels(polygon, minimum, scale):
    result = (np.asarray(polygon) - minimum) * scale + 40.0
    result[:, 1] = 500.0 - result[:, 1]
    return np.rint(result).astype(np.int32)


__all__ = ['save_relation_diagnostic']
