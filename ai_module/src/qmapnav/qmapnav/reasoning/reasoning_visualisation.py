"""Bounded reasoning reasoning tables, traces, and top-down diagnostics."""

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from qmapnav.reasoning.ambiguity import AmbiguityAssessment
from qmapnav.reasoning.corridor_evaluation import CorridorEvaluation
from qmapnav.reasoning.spatial_relations import DistanceRanking
from qmapnav.reasoning.support_geometry import SupportGeometry


@dataclass(frozen=True)
class ReasoningDiagnosticPaths:
    """Files written for one reconstructable reference decision."""

    top_down_png: Path
    candidate_scores_csv: Path
    resolution_json: Path
    corridor_json: Path


def save_reasoning_diagnostics(
    directory: Path,
    entities: Sequence[SupportGeometry],
    assessment: AmbiguityAssessment,
    *,
    corridors: Sequence[CorridorEvaluation] = (),
    distance_ranking: DistanceRanking | None = None,
) -> ReasoningDiagnosticPaths:
    """Save complete ranking evidence and a labelled top-down map view."""
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = ReasoningDiagnosticPaths(
        output / 'top_down_relations.png',
        output / 'candidate_scores.csv',
        output / 'reference_resolution.json',
        output / 'corridor_hypotheses.json',
    )
    _save_score_table(paths.candidate_scores_csv, assessment)
    _write_json(paths.resolution_json, {
        'event': 'reference_resolution',
        **assessment.to_dict(),
    })
    _write_json(paths.corridor_json, {
        'event': 'pair_resolution',
        'pairs_considered': len(corridors),
        'pairs': [item.to_dict() for item in corridors],
    })
    image = _top_down_image(
        entities, assessment, corridors, distance_ranking
    )
    if not cv2.imwrite(str(paths.top_down_png), image):
        raise OSError(f'failed to save {paths.top_down_png}')
    return paths


def _save_score_table(path, assessment):
    rows = []
    for rank, hypothesis in enumerate(
        assessment.resolution.ranked_hypotheses, start=1
    ):
        rows.append({
            'rank': rank,
            'candidate_ids': '|'.join(hypothesis.candidate_ids),
            'score': hypothesis.score,
            'confidence': hypothesis.confidence,
            'satisfied_constraints': '|'.join(
                hypothesis.satisfied_constraints
            ),
            'violated_constraints': '|'.join(
                hypothesis.violated_constraints
            ),
            'unresolved_constraints': '|'.join(
                hypothesis.unresolved_constraints
            ),
            'score_components': json.dumps(
                dict(hypothesis.evidence), sort_keys=True
            ),
            'raw_margin': assessment.raw_margin if rank <= 2 else '',
            'normalized_margin': (
                assessment.normalized_margin if rank <= 2 else ''
            ),
        })
    fieldnames = (
        'rank',
        'candidate_ids',
        'score',
        'confidence',
        'satisfied_constraints',
        'violated_constraints',
        'unresolved_constraints',
        'score_components',
        'raw_margin',
        'normalized_margin',
    )
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, payload):
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def _top_down_image(entities, assessment, corridors, distance_ranking):
    size = 900
    margin = 70
    image = np.full((size, size, 3), 248, dtype=np.uint8)
    points = [point for entity in entities for point in entity.footprint_xy]
    points.extend(
        point
        for item in corridors if item.gate is not None
        for point in item.gate.polygon_xy
    )
    if not points:
        points = [(-1.0, -1.0), (1.0, 1.0)]
    points = np.asarray(points, dtype=np.float64)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    extent = np.maximum(maximum - minimum, 1.0)
    scale = min((size - 2 * margin) / extent[0],
                (size - 2 * margin) / extent[1])

    def pixel(point):
        value = (np.asarray(point) - minimum) * scale + margin
        return int(round(value[0])), size - int(round(value[1]))

    ranked_ids = [
        item.candidate_ids[0]
        for item in assessment.resolution.ranked_hypotheses[:2]
    ]
    by_id = {entity.entity_id: entity for entity in entities}
    for entity in entities:
        polygon = np.asarray(
            [pixel(point) for point in entity.footprint_xy], dtype=np.int32
        )
        rank = ranked_ids.index(entity.entity_id) if (
            entity.entity_id in ranked_ids
        ) else None
        colour = (20, 140, 20) if rank == 0 else (
            (0, 165, 255) if rank == 1 else (120, 120, 120)
        )
        cv2.polylines(image, (polygon,), True, colour, 3)
        cv2.putText(
            image,
            f'{entity.entity_id}:{entity.semantic_class}',
            pixel(entity.centre_xyz[:2]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            colour,
            1,
            cv2.LINE_AA,
        )
    for item in corridors:
        if item.gate is None:
            continue
        polygon = np.asarray(
            [pixel(point) for point in item.gate.polygon_xy], dtype=np.int32
        )
        colour = (20, 170, 20) if item.pair.traversable else (20, 20, 210)
        cv2.polylines(image, (polygon,), True, colour, 2)
        cv2.line(
            image,
            pixel(item.gate.boundary_first_xy),
            pixel(item.gate.boundary_second_xy),
            colour,
            2,
        )
    if distance_ranking is not None:
        for item in distance_ranking.ranked:
            target = by_id.get(item.target_id)
            anchor = by_id.get(item.anchor_id)
            if target is None or anchor is None:
                continue
            cv2.line(
                image,
                pixel(target.centre_xyz[:2]),
                pixel(anchor.centre_xyz[:2]),
                (180, 90, 30),
                1,
            )
    label = (
        f'{assessment.resolution.resolution_status} '
        f'raw={assessment.raw_margin:.3f} '
        f'norm={assessment.normalized_margin:.3f}'
    )
    cv2.putText(
        image,
        label,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    return image


__all__ = ['ReasoningDiagnosticPaths', 'save_reasoning_diagnostics']
