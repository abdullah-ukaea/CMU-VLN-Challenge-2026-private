"""Run Day 9 reasoning diagnostics on released Office 1 annotations."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from qmapnav.evaluation.dataset_loader import load_oracle_scene
from qmapnav.evaluation.dataset_loader import load_questions
from qmapnav.mapping.geometry_evaluation import upright_box_corners_xy
from qmapnav.reasoning.corridor_evaluation import CorridorConfig
from qmapnav.reasoning.corridor_evaluation import evaluate_corridor
from qmapnav.reasoning.corridor_evaluation import rank_corridors
from qmapnav.reasoning.reasoning_visualisation import save_reasoning_diagnostics
from qmapnav.reasoning.reference_resolver import resolve_distance_ranking
from qmapnav.reasoning.route_planner import build_planning_grid
from qmapnav.reasoning.route_planner import OraclePlannerConfig
from qmapnav.reasoning.spatial_relations import evaluate_near
from qmapnav.reasoning.spatial_relations import rank_distances
from qmapnav.reasoning.support_geometry import SupportGeometry


_NON_BLOCKERS = frozenset({
    'ceiling', 'door', 'door_frame', 'floor', 'window'
})


def evaluate_office1(
    questions_path: Path,
    simulation_root: Path,
    output_directory: Path,
    *,
    robot_width_m: float = 0.55,
) -> dict[str, object]:
    """Evaluate real annotated distractors and save reconstructable evidence."""
    questions = load_questions(questions_path)
    scene = load_oracle_scene(
        'office_1',
        questions=questions,
        simulation_root=simulation_root,
        require_vla_annotations=False,
    )
    entities = tuple(_geometry(item) for item in scene.objects)
    chairs = tuple(
        item for item in entities if item.semantic_class == 'chair'
    )
    tables = tuple(
        item for item in entities if item.semantic_class == 'table'
    )
    windows = tuple(
        item for item in entities if item.semantic_class == 'window'
    )
    if len(chairs) < 2 or len(tables) < 2 or not windows:
        raise RuntimeError('Office 1 lacks required Day 9 distractors')

    closest = rank_distances(chairs, windows, 'closest')
    farthest = rank_distances(chairs, windows, 'farthest')
    assessment = resolve_distance_ranking('office1_chair_window', closest)
    planner_config = OraclePlannerConfig(
        robot_radius=robot_width_m / 2.0,
        resolution=0.10,
    )
    grid = build_planning_grid(scene, (0.0, 0.0), planner_config)
    blockers = tuple(
        item for item in entities
        if item.semantic_class not in _NON_BLOCKERS
    )
    corridor_config = CorridorConfig(robot_width_m=robot_width_m)
    corridors = rank_corridors(tuple(
        evaluate_corridor(
            first,
            second,
            grid,
            corridor_config,
            blockers=blockers,
        )
        for first, second in combinations(tables, 2)
    ))
    near_scores = [
        evaluate_near(first, second).score
        for first, second in combinations(chairs, 2)
    ]
    diagnostics = save_reasoning_diagnostics(
        output_directory,
        entities,
        assessment,
        corridors=corridors,
        distance_ranking=closest,
    )
    report = {
        'scene_id': scene.scene_id,
        'source_object_count': len(scene.objects),
        'chair_candidate_count': len(chairs),
        'window_candidate_count': len(windows),
        'chair_window_combinations': len(closest.ranked),
        'closest': {
            'target_id': closest.ranked[0].target_id,
            'anchor_id': closest.ranked[0].anchor_id,
            'distance_m': closest.ranked[0].distance_m,
            'raw_margin': closest.raw_margin,
            'normalized_margin': closest.normalized_margin,
            'status': assessment.resolution.resolution_status,
        },
        'farthest': {
            'target_id': farthest.ranked[0].target_id,
            'anchor_id': farthest.ranked[0].anchor_id,
            'distance_m': farthest.ranked[0].distance_m,
            'raw_margin': farthest.raw_margin,
            'normalized_margin': farthest.normalized_margin,
        },
        'chair_pair_count': len(near_scores),
        'near_score_range': [min(near_scores), max(near_scores)],
        'table_pair_count': len(corridors),
        'traversable_table_pair_count': sum(
            item.pair.traversable for item in corridors
        ),
        'table_pairs': [item.to_dict() for item in corridors],
        'diagnostics': {
            name: str(value)
            for name, value in diagnostics.__dict__.items()
        },
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / 'office1_reasoning_report.json'
    with report_path.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    return report


def _geometry(obj):
    centre = np.asarray(obj.centre_xyz, dtype=np.float64)
    dimensions = np.asarray(obj.dimensions_xyz, dtype=np.float64)
    footprint = upright_box_corners_xy(centre, dimensions, obj.yaw)
    return SupportGeometry(
        obj.object_id,
        obj.class_name,
        centre,
        dimensions,
        obj.yaw,
        footprint,
        float(centre[2] - dimensions[2] / 2.0),
        float(centre[2] + dimensions[2] / 2.0),
        1.0,
        'oracle',
        'simulator_annotation',
    )


def main() -> None:
    """Run the Office 1 reasoning audit from command-line paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--simulation-root', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--robot-width-m', type=float, default=0.55)
    arguments = parser.parse_args()
    report = evaluate_office1(
        arguments.questions,
        arguments.simulation_root,
        arguments.output_directory,
        robot_width_m=arguments.robot_width_m,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
