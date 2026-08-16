"""Tests for perfect-object task resolution and semantic route planning."""

from math import pi

import pytest

from qmapnav.evaluation import ColourAttribute
from qmapnav.evaluation import OracleObject
from qmapnav.evaluation import OracleRelation
from qmapnav.evaluation import OracleScene
from qmapnav.evaluation.oracle import OracleReasoningError
from qmapnav.evaluation.oracle import solve_numerical
from qmapnav.evaluation.oracle import solve_object_reference
from qmapnav.evaluation.oracle_route_planner import build_planning_grid
from qmapnav.evaluation.oracle_route_planner import OraclePlannerConfig
from qmapnav.evaluation.oracle_route_planner import plan_semantic_route
from qmapnav.evaluation.oracle_route_planner import RoutePlanningError
from qmapnav.language import parse_question
from qmapnav.reasoning import make_approach_region
from qmapnav.reasoning import make_between_gate
from qmapnav.reasoning import make_near_region
from qmapnav.reasoning import object_footprint


def _object(
    object_id: str,
    class_name: str,
    x: float,
    y: float,
    *,
    dimensions: tuple[float, float, float] = (0.5, 0.5, 1.0),
    yaw: float = 0.0,
    colour: str | None = None,
) -> OracleObject:
    colours = ()
    if colour is not None:
        colours = (
            ColourAttribute(
                label=colour,
                rgb=(128, 128, 128),
                proportion=1.0,
            ),
        )
    return OracleObject(
        object_id=object_id,
        class_name=class_name,
        centre_xyz=(x, y, dimensions[2] / 2.0),
        dimensions_xyz=dimensions,
        yaw=yaw,
        colours=colours,
    )


def _scene(
    objects: tuple[OracleObject, ...],
    relations: tuple[OracleRelation, ...] = (),
) -> OracleScene:
    return OracleScene(
        scene_id='test_scene',
        objects=objects,
        relations=relations,
        regions=(),
        questions=(),
    )


def test_object_footprint_preserves_oriented_box_geometry() -> None:
    obj = _object(
        'table',
        'table',
        1.0,
        2.0,
        dimensions=(2.0, 1.0, 0.8),
        yaw=pi / 2.0,
    )

    footprint = object_footprint(obj)

    assert footprint.centre == pytest.approx((1.0, 2.0))
    assert footprint.bounds == pytest.approx((0.5, 1.0, 1.5, 3.0))
    assert footprint.contains((1.0, 2.9))
    assert not footprint.contains((1.6, 2.0))


def test_near_and_approach_regions_exclude_object_interior() -> None:
    obj = _object('sofa', 'sofa', 0.0, 0.0, dimensions=(2.0, 1.0, 1.0))

    near = make_near_region(obj, min_distance=0.5, max_distance=1.5)
    approach = make_approach_region(
        obj,
        minimum_clearance=0.4,
        maximum_distance=1.0,
    )

    assert near.contains((2.0, 0.0))
    assert not near.contains((0.0, 0.0))
    assert not near.contains((3.0, 0.0))
    assert approach.contains((1.8, 0.0))
    assert not approach.contains((1.2, 0.0))


def test_between_gate_reports_clearance_and_invalid_narrow_gap() -> None:
    left = _object('left', 'table', 0.0, 0.0, dimensions=(1.0, 1.0, 1.0))
    right = _object('right', 'table', 2.0, 0.0, dimensions=(1.0, 1.0, 1.0))
    narrow = _object(
        'narrow',
        'table',
        1.4,
        0.0,
        dimensions=(1.0, 1.0, 1.0),
    )

    valid = make_between_gate(left, right, robot_diameter=0.6)
    invalid = make_between_gate(left, narrow, robot_diameter=0.6)

    assert valid.valid
    assert valid.gap_width == pytest.approx(1.0)
    assert valid.region is not None
    assert valid.region.contains((1.0, 0.0))
    assert not invalid.valid
    assert invalid.region is None
    assert 'narrower' in invalid.reason


def test_numerical_solver_propagates_inbound_support_relation() -> None:
    chair_1 = _object('1', 'chair', 0.0, 0.0)
    chair_2 = _object('2', 'chair', 2.0, 0.0)
    pillow = _object('3', 'pillow', 0.0, 0.0)
    scene = _scene(
        (chair_1, chair_2, pillow),
        (OracleRelation('on', '3', ('1',)),),
    )
    task = parse_question('Count the number of chairs with pillows on them.')

    result = solve_numerical(task, scene)

    assert result.count == 1
    assert result.matching_object_ids == ('1',)
    decisions = {item.object_id: item for item in result.candidate_decisions}
    assert decisions['1'].accepted
    assert not decisions['2'].accepted
    assert 'failed on' in decisions['2'].reasons[0]


def test_numerical_solver_filters_dominant_colour() -> None:
    scene = _scene(
        (
            _object('1', 'pillow', 0.0, 0.0, colour='red'),
            _object('2', 'pillow', 1.0, 0.0, colour='blue'),
        )
    )
    task = parse_question('How many red pillows are there?')

    result = solve_numerical(task, scene)

    assert result.count == 1
    assert result.matching_object_ids == ('1',)
    rejected = result.candidate_decisions[1]
    assert rejected.reasons == ('colour is not red',)


def test_colour_solver_recovers_released_maroon_and_dark_rgb_aliases() -> None:
    maroon = OracleObject(
        object_id='1',
        class_name='pillow',
        centre_xyz=(0.0, 0.0, 0.5),
        dimensions_xyz=(0.5, 0.5, 0.5),
        yaw=0.0,
        colours=(ColourAttribute('maroon', (178, 34, 34), 1.0),),
    )
    dark_grey = OracleObject(
        object_id='2',
        class_name='pillow',
        centre_xyz=(1.0, 0.0, 0.5),
        dimensions_xyz=(0.5, 0.5, 0.5),
        yaw=0.0,
        colours=(ColourAttribute('grey', (47, 79, 79), 1.0),),
    )
    scene = _scene((maroon, dark_grey))

    red = solve_numerical(parse_question('How many red pillows are there?'), scene)
    black = solve_numerical(
        parse_question('How many black pillows are there?'),
        scene,
    )

    assert red.matching_object_ids == ('1',)
    assert black.matching_object_ids == ('2',)


def test_object_solver_applies_between_then_closest_ranking() -> None:
    objects = (
        _object('1', 'chair', 0.0, 0.0, colour='orange'),
        _object('2', 'chair', 5.0, 0.0, colour='orange'),
        _object('3', 'table', 4.0, -1.0),
        _object('4', 'sink', 4.0, 1.0),
        _object('5', 'window', 5.2, 0.0),
    )
    scene = _scene(
        objects,
        (OracleRelation('between', '2', ('3', '4')),),
    )
    task = parse_question(
        'Find the orange chair between the table and sink that is closest '
        'to the window.'
    )

    result = solve_object_reference(task, scene)

    assert result.selected_object_id == '2'
    assert result.selected_object == objects[1]
    assert result.confidence_margin == 1.0
    assert result.warnings == ()


def test_object_solver_reports_ambiguity_and_uses_stable_id() -> None:
    scene = _scene(
        (
            _object('10', 'vase', 0.0, 0.0),
            _object('2', 'vase', 1.0, 0.0),
        )
    )

    result = solve_object_reference(parse_question('Find the vase.'), scene)

    assert result.selected_object_id == '2'
    assert result.confidence_margin == 0.0
    assert 'found 2 candidates' in result.warnings[-1]


def test_oracle_solvers_reject_wrong_task_type() -> None:
    scene = _scene((_object('1', 'chair', 0.0, 0.0),))
    task = parse_question('Find the chair.')

    with pytest.raises(OracleReasoningError, match='numerical task'):
        solve_numerical(task, scene)


def test_grid_inflates_objects_and_preserves_free_space() -> None:
    scene = _scene((_object('1', 'table', 1.0, 0.0),))
    config = OraclePlannerConfig(resolution=0.2, robot_radius=0.3)

    grid = build_planning_grid(scene, (0.0, 0.0), config)

    assert not grid.is_free(grid.point_to_cell((1.0, 0.0)))
    assert grid.is_free(grid.point_to_cell((0.0, 0.0)))


def test_semantic_route_selects_regions_in_textual_order() -> None:
    scene = _scene(
        (
            _object('1', 'plant', 2.0, 0.0),
            _object('2', 'table', 4.0, -1.0),
            _object('3', 'table', 4.0, 1.0),
            _object('4', 'window', 6.0, 0.0),
            _object('5', 'sofa', 3.0, 2.5, dimensions=(1.5, 0.8, 1.0)),
        )
    )
    task = parse_question(
        'First go near the plant, then pass between the two tables, avoid '
        'the sofa and stop near the window.'
    )

    plan = plan_semantic_route(task, scene, (0.0, 0.0))

    assert len(plan.required_regions) == 3
    assert [item.region_type for item in plan.required_regions] == [
        'near',
        'between_gate',
        'near',
    ]
    assert plan.resolved_step_object_ids == (
        (0, ('1',)),
        (1, ('2', '3')),
        (2, ('4',)),
    )
    assert [item.region_type for item in plan.forbidden_regions] == [
        'forbidden_near'
    ]
    assert all(
        plan.grid.is_free(plan.grid.point_to_cell(point))
        for point in plan.waypoints_xy
    )
    assert all(
        not plan.forbidden_regions[0].contains(point)
        for point in plan.waypoints_xy
    )


def test_route_planner_rejects_non_instruction_and_missing_candidates() -> None:
    scene = _scene((_object('1', 'chair', 0.0, 0.0),))

    with pytest.raises(RoutePlanningError, match='instruction-following'):
        plan_semantic_route(parse_question('Find the chair.'), scene, (1.0, 0.0))

    missing_task = parse_question('Go to the window.')
    with pytest.raises(RoutePlanningError, match='no oracle object candidates'):
        plan_semantic_route(missing_task, scene, (1.0, 0.0))


def test_route_planner_reports_unreachable_inferred_between_pair() -> None:
    scene = _scene(
        (
            _object('1', 'table', 1.0, 0.0, dimensions=(1.0, 1.0, 1.0)),
            _object('2', 'table', 1.5, 0.0, dimensions=(1.0, 1.0, 1.0)),
            _object('3', 'window', 3.0, 0.0),
        )
    )
    task = parse_question(
        'Pass between the two tables and stop at the window.'
    )

    with pytest.raises(RoutePlanningError, match='route step 0'):
        plan_semantic_route(task, scene, (0.0, 0.0))
