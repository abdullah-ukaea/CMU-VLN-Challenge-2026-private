"""Tests for explicit path-level between gates and physical rejection."""

from day9_helpers import geometry
from qmapnav.reasoning.corridor_evaluation import CorridorConfig
from qmapnav.reasoning.corridor_evaluation import evaluate_corridor
from qmapnav.reasoning.corridor_evaluation import rank_corridors
from qmapnav.reasoning.route_planner import PlanningGrid


def _grid(occupied=()):
    return PlanningGrid(0.1, (-5.0, -5.0), 100, 100, frozenset(occupied))


def _table(entity_id, x, y=0.0):
    return geometry(
        entity_id,
        x,
        y,
        length=0.5,
        width=0.8,
        semantic_class='table',
    )


def test_clear_pair_yields_explicit_traversable_gate():
    result = evaluate_corridor(
        _table('table_1', -1.0),
        _table('table_2', 1.0),
        _grid(),
        CorridorConfig(robot_width_m=0.5),
    )
    assert result.pair.traversable
    assert result.gate is not None
    assert result.gate.centre_xy == (0.0, 0.0)
    assert result.approach_reachable and result.exit_reachable


def test_narrow_gate_is_rejected_but_pair_geometry_remains():
    result = evaluate_corridor(
        _table('table_1', -0.35),
        _table('table_2', 0.35),
        _grid(),
        CorridorConfig(robot_width_m=0.5),
    )
    assert not result.pair.traversable
    assert result.gate is not None
    assert result.pair.footprint_gap_m > 0.0
    assert 'corridor_too_narrow_for_robot_clearance' in result.reasons


def test_occupied_gate_is_rejected():
    occupied = {
        (x_index, y_index)
        for x_index in range(47, 53)
        for y_index in range(47, 53)
    }
    result = evaluate_corridor(
        _table('table_1', -1.0),
        _table('table_2', 1.0),
        _grid(occupied),
        CorridorConfig(robot_width_m=0.5),
    )
    assert not result.pair.traversable
    assert 'corridor_occupancy_below_free_threshold' in result.reasons


def test_approach_and_exit_are_checked_separately():
    # Gate centre is (0, 0), crossing is Y, approach sample is around y=-0.4.
    result = evaluate_corridor(
        _table('table_1', -1.0),
        _table('table_2', 1.0),
        _grid({(50, 45)}),
        CorridorConfig(robot_width_m=0.5),
    )
    assert not result.approach_reachable
    assert result.exit_reachable
    assert 'approach_region_unreachable' in result.reasons


def test_third_object_blocking_gap_is_recorded():
    blocker = geometry(
        'box_3', 0.0, 0.0, length=0.2, width=0.2,
        semantic_class='box',
    )
    result = evaluate_corridor(
        _table('table_1', -1.0),
        _table('table_2', 1.0),
        _grid(),
        CorridorConfig(robot_width_m=0.5),
        blockers=(blocker,),
    )
    assert not result.pair.traversable
    assert result.blocker_ids == ('box_3',)


def test_traversable_gate_ranks_ahead_of_rejected_pair():
    config = CorridorConfig(robot_width_m=0.5)
    clear = evaluate_corridor(
        _table('table_1', -1.0), _table('table_2', 1.0), _grid(), config
    )
    narrow = evaluate_corridor(
        _table('table_3', -0.35), _table('table_4', 0.35), _grid(), config
    )
    assert rank_corridors((narrow, clear))[0].pair.traversable
