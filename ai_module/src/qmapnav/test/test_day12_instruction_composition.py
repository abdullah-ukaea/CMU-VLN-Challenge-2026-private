"""Causal regressions for the live Day 11 instruction composition path."""

from collections.abc import Iterator
from types import SimpleNamespace

from day11_helpers import make_candidate
from day11_helpers import make_observation
from day11_helpers import open_grid
import numpy as np
import pytest
from qmapnav.common.decision_trace import InMemoryTraceRecorder
from qmapnav.language import parse_question
from qmapnav.mapping import ObjectMap
from qmapnav.mapping import StructuralMap
from qmapnav.mission import InstructionEpisodeCoordinator
from qmapnav.navigation import SemanticStageState
from rclpy.parameter import Parameter


ROUTE_QUESTION = (
    'Go to the potted plant furthest from the projector screen then stop '
    'at the water cooler near the window.'
)


class _RecordingPublisher:
    """Record published messages without requiring a DDS subscriber."""

    def __init__(self) -> None:
        self.messages = []

    def publish(self, message: object) -> None:
        """Record one outgoing message."""
        self.messages.append(message)


def _question(text: str) -> object:
    from std_msgs.msg import String

    message = String()
    message.data = text
    return message


def _pose(x: float, y: float) -> object:
    from nav_msgs.msg import Odometry

    message = Odometry()
    message.pose.pose.position.x = x
    message.pose.pose.position.y = y
    message.pose.pose.orientation.w = 1.0
    return message


def _add(
    object_map: ObjectMap,
    detection_id: str,
    class_name: str,
    xy: tuple[float, float],
) -> int:
    candidate = make_candidate(
        detection_id,
        (xy[0], xy[1], 0.8),
        class_name=class_name,
        dimensions=(0.6, 0.6, 1.2),
    )
    return object_map.add_or_update(
        candidate,
        make_observation(candidate, f'view_{detection_id}'),
    )


def _relation_scene(object_map: ObjectMap) -> dict[str, int]:
    """Populate two relation-bearing targets and plausible distractors."""
    ids = {
        'screen': _add(object_map, 'screen', 'projector_screen', (0.0, 0.0)),
        'plant_near': _add(
            object_map, 'plant_near', 'potted_plant', (1.0, 0.0)
        ),
        'plant_far': _add(
            object_map, 'plant_far', 'potted_plant', (5.0, 0.0)
        ),
        'window': _add(object_map, 'window', 'window', (-3.0, -3.0)),
        'cooler_near': _add(
            object_map, 'cooler_near', 'water_cooler', (-2.7, -3.0)
        ),
        'cooler_far': _add(
            object_map, 'cooler_far', 'water_cooler', (0.0, 4.0)
        ),
    }
    return ids


def _coordinator_action(object_map: ObjectMap):
    coordinator = InstructionEpisodeCoordinator()
    coordinator.start(parse_question(ROUTE_QUESTION))
    action = coordinator.evaluate(
        object_map,
        StructuralMap(),
        grid=open_grid(half_extent=8.0),
        current_pose_xy_yaw=(0.0, -2.0, 0.0),
        time_remaining_sec=500.0,
    )
    return action


def test_default_stage_resolver_preserves_each_relation_closure() -> None:
    object_map = ObjectMap()
    ids = _relation_scene(object_map)

    action = _coordinator_action(object_map)

    assert action.action == 'route'
    assert [
        item.instance.instance_id for item in action.stage_resolutions
    ] == [ids['plant_far'], ids['cooler_near']]
    assert [item.resolved_instance_id for item in action.plan.stages] == [
        str(ids['plant_far']), str(ids['cooler_near'])
    ]


@pytest.fixture
def live_node(monkeypatch) -> Iterator[tuple[object, dict[str, int], object]]:
    """Create a production node with deterministic live map evidence."""
    from qmapnav.mission import node as node_module
    from qmapnav.mission.node import QMapNavNode
    import rclpy

    trace = InMemoryTraceRecorder()
    rclpy.init()
    object_map = ObjectMap()
    ids = _relation_scene(object_map)
    node = QMapNavNode(
        object_map=object_map,
        structural_map=StructuralMap(),
        trace_recorder=trace,
        parameter_overrides=[
            Parameter('instruction_initial_observations', value=1),
        ],
    )
    monkeypatch.setattr(
        node_module,
        'occupancy_from_scan_accumulator',
        lambda *args, **kwargs: open_grid(half_extent=8.0),
    )
    lifting = SimpleNamespace(
        candidates=(),
        results=(),
        ground_estimate=SimpleNamespace(reason='test_evidence'),
    )
    node._lifting_pipeline = SimpleNamespace(
        process=lambda result: lifting
    )
    monkeypatch.setattr(node, '_update_persistent_maps', lambda *args: None)
    monkeypatch.setattr(node, '_save_projection_debug', lambda *args: None)
    node._waypoint_publisher = _RecordingPublisher()
    try:
        yield node, ids, trace
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _projection() -> object:
    """Return the minimum valid upstream projection event for composition."""
    from qmapnav.mapping.lidar_camera_projection import ProjectionDiagnostics
    from qmapnav.mapping.timed_buffers import TimedPanorama

    diagnostics = ProjectionDiagnostics(
        input_point_count=0,
        range_valid_count=0,
        vertical_valid_count=0,
        projected_point_count=0,
        image_scan_delta_ms=0.0,
        pose_mode='exact',
        pose_before_delta_ms=0.0,
        pose_after_delta_ms=0.0,
        timing_warning=False,
    )
    panorama = TimedPanorama(
        image_id='instruction_frame_1',
        timestamp_ns=1,
        frame_id='camera',
        image_rgb=np.zeros((2, 4, 3), dtype=np.uint8),
        receipt_timestamp_ns=1,
    )
    pose = SimpleNamespace(
        position_xyz=np.array([0.0, -2.0, 0.0]),
        orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    return SimpleNamespace(
        panorama=panorama,
        association=SimpleNamespace(pose=pose),
        current=SimpleNamespace(diagnostics=diagnostics),
        accumulated=SimpleNamespace(diagnostics=diagnostics),
    )


def _cause_live_instruction_decision(node: object) -> None:
    """Drive only real upstream callbacks into the instruction composition."""
    node._on_pose(_pose(0.0, -2.0))
    node._on_question(_question(ROUTE_QUESTION))
    node._on_projection_result(_projection())


def test_question_and_map_evidence_causally_publish_relation_aware_stage_a(
    live_node,
) -> None:
    node, ids, trace = live_node

    _cause_live_instruction_decision(node)

    assert node._instruction_state == 'semantic_route_active'
    assert node._instruction_plan.stages[0].resolved_instance_id == str(
        ids['plant_far']
    )
    assert node._instruction_plan.stages[1].resolved_instance_id == str(
        ids['cooler_near']
    )
    assert len(node._waypoint_publisher.messages) == 1
    assert any(
        event.event == 'instruction_route_started' for event in trace.events
    )


def test_stage_b_is_published_only_after_stage_a_semantic_entry(
    live_node,
) -> None:
    node, _, trace = live_node
    _cause_live_instruction_decision(node)
    initial_count = len(node._waypoint_publisher.messages)
    plan = node._instruction_plan

    terminal = plan.stages[1].selected_goal_pose
    node._on_pose(_pose(terminal[0], terminal[1]))
    assert len(node._waypoint_publisher.messages) == initial_count
    assert node._instruction_stage_executor.state is (
        SemanticStageState.EXECUTE_STAGE_A
    )

    stage_a = plan.stages[0].selected_goal_pose
    node._on_pose(_pose(stage_a[0], stage_a[1]))
    assert len(node._waypoint_publisher.messages) == initial_count + 1
    assert node._instruction_stage_executor.state is (
        SemanticStageState.EXECUTE_STAGE_B
    )
    assert any(
        event.event == 'instruction_stage_b_started'
        for event in trace.events
    )

    node._on_pose(_pose(terminal[0], terminal[1]))
    assert node._instruction_state == 'complete'


def test_unresolved_first_stage_commits_bounded_terminal_fallback(
    live_node,
) -> None:
    node, _, _ = live_node
    node.reset_persistent_maps()
    _add(node.object_map, 'fallback_screen', 'projector_screen', (0.0, 0.0))
    _add(node.object_map, 'fallback_window', 'window', (-3.0, -3.0))
    cooler_id = _add(
        node.object_map,
        'fallback_cooler',
        'water_cooler',
        (-2.7, -3.0),
    )
    node._episode_time_limit = 100.0

    _cause_live_instruction_decision(node)

    assert node._instruction_plan.route_status == 'terminal_only'
    assert node._instruction_plan.stages[0].resolved_instance_id == str(
        cooler_id
    )
    assert len(node._waypoint_publisher.messages) == 1


def test_unresolved_terminal_publishes_stage_a_information_route(
    live_node,
) -> None:
    node, _, trace = live_node
    node.reset_persistent_maps()
    _add(node.object_map, 'partial_screen', 'projector_screen', (0.0, 0.0))
    _add(node.object_map, 'partial_window', 'window', (-3.0, -3.0))
    _add(node.object_map, 'partial_plant_near', 'potted_plant', (1.0, 0.0))
    plant_id = _add(
        node.object_map,
        'partial_plant_far',
        'potted_plant',
        (5.0, 0.0),
    )

    _cause_live_instruction_decision(node)

    assert node._instruction_plan.route_status == 'stage_a_only'
    assert node._instruction_plan.stages[0].resolved_instance_id == str(
        plant_id
    )
    assert len(node._waypoint_publisher.messages) == 1
    goal = node._instruction_plan.stages[0].selected_goal_pose
    node._on_pose(_pose(goal[0], goal[1]))
    assert node._instruction_state == 'reobservation'
    assert node._instruction_stage_executor is None
    assert any(
        event.event == 'instruction_stage_a_information_complete'
        for event in trace.events
    )


def test_repeated_question_and_map_updates_do_not_recommit_route(
    live_node,
) -> None:
    node, _, trace = live_node
    _cause_live_instruction_decision(node)
    first_count = len(node._waypoint_publisher.messages)

    for _ in range(3):
        node._on_question(_question(ROUTE_QUESTION))
        node._on_projection_result(_projection())

    assert len(node._waypoint_publisher.messages) == first_count
    assert node.question_latch.duplicate_count == 3
    assert sum(
        event.event == 'instruction_route_started' for event in trace.events
    ) == 1
