"""ROS 2 composition root for Q-MapNav."""

from collections.abc import Callable
from collections.abc import Iterable

from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from qmapnav.common import TaskSpecification
from qmapnav.language import parse_question
from qmapnav.mission.question_latch import QuestionLatch
from qmapnav.mission.question_latch import QuestionLatchStatus
from qmapnav.navigation import DEFAULT_ARRIVAL_RADIUS
from qmapnav.navigation import SequentialWaypointExecutor
from qmapnav.navigation import Waypoint2D
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class QMapNavNode(Node):
    """Own the ROS lifecycle and composition of Q-MapNav subsystems."""

    def __init__(
        self,
        question_parser: Callable[[str], TaskSpecification] = parse_question,
    ) -> None:
        super().__init__('qmapnav')
        self.declare_parameter('arrival_radius', DEFAULT_ARRIVAL_RADIUS)
        arrival_radius = self.get_parameter('arrival_radius').value

        self._question_parser = question_parser
        self._question_latch = QuestionLatch()
        self._task_specification: TaskSpecification | None = None
        self._waypoint_executor = SequentialWaypointExecutor(arrival_radius)

        self._question_subscription = self.create_subscription(
            String,
            '/challenge_question',
            self._on_question,
            5,
        )
        self._pose_subscription = self.create_subscription(
            Odometry,
            '/state_estimation',
            self._on_pose,
            5,
        )
        self._waypoint_publisher = self.create_publisher(
            Pose2D,
            '/way_point_with_heading',
            5,
        )
        self.get_logger().info('Q-MapNav node initialized')

    @property
    def question_latch(self) -> QuestionLatch:
        """Expose read-only latch state for composition and evaluation."""
        return self._question_latch

    @property
    def task_specification(self) -> TaskSpecification | None:
        """Return the task parsed from the one accepted question."""
        return self._task_specification

    @property
    def waypoint_executor(self) -> SequentialWaypointExecutor:
        """Expose read-only route execution state for composition."""
        return self._waypoint_executor

    def start_route(self, route: Iterable[Waypoint2D]) -> None:
        """Start a route and publish exactly its first active waypoint."""
        first_goal = self._waypoint_executor.start(route)
        self._publish_waypoint(first_goal)

    def _on_question(self, message: String) -> None:
        decision = self._question_latch.offer(message.data)
        if decision.status is QuestionLatchStatus.ACCEPTED:
            self._task_specification = self._question_parser(decision.question)
            self.get_logger().info('Accepted challenge question')
        elif decision.status is QuestionLatchStatus.DUPLICATE:
            self.get_logger().debug('Ignored repeated challenge question')
        elif decision.status is QuestionLatchStatus.CONFLICT:
            self.get_logger().warning(
                'Ignored different question during active episode'
            )

    def _on_pose(self, message: Odometry) -> None:
        position = message.pose.pose.position
        next_goal = self._waypoint_executor.update_pose(position.x, position.y)
        if next_goal is not None:
            self._publish_waypoint(next_goal)

    def _publish_waypoint(self, waypoint: Waypoint2D) -> None:
        message = Pose2D()
        message.x = waypoint.x
        message.y = waypoint.y
        message.theta = waypoint.heading
        self._waypoint_publisher.publish(message)


def main(args: list[str] | None = None) -> None:
    """Run the Q-MapNav ROS node until shutdown."""
    rclpy.init(args=args)
    node = QMapNavNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
