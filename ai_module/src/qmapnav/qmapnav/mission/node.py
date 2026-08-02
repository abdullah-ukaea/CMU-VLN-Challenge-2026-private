"""ROS 2 composition root for Q-MapNav."""

import rclpy
from rclpy.node import Node


class QMapNavNode(Node):
    """Own the ROS lifecycle and composition of Q-MapNav subsystems."""

    def __init__(self) -> None:
        super().__init__('qmapnav')
        self.get_logger().info('Q-MapNav node initialized')


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
