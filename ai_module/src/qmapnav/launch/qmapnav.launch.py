from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package='qmapnav',
                executable='qmapnav_node',
                name='qmapnav',
                output='screen',
            )
        ]
    )
