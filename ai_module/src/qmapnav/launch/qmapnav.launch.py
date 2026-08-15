from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    config = (
        Path(get_package_share_directory('qmapnav'))
        / 'configs'
        / 'submission_v1.yaml'
    )
    return LaunchDescription(
        [
            Node(
                package='qmapnav',
                executable='qmapnav_node',
                name='qmapnav',
                output='screen',
                parameters=[str(config)],
            )
        ]
    )
