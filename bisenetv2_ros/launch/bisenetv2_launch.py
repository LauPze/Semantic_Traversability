from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='bisenetv2_ros',
            executable='bisenetv2_node',
            name='bisenetv2_node',
            output='screen',
            parameters=[],
        )
    ])
