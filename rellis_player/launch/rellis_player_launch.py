import os
from launch import LaunchDescription
from launch_ros.actions import Node

RELLIS_BASE = "/ros2_ws/data/rellis"
VELODYNE_PATH = os.path.join(KITTI_BASE, "vel_cloud_node")
IMAGE_PATH = os.path.join(KITTI_BASE, "image_raw")  

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rellis_player',
            executable='rellis_player_node',
            output='screen',
            parameters=[
                {'rellis_velodyne_path': VELODYNE_PATH},
                {'rellis_image_path': IMAGE_PATH},
            ]
        )
    ])
