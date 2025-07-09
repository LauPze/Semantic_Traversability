import os
from launch import LaunchDescription
from launch_ros.actions import Node

KITTI_BASE = "/ros2_ws/data/kitti/2011_09_26/2011_09_26_drive_0001_sync"
VELODYNE_PATH = os.path.join(KITTI_BASE, "velodyne_points/data")
IMAGE_PATH = os.path.join(KITTI_BASE, "image_02/data")  # ou image_03 selon la caméra

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='kitti_player',
            executable='kitti_player_node',
            output='screen',
            parameters=[
                {'kitti_velodyne_path': VELODYNE_PATH},
                {'kitti_image_path': IMAGE_PATH},
            ]
        )
    ])
