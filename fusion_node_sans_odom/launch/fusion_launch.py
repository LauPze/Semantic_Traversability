from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='fusion_node',
            executable='fusion_node',
            name='fusion_node',
            parameters=[{
                'calib_velo_to_cam_path': '/home/Laura/data/kitti/2011_09_26/calib_velo_to_cam.txt',
                'calib_cam_intr_path': '/home/Laura/data/kitti/2011_09_26/calib_cam_to_cam.txt',
            }]
        )
    ])
