from setuptools import setup

package_name = 'kitti_player'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/kitti_player_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Laura PAUZIE',
    maintainer_email='laura.pauzie@student.isae-supaero.com',
    description='Node to play KITTI dataset (LiDAR + images) for ROS2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'kitti_player_node = kitti_player.kitti_player_node:main',
        ],
    },
)


