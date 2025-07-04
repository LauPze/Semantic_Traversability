from setuptools import setup, find_packages
import os

package_name = 'bisenetv2_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(include=[package_name, package_name + '.*']),
    data_files=[
        # Installation du modèle TorchScript (.pt), adapte le nom si besoin
        ('share/' + package_name + '/models', ['models/bisenetv2_scripted.pt']),
        ('share/' + package_name + '/launch', ['launch/bisenetv2_launch.py']),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Laura PAUZIE',
    maintainer_email='laura.pauzie@student.isae-supaero.com',
    description='Node ROS 2 pour BiSeNetV2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bisenetv2_node = bisenetv2_ros.bisenetv2_node:main',
        ],
    },
)

