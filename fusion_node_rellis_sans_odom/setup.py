from setuptools import setup
from glob import glob
import os

package_name = 'fusion_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        # Ajouter les fichiers launch
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'data'), glob('data/**/*', recursive=True)),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Laura PAUZIE',
    maintainer_email='laura.pauzie@student.isae-supaero.com',
    description='Fusion Lidar + segmentation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fusion_node = fusion_node.fusion_node:main',
        ],
    },
)

