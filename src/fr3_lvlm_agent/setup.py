# ~/ws_moveit2/src/fr3_lvlm_agent/setup.py
from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'fr3_lvlm_agent'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files from the launch/ directory
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Include checkpoints recursively if needed, or flat
        (os.path.join('share', package_name, 'checkpoints'), glob(f'{package_name}/checkpoints/*')),
        (os.path.join('share', package_name, 'config'), glob(f'{package_name}/config/*.json')),
    ],
    install_requires=[
        'setuptools',
        'openai>=1.0.0',
        'pillow',
        'numpy',
    ],
    zip_safe=True,
    maintainer='lonnie',
    maintainer_email='lonnie@todo.todo',
    description='Shared perception, reasoning, and UI utilities for the FR3 zero-shot agent',
    license='Apache-2.0',
    extras_require={
        'llava': [
            'transformers>=4.35.0',
            'torch>=2.0.0',
            'accelerate',
        ],
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_info_repub = fr3_lvlm_agent.camera_info_repub:main',
            'lvlm_command_gui = fr3_lvlm_agent.command_gui:main',
        ],
    },
)
