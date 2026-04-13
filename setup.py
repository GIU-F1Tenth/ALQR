from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'lqr_controller'

lqr_packages = find_packages(include=['lqr_controller', 'lqr_controller.*'])
horizon_mapper_packages = find_packages(
    where='path_planner',
    include=['horizon_mapper', 'horizon_mapper.*'],
)

setup(
    name=package_name,
    version='1.0.0',
    packages=lqr_packages + horizon_mapper_packages,
    package_dir={
        'horizon_mapper': 'path_planner/horizon_mapper',
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'path_planner', 'launch'), glob('path_planner/launch/*.launch.py')),
        (os.path.join('share', package_name, 'path_planner', 'config'), glob('path_planner/config/*.yaml')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*.py')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*.sh')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*.ini')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*.txt')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/README.md')),
    ],
    install_requires=['setuptools', 'numpy', 'scipy', 'pyyaml', 'matplotlib'],
    zip_safe=True,
    maintainer='Mohammed Azab',
    maintainer_email='mohammed@azab.io',
    description='Adaptive Linear Quadratic Regulator (LQR) controller for F1TENTH autonomous racing with real-time parameter tuning and GUI visualization.',
    license='MIT',
    extras_require={
        'test': ['pytest', 'pytest-cov'],
    },
    entry_points={
        'console_scripts': [
            'lqr_node = lqr_controller.lqr_node:main',
            'horizon_mapper_node = horizon_mapper.horizon_mapper_node:main',
        ],
    },
)
