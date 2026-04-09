from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'lqg_controller'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
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
    description='Linear Quadratic Regulator (LQR) and Linear Quadratic Gaussian (LQG) controllers for F1TENTH autonomous racing with state estimation and GUI visualization.',
    license='MIT',
    tests_require=['pytest', 'pytest-cov'],
    entry_points={
        'console_scripts': [
            'lqr_node = lqg_controller.lqr_node:main',
            'lqg_node = lqg_controller.lqg_node:main',
        ],
    },
)
