from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'lqr_controller'

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
    ],
    install_requires=['setuptools', 'numpy', 'scipy', 'pyyaml'],
    zip_safe=True,
    maintainer='Mohammed Azab',
    maintainer_email='mohammed@azab.io',
    description='Linear Quadratic Regulator (LQR) controller for autonomous systems.',
    license='MIT',
    tests_require=['pytest', 'pytest-cov'],
    entry_points={
        'console_scripts': [
            'lqr_node = lqr_controller.lqr_node:main'
        ],
    },
)
