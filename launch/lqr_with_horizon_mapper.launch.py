#!/usr/bin/env python3

"""
Launch file for LQR Controller with Horizon Mapper

This launch file starts both the horizon mapper and LQR controller for complete
trajectory tracking functionality.

Author: Mohammed Azab <mohammed@azab.io>
License: MIT
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for LQR controller with horizon mapper."""

    # Package directories
    lqr_pkg_share = FindPackageShare('lqr_controller')
    horizon_pkg_share = FindPackageShare('horizon_mapper')

    # Config file paths
    lqr_config_file = PathJoinSubstitution([
        lqr_pkg_share, 'config', 'lqr_params.yaml'
    ])

    # Launch arguments
    lqr_config_arg = DeclareLaunchArgument(
        'lqr_config_file',
        default_value=lqr_config_file,
        description='Path to LQR controller configuration file'
    )

    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='Enable debug logging'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    # Horizon mapper launch (if available)
    try:
        horizon_mapper_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                horizon_pkg_share, '/launch/horizon_mapper.launch.py'
            ]),
            launch_arguments={
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'debug': LaunchConfiguration('debug')
            }.items()
        )
    except BaseException:
        # If horizon mapper launch file doesn't exist, create a basic node
        horizon_mapper_launch = Node(
            package='horizon_mapper',
            executable='horizon_mapper_node',
            name='horizon_mapper_node',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ],
            output='screen'
        )

    # LQR controller node
    lqr_controller_node = Node(
        package='lqr_controller',
        executable='lqr_node',
        name='lqr_controller_node',
        parameters=[
            LaunchConfiguration('lqr_config_file'),
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'debug_logging_enabled': LaunchConfiguration('debug')
            }
        ],
        output='screen',
        emulate_tty=True,
        respawn=True,
        respawn_delay=2.0
    )

    return LaunchDescription([
        lqr_config_arg,
        debug_arg,
        use_sim_time_arg,
        horizon_mapper_launch,
        lqr_controller_node
    ])
