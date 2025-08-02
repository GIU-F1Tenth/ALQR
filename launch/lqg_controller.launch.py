#!/usr/bin/env python3

"""
Launch file for LQG Controller

This launch file starts the LQG controller node with configurable parameters.

Author: Mohammed Azab <mohammed@azab.io>
License: MIT
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for LQG controller."""

    # Package directory
    pkg_share = FindPackageShare('lqg_controller')

    # Default config file path
    default_config_file = PathJoinSubstitution([
        pkg_share, 'config', 'lqg_params.yaml'
    ])

    # Launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config_file,
        description='Path to LQG controller configuration file'
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

    # LQG controller node
    lqg_controller_node = Node(
        package='lqg_controller',
        executable='lqg_node',
        name='lqg_controller_node',
        parameters=[
            LaunchConfiguration('config_file'),
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
        config_file_arg,
        debug_arg,
        use_sim_time_arg,
        lqg_controller_node
    ])
