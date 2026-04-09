#!/usr/bin/env python3

"""
Launch file for LQR Controller

Starts the horizon_mapper (path planner) node first, then the adaptive LQR
controller node once the path planner is ready.

Author: Mohammed Azab <mohammed@azab.io>
License: MIT
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for horizon_mapper + LQR controller."""

    # Package directories
    lqr_pkg_share = FindPackageShare('lqr_controller')
    horizon_pkg_share = FindPackageShare('horizon_mapper')

    # Default config file path
    default_config_file = PathJoinSubstitution([
        lqr_pkg_share, 'config', 'lqr_params.yaml'
    ])

    # Launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config_file,
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

    # horizon_mapper (path planner) node — must start before LQR
    horizon_mapper_node = Node(
        package='horizon_mapper',
        executable='horizon_mapper_node',
        name='horizon_mapper_node',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen',
        emulate_tty=True,
        respawn=True,
        respawn_delay=2.0
    )

    # LQR controller node — starts after horizon_mapper is up
    lqr_controller_node = Node(
        package='lqr_controller',
        executable='lqr_node',
        name='lqr_controller_node',
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

    # Register LQR to start only after horizon_mapper process has started
    lqr_after_horizon = RegisterEventHandler(
        OnProcessStart(
            target_action=horizon_mapper_node,
            on_start=[lqr_controller_node]
        )
    )

    return LaunchDescription([
        config_file_arg,
        debug_arg,
        use_sim_time_arg,
        horizon_mapper_node,
        lqr_after_horizon,
    ])
