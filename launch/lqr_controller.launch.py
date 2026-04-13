#!/usr/bin/env python3

"""
Launch file for the LQR controller stack.

Starts the horizon_mapper (path planner) and adaptive LQR controller nodes with
their matching parameter files.

Author: Mohammed Azab <mohammed@azab.io>
License: MIT
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for horizon_mapper + LQR controller."""

    # Package directories
    lqr_pkg_share = FindPackageShare('lqr_controller')

    # Default config file paths
    default_lqr_config_file = PathJoinSubstitution([
        lqr_pkg_share, 'config', 'lqr_params.yaml'
    ])
    default_horizon_config_file = PathJoinSubstitution([
        lqr_pkg_share, 'path_planner', 'config', 'horizon_mapper.yaml'
    ])

    # Launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_lqr_config_file,
        description='Path to LQR controller configuration file'
    )

    horizon_config_file_arg = DeclareLaunchArgument(
        'horizon_config_file',
        default_value=default_horizon_config_file,
        description='Path to horizon mapper configuration file'
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

    # horizon_mapper (path planner) node
    horizon_mapper_node = Node(
        package='horizon_mapper',
        executable='horizon_mapper_node',
        name='horizon_mapper_node',
        parameters=[
            LaunchConfiguration('horizon_config_file'),
            {
                'enable_logging': LaunchConfiguration('debug'),
                'enable_debugging': LaunchConfiguration('debug'),
            },
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen',
        emulate_tty=True, 
        respawn=True,
        respawn_delay=2.0
    )

    # LQR controller node
    lqr_controller_node = Node(
        package='lqr_controller',
        executable='lqr_node',
        name='adaptive_lqr_controller_node',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'debug': LaunchConfiguration('debug')
            }
        ],
        output='screen',
        emulate_tty=True,
        respawn=True,
        respawn_delay=2.0
    )

    return LaunchDescription([
        config_file_arg,
        horizon_config_file_arg,
        debug_arg,
        use_sim_time_arg,
        horizon_mapper_node,
        lqr_controller_node,
    ])
