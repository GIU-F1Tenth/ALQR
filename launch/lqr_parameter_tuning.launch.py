#!/usr/bin/env python3

"""
Launch file for LQR Parameter Tuning GUI with ROS2 integration.

This launch file starts the LQR controller node and the parameter tuning GUI
to allow real-time parameter adjustment during simulation or physical testing.

Author: Mohammed Azab <mohammed@azab.io>
License: MIT
Version: 1.0.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate launch description for LQR parameter tuning."""
    
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='lqr_params.yaml',
        description='Configuration file name'
    )
    
    start_gui_arg = DeclareLaunchArgument(
        'start_gui',
        default_value='true',
        description='Start parameter tuning GUI'
    )
    
    # Get package directory
    package_dir = get_package_share_directory('lqr_controller')
    config_dir = os.path.join(package_dir, 'config')
    
    # LQR Controller Node
    lqr_controller_node = Node(
        package='lqr_controller',
        executable='lqr_node.py',
        name='lqr_controller',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'config_file': os.path.join(config_dir, LaunchConfiguration('config_file'))
        }],
        emulate_tty=True
    )
    
    # Parameter Tuning GUI
    parameter_gui_node = Node(
        package='lqr_controller',
        executable='lqr_parameter_gui.py',
        name='lqr_parameter_gui',
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_gui')),
        emulate_tty=True
    )
    
    # Alternative: Start standalone GUI as external process
    standalone_gui_process = ExecuteProcess(
        cmd=['python3', os.path.join(
            get_package_share_directory('lqr_controller'), 
            'scripts', 
            'lqr_parameter_gui_standalone.py'
        )],
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_gui'))
    )
    
    return LaunchDescription([
        use_sim_time_arg,
        config_file_arg,
        start_gui_arg,
        lqr_controller_node,
        parameter_gui_node,
        # Uncomment the line below and comment the line above to use standalone GUI
        # standalone_gui_process,
    ])
