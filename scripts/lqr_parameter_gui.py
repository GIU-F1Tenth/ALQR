#!/usr/bin/env python3

"""
LQR Parameter Tuning GUI

Real-time parameter adjustment interface for F1TENTH LQR controller.
Allows fine-tuning of Q and R matrices, control limits, and other parameters
while the vehicle is running in simulation or on the physical car.

Author: Mohammed Azab <mohammed@azab.io>
License: MIT
Version: 1.0.0
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import yaml
import threading
import time
from typing import Dict, Any, Optional, Callable
import os
import sys

# ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rcl_interfaces.msg import SetParametersResult
    from rcl_interfaces.srv import SetParameters, GetParameters
    from rcl_interfaces.msg import ParameterDescriptor, ParameterType
    from std_msgs.msg import Float32MultiArray, String
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry
    ROS2_AVAILABLE = True
except ImportError:
    print("ROS2 not available. GUI will run in standalone mode.")
    ROS2_AVAILABLE = False


# Import config for default values
def _load_config():
    """Load configuration module safely."""
    try:
        import importlib.util
        # Try multiple possible config locations
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.py'),
            os.path.join(os.path.dirname(__file__), '..', 'config', 'config.py'),
            './config/config.py',
            '../config/config.py'
        ]
        
        for config_path in possible_paths:
            if os.path.exists(config_path):
                spec = importlib.util.spec_from_file_location("lqr_config", config_path)
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)
                print(f"✅ Config loaded from: {config_path}")
                return config_module
                
        print("⚠️  Config file not found, using defaults")
        return None
    except Exception as e:
        print(f"⚠️  Config load error: {e}")
        return None

config = _load_config()
CONFIG_AVAILABLE = config is not None

class LQRParameterGUI:
    """Main GUI class for LQR parameter tuning."""
    
    def __init__(self, master):
        self.master = master
        self.master.title("LQR Controller Parameter Tuning - F1TENTH")
        self.master.geometry("1200x800")
        self.master.configure(bg='#f0f0f0')
        
        # ROS2 node for parameter updates
        self.ros_node = None
        self.ros_thread = None
        self.parameter_client = None
        
        # Parameter storage
        self.parameters = self._load_default_parameters()
        self.parameter_widgets = {}
        self.real_time_plots = {}
        
        # Status tracking
        self.connected_to_ros = False
        self.last_update_time = 0
        
        # Initialize monitoring labels dict
        self.monitor_labels = {}
        self.monitor_enabled = tk.BooleanVar(value=False)
        
        # Create GUI
        self._create_widgets()
        
        # Setup ROS after GUI is created
        if ROS2_AVAILABLE:
            self._setup_ros()
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
    
    def _load_default_parameters(self) -> Dict[str, Any]:
        """Load default parameters from config or use hardcoded values."""
        if CONFIG_AVAILABLE:
            try:
                return {
                    # Vehicle Parameters
                    'wheelbase': getattr(config, 'wheelbase', 0.33),
                    'dt': getattr(config, 'dt', 0.05),
                    
                    # Control Limits
                    'max_acceleration': getattr(config, 'max_acceleration', 5.0),
                    'max_deceleration': getattr(config, 'max_deceleration', 9.0),
                    'max_steering_angle': getattr(config, 'max_steering_angle', 0.9),
                    'min_speed': getattr(config, 'min_speed', 0.1),
                    'max_speed': getattr(config, 'max_speed', 15.0),
                    
                    # LQR Q Matrix Weights (State Cost)
                    'q_position_x': getattr(config, 'position_weight', 5.0),
                    'q_position_y': getattr(config, 'position_weight', 5.0),
                    'q_velocity': getattr(config, 'velocity_weight', 1.0),
                    'q_heading': getattr(config, 'heading_weight', 6.0),
                    
                    # LQR R Matrix Weights (Control Cost)
                    'r_acceleration': getattr(config, 'acceleration_weight', 0.3),
                    'r_steering': getattr(config, 'steering_weight', 4.0),
                    
                    # Control Parameters
                    'control_hz': getattr(config, 'control_hz', 20.0),
                    'lookahead_distance': getattr(config, 'lookahead_distance', 1.5),
                    'enable_feedforward': getattr(config, 'enable_feedforward', True),
                    
                    # Anti-Wobble Parameters
                    'min_lookahead_distance': getattr(config, 'min_lookahead_distance', 0.7),
                    'max_lookahead_distance': getattr(config, 'max_lookahead_distance', 2.5),
                    'lookahead_time': getattr(config, 'lookahead_time', 0.8),
                    'enable_steering_rate_limit': getattr(config, 'enable_steering_rate_limit', True),
                    'max_steering_rate': getattr(config, 'max_steering_rate', 1.5),
                    
                    # Curve Detection
                    'enable_curve_detection': getattr(config, 'enable_curve_detection', True),
                    'curve_lookahead_points': getattr(config, 'curve_lookahead_points', 5),
                    'max_curvature_threshold': getattr(config, 'max_curvature_threshold', 1.0),
                    'curve_speed_factor': getattr(config, 'curve_speed_factor', 0.7),
                    
                    # Safety Parameters
                    'enable_safety_checks': getattr(config, 'enable_safety_checks', False),
                    'safety_timeout': getattr(config, 'safety_timeout', 1.0),
                    'emergency_brake_threshold': getattr(config, 'emergency_brake_threshold', 2.0),
                }
            except Exception as e:
                print(f"Error loading config attributes: {e}")
                return self._get_hardcoded_defaults()
        else:
            return self._get_hardcoded_defaults()
    
    def _get_hardcoded_defaults(self) -> Dict[str, Any]:
        """Get hardcoded default parameters."""
        return {
            # Vehicle Parameters
            'wheelbase': 0.33,
            'dt': 0.05,
            
            # Control Limits
            'max_acceleration': 5.0,
            'max_deceleration': 9.0,
            'max_steering_angle': 0.9,
            'min_speed': 0.1,
            'max_speed': 15.0,
            
            # LQR Q Matrix Weights (State Cost)
            'q_position_x': 5.0,
            'q_position_y': 5.0,
            'q_velocity': 1.0,
            'q_heading': 6.0,
            
            # LQR R Matrix Weights (Control Cost)
            'r_acceleration': 0.3,
            'r_steering': 4.0,
            
            # Control Parameters
            'control_hz': 20.0,
            'lookahead_distance': 1.5,
            'enable_feedforward': True,
            
            # Anti-Wobble Parameters
            'min_lookahead_distance': 0.7,
            'max_lookahead_distance': 2.5,
            'lookahead_time': 0.8,
            'enable_steering_rate_limit': True,
            'max_steering_rate': 1.5,
            
            # Curve Detection
            'enable_curve_detection': True,
            'curve_lookahead_points': 5,
            'max_curvature_threshold': 1.0,
            'curve_speed_factor': 0.7,
            
            # Safety Parameters
            'enable_safety_checks': False,
            'safety_timeout': 1.0,
            'emergency_brake_threshold': 2.0,
        }
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        # Create main frame with scrollbar
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create canvas and scrollbar for scrolling
        canvas = tk.Canvas(main_frame, bg='#f0f0f0')
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Status bar
        self._create_status_bar(scrollable_frame)
        
        # Control buttons
        self._create_control_buttons(scrollable_frame)
        
        # Parameter sections
        self._create_lqr_section(scrollable_frame)
        self._create_control_limits_section(scrollable_frame)
        self._create_vehicle_section(scrollable_frame)
        self._create_anti_wobble_section(scrollable_frame)
        self._create_curve_detection_section(scrollable_frame)
        self._create_safety_section(scrollable_frame)
        
        # Real-time monitoring section
        self._create_monitoring_section(scrollable_frame)
        
        # Bind mousewheel to canvas (Linux compatible)
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _on_mousewheel_linux(event):
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
        
        # Bind for different platforms
        canvas.bind_all("<MouseWheel>", _on_mousewheel)  # Windows
        canvas.bind_all("<Button-4>", _on_mousewheel_linux)  # Linux
        canvas.bind_all("<Button-5>", _on_mousewheel_linux)  # Linux
    
    def _create_status_bar(self, parent):
        """Create status bar showing connection status."""
        status_frame = ttk.LabelFrame(parent, text="Connection Status", padding=10)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = tk.Label(
            status_frame, 
            text="Initializing...", 
            fg="orange",
            font=("Arial", 10, "bold")
        )
        self.status_label.pack(side=tk.LEFT)
        
        self.ros_status_label = tk.Label(
            status_frame,
            text="ROS2: Disconnected",
            fg="red",
            font=("Arial", 9)
        )
        self.ros_status_label.pack(side=tk.RIGHT)
    
    def _create_control_buttons(self, parent):
        """Create control buttons for save/load/reset."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Save/Load buttons
        ttk.Button(
            button_frame, 
            text="Save Config", 
            command=self._save_config
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="Load Config", 
            command=self._load_config_file
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="Reset to Defaults", 
            command=self._reset_to_defaults
        ).pack(side=tk.LEFT, padx=5)
        
        # ROS2 connection button
        self.connect_button = ttk.Button(
            button_frame, 
            text="Connect to ROS2", 
            command=self._toggle_ros_connection
        )
        self.connect_button.pack(side=tk.RIGHT, padx=5)
        
        # Update parameters button
        ttk.Button(
            button_frame, 
            text="Update Parameters", 
            command=self._update_ros_parameters
        ).pack(side=tk.RIGHT, padx=5)
    
    def _create_lqr_section(self, parent):
        """Create LQR weights configuration section."""
        lqr_frame = ttk.LabelFrame(parent, text="LQR Cost Matrices", padding=10)
        lqr_frame.pack(fill=tk.X, pady=5)
        
        # Q Matrix (State Cost) section
        q_frame = ttk.LabelFrame(lqr_frame, text="Q Matrix - State Cost Weights", padding=5)
        q_frame.pack(fill=tk.X, pady=5)
        
        q_params = [
            ('q_position_x', 'Position X Weight', 0.1, 50.0, 0.1),
            ('q_position_y', 'Position Y Weight', 0.1, 50.0, 0.1),
            ('q_velocity', 'Velocity Weight', 0.1, 10.0, 0.1),
            ('q_heading', 'Heading Weight', 0.1, 20.0, 0.1),
        ]
        
        for param, label, min_val, max_val, resolution in q_params:
            self._create_parameter_slider(q_frame, param, label, min_val, max_val, resolution)
        
        # R Matrix (Control Cost) section
        r_frame = ttk.LabelFrame(lqr_frame, text="R Matrix - Control Cost Weights", padding=5)
        r_frame.pack(fill=tk.X, pady=5)
        
        r_params = [
            ('r_acceleration', 'Acceleration Weight', 0.01, 5.0, 0.01),
            ('r_steering', 'Steering Weight', 0.1, 10.0, 0.1),
        ]
        
        for param, label, min_val, max_val, resolution in r_params:
            self._create_parameter_slider(r_frame, param, label, min_val, max_val, resolution)
    
    def _create_control_limits_section(self, parent):
        """Create control limits configuration section."""
        limits_frame = ttk.LabelFrame(parent, text="Control Limits", padding=10)
        limits_frame.pack(fill=tk.X, pady=5)
        
        limits_params = [
            ('max_acceleration', 'Max Acceleration (m/s²)', 1.0, 15.0, 0.1),
            ('max_deceleration', 'Max Deceleration (m/s²)', 1.0, 15.0, 0.1),
            ('max_steering_angle', 'Max Steering Angle (rad)', 0.1, 1.5, 0.01),
            ('min_speed', 'Min Speed (m/s)', 0.0, 2.0, 0.01),
            ('max_speed', 'Max Speed (m/s)', 1.0, 25.0, 0.1),
        ]
        
        for param, label, min_val, max_val, resolution in limits_params:
            self._create_parameter_slider(limits_frame, param, label, min_val, max_val, resolution)
    
    def _create_vehicle_section(self, parent):
        """Create vehicle parameters section."""
        vehicle_frame = ttk.LabelFrame(parent, text="Vehicle Parameters", padding=10)
        vehicle_frame.pack(fill=tk.X, pady=5)
        
        vehicle_params = [
            ('wheelbase', 'Wheelbase (m)', 0.1, 1.0, 0.01),
            ('dt', 'Time Step (s)', 0.01, 0.2, 0.001),
            ('control_hz', 'Control Frequency (Hz)', 5.0, 100.0, 1.0),
            ('lookahead_distance', 'Lookahead Distance (m)', 0.1, 5.0, 0.1),
        ]
        
        for param, label, min_val, max_val, resolution in vehicle_params:
            self._create_parameter_slider(vehicle_frame, param, label, min_val, max_val, resolution)
        
        # Boolean parameter for feedforward
        self._create_boolean_parameter(vehicle_frame, 'enable_feedforward', 'Enable Feedforward')
    
    def _create_anti_wobble_section(self, parent):
        """Create anti-wobble parameters section."""
        wobble_frame = ttk.LabelFrame(parent, text="Anti-Wobble Parameters", padding=10)
        wobble_frame.pack(fill=tk.X, pady=5)
        
        wobble_params = [
            ('min_lookahead_distance', 'Min Lookahead Distance (m)', 0.1, 2.0, 0.1),
            ('max_lookahead_distance', 'Max Lookahead Distance (m)', 1.0, 5.0, 0.1),
            ('lookahead_time', 'Lookahead Time (s)', 0.1, 2.0, 0.1),
            ('max_steering_rate', 'Max Steering Rate (rad/s)', 0.1, 5.0, 0.1),
        ]
        
        for param, label, min_val, max_val, resolution in wobble_params:
            self._create_parameter_slider(wobble_frame, param, label, min_val, max_val, resolution)
        
        self._create_boolean_parameter(wobble_frame, 'enable_steering_rate_limit', 'Enable Steering Rate Limit')
    
    def _create_curve_detection_section(self, parent):
        """Create curve detection parameters section."""
        curve_frame = ttk.LabelFrame(parent, text="Curve Detection Parameters", padding=10)
        curve_frame.pack(fill=tk.X, pady=5)
        
        curve_params = [
            ('curve_lookahead_points', 'Curve Lookahead Points', 1, 20, 1),
            ('max_curvature_threshold', 'Max Curvature Threshold', 0.1, 5.0, 0.1),
            ('curve_speed_factor', 'Curve Speed Factor', 0.1, 1.0, 0.01),
        ]
        
        for param, label, min_val, max_val, resolution in curve_params:
            self._create_parameter_slider(curve_frame, param, label, min_val, max_val, resolution)
        
        self._create_boolean_parameter(curve_frame, 'enable_curve_detection', 'Enable Curve Detection')
    
    def _create_safety_section(self, parent):
        """Create safety parameters section."""
        safety_frame = ttk.LabelFrame(parent, text="Safety Parameters", padding=10)
        safety_frame.pack(fill=tk.X, pady=5)
        
        safety_params = [
            ('safety_timeout', 'Safety Timeout (s)', 0.1, 5.0, 0.1),
            ('emergency_brake_threshold', 'Emergency Brake Threshold (m/s²)', 1.0, 10.0, 0.1),
        ]
        
        for param, label, min_val, max_val, resolution in safety_params:
            self._create_parameter_slider(safety_frame, param, label, min_val, max_val, resolution)
        
        self._create_boolean_parameter(safety_frame, 'enable_safety_checks', 'Enable Safety Checks')
    
    def _create_monitoring_section(self, parent):
        """Create real-time monitoring section."""
        monitor_frame = ttk.LabelFrame(parent, text="Real-Time Monitoring", padding=10)
        monitor_frame.pack(fill=tk.X, pady=5)
        
        # Status display
        status_display_frame = ttk.Frame(monitor_frame)
        status_display_frame.pack(fill=tk.X, pady=5)
        
        monitor_params = [
            ('lateral_error', 'Lateral Error (m)'),
            ('heading_error', 'Heading Error (rad)'),
            ('velocity_error', 'Velocity Error (m/s)'),
            ('control_cost', 'Control Cost'),
            ('state_cost', 'State Cost'),
        ]
        
        for i, (param, label) in enumerate(monitor_params):
            row = i // 2
            col = i % 2
            
            frame = ttk.Frame(status_display_frame)
            frame.grid(row=row, column=col, padx=10, pady=2, sticky="w")
            
            ttk.Label(frame, text=f"{label}:").pack(side=tk.LEFT)
            
            value_label = tk.Label(
                frame, 
                text="--", 
                font=("Arial", 10, "bold"),
                fg="blue"
            )
            value_label.pack(side=tk.LEFT, padx=5)
            
            self.monitor_labels[param] = value_label
        
        # Control buttons for monitoring
        control_frame = ttk.Frame(monitor_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(
            control_frame,
            text="Enable Real-Time Monitoring",
            variable=self.monitor_enabled
        ).pack(side=tk.LEFT)
    
    def _create_parameter_slider(self, parent, param_name, label, min_val, max_val, resolution):
        """Create a parameter slider with label and value display."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        
        # Label
        ttk.Label(frame, text=f"{label}:", width=25).pack(side=tk.LEFT)
        
        # Current value display
        value_var = tk.DoubleVar(value=self.parameters.get(param_name, min_val))
        value_label = tk.Label(
            frame, 
            textvariable=value_var, 
            width=8, 
            relief=tk.SUNKEN,
            font=("Arial", 9, "bold")
        )
        value_label.pack(side=tk.RIGHT, padx=5)
        
        # Slider
        slider = tk.Scale(
            frame,
            from_=min_val,
            to=max_val,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            variable=value_var,
            command=lambda val, param=param_name: self._on_parameter_change(param, float(val))
        )
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        # Store widget references
        self.parameter_widgets[param_name] = {
            'slider': slider,
            'value_var': value_var,
            'label': value_label
        }
    
    def _create_boolean_parameter(self, parent, param_name, label):
        """Create a boolean parameter checkbox."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        
        value_var = tk.BooleanVar(value=self.parameters.get(param_name, False))
        
        checkbox = ttk.Checkbutton(
            frame,
            text=label,
            variable=value_var,
            command=lambda param=param_name: self._on_parameter_change(param, value_var.get())
        )
        checkbox.pack(side=tk.LEFT)
        
        self.parameter_widgets[param_name] = {
            'checkbox': checkbox,
            'value_var': value_var
        }
    
    def _on_parameter_change(self, param_name, value):
        """Handle parameter value changes."""
        self.parameters[param_name] = value
        self.last_update_time = time.time()
        
        print(f"Parameter changed: {param_name} = {value}")
        
        # If connected to ROS, update parameters immediately for critical ones
        critical_params = ['q_position_x', 'q_position_y', 'q_velocity', 'q_heading', 
                          'r_acceleration', 'r_steering', 'max_acceleration', 
                          'max_steering_angle', 'lookahead_distance']
        
        if self.connected_to_ros and param_name in critical_params:
            self._update_single_ros_parameter(param_name, value)
    
    def _setup_ros(self):
        """Initialize ROS2 components."""
        if not ROS2_AVAILABLE:
            self._update_status("ROS2 not available", "red")
            return
        
        try:
            # Initialize rclpy if not already done
            if not rclpy.ok():
                rclpy.init()
            
            # Create parameter node
            self.ros_node = LQRParameterNode()
            self.ros_node.set_gui_callback(self._update_monitoring_data)
            
            # Start ROS2 spinning in separate thread
            self.ros_thread = threading.Thread(
                target=lambda: rclpy.spin(self.ros_node), 
                daemon=True
            )
            self.ros_thread.start()
            
            # Setup parameter client for communicating with LQR controller
            self._setup_parameter_client()
            
            self.connected_to_ros = True
            self._update_status("ROS2 initialized", "green")
            self._update_ros_status("Connected", "green")
            
        except Exception as e:
            self._update_status(f"ROS2 init failed: {e}", "red")
            self._update_ros_status("Failed", "red")
            print(f"ROS2 setup error: {e}")
    
    def _setup_parameter_client(self):
        """Setup parameter client for LQR controller communication."""
        if self.ros_node:
            self.parameter_client = self.ros_node.create_client(
                SetParameters,
                'lqr_controller_node/set_parameters'
            )
    
    def _toggle_ros_connection(self):
        """Toggle ROS2 connection."""
        if self.connected_to_ros:
            self._disconnect_ros()
        else:
            self._setup_ros()
    
    def _disconnect_ros(self):
        """Disconnect from ROS2."""
        self.connected_to_ros = False
        if self.ros_node:
            self.ros_node.destroy_node()
            self.ros_node = None
        
        self._update_status("Disconnected from ROS2", "orange")
        self._update_ros_status("Disconnected", "red")
        self.connect_button.configure(text="Connect to ROS2")
    
    def _update_ros_parameters(self):
        """Update all parameters in ROS2 node."""
        if not self.connected_to_ros or not self.parameter_client:
            messagebox.showwarning("Warning", "Not connected to ROS2")
            return
        
        try:
            # Create parameter list for ROS2
            parameters_to_set = []
            
            for param_name, value in self.parameters.items():
                if isinstance(value, bool):
                    param = Parameter(param_name, Parameter.Type.BOOL, value)
                elif isinstance(value, int):
                    param = Parameter(param_name, Parameter.Type.INTEGER, value)
                else:
                    param = Parameter(param_name, Parameter.Type.DOUBLE, float(value))
                
                parameters_to_set.append(param)
            
            # Create service request
            request = SetParameters.Request()
            request.parameters = [param.to_parameter_msg() for param in parameters_to_set]
            
            # Call service asynchronously
            if self.parameter_client.wait_for_service(timeout_sec=1.0):
                future = self.parameter_client.call_async(request)
                self._update_status("Parameters update sent...", "orange")
            else:
                self._update_status("Parameter service not available", "red")
                
        except Exception as e:
            self._update_status(f"Parameter update failed: {e}", "red")
            messagebox.showerror("Error", f"Failed to update parameters: {e}")
    
    def _update_single_ros_parameter(self, param_name, value):
        """Update a single parameter in ROS2 node."""
        if not self.connected_to_ros or not self.parameter_client:
            return
        
        try:
            if isinstance(value, bool):
                param = Parameter(param_name, Parameter.Type.BOOL, value)
            elif isinstance(value, int):
                param = Parameter(param_name, Parameter.Type.INTEGER, value)
            else:
                param = Parameter(param_name, Parameter.Type.DOUBLE, float(value))
            
            # Create service request
            request = SetParameters.Request()
            request.parameters = [param.to_parameter_msg()]
            
            # Call service asynchronously
            if self.parameter_client.wait_for_service(timeout_sec=0.1):
                future = self.parameter_client.call_async(request)
                
        except Exception as e:
            print(f"Failed to update parameter {param_name}: {e}")
    
    def _save_config(self):
        """Save current parameters to YAML file."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
            title="Save LQR Configuration"
        )
        
        if filename:
            try:
                # Create ROS2 parameter format
                config_data = {
                    'lqr_controller_node': {
                        'ros__parameters': self.parameters
                    }
                }
                
                with open(filename, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                
                self._update_status(f"Configuration saved to {filename}", "green")
                messagebox.showinfo("Success", f"Configuration saved to {filename}")
                
            except Exception as e:
                self._update_status(f"Save failed: {e}", "red")
                messagebox.showerror("Error", f"Failed to save configuration: {e}")
    
    def _load_config_file(self):
        """Load parameters from YAML file."""
        filename = filedialog.askopenfilename(
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
            title="Load LQR Configuration"
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Extract parameters from ROS2 format or flat format
                if 'lqr_controller_node' in config_data and 'ros__parameters' in config_data['lqr_controller_node']:
                    new_params = config_data['lqr_controller_node']['ros__parameters']
                else:
                    new_params = config_data
                
                # Update parameters
                for param_name, value in new_params.items():
                    if param_name in self.parameters:
                        self.parameters[param_name] = value
                
                self._update_gui_from_parameters()
                
                self._update_status(f"Configuration loaded from {filename}", "green")
                messagebox.showinfo("Success", f"Configuration loaded from {filename}")
                
            except Exception as e:
                self._update_status(f"Load failed: {e}", "red")
                messagebox.showerror("Error", f"Failed to load configuration: {e}")
    
    def _update_gui_from_parameters(self):
        """Update GUI widgets from current parameters."""
        for param_name, widgets in self.parameter_widgets.items():
            if param_name in self.parameters:
                value = self.parameters[param_name]
                widgets['value_var'].set(value)
    
    def _reset_to_defaults(self):
        """Reset all parameters to default values."""
        if messagebox.askyesno("Confirm Reset", "Reset all parameters to default values?"):
            self.parameters = self._load_default_parameters()
            self._update_gui_from_parameters()
            self._update_status("Parameters reset to defaults", "green")
    
    def _update_monitoring_data(self, data):
        """Update monitoring displays with real-time data."""
        if not self.monitor_enabled.get():
            return
        
        # Update monitoring labels in GUI thread
        def update_labels():
            for param, value in data.items():
                if param in self.monitor_labels:
                    if isinstance(value, float):
                        self.monitor_labels[param].config(text=f"{value:.4f}")
                    else:
                        self.monitor_labels[param].config(text=str(value))
        
        self.master.after(0, update_labels)
    
    def _update_status(self, message, color):
        """Update status label."""
        def update():
            self.status_label.config(text=message, fg=color)
        self.master.after(0, update)
    
    def _update_ros_status(self, message, color):
        """Update ROS status label."""
        def update():
            self.ros_status_label.config(text=f"ROS2: {message}", fg=color)
            
            if color == "green":
                self.connect_button.configure(text="Disconnect ROS2")
            else:
                self.connect_button.configure(text="Connect to ROS2")
        
        self.master.after(0, update)
    
    def _update_loop(self):
        """Main update loop for GUI."""
        while True:
            try:
                time.sleep(0.1)
                
            except Exception as e:
                print(f"GUI update error: {e}")
                time.sleep(1.0)


class LQRParameterNode(Node):
    """ROS2 node for parameter management and monitoring."""
    
    def __init__(self):
        super().__init__('lqr_parameter_gui_node')
        
        # GUI callback for monitoring data
        self.gui_callback = None
        
        # Subscribers for real-time data
        self.odom_sub = self.create_subscription(
            Odometry,
            '/car_state/odom',
            self._odom_callback,
            10
        )
        
        # Timer for periodic updates
        self.create_timer(0.1, self._timer_callback)
        
        # Data storage
        self.latest_odom = None
        self.monitoring_data = {}
        
        self.get_logger().info("LQR Parameter GUI Node initialized")
    
    def set_gui_callback(self, callback):
        """Set callback function for updating GUI."""
        self.gui_callback = callback
    
    def _odom_callback(self, msg):
        """Handle odometry messages."""
        self.latest_odom = msg
        
        # Extract monitoring data
        if self.gui_callback:
            # Calculate some basic monitoring values
            # In a real implementation, you'd get these from the LQR controller
            monitoring_data = {
                'lateral_error': 0.0,  # Placeholder
                'heading_error': 0.0,  # Placeholder  
                'velocity_error': 0.0,  # Placeholder
                'control_cost': 0.0,   # Placeholder
                'state_cost': 0.0,     # Placeholder
            }
            
            self.gui_callback(monitoring_data)
    
    def _timer_callback(self):
        """Periodic timer callback."""
        pass  # Could be used for periodic tasks


def main():
    """Main entry point for the GUI application."""
    print("Starting LQR Parameter Tuning GUI...")
    
    root = tk.Tk()
    app = LQRParameterGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("GUI application interrupted")
    finally:
        if ROS2_AVAILABLE and rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
