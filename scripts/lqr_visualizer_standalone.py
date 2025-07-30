#!/usr/bin/env python3

"""
LQR Controller Standalone GUI Visualizer

This script provides a standalone visualization tool for testing the LQR controller
without requiring a full ROS2 setup. It generates synthetic data to demonstrate
the visualization capabilities.

Author: Mohammed Azab <mohammed@azab.io>
License: MIT
Version: 1.0.0
"""

import sys
import os
import numpy as np
import time
import threading
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass
import math

# GUI imports
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation


@dataclass
class VehicleState:
    """Data class for vehicle state information."""
    x: float = 0.0
    y: float = 0.0
    velocity: float = 0.0
    yaw: float = 0.0
    steering_angle: float = 0.0
    acceleration: float = 0.0
    timestamp: float = 0.0


@dataclass
class ControlCommand:
    """Data class for control command information."""
    acceleration: float = 0.0
    steering_angle: float = 0.0
    speed: float = 0.0
    timestamp: float = 0.0


@dataclass
class PerformanceMetrics:
    """Data class for performance metrics."""
    control_frequency: float = 20.0
    avg_control_time: float = 0.002
    max_control_time: float = 0.005
    consecutive_failures: int = 0
    path_ready: bool = True
    controller_active: bool = True
    emergency_stop: bool = False
    state_error: float = 0.0


class SyntheticDataGenerator:
    """Generates synthetic data for testing the visualizer."""
    
    def __init__(self, data_callback):
        self.data_callback = data_callback
        self.running = True
        self.start_time = time.time()
        
        # Vehicle parameters
        self.wheelbase = 0.33
        self.dt = 0.05
        
        # Current state
        self.vehicle_state = VehicleState()
        self.control_command = ControlCommand()
        self.metrics = PerformanceMetrics()
        
        # Reference trajectory (figure-8 track)
        self.generate_reference_trajectory()
        
        # Start data generation thread
        self.thread = threading.Thread(target=self.generate_data, daemon=True)
        self.thread.start()
    
    def generate_reference_trajectory(self):
        """Generate a figure-8 reference trajectory."""
        
        self.reference_trajectory = []
        
        # Figure-8 parameters
        a = 5.0  # Scale factor
        b = 3.0  # Scale factor
        
        for i in range(200):
            t = i * 2 * np.pi / 200
            
            # Figure-8 parametric equations
            x = a * np.sin(t)
            y = b * np.sin(2 * t) / 2
            
            # Calculate velocity and heading
            dx_dt = a * np.cos(t)
            dy_dt = b * np.cos(2 * t)
            
            velocity = np.sqrt(dx_dt**2 + dy_dt**2) * 0.1  # Scale down velocity
            velocity = max(0.5, min(3.0, velocity))  # Clamp velocity
            
            theta = np.arctan2(dy_dt, dx_dt)
            
            # Calculate steering angle (simplified)
            if i > 0:
                prev_theta = self.reference_trajectory[-1]['theta']
                dtheta = theta - prev_theta
                # Normalize angle difference
                while dtheta > np.pi:
                    dtheta -= 2 * np.pi
                while dtheta < -np.pi:
                    dtheta += 2 * np.pi
                delta = dtheta / self.dt
                delta = max(-0.5, min(0.5, delta))  # Clamp steering
            else:
                delta = 0.0
            
            self.reference_trajectory.append({
                'x': x,
                'y': y,
                'v': velocity,
                'theta': theta,
                'delta': delta
            })
    
    def generate_data(self):
        """Generate synthetic data in real-time."""
        
        # Initial state
        self.vehicle_state.x = 0.0
        self.vehicle_state.y = 0.0
        self.vehicle_state.velocity = 1.0
        self.vehicle_state.yaw = 0.0
        
        while self.running:
            current_time = time.time()
            elapsed = current_time - self.start_time
            
            # Update vehicle state with simple kinematic model
            self.update_vehicle_state(elapsed)
            
            # Generate control command
            self.generate_control_command(elapsed)
            
            # Update performance metrics
            self.update_metrics(elapsed)
            
            # Send data to GUI
            if self.data_callback:
                self.data_callback('vehicle_state', self.vehicle_state)
                self.data_callback('control_command', self.control_command)
                self.data_callback('performance_metrics', self.metrics)
                self.data_callback('reference_trajectory', self.reference_trajectory)
                
                # State error (distance to closest reference point)
                state_error = self.calculate_state_error()
                self.data_callback('state_error', state_error)
            
            time.sleep(self.dt)
    
    def update_vehicle_state(self, elapsed_time):
        """Update vehicle state using simple kinematic model."""
        
        # Add some noise and dynamics
        noise_scale = 0.01
        
        # Simple kinematic bicycle model
        self.vehicle_state.x += self.vehicle_state.velocity * np.cos(self.vehicle_state.yaw) * self.dt
        self.vehicle_state.y += self.vehicle_state.velocity * np.sin(self.vehicle_state.yaw) * self.dt
        self.vehicle_state.yaw += (self.vehicle_state.velocity / self.wheelbase) * np.tan(self.vehicle_state.steering_angle) * self.dt
        
        # Add noise
        self.vehicle_state.x += np.random.normal(0, noise_scale)
        self.vehicle_state.y += np.random.normal(0, noise_scale)
        self.vehicle_state.yaw += np.random.normal(0, noise_scale * 0.1)
        
        # Update velocity based on acceleration
        self.vehicle_state.velocity += self.vehicle_state.acceleration * self.dt
        self.vehicle_state.velocity = max(0.1, min(5.0, self.vehicle_state.velocity))
        
        self.vehicle_state.timestamp = time.time()
    
    def generate_control_command(self, elapsed_time):
        """Generate realistic control commands."""
        
        # Simple controller that tries to follow the reference trajectory
        closest_ref = self.find_closest_reference_point()
        
        if closest_ref is not None:
            ref_point = self.reference_trajectory[closest_ref]
            
            # Simple proportional control
            dx = ref_point['x'] - self.vehicle_state.x
            dy = ref_point['y'] - self.vehicle_state.y
            distance_error = np.sqrt(dx**2 + dy**2)
            
            # Heading error
            desired_heading = np.arctan2(dy, dx)
            heading_error = desired_heading - self.vehicle_state.yaw
            
            # Normalize heading error
            while heading_error > np.pi:
                heading_error -= 2 * np.pi
            while heading_error < -np.pi:
                heading_error += 2 * np.pi
            
            # Generate control commands
            self.control_command.acceleration = 0.5 * (ref_point['v'] - self.vehicle_state.velocity)
            self.control_command.steering_angle = 0.3 * heading_error + ref_point['delta']
            
            # Clamp values
            self.control_command.acceleration = max(-3.0, min(3.0, self.control_command.acceleration))
            self.control_command.steering_angle = max(-0.5, min(0.5, self.control_command.steering_angle))
            
            # Update vehicle state with control
            self.vehicle_state.acceleration = self.control_command.acceleration
            self.vehicle_state.steering_angle = self.control_command.steering_angle
        
        self.control_command.speed = self.vehicle_state.velocity
        self.control_command.timestamp = time.time()
    
    def find_closest_reference_point(self):
        """Find the closest point in the reference trajectory."""
        
        if not self.reference_trajectory:
            return None
        
        min_distance = float('inf')
        closest_index = 0
        
        for i, point in enumerate(self.reference_trajectory):
            dx = self.vehicle_state.x - point['x']
            dy = self.vehicle_state.y - point['y']
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        return closest_index
    
    def calculate_state_error(self):
        """Calculate state error magnitude."""
        
        closest_ref = self.find_closest_reference_point()
        
        if closest_ref is not None:
            ref_point = self.reference_trajectory[closest_ref]
            dx = self.vehicle_state.x - ref_point['x']
            dy = self.vehicle_state.y - ref_point['y']
            dv = self.vehicle_state.velocity - ref_point['v']
            dtheta = self.vehicle_state.yaw - ref_point['theta']
            
            # Normalize angle difference
            while dtheta > np.pi:
                dtheta -= 2 * np.pi
            while dtheta < -np.pi:
                dtheta += 2 * np.pi
            
            return np.sqrt(dx**2 + dy**2 + dv**2 + dtheta**2)
        
        return 0.0
    
    def update_metrics(self, elapsed_time):
        """Update performance metrics."""
        
        # Simulate realistic metrics
        self.metrics.control_frequency = 20.0 + np.random.normal(0, 0.5)
        self.metrics.avg_control_time = 0.002 + np.random.normal(0, 0.0005)
        self.metrics.max_control_time = 0.005 + np.random.normal(0, 0.001)
        
        # Occasionally simulate failures
        if np.random.random() < 0.01:  # 1% chance
            self.metrics.consecutive_failures += 1
        else:
            self.metrics.consecutive_failures = max(0, self.metrics.consecutive_failures - 1)
        
        # Simulate occasional emergency stops
        if np.random.random() < 0.001:  # 0.1% chance
            self.metrics.emergency_stop = True
        elif np.random.random() < 0.01:  # 1% chance to recover
            self.metrics.emergency_stop = False
        
        self.metrics.controller_active = not self.metrics.emergency_stop
        self.metrics.state_error = self.calculate_state_error()
    
    def stop(self):
        """Stop data generation."""
        self.running = False


class LQRVisualizerGUI:
    """Main GUI class for LQR controller visualization."""

    def __init__(self, use_synthetic_data=True):
        self.root = tk.Tk()
        self.root.title("LQR Controller Visualizer")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Data storage with history
        self.max_history = 1000
        self.vehicle_history = deque(maxlen=self.max_history)
        self.control_history = deque(maxlen=self.max_history)
        self.error_history = deque(maxlen=self.max_history)
        self.reference_trajectory = []
        
        # Current data
        self.current_vehicle_state = VehicleState()
        self.current_control = ControlCommand()
        self.current_metrics = PerformanceMetrics()
        
        # Setup GUI
        self.setup_gui()
        
        # Data source
        if use_synthetic_data:
            self.data_generator = SyntheticDataGenerator(self.data_callback)
        else:
            # Setup ROS2 node (commented out for standalone version)
            self.setup_ros_node()
        
        # Animation timer
        self.animation_running = True
        self.update_plots()

    # [Rest of the GUI methods are identical to the ROS2 version]
    # Including all the methods from the previous file...
    
    def setup_gui(self):
        """Setup the main GUI layout."""
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel at top
        self.setup_control_panel(main_frame)
        
        # Create notebook for different tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Setup different tabs
        self.setup_trajectory_tab()
        self.setup_control_tab()
        self.setup_performance_tab()
        self.setup_diagnostics_tab()

    def setup_control_panel(self, parent):
        """Setup the control panel with status indicators."""
        
        control_frame = ttk.LabelFrame(parent, text="Controller Status", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Status indicators
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(fill=tk.X)
        
        # Controller status
        ttk.Label(status_frame, text="Controller:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.controller_status_label = ttk.Label(status_frame, text="Unknown", foreground="gray")
        self.controller_status_label.grid(row=0, column=1, padx=5, sticky=tk.W)
        
        # Path ready status
        ttk.Label(status_frame, text="Path Ready:").grid(row=0, column=2, padx=5, sticky=tk.W)
        self.path_ready_label = ttk.Label(status_frame, text="Unknown", foreground="gray")
        self.path_ready_label.grid(row=0, column=3, padx=5, sticky=tk.W)
        
        # Emergency stop status
        ttk.Label(status_frame, text="Emergency Stop:").grid(row=0, column=4, padx=5, sticky=tk.W)
        self.emergency_stop_label = ttk.Label(status_frame, text="Unknown", foreground="gray")
        self.emergency_stop_label.grid(row=0, column=5, padx=5, sticky=tk.W)
        
        # Current values frame
        values_frame = ttk.Frame(control_frame)
        values_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Vehicle state values
        ttk.Label(values_frame, text="Position:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.position_label = ttk.Label(values_frame, text="(0.0, 0.0)")
        self.position_label.grid(row=0, column=1, padx=5, sticky=tk.W)
        
        ttk.Label(values_frame, text="Velocity:").grid(row=0, column=2, padx=5, sticky=tk.W)
        self.velocity_label = ttk.Label(values_frame, text="0.0 m/s")
        self.velocity_label.grid(row=0, column=3, padx=5, sticky=tk.W)
        
        ttk.Label(values_frame, text="Steering:").grid(row=0, column=4, padx=5, sticky=tk.W)
        self.steering_label = ttk.Label(values_frame, text="0.0 rad")
        self.steering_label.grid(row=0, column=5, padx=5, sticky=tk.W)
        
        # Control values
        ttk.Label(values_frame, text="Acceleration:").grid(row=1, column=0, padx=5, sticky=tk.W)
        self.acceleration_label = ttk.Label(values_frame, text="0.0 m/s²")
        self.acceleration_label.grid(row=1, column=1, padx=5, sticky=tk.W)
        
        ttk.Label(values_frame, text="State Error:").grid(row=1, column=2, padx=5, sticky=tk.W)
        self.state_error_label = ttk.Label(values_frame, text="0.0")
        self.state_error_label.grid(row=1, column=3, padx=5, sticky=tk.W)

    def setup_trajectory_tab(self):
        """Setup the trajectory visualization tab."""
        
        traj_frame = ttk.Frame(self.notebook)
        self.notebook.add(traj_frame, text="Trajectory")
        
        # Create matplotlib figure
        self.traj_fig = Figure(figsize=(12, 8), dpi=100)
        self.traj_ax = self.traj_fig.add_subplot(111)
        
        self.traj_canvas = FigureCanvasTkAgg(self.traj_fig, traj_frame)
        self.traj_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plot
        self.traj_ax.set_title("Vehicle Trajectory and Reference Path")
        self.traj_ax.set_xlabel("X Position [m]")
        self.traj_ax.set_ylabel("Y Position [m]")
        self.traj_ax.grid(True, alpha=0.3)
        self.traj_ax.set_aspect('equal')

    def setup_control_tab(self):
        """Setup the control inputs visualization tab."""
        
        control_frame = ttk.Frame(self.notebook)
        self.notebook.add(control_frame, text="Control Inputs")
        
        # Create matplotlib figure with subplots
        self.control_fig = Figure(figsize=(12, 8), dpi=100)
        
        # Acceleration plot
        self.accel_ax = self.control_fig.add_subplot(2, 1, 1)
        self.accel_ax.set_title("Acceleration Command")
        self.accel_ax.set_ylabel("Acceleration [m/s²]")
        self.accel_ax.grid(True, alpha=0.3)
        
        # Steering plot
        self.steer_ax = self.control_fig.add_subplot(2, 1, 2)
        self.steer_ax.set_title("Steering Angle Command")
        self.steer_ax.set_xlabel("Time [s]")
        self.steer_ax.set_ylabel("Steering Angle [rad]")
        self.steer_ax.grid(True, alpha=0.3)
        
        self.control_fig.tight_layout()
        
        self.control_canvas = FigureCanvasTkAgg(self.control_fig, control_frame)
        self.control_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_performance_tab(self):
        """Setup the performance metrics visualization tab."""
        
        perf_frame = ttk.Frame(self.notebook)
        self.notebook.add(perf_frame, text="Performance")
        
        # Create matplotlib figure
        self.perf_fig = Figure(figsize=(12, 8), dpi=100)
        
        # State error plot
        self.error_ax = self.perf_fig.add_subplot(2, 1, 1)
        self.error_ax.set_title("State Error")
        self.error_ax.set_ylabel("Error Magnitude")
        self.error_ax.grid(True, alpha=0.3)
        
        # Velocity tracking plot
        self.vel_ax = self.perf_fig.add_subplot(2, 1, 2)
        self.vel_ax.set_title("Velocity Profile")
        self.vel_ax.set_xlabel("Time [s]")
        self.vel_ax.set_ylabel("Velocity [m/s]")
        self.vel_ax.grid(True, alpha=0.3)
        
        self.perf_fig.tight_layout()
        
        self.perf_canvas = FigureCanvasTkAgg(self.perf_fig, perf_frame)
        self.perf_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_diagnostics_tab(self):
        """Setup the diagnostics information tab."""
        
        diag_frame = ttk.Frame(self.notebook)
        self.notebook.add(diag_frame, text="Diagnostics")
        
        # Create text widget for diagnostics
        self.diag_text = tk.Text(diag_frame, wrap=tk.WORD, font=("Courier", 10))
        
        # Scrollbar for text widget
        scrollbar = ttk.Scrollbar(diag_frame, orient=tk.VERTICAL, command=self.diag_text.yview)
        self.diag_text.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        self.diag_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_ros_node(self):
        """Setup ROS2 node (placeholder for standalone version)."""
        pass

    def data_callback(self, data_type: str, data):
        """Callback for receiving data from data source."""
        
        try:
            if data_type == 'vehicle_state':
                self.current_vehicle_state = data
                self.vehicle_history.append((time.time(), data))
                
            elif data_type == 'control_command':
                self.current_control = data
                self.control_history.append((time.time(), data))
                
            elif data_type == 'reference_trajectory':
                self.reference_trajectory = data
                
            elif data_type == 'performance_metrics':
                self.current_metrics = data
                
            elif data_type == 'state_error':
                self.error_history.append((time.time(), data))
                
        except Exception as e:
            print(f"Error in data callback: {e}")

    # [Include all the update methods from the previous version]
    def update_status_labels(self):
        """Update status labels in control panel."""
        
        try:
            # Controller status
            if self.current_metrics.controller_active:
                self.controller_status_label.config(text="Active", foreground="green")
            elif self.current_metrics.emergency_stop:
                self.controller_status_label.config(text="Emergency Stop", foreground="red")
            else:
                self.controller_status_label.config(text="Inactive", foreground="orange")
            
            # Path ready status
            if self.current_metrics.path_ready:
                self.path_ready_label.config(text="Ready", foreground="green")
            else:
                self.path_ready_label.config(text="Not Ready", foreground="red")
            
            # Emergency stop status
            if self.current_metrics.emergency_stop:
                self.emergency_stop_label.config(text="ACTIVE", foreground="red")
            else:
                self.emergency_stop_label.config(text="Normal", foreground="green")
            
            # Vehicle state values
            self.position_label.config(
                text=f"({self.current_vehicle_state.x:.2f}, {self.current_vehicle_state.y:.2f})")
            self.velocity_label.config(text=f"{self.current_vehicle_state.velocity:.2f} m/s")
            self.steering_label.config(text=f"{self.current_vehicle_state.steering_angle:.3f} rad")
            self.acceleration_label.config(text=f"{self.current_vehicle_state.acceleration:.2f} m/s²")
            self.state_error_label.config(text=f"{self.current_metrics.state_error:.3f}")
            
        except Exception as e:
            print(f"Error updating status labels: {e}")

    def update_trajectory_plot(self):
        """Update the trajectory plot."""
        
        try:
            self.traj_ax.clear()
            
            # Plot reference trajectory
            if self.reference_trajectory:
                ref_x = [point['x'] for point in self.reference_trajectory]
                ref_y = [point['y'] for point in self.reference_trajectory]
                self.traj_ax.plot(ref_x, ref_y, 'b-', linewidth=2, label='Reference Trajectory', alpha=0.7)
            
            # Plot vehicle history
            if len(self.vehicle_history) > 1:
                hist_x = [state[1].x for state in self.vehicle_history]
                hist_y = [state[1].y for state in self.vehicle_history]
                self.traj_ax.plot(hist_x, hist_y, 'r-', linewidth=1, label='Vehicle Path', alpha=0.8)
            
            # Plot current vehicle position
            if self.current_vehicle_state.timestamp > 0:
                self.traj_ax.plot(self.current_vehicle_state.x, self.current_vehicle_state.y, 
                                'ro', markersize=8, label='Current Position')
                
                # Draw vehicle orientation arrow
                arrow_length = 0.5
                dx = arrow_length * np.cos(self.current_vehicle_state.yaw)
                dy = arrow_length * np.sin(self.current_vehicle_state.yaw)
                self.traj_ax.arrow(self.current_vehicle_state.x, self.current_vehicle_state.y,
                                 dx, dy, head_width=0.1, head_length=0.1, fc='red', ec='red')
            
            self.traj_ax.set_title("Vehicle Trajectory and Reference Path")
            self.traj_ax.set_xlabel("X Position [m]")
            self.traj_ax.set_ylabel("Y Position [m]")
            self.traj_ax.grid(True, alpha=0.3)
            self.traj_ax.legend()
            self.traj_ax.set_aspect('equal')
            
            self.traj_canvas.draw()
            
        except Exception as e:
            print(f"Error updating trajectory plot: {e}")

    def update_control_plots(self):
        """Update the control input plots."""
        
        try:
            if len(self.control_history) < 2:
                return
            
            # Get time and control data
            times = [t[0] for t in self.control_history]
            accelerations = [t[1].acceleration for t in self.control_history]
            steering_angles = [t[1].steering_angle for t in self.control_history]
            
            # Normalize time to start from 0
            if times:
                start_time = times[0]
                times = [t - start_time for t in times]
            
            # Clear and plot acceleration
            self.accel_ax.clear()
            self.accel_ax.plot(times, accelerations, 'b-', linewidth=2)
            self.accel_ax.set_title("Acceleration Command")
            self.accel_ax.set_ylabel("Acceleration [m/s²]")
            self.accel_ax.grid(True, alpha=0.3)
            self.accel_ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            # Clear and plot steering
            self.steer_ax.clear()
            self.steer_ax.plot(times, steering_angles, 'r-', linewidth=2)
            self.steer_ax.set_title("Steering Angle Command")
            self.steer_ax.set_xlabel("Time [s]")
            self.steer_ax.set_ylabel("Steering Angle [rad]")
            self.steer_ax.grid(True, alpha=0.3)
            self.steer_ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            self.control_fig.tight_layout()
            self.control_canvas.draw()
            
        except Exception as e:
            print(f"Error updating control plots: {e}")

    def update_performance_plots(self):
        """Update the performance plots."""
        
        try:
            # State error plot
            if len(self.error_history) > 1:
                error_times = [t[0] for t in self.error_history]
                error_values = [t[1] for t in self.error_history]
                
                if error_times:
                    start_time = error_times[0]
                    error_times = [t - start_time for t in error_times]
                
                self.error_ax.clear()
                self.error_ax.plot(error_times, error_values, 'g-', linewidth=2)
                self.error_ax.set_title("State Error")
                self.error_ax.set_ylabel("Error Magnitude")
                self.error_ax.grid(True, alpha=0.3)
            
            # Velocity plot
            if len(self.vehicle_history) > 1:
                vel_times = [t[0] for t in self.vehicle_history]
                velocities = [t[1].velocity for t in self.vehicle_history]
                
                if vel_times:
                    start_time = vel_times[0]
                    vel_times = [t - start_time for t in vel_times]
                
                self.vel_ax.clear()
                self.vel_ax.plot(vel_times, velocities, 'm-', linewidth=2, label='Actual')
                
                # Add reference velocity if available
                if self.reference_trajectory:
                    ref_velocities = [point['v'] for point in self.reference_trajectory]
                    if len(ref_velocities) > 0:
                        ref_times = np.linspace(0, vel_times[-1] if vel_times else 10, len(ref_velocities))
                        self.vel_ax.plot(ref_times, ref_velocities, 'b--', linewidth=2, 
                                       label='Reference', alpha=0.7)
                
                self.vel_ax.set_title("Velocity Profile")
                self.vel_ax.set_xlabel("Time [s]")
                self.vel_ax.set_ylabel("Velocity [m/s]")
                self.vel_ax.grid(True, alpha=0.3)
                self.vel_ax.legend()
            
            self.perf_fig.tight_layout()
            self.perf_canvas.draw()
            
        except Exception as e:
            print(f"Error updating performance plots: {e}")

    def update_diagnostics_text(self):
        """Update the diagnostics text display."""
        
        try:
            # Clear text
            self.diag_text.delete(1.0, tk.END)
            
            # Add current diagnostics
            diag_info = f"""LQR Controller Diagnostics
{'='*50}

Controller Status:
  Active: {self.current_metrics.controller_active}
  Emergency Stop: {self.current_metrics.emergency_stop}
  Path Ready: {self.current_metrics.path_ready}

Performance Metrics:
  Control Frequency: {self.current_metrics.control_frequency:.1f} Hz
  Average Control Time: {self.current_metrics.avg_control_time*1000:.2f} ms
  Maximum Control Time: {self.current_metrics.max_control_time*1000:.2f} ms
  Consecutive Failures: {self.current_metrics.consecutive_failures}
  State Error: {self.current_metrics.state_error:.4f}

Vehicle State:
  Position: ({self.current_vehicle_state.x:.3f}, {self.current_vehicle_state.y:.3f})
  Velocity: {self.current_vehicle_state.velocity:.3f} m/s
  Yaw: {self.current_vehicle_state.yaw:.3f} rad
  Steering Angle: {self.current_vehicle_state.steering_angle:.3f} rad

Control Commands:
  Acceleration: {self.current_control.acceleration:.3f} m/s²
  Steering: {self.current_control.steering_angle:.3f} rad
  Speed Command: {self.current_control.speed:.3f} m/s

Reference Trajectory:
  Number of Points: {len(self.reference_trajectory)}

Data History:
  Vehicle History Points: {len(self.vehicle_history)}
  Control History Points: {len(self.control_history)}
  Error History Points: {len(self.error_history)}

Last Updated: {time.strftime('%H:%M:%S')}
"""
            
            self.diag_text.insert(tk.END, diag_info)
            
        except Exception as e:
            print(f"Error updating diagnostics text: {e}")

    def update_plots(self):
        """Main update function called periodically."""
        
        if not self.animation_running:
            return
        
        try:
            # Update all displays
            self.update_status_labels()
            self.update_trajectory_plot()
            self.update_control_plots()
            self.update_performance_plots()
            self.update_diagnostics_text()
            
        except Exception as e:
            print(f"Error in update_plots: {e}")
        
        # Schedule next update
        self.root.after(100, self.update_plots)  # Update at 10 Hz

    def on_closing(self):
        """Handle application closing."""
        
        self.animation_running = False
        
        try:
            if hasattr(self, 'data_generator'):
                self.data_generator.stop()
        except:
            pass
        
        self.root.destroy()

    def run(self):
        """Run the GUI application."""
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


def main():
    """Main entry point."""
    
    try:
        print("Starting LQR Controller Visualizer (Standalone Mode)")
        print("This demo shows synthetic data to demonstrate the GUI features.")
        
        # Create and run GUI with synthetic data
        app = LQRVisualizerGUI(use_synthetic_data=True)
        app.run()
        
    except KeyboardInterrupt:
        print("\nLQR Visualizer interrupted by user")
    except Exception as e:
        print(f"LQR Visualizer error: {e}")
        if tk._default_root:
            messagebox.showerror("Error", f"Application error: {e}")


if __name__ == '__main__':
    main()
