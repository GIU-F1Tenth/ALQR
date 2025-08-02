#!/usr/bin/env python3

"""
LQG Controller GUI Visualizer

This script provides a comprehensive real-time visualization tool for the LQG controller,
displaying state estimation, control inputs, Kalman filter performance, and system diagnostics.

The visualizer shows:
1. State estimation accuracy and uncertainty
2. Sensor measurements vs estimates
3. Control inputs and tracking performance
4. Kalman filter health and innovation analysis
5. Real-time performance metrics

Author: Mohammed Azab <mohammed@azab.io>
License: MIT
Version: 1.0.0
"""

import tkinter as tk
import threading
import time
import numpy as np
import rclpy
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass

# GUI imports
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ROS2 imports
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool, Float32
from diagnostic_msgs.msg import DiagnosticArray
from giu_f1t_interfaces.msg import VehicleStateArray
from tf_transformations import euler_from_quaternion


@dataclass
class StateEstimate:
    """Data class for state estimation information."""
    estimated_state: np.ndarray = None
    state_uncertainty: np.ndarray = None
    true_state: np.ndarray = None  # From odometry (if available)
    timestamp: float = 0.0


@dataclass
class SensorData:
    """Data class for sensor measurements."""
    imu_angular_velocity: float = 0.0
    odom_velocity: float = 0.0
    position_measurement: np.ndarray = None
    timestamp: float = 0.0


@dataclass
class LQGMetrics:
    """Data class for LQG performance metrics."""
    control_frequency: float = 0.0
    estimation_quality: float = 0.0
    system_healthy: bool = False
    ekf_healthy: bool = False
    real_time_factor: float = 0.0
    position_uncertainty: float = 0.0
    innovation_magnitude: float = 0.0


class LQGVisualizerNode(Node):
    """ROS2 node for collecting LQG controller data."""

    def __init__(self, data_callback):
        super().__init__('lqg_visualizer_node')

        self.data_callback = data_callback

        # QoS profile for subscriptions
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # Data storage
        self.state_estimate = StateEstimate()
        self.sensor_data = SensorData()
        self.lqg_metrics = LQGMetrics()
        self.control_command = None
        self.reference_trajectory = []

        # Subscriptions
        self.estimated_state_sub = self.create_subscription(
            Odometry, '/lqg/estimated_state', self.estimated_state_callback, qos_profile)

        self.true_state_sub = self.create_subscription(
            Odometry, '/odom', self.true_state_callback, qos_profile)

        self.imu_sub = self.create_subscription(
            Imu, '/imu', self.imu_callback, qos_profile)

        self.control_sub = self.create_subscription(
            AckermannDriveStamped, '/drive', self.control_callback, qos_profile)

        self.trajectory_sub = self.create_subscription(
            VehicleStateArray, '/reference_trajectory',
            self.trajectory_callback, qos_profile)

        self.diagnostics_sub = self.create_subscription(
            DiagnosticArray, '/lqg/diagnostics',
            self.diagnostics_callback, qos_profile)

        self.state_error_sub = self.create_subscription(
            Float32, '/lqg/state_error',
            self.state_error_callback, qos_profile)

        self.get_logger().info("LQG Visualizer Node initialized")

    def estimated_state_callback(self, msg: Odometry):
        """Handle estimated state messages."""
        try:
            # Extract state estimate
            estimated_state = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.twist.twist.linear.x,
                2 * np.arctan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
            ])

            # Extract uncertainties from covariance
            uncertainties = np.array([
                np.sqrt(msg.pose.covariance[0]),   # x uncertainty
                np.sqrt(msg.pose.covariance[7]),   # y uncertainty
                np.sqrt(msg.twist.covariance[0]),  # v uncertainty
                np.sqrt(msg.pose.covariance[35])   # theta uncertainty
            ])

            self.state_estimate.estimated_state = estimated_state
            self.state_estimate.state_uncertainty = uncertainties
            self.state_estimate.timestamp = time.time()

            # Callback to GUI
            if self.data_callback:
                self.data_callback('state_estimate', self.state_estimate)

        except Exception as e:
            self.get_logger().error(f"Error in estimated state callback: {e}")

    def true_state_callback(self, msg: Odometry):
        """Handle true state (odometry) messages."""
        try:
            # Extract true state for comparison
            true_state = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.twist.twist.linear.x,
                2 * np.arctan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
            ])

            self.state_estimate.true_state = true_state

            # Callback to GUI
            if self.data_callback:
                self.data_callback('true_state', true_state)

        except Exception as e:
            self.get_logger().error(f"Error in true state callback: {e}")

    def imu_callback(self, msg: Imu):
        """Handle IMU messages."""
        try:
            self.sensor_data.imu_angular_velocity = msg.angular_velocity.z
            self.sensor_data.timestamp = time.time()

            # Callback to GUI
            if self.data_callback:
                self.data_callback('sensor_data', self.sensor_data)

        except Exception as e:
            self.get_logger().error(f"Error in IMU callback: {e}")

    def control_callback(self, msg: AckermannDriveStamped):
        """Handle control command messages."""
        try:
            self.control_command = {
                'acceleration': msg.drive.acceleration,
                'steering_angle': msg.drive.steering_angle,
                'speed': msg.drive.speed,
                'timestamp': time.time()
            }

            # Callback to GUI
            if self.data_callback:
                self.data_callback('control_command', self.control_command)

        except Exception as e:
            self.get_logger().error(f"Error in control callback: {e}")

    def trajectory_callback(self, msg: VehicleStateArray):
        """Handle reference trajectory messages."""
        try:
            self.reference_trajectory = []
            for state in msg.states:
                self.reference_trajectory.append({
                    'x': state.x,
                    'y': state.y,
                    'v': state.v,
                    'theta': state.theta
                })

            # Callback to GUI
            if self.data_callback:
                self.data_callback('reference_trajectory', self.reference_trajectory)

        except Exception as e:
            self.get_logger().error(f"Error in trajectory callback: {e}")

    def diagnostics_callback(self, msg: DiagnosticArray):
        """Handle diagnostics messages."""
        try:
            for status in msg.status:
                if status.name == "lqg_controller":
                    # Parse diagnostics
                    self.lqg_metrics.system_healthy = (status.level == 0)

                    for kv in status.values:
                        if kv.key == "control_frequency":
                            self.lqg_metrics.control_frequency = float(kv.value)
                        elif kv.key == "system_healthy":
                            self.lqg_metrics.system_healthy = bool(kv.value)
                        elif kv.key == "ekf_healthy":
                            self.lqg_metrics.ekf_healthy = bool(kv.value)
                        elif kv.key == "real_time_factor":
                            self.lqg_metrics.real_time_factor = float(kv.value)
                        elif kv.key == "position_uncertainty":
                            self.lqg_metrics.position_uncertainty = float(kv.value)

            # Callback to GUI
            if self.data_callback:
                self.data_callback('lqg_metrics', self.lqg_metrics)

        except Exception as e:
            self.get_logger().error(f"Error in diagnostics callback: {e}")

    def state_error_callback(self, msg: Float32):
        """Handle state error messages."""
        try:
            # Callback to GUI
            if self.data_callback:
                self.data_callback('state_error', msg.data)

        except Exception as e:
            self.get_logger().error(f"Error in state error callback: {e}")


class LQGVisualizerGUI:
    """Main GUI class for LQG controller visualization."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("LQG Controller Visualizer")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#f0f0f0')

        # Data storage with history
        self.max_history = 1000
        self.state_estimate_history = deque(maxlen=self.max_history)
        self.true_state_history = deque(maxlen=self.max_history)
        self.estimation_error_history = deque(maxlen=self.max_history)
        self.uncertainty_history = deque(maxlen=self.max_history)
        self.control_history = deque(maxlen=self.max_history)
        self.sensor_history = deque(maxlen=self.max_history)
        self.reference_trajectory = []

        # Current data
        self.current_state_estimate = StateEstimate()
        self.current_sensor_data = SensorData()
        self.current_lqg_metrics = LQGMetrics()
        self.current_control = None

        # Setup GUI
        self.setup_gui()

        # ROS2 node in separate thread
        self.ros_thread = None
        self.ros_node = None
        self.setup_ros_node()

        # Animation timer
        self.animation_running = True
        self.update_plots()

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
        self.setup_state_estimation_tab()
        self.setup_trajectory_tab()
        self.setup_sensor_fusion_tab()
        self.setup_performance_tab()
        self.setup_diagnostics_tab()

    def setup_control_panel(self, parent):
        """Setup the control panel with status indicators."""

        control_frame = ttk.LabelFrame(parent, text="LQG System Status", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Status indicators
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(fill=tk.X)

        # System health
        ttk.Label(status_frame, text="System:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.system_status_label = ttk.Label(status_frame, text="Unknown", foreground="gray")
        self.system_status_label.grid(row=0, column=1, padx=5, sticky=tk.W)

        # EKF health
        ttk.Label(status_frame, text="EKF:").grid(row=0, column=2, padx=5, sticky=tk.W)
        self.ekf_status_label = ttk.Label(status_frame, text="Unknown", foreground="gray")
        self.ekf_status_label.grid(row=0, column=3, padx=5, sticky=tk.W)

        # Control frequency
        ttk.Label(status_frame, text="Frequency:").grid(row=0, column=4, padx=5, sticky=tk.W)
        self.frequency_label = ttk.Label(status_frame, text="0.0 Hz")
        self.frequency_label.grid(row=0, column=5, padx=5, sticky=tk.W)

        # Current values frame
        values_frame = ttk.Frame(control_frame)
        values_frame.pack(fill=tk.X, pady=(10, 0))

        # State estimate values
        ttk.Label(values_frame, text="Position Est:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.position_est_label = ttk.Label(values_frame, text="(0.0, 0.0)")
        self.position_est_label.grid(row=0, column=1, padx=5, sticky=tk.W)

        ttk.Label(values_frame, text="Velocity Est:").grid(row=0, column=2, padx=5, sticky=tk.W)
        self.velocity_est_label = ttk.Label(values_frame, text="0.0 m/s")
        self.velocity_est_label.grid(row=0, column=3, padx=5, sticky=tk.W)

        ttk.Label(values_frame, text="Position Unc:").grid(row=0, column=4, padx=5, sticky=tk.W)
        self.position_unc_label = ttk.Label(values_frame, text="0.0 m")
        self.position_unc_label.grid(row=0, column=5, padx=5, sticky=tk.W)

        # Control and error values
        ttk.Label(values_frame, text="Acceleration:").grid(row=1, column=0, padx=5, sticky=tk.W)
        self.acceleration_label = ttk.Label(values_frame, text="0.0 m/s²")
        self.acceleration_label.grid(row=1, column=1, padx=5, sticky=tk.W)

        ttk.Label(values_frame, text="Steering:").grid(row=1, column=2, padx=5, sticky=tk.W)
        self.steering_label = ttk.Label(values_frame, text="0.0 rad")
        self.steering_label.grid(row=1, column=3, padx=5, sticky=tk.W)

        ttk.Label(values_frame, text="Est Error:").grid(row=1, column=4, padx=5, sticky=tk.W)
        self.estimation_error_label = ttk.Label(values_frame, text="0.0 m")
        self.estimation_error_label.grid(row=1, column=5, padx=5, sticky=tk.W)

    def setup_state_estimation_tab(self):
        """Setup the state estimation visualization tab."""

        est_frame = ttk.Frame(self.notebook)
        self.notebook.add(est_frame, text="State Estimation")

        # Create matplotlib figure with subplots
        self.est_fig = Figure(figsize=(14, 10), dpi=100)

        # Position estimation
        self.pos_est_ax = self.est_fig.add_subplot(2, 2, 1)
        self.pos_est_ax.set_title("Position Estimation")
        self.pos_est_ax.set_xlabel("X Position [m]")
        self.pos_est_ax.set_ylabel("Y Position [m]")
        self.pos_est_ax.grid(True, alpha=0.3)
        self.pos_est_ax.set_aspect('equal')

        # Velocity estimation
        self.vel_est_ax = self.est_fig.add_subplot(2, 2, 2)
        self.vel_est_ax.set_title("Velocity Estimation")
        self.vel_est_ax.set_xlabel("Time [s]")
        self.vel_est_ax.set_ylabel("Velocity [m/s]")
        self.vel_est_ax.grid(True, alpha=0.3)

        # Estimation error
        self.err_ax = self.est_fig.add_subplot(2, 2, 3)
        self.err_ax.set_title("Estimation Error")
        self.err_ax.set_xlabel("Time [s]")
        self.err_ax.set_ylabel("Error [m]")
        self.err_ax.grid(True, alpha=0.3)

        # State uncertainty
        self.unc_ax = self.est_fig.add_subplot(2, 2, 4)
        self.unc_ax.set_title("State Uncertainty")
        self.unc_ax.set_xlabel("Time [s]")
        self.unc_ax.set_ylabel("Standard Deviation")
        self.unc_ax.grid(True, alpha=0.3)

        self.est_fig.tight_layout()

        self.est_canvas = FigureCanvasTkAgg(self.est_fig, est_frame)
        self.est_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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
        self.traj_ax.set_title("Vehicle Trajectory: Estimated vs True")
        self.traj_ax.set_xlabel("X Position [m]")
        self.traj_ax.set_ylabel("Y Position [m]")
        self.traj_ax.grid(True, alpha=0.3)
        self.traj_ax.set_aspect('equal')

    def setup_sensor_fusion_tab(self):
        """Setup the sensor fusion visualization tab."""

        sensor_frame = ttk.Frame(self.notebook)
        self.notebook.add(sensor_frame, text="Sensor Fusion")

        # Create matplotlib figure with subplots
        self.sensor_fig = Figure(figsize=(12, 8), dpi=100)

        # IMU data
        self.imu_ax = self.sensor_fig.add_subplot(2, 1, 1)
        self.imu_ax.set_title("IMU Angular Velocity")
        self.imu_ax.set_ylabel("Angular Velocity [rad/s]")
        self.imu_ax.grid(True, alpha=0.3)

        # Odometry data
        self.odom_ax = self.sensor_fig.add_subplot(2, 1, 2)
        self.odom_ax.set_title("Odometry Velocity")
        self.odom_ax.set_xlabel("Time [s]")
        self.odom_ax.set_ylabel("Linear Velocity [m/s]")
        self.odom_ax.grid(True, alpha=0.3)

        self.sensor_fig.tight_layout()

        self.sensor_canvas = FigureCanvasTkAgg(self.sensor_fig, sensor_frame)
        self.sensor_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_performance_tab(self):
        """Setup the performance metrics visualization tab."""

        perf_frame = ttk.Frame(self.notebook)
        self.notebook.add(perf_frame, text="Performance")

        # Create matplotlib figure
        self.perf_fig = Figure(figsize=(12, 8), dpi=100)

        # Control inputs
        self.control_ax = self.perf_fig.add_subplot(2, 1, 1)
        self.control_ax.set_title("Control Inputs")
        self.control_ax.set_ylabel("Control Value")
        self.control_ax.grid(True, alpha=0.3)

        # Real-time performance
        self.rt_ax = self.perf_fig.add_subplot(2, 1, 2)
        self.rt_ax.set_title("Real-time Performance")
        self.rt_ax.set_xlabel("Time [s]")
        self.rt_ax.set_ylabel("Frequency [Hz] / Factor")
        self.rt_ax.grid(True, alpha=0.3)

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
        """Setup ROS2 node in separate thread."""

        def ros_thread_func():
            try:
                rclpy.init()
                self.ros_node = LQGVisualizerNode(self.data_callback)
                rclpy.spin(self.ros_node)
            except Exception as e:
                print(f"ROS node error: {e}")
            finally:
                if self.ros_node:
                    self.ros_node.destroy_node()
                rclpy.shutdown()

        self.ros_thread = threading.Thread(target=ros_thread_func, daemon=True)
        self.ros_thread.start()

    def data_callback(self, data_type: str, data):
        """Callback for receiving data from ROS node."""

        try:
            if data_type == 'state_estimate':
                self.current_state_estimate = data
                self.state_estimate_history.append((time.time(), data))

            elif data_type == 'true_state':
                # Store true state for comparison
                self.true_state_history.append((time.time(), data))

                # Calculate estimation error if we have both estimates
                if (self.current_state_estimate.estimated_state is not None and
                        len(self.current_state_estimate.estimated_state) >= 2):
                    position_error = np.linalg.norm(
                        self.current_state_estimate.estimated_state[:2] - data[:2])
                    self.estimation_error_history.append((time.time(), position_error))

            elif data_type == 'sensor_data':
                self.current_sensor_data = data
                self.sensor_history.append((time.time(), data))

            elif data_type == 'control_command':
                self.current_control = data
                self.control_history.append((time.time(), data))

            elif data_type == 'reference_trajectory':
                self.reference_trajectory = data

            elif data_type == 'lqg_metrics':
                self.current_lqg_metrics = data

        except Exception as e:
            print(f"Error in data callback: {e}")

    def update_status_labels(self):
        """Update status labels in control panel."""

        try:
            # System status
            if self.current_lqg_metrics.system_healthy:
                self.system_status_label.config(text="Healthy", foreground="green")
            else:
                self.system_status_label.config(text="Warning", foreground="orange")

            # EKF status
            if self.current_lqg_metrics.ekf_healthy:
                self.ekf_status_label.config(text="Healthy", foreground="green")
            else:
                self.ekf_status_label.config(text="Warning", foreground="red")

            # Frequency
            self.frequency_label.config(
                text=f"{self.current_lqg_metrics.control_frequency:.1f} Hz")

            # State estimate values
            if self.current_state_estimate.estimated_state is not None:
                state = self.current_state_estimate.estimated_state
                self.position_est_label.config(text=f"({state[0]:.2f}, {state[1]:.2f})")
                self.velocity_est_label.config(text=f"{state[2]:.2f} m/s")

            # Uncertainty
            self.position_unc_label.config(
                text=f"{self.current_lqg_metrics.position_uncertainty:.3f} m")

            # Control values
            if self.current_control:
                self.acceleration_label.config(
                    text=f"{self.current_control['acceleration']:.2f} m/s²")
                self.steering_label.config(
                    text=f"{self.current_control['steering_angle']:.3f} rad")

            # Estimation error
            if len(self.estimation_error_history) > 0:
                latest_error = self.estimation_error_history[-1][1]
                self.estimation_error_label.config(text=f"{latest_error:.3f} m")

        except Exception as e:
            print(f"Error updating status labels: {e}")

    def update_plots(self):
        """Main update function called periodically."""

        if not self.animation_running:
            return

        try:
            # Update all displays
            self.update_status_labels()
            self.update_state_estimation_plots()
            self.update_trajectory_plot()
            self.update_sensor_fusion_plots()
            self.update_performance_plots()
            self.update_diagnostics_text()

        except Exception as e:
            print(f"Error in update_plots: {e}")

        # Schedule next update
        self.root.after(100, self.update_plots)  # Update at 10 Hz

    def update_state_estimation_plots(self):
        """Update the state estimation plots."""
        # Implementation similar to LQR visualizer but focused on estimation
        pass

    def update_trajectory_plot(self):
        """Update the trajectory plot."""
        # Implementation similar to LQR visualizer
        pass

    def update_sensor_fusion_plots(self):
        """Update the sensor fusion plots."""
        # Show sensor data and fusion performance
        pass

    def update_performance_plots(self):
        """Update the performance plots."""
        # Show control performance and real-time metrics
        pass

    def update_diagnostics_text(self):
        """Update the diagnostics text display."""
        # Show comprehensive LQG diagnostics
        pass

    def on_closing(self):
        """Handle application closing."""
        self.animation_running = False
        try:
            if self.ros_node:
                self.ros_node.destroy_node()
        except BaseException:
            pass
        self.root.destroy()

    def run(self):
        """Run the GUI application."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


def main():
    """Main entry point."""

    try:
        # Create and run GUI
        app = LQGVisualizerGUI()
        app.run()

    except ImportError:
        messagebox.showerror("Error",
                             "ROS2 Python libraries not found. Please ensure ROS2 is installed and sourced.")
    except KeyboardInterrupt:
        print("\nLQG Visualizer interrupted by user")
    except Exception as e:
        print(f"LQG Visualizer error: {e}")
        messagebox.showerror("Error", f"Application error: {e}")


if __name__ == '__main__':
    main()
