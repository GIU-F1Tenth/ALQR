#!/usr/bin/env python3

"""
F1TENTH LQG Controller ROS2 Node

This ROS2 node implements a Linear Quadratic Gaussian (LQG) controller that combines:
1. Extended Kalman Filter for state estimation from noisy sensor data
2. LQR controller for optimal control computation

The node provides real-time control at ≥50 Hz for F1TENTH autonomous racing.

Key Features:
- Multi-sensor fusion (IMU, odometry, optional position updates)
- Real-time state estimation using Extended Kalman Filter
- Optimal control using existing LQR controller
- Comprehensive diagnostics and performance monitoring
- Safety monitoring and emergency stop capabilities

Author: Mohammed Azab <mohammed@azab.io>
License: MIT
Version: 1.0.0
"""

import rclpy
import numpy as np
import time
import traceback
from typing import Dict, List, Tuple, Optional
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# ROS2 message types
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool, Float32, Header
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from giu_f1t_interfaces.msg import VehicleStateArray

# Transformations
from tf_transformations import euler_from_quaternion

# LQG Controller components
try:
    from .lqg_controller import LQGController
    from .kinematic_bicycle_model import KinematicBicycleModel
except ImportError:
    # Fallback for standalone execution
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    from lqg_controller import LQGController
    from kinematic_bicycle_model import KinematicBicycleModel


class LQGControllerNode(Node):
    """
    ROS2 node implementing LQG controller for F1TENTH autonomous racing.

    The node subscribes to sensor topics, estimates vehicle state using EKF,
    computes optimal control using LQR, and publishes control commands.
    """

    def __init__(self):
        super().__init__('lqg_controller')

        # Load parameters
        self._load_parameters()

        # Initialize LQG controller
        self._initialize_lqg_controller()

        # Initialize ROS2 publishers and subscribers
        self._setup_ros_interface()

        # Control loop state
        self.reference_trajectory = None
        self.current_reference_state = np.zeros(4)
        self.emergency_stop = False
        self.path_ready = False

        # Performance monitoring
        self.control_loop_times = []
        self.control_frequency_counter = 0
        self.last_frequency_check = time.time()
        self.target_frequency = 1.0 / self.dt

        # Sensor data storage
        self.latest_imu_data = None
        self.latest_odom_data = None
        self.latest_position_data = None

        # Initialize control timer (50+ Hz)
        self.control_timer = self.create_timer(self.dt, self.control_loop_callback)

        # Diagnostics timer (5 Hz)
        self.diagnostics_timer = self.create_timer(0.2, self.publish_diagnostics)

        self.get_logger().info("🚀 LQG Controller Node initialized successfully")
        self.get_logger().info(f"📊 Target control frequency: {self.target_frequency:.1f} Hz")

    def _load_parameters(self):
        """Load parameters from ROS2 parameter server or config file."""

        # Default parameters
        default_params = {
            # Vehicle parameters
            'wheelbase': 0.33,
            'dt': 0.02,  # 50 Hz

            # LQR cost matrices
            'lqr_Q_diag': [10.0, 10.0, 1.0, 5.0],  # [x, y, v, theta] weights
            'lqr_R_diag': [0.1, 1.0],              # [acceleration, steering] weights
            'max_acceleration': 5.0,
            'max_steering': 0.5,

            # EKF parameters
            'initial_state': [0.0, 0.0, 0.0, 0.0],  # [x, y, v, theta]
            'initial_covariance_diag': [1.0, 1.0, 0.5, 0.1],
            'process_noise_diag': [0.01, 0.01, 0.1, 0.05],
            'measurement_noise_imu': 0.1,
            'measurement_noise_odom': 0.05,
            'measurement_noise_position': 0.1,

            # Safety parameters
            'max_speed': 8.0,
            'min_speed': -2.0,
            'emergency_brake_deceleration': 3.0,
            'obstacle_detection_enabled': True,
            'min_obstacle_distance': 1.0,

            # Topic names
            'control_topic': '/drive',
            'odom_topic': '/car_state/odom',
            'imu_topic': '/imu',
            'trajectory_topic': '/horizon_mapper/reference_trajectory',
            'path_ready_topic': '/horizon_mapper/path_ready',
            'scan_topic': '/scan'
        }

        # Override with ROS2 parameters (no config.py usage)
        for param_name, default_value in default_params.items():
            if isinstance(default_value, list):
                param_value = self.declare_parameter(param_name, default_value).value
            else:
                param_value = self.declare_parameter(param_name, default_value).value
            setattr(self, param_name, param_value)

        self.get_logger().info(f"🔧 Loaded {len(default_params)} parameters")

    def _initialize_lqg_controller(self):
        """Initialize the LQG controller with loaded parameters."""

        try:
            # Prepare LQR cost matrices
            Q = np.diag(self.lqr_Q_diag)
            R = np.diag(self.lqr_R_diag)

            # Prepare EKF initial conditions
            initial_state = np.array(self.initial_state)
            initial_covariance = np.diag(self.initial_covariance_diag)
            process_noise = np.diag(self.process_noise_diag)

            # Create LQG controller
            self.lqg_controller = LQGController(
                wheelbase=self.wheelbase,
                dt=self.dt,
                Q=Q,
                R=R,
                max_acceleration=self.max_acceleration,
                max_steering=self.max_steering,
                initial_state=initial_state,
                initial_covariance=initial_covariance,
                process_noise=process_noise,
                measurement_noise_imu=self.measurement_noise_imu,
                measurement_noise_odom=self.measurement_noise_odom,
                measurement_noise_position=self.measurement_noise_position,
                enable_logging=True,
                logger=self.get_logger()
            )

            self.get_logger().info("✅ LQG Controller initialized")

        except Exception as e:
            self.get_logger().error(f"❌ Failed to initialize LQG controller: {e}")
            raise

    def _setup_ros_interface(self):
        """Setup ROS2 publishers and subscribers."""

        # QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        control_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=5
        )

        # Publishers
        self.control_pub = self.create_publisher(
            AckermannDriveStamped, self.control_topic, control_qos)

        self.state_error_pub = self.create_publisher(
            Float32, '/lqg_controller/state_error', control_qos)

        self.estimated_state_pub = self.create_publisher(
            Odometry, '/lqg_controller/estimated_state', control_qos)

        self.diagnostics_pub = self.create_publisher(
            DiagnosticArray, '/lqg_controller/diagnostics', control_qos)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, sensor_qos)

        self.imu_sub = self.create_subscription(
            Imu, self.imu_topic, self.imu_callback, sensor_qos)

        self.trajectory_sub = self.create_subscription(
            VehicleStateArray, self.trajectory_topic,
            self.trajectory_callback, control_qos)

        self.path_ready_sub = self.create_subscription(
            Bool, self.path_ready_topic, self.path_ready_callback, control_qos)

        # Optional LiDAR subscription for safety
        if self.obstacle_detection_enabled:
            self.scan_sub = self.create_subscription(
                LaserScan, self.scan_topic, self.scan_callback, sensor_qos)

        self.get_logger().info("📡 ROS2 interface setup complete")

    def odom_callback(self, msg: Odometry):
        """Handle odometry messages for velocity measurements."""
        try:
            # Extract linear velocity
            linear_vel = msg.twist.twist.linear
            velocity = np.sqrt(linear_vel.x**2 + linear_vel.y**2)

            # Store for LQG controller
            self.latest_odom_data = {
                'velocity': velocity,
                'timestamp': time.time(),
                'position': np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
            }

        except Exception as e:
            self.get_logger().warn(f"Error in odometry callback: {e}")

    def imu_callback(self, msg: Imu):
        """Handle IMU messages for angular velocity measurements."""
        try:
            # Extract angular velocity
            angular_velocity = msg.angular_velocity.z

            # Store for LQG controller
            self.latest_imu_data = {
                'angular_velocity': angular_velocity,
                'timestamp': time.time()
            }

        except Exception as e:
            self.get_logger().warn(f"Error in IMU callback: {e}")

    def trajectory_callback(self, msg: VehicleStateArray):
        """Handle reference trajectory messages."""
        try:
            if len(msg.states) > 0:
                self.reference_trajectory = []
                for state in msg.states:
                    self.reference_trajectory.append({
                        'x': state.x,
                        'y': state.y,
                        'v': state.v,
                        'theta': state.theta
                    })

                # Use first state as current reference (or implement trajectory tracking)
                first_state = msg.states[0]
                self.current_reference_state = np.array([
                    first_state.x, first_state.y, first_state.v, first_state.theta
                ])

        except Exception as e:
            self.get_logger().warn(f"Error in trajectory callback: {e}")

    def path_ready_callback(self, msg: Bool):
        """Handle path ready status messages."""
        self.path_ready = msg.data

    def scan_callback(self, msg: LaserScan):
        """Handle LiDAR scan messages for obstacle detection."""
        # LiDAR processing can be added here if needed for LQG controller
        pass

    def control_loop_callback(self):
        """Main control loop callback - runs at target frequency."""

        loop_start_time = time.time()

        try:
            # Check if we have trajectory data
            if not self.path_ready or self.reference_trajectory is None:
                self._publish_zero_control("No path available")
                return

            # Check emergency stop
            if self.emergency_stop:
                self._publish_emergency_stop()
                return

            # Prepare sensor measurements
            imu_angular_velocity = None
            odom_velocity = None
            position_measurement = None

            # Get latest IMU data
            if (self.latest_imu_data and
                    time.time() - self.latest_imu_data['timestamp'] < 0.1):  # Fresh data
                imu_angular_velocity = self.latest_imu_data['angular_velocity']

            # Get latest odometry data
            if (self.latest_odom_data and
                    time.time() - self.latest_odom_data['timestamp'] < 0.1):  # Fresh data
                odom_velocity = self.latest_odom_data['velocity']
                # Optionally use position measurement
                position_measurement = self.latest_odom_data['position']

            # Compute LQG control
            control_input, estimated_state = self.lqg_controller.compute_control(
                reference_state=self.current_reference_state,
                imu_angular_velocity=imu_angular_velocity,
                odom_velocity=odom_velocity,
                position_measurement=position_measurement
            )

            # Update state estimate for next iteration
            self.lqg_controller.update_state_estimate(control_input)

            # Publish control command
            self._publish_control(control_input)

            # Publish estimated state
            self._publish_estimated_state(estimated_state)

            # Publish state error
            self._publish_state_error(estimated_state, self.current_reference_state)

            # Performance tracking
            loop_time = time.time() - loop_start_time
            self.control_loop_times.append(loop_time)
            self.control_frequency_counter += 1

            # Check for real-time performance
            if loop_time > self.dt * 0.8:  # Using >80% of available time
                self.get_logger().warn(f"Control loop slow: {loop_time*1000:.2f}ms > {self.dt*800:.2f}ms")

        except Exception as e:
            self.get_logger().error(f"Error in control loop: {e}")
            self.get_logger().error(traceback.format_exc())
            self._publish_zero_control("Control loop error")

    def _publish_control(self, control_input: np.ndarray):
        """Publish control command."""
        try:
            msg = AckermannDriveStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"

            msg.drive.acceleration = float(control_input[0])
            msg.drive.steering_angle = float(control_input[1])
            msg.drive.speed = float(control_input[0] * self.dt)  # Simple speed estimate

            self.control_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Error publishing control: {e}")

    def _publish_zero_control(self, reason: str):
        """Publish zero control command."""
        try:
            msg = AckermannDriveStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"

            msg.drive.acceleration = 0.0
            msg.drive.steering_angle = 0.0
            msg.drive.speed = 0.0

            self.control_pub.publish(msg)

            if reason:
                self.get_logger().debug(f"Zero control: {reason}")

        except Exception as e:
            self.get_logger().error(f"Error publishing zero control: {e}")

    def _publish_emergency_stop(self):
        """Publish emergency stop command."""
        try:
            msg = AckermannDriveStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"

            # Emergency brake
            msg.drive.acceleration = -self.emergency_brake_deceleration
            msg.drive.steering_angle = 0.0
            msg.drive.speed = 0.0

            self.control_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Error publishing emergency stop: {e}")

    def _publish_estimated_state(self, estimated_state: np.ndarray):
        """Publish estimated state as odometry message."""
        try:
            msg = Odometry()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "map"
            msg.child_frame_id = "base_link"

            # Position
            msg.pose.pose.position.x = float(estimated_state[0])
            msg.pose.pose.position.y = float(estimated_state[1])
            msg.pose.pose.position.z = 0.0

            # Orientation (from theta)
            theta = estimated_state[3]
            msg.pose.pose.orientation.z = np.sin(theta / 2.0)
            msg.pose.pose.orientation.w = np.cos(theta / 2.0)

            # Velocity
            msg.twist.twist.linear.x = float(estimated_state[2])

            # Add covariance information
            _, uncertainties = self.lqg_controller.get_state_estimate()
            msg.pose.covariance[0] = uncertainties[0]**2  # x variance
            msg.pose.covariance[7] = uncertainties[1]**2  # y variance
            msg.pose.covariance[35] = uncertainties[3]**2  # theta variance
            msg.twist.covariance[0] = uncertainties[2]**2  # v variance

            self.estimated_state_pub.publish(msg)

        except Exception as e:
            self.get_logger().warn(f"Error publishing estimated state: {e}")

    def _publish_state_error(self, estimated_state: np.ndarray, reference_state: np.ndarray):
        """Publish state tracking error."""
        try:
            error = np.linalg.norm(estimated_state - reference_state)

            msg = Float32()
            msg.data = float(error)

            self.state_error_pub.publish(msg)

        except Exception as e:
            self.get_logger().warn(f"Error publishing state error: {e}")

    def publish_diagnostics(self):
        """Publish comprehensive diagnostics information."""
        try:
            # Get performance metrics
            lqg_metrics = self.lqg_controller.get_performance_metrics()

            # Calculate control frequency
            current_time = time.time()
            time_elapsed = current_time - self.last_frequency_check
            if time_elapsed >= 1.0:  # Update every second
                actual_frequency = self.control_frequency_counter / time_elapsed
                self.last_frequency_check = current_time
                self.control_frequency_counter = 0
            else:
                actual_frequency = 0.0

            # Create diagnostics message
            diagnostics_msg = DiagnosticArray()
            diagnostics_msg.header.stamp = self.get_clock().now().to_msg()

            # LQG Controller status
            lqg_status = DiagnosticStatus()
            lqg_status.name = "lqg_controller"
            lqg_status.hardware_id = "lqg_controller_node"

            # Determine overall status
            if (lqg_metrics.get('system_healthy', False) and
                    self.path_ready and not self.emergency_stop):
                lqg_status.level = DiagnosticStatus.OK
                lqg_status.message = "LQG Controller operating normally"
            elif self.emergency_stop:
                lqg_status.level = DiagnosticStatus.ERROR
                lqg_status.message = "Emergency stop active"
            else:
                lqg_status.level = DiagnosticStatus.WARN
                lqg_status.message = "LQG Controller degraded performance"

            # Add key-value diagnostics
            diagnostics_values = [
                KeyValue(key="control_frequency", value=f"{actual_frequency:.1f}"),
                KeyValue(key="path_ready", value=str(self.path_ready)),
                KeyValue(key="emergency_stop", value=str(self.emergency_stop)),
                KeyValue(key="system_healthy", value=str(lqg_metrics.get('system_healthy', False))),
                KeyValue(key="ekf_healthy", value=str(lqg_metrics.get('ekf_is_healthy', False))),
                KeyValue(key="position_uncertainty", value=f"{lqg_metrics.get('current_position_uncertainty', 0.0):.3f}"),
                KeyValue(key="filter_health_percentage", value=f"{lqg_metrics.get('filter_health_percentage', 0.0):.1f}"),
            ]

            # Add timing information
            if self.control_loop_times:
                avg_loop_time = np.mean(self.control_loop_times[-100:])  # Last 100 samples
                max_loop_time = np.max(self.control_loop_times[-100:])
                diagnostics_values.extend([
                    KeyValue(key="avg_loop_time", value=f"{avg_loop_time*1000:.2f}"),
                    KeyValue(key="max_loop_time", value=f"{max_loop_time*1000:.2f}"),
                    KeyValue(key="real_time_factor", value=f"{lqg_metrics.get('real_time_factor', 0.0):.3f}")
                ])

            lqg_status.values = diagnostics_values
            diagnostics_msg.status.append(lqg_status)

            self.diagnostics_pub.publish(diagnostics_msg)

        except Exception as e:
            self.get_logger().warn(f"Error publishing diagnostics: {e}")


def main(args=None):
    """Main entry point for LQG controller node."""

    rclpy.init(args=args)

    try:
        # Create and run the LQG controller node
        lqg_node = LQGControllerNode()

        print("🚀 LQG Controller Node running...")
        print("📊 Publishing diagnostics on /lqg_controller/diagnostics")
        print("🎮 Publishing control commands on /drive")
        print("📍 Publishing estimated state on /lqg_controller/estimated_state")
        print("📈 Publishing state error on /lqg_controller/state_error")
        print("Press Ctrl+C to stop...")

        rclpy.spin(lqg_node)

    except KeyboardInterrupt:
        print("\n🛑 LQG Controller Node interrupted by user")
    except Exception as e:
        print(f"❌ LQG Controller Node error: {e}")
        traceback.print_exc()
    finally:
        try:
            lqg_node.destroy_node()
        except BaseException:
            pass
        rclpy.shutdown()
        print("👋 LQG Controller Node shutdown complete")


if __name__ == '__main__':
    main()
