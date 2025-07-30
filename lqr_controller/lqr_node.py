#!/usr/bin/env python3

"""
F1TENTH LQR Controller Node

This ROS2 node implements a Linear Quadratic Regulator (LQR) controller for trajectory tracking
in F1TENTH autonomous racing cars.

Author: Mohammed Azab <mohammed@azab.io>
License: MIT
Version: 1.0.0
"""

import rclpy
import numpy as np
import time
import traceback
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool, Float32
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from giu_f1t_interfaces.msg import VehicleStateArray
from tf_transformations import euler_from_quaternion

try:
    from .lqr_controller import LQRController
    from .kinematic_bicycle_model import KinematicBicycleModel
except ImportError:
    from lqr_controller import LQRController
    from kinematic_bicycle_model import KinematicBicycleModel
    

# Import configuration defaults
import sys, os
try:
    # Try multiple paths to find config
    possible_config_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'config'),  # Source tree
        '/home/mohammedazab/ws/src/race_stack/lqr_contoller/config',  # Absolute path (fixed typo)
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')  # Alternative relative
    ]
    
    config_imported = False
    for config_path in possible_config_paths:
        abs_path = os.path.abspath(config_path)
        if os.path.exists(abs_path):
            if abs_path not in sys.path:
                sys.path.insert(0, abs_path)
            try:
                import config
                print(f"✅ Using config.py from {abs_path}")
                config_imported = True
                break
            except ImportError as e:
                print(f"⚠️ Failed to import from {abs_path}: {e}")
                continue
        else:
            print(f"❌ Path does not exist: {abs_path}")
    
    if not config_imported:
        raise ImportError("Config module not found in any expected location")
    
    # Create a config object from the module variables
    class ConfigWrapper:
        def __init__(self, config_module):
            # Vehicle Parameters
            self.wheelbase = getattr(config_module, 'wheelbase', 0.33)
            self.dt = getattr(config_module, 'dt', 0.05)
            
            # Control Limits
            self.max_acceleration = getattr(config_module, 'max_acceleration', 5.0)
            self.max_deceleration = getattr(config_module, 'max_deceleration', 5.0)
            self.max_steering_angle = getattr(config_module, 'max_steering_angle', 0.5)
            self.min_speed = getattr(config_module, 'min_speed', 0.1)
            self.max_speed = getattr(config_module, 'max_speed', 8.0)
            
            # LQR Cost Function Weights
            self.position_weight = getattr(config_module, 'position_weight', 10.0)
            self.velocity_weight = getattr(config_module, 'velocity_weight', 1.0)
            self.heading_weight = getattr(config_module, 'heading_weight', 5.0)
            self.acceleration_weight = getattr(config_module, 'acceleration_weight', 0.1)
            self.steering_weight = getattr(config_module, 'steering_weight', 1.0)
            
            # Control Parameters
            self.control_hz = getattr(config_module, 'control_hz', 20.0)
            self.lookahead_distance = getattr(config_module, 'lookahead_distance', 0.5)
            self.enable_feedforward = getattr(config_module, 'enable_feedforward', True)
            
            # Safety Parameters
            self.enable_safety_checks = getattr(config_module, 'enable_safety_checks', True)
            self.safety_timeout = getattr(config_module, 'safety_timeout', 1.0)
            self.emergency_brake_threshold = getattr(config_module, 'emergency_brake_threshold', 2.0)
            
            # ROS2 Topics
            self.odom_topic = getattr(config_module, 'odom_topic', "/car_state/odom")
            self.reference_topic = getattr(config_module, 'reference_topic', "/horizon_mapper/reference_trajectory")
            self.status_topic = getattr(config_module, 'status_topic', "/horizon_mapper/path_ready")
            self.control_topic = getattr(config_module, 'control_topic', "/drive")
            self.pose_estimate_topic = getattr(config_module, 'pose_estimate_topic', "/initialpose")
            
            # Quality of Service
            self.qos_depth = getattr(config_module, 'qos_depth', 10)
            
            # Logging and Debug
            self.enable_logging = getattr(config_module, 'enable_logging', True)
            self.debug_logging_enabled = getattr(config_module, 'debug_logging_enabled', False)
            self.performance_logging_enabled = getattr(config_module, 'performance_logging_enabled', True)
            self.log_frequency_divider = getattr(config_module, 'log_frequency_divider', 10)
    
    default_config = ConfigWrapper(config)
        
except ImportError as e:
    # Fallback if config.py is not available
    class DefaultConfig:
        def __init__(self):
            # Vehicle Parameters
            self.wheelbase = 0.33
            self.dt = 0.05
            
            # Control Limits
            self.max_acceleration = 5.0
            self.max_deceleration = 5.0
            self.max_steering_angle = 0.5
            self.min_speed = 0.1
            self.max_speed = 8.0
            
            # LQR Cost Function Weights
            self.position_weight = 10.0
            self.velocity_weight = 1.0
            self.heading_weight = 5.0
            self.acceleration_weight = 0.1
            self.steering_weight = 1.0
            
            # Control Parameters
            self.control_hz = 20.0
            self.lookahead_distance = 0.5
            self.enable_feedforward = True
            
            # Safety Parameters
            self.enable_safety_checks = True
            self.safety_timeout = 1.0
            self.emergency_brake_threshold = 2.0
            
            # ROS2 Topics
            self.odom_topic = "/car_state/odom"
            self.reference_topic = "/horizon_mapper/reference_trajectory"
            self.status_topic = "/horizon_mapper/path_ready"
            self.control_topic = "/drive"
            self.pose_estimate_topic = "/initialpose"
            
            # Quality of Service
            self.qos_depth = 10
            
            # Logging and Debug
            self.enable_logging = True
            self.debug_logging_enabled = False
            self.performance_logging_enabled = True
            self.log_frequency_divider = 10

    default_config = DefaultConfig()


class LQRNode(Node):
    """
    ROS2 node implementing LQR controller for F1TENTH trajectory tracking.

    This node:
    1. Subscribes to vehicle odometry and reference trajectories
    2. Processes reference trajectory into state vectors
    3. Computes optimal control using LQR
    4. Publishes control commands and diagnostics
    """

    def __init__(self):
        super().__init__('lqr_controller_node')

        self._declare_parameters()
        self._load_parameters()
        self._initialize_lqr_controller()
        self._initialize_state()
        self._setup_subscriptions()
        self._setup_publishers()
        self._setup_timers()

        self.get_logger().info("LQR Controller Node has been started 🏁")

    def _declare_parameters(self):
        """Declare all ROS2 parameters with defaults."""

        # Vehicle parameters
        self.declare_parameter('wheelbase', default_config.wheelbase)
        self.declare_parameter('dt', default_config.dt)

        # Control limits
        self.declare_parameter('max_acceleration', default_config.max_acceleration)
        self.declare_parameter('max_deceleration', default_config.max_deceleration)
        self.declare_parameter('max_steering_angle', default_config.max_steering_angle)
        self.declare_parameter('min_speed', default_config.min_speed)
        self.declare_parameter('max_speed', default_config.max_speed)

        # LQR cost function weights
        self.declare_parameter('lqr_weights.position_weight', default_config.position_weight)
        self.declare_parameter('lqr_weights.velocity_weight', default_config.velocity_weight)
        self.declare_parameter('lqr_weights.heading_weight', default_config.heading_weight)
        self.declare_parameter('lqr_weights.acceleration_weight', default_config.acceleration_weight)
        self.declare_parameter('lqr_weights.steering_weight', default_config.steering_weight)

        # Control parameters
        self.declare_parameter('control_hz', default_config.control_hz)
        self.declare_parameter('lookahead_distance', default_config.lookahead_distance)
        self.declare_parameter('enable_feedforward', default_config.enable_feedforward)

        # Safety parameters
        self.declare_parameter('enable_safety_checks', default_config.enable_safety_checks)
        self.declare_parameter('safety_timeout', default_config.safety_timeout)
        self.declare_parameter('emergency_brake_threshold', default_config.emergency_brake_threshold)

        # Topics
        self.declare_parameter('odom_topic', default_config.odom_topic)
        self.declare_parameter('reference_topic', default_config.reference_topic)
        self.declare_parameter('status_topic', default_config.status_topic)
        self.declare_parameter('control_topic', default_config.control_topic)
        self.declare_parameter('pose_estimate_topic', default_config.pose_estimate_topic)

        # QoS and logging
        self.declare_parameter('qos_depth', default_config.qos_depth)
        self.declare_parameter('enable_logging', default_config.enable_logging)
        self.declare_parameter('debug_logging_enabled', default_config.debug_logging_enabled)
        self.declare_parameter('performance_logging_enabled', default_config.performance_logging_enabled)
        self.declare_parameter('log_frequency_divider', default_config.log_frequency_divider)

    def _load_parameters(self):
        """Load all parameters from ROS2 parameter server."""

        # Vehicle parameters
        self.wheelbase = self.get_parameter('wheelbase').value
        self.dt = self.get_parameter('dt').value

        # Control limits
        self.max_acceleration = self.get_parameter('max_acceleration').value
        self.max_deceleration = self.get_parameter('max_deceleration').value
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        self.min_speed = self.get_parameter('min_speed').value
        self.max_speed = self.get_parameter('max_speed').value

        # LQR weights
        self.lqr_weights = {
            'position_weight': self.get_parameter('lqr_weights.position_weight').value,
            'velocity_weight': self.get_parameter('lqr_weights.velocity_weight').value,
            'heading_weight': self.get_parameter('lqr_weights.heading_weight').value,
            'acceleration_weight': self.get_parameter('lqr_weights.acceleration_weight').value,
            'steering_weight': self.get_parameter('lqr_weights.steering_weight').value,
        }

        # Control parameters
        self.control_hz = self.get_parameter('control_hz').value
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.enable_feedforward = self.get_parameter('enable_feedforward').value

        # Safety parameters
        self.enable_safety_checks = self.get_parameter('enable_safety_checks').value
        self.safety_timeout = self.get_parameter('safety_timeout').value
        self.emergency_brake_threshold = self.get_parameter('emergency_brake_threshold').value

        # Topics
        self.odom_topic = self.get_parameter('odom_topic').value
        self.reference_topic = self.get_parameter('reference_topic').value
        self.status_topic = self.get_parameter('status_topic').value
        self.control_topic = self.get_parameter('control_topic').value
        self.pose_estimate_topic = self.get_parameter('pose_estimate_topic').value

        # QoS and logging
        self.qos_depth = self.get_parameter('qos_depth').value
        self.enable_logging = self.get_parameter('enable_logging').value
        self.debug_logging_enabled = self.get_parameter('debug_logging_enabled').value
        self.performance_logging_enabled = self.get_parameter('performance_logging_enabled').value
        self.log_frequency_divider = self.get_parameter('log_frequency_divider').value

    def _initialize_lqr_controller(self):
        """Initialize the LQR controller with loaded parameters."""

        try:
            # Set up cost matrices from parameters
            Q = np.diag([
                self.lqr_weights['position_weight'],  # x position
                self.lqr_weights['position_weight'],  # y position
                self.lqr_weights['velocity_weight'],  # velocity
                self.lqr_weights['heading_weight']    # heading
            ])

            R = np.diag([
                self.lqr_weights['acceleration_weight'],  # acceleration
                self.lqr_weights['steering_weight']       # steering
            ])

            # Initialize LQR controller
            self.lqr_controller = LQRController(
                wheelbase=self.wheelbase,
                dt=self.dt,
                Q=Q,
                R=R,
                max_acceleration=self.max_acceleration,
                max_steering=self.max_steering_angle,
                enable_logging=self.enable_logging,
                logger=self.get_logger()
            )

            # Initialize kinematic model for reference computation
            self.kinematic_model = KinematicBicycleModel(self.wheelbase, self.dt)

            self.get_logger().info("✅ LQR Controller initialized successfully")
            self.get_logger().info(f"   - Wheelbase: {self.wheelbase}m")
            self.get_logger().info(f"   - Control frequency: {self.control_hz}Hz")
            self.get_logger().info(f"   - Max acceleration: {self.max_acceleration}m/s²")
            self.get_logger().info(f"   - Max steering: {self.max_steering_angle}rad")

        except Exception as e:
            self.get_logger().error(f"❌ Failed to initialize LQR controller: {e}")
            raise e

    def _initialize_state(self):
        """Initialize node state variables."""

        # Vehicle state
        self.current_pose = None
        self.current_velocity = 0.0
        self.current_yaw = 0.0
        self.current_steering_angle = 0.0

        # Reference trajectory
        self.reference_trajectory = []
        self.path_ready = False

        # Safety and performance tracking
        self.last_trajectory_time = None
        self.last_odom_time = None
        self.control_active = False
        self.emergency_stop = False
        self.emergency_stop_time = None
        self.emergency_recovery_timeout = 2.0

        # Performance metrics
        self.control_loop_times = []
        self.state_initialized = False
        self.first_odom_received = False
        self.consecutive_failures = 0
        self.last_successful_solve_time = None

        # Debug logging counters
        self.control_iteration_count = 0

        self.get_logger().info("Node state initialized")

    def _setup_subscriptions(self):
        """Set up ROS2 subscriptions."""

        # Vehicle odometry subscription
        self.odom_subscription = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            self.qos_depth
        )

        # Reference trajectory subscription
        self.trajectory_subscription = self.create_subscription(
            VehicleStateArray,
            self.reference_topic,
            self.trajectory_callback,
            self.qos_depth
        )

        # Path ready status subscription
        self.status_subscription = self.create_subscription(
            Bool,
            self.status_topic,
            self.status_callback,
            self.qos_depth
        )

        # RViz pose estimate subscription (for debugging)
        self.pose_estimate_subscription = self.create_subscription(
            PoseStamped,
            self.pose_estimate_topic,
            self.pose_estimate_callback,
            1
        )

        self.get_logger().info("Subscriptions set up successfully")

    def _setup_publishers(self):
        """Set up ROS2 publishers."""

        # Control command publisher
        self.control_publisher = self.create_publisher(
            AckermannDriveStamped,
            self.control_topic,
            self.qos_depth
        )

        # Diagnostics publisher
        self.diagnostics_publisher = self.create_publisher(
            DiagnosticArray,
            '/lqr_controller/diagnostics',
            self.qos_depth
        )

        # Debug publishers
        self.state_error_publisher = self.create_publisher(
            Float32,
            '/lqr_controller/state_error',
            self.qos_depth
        )

        self.get_logger().info("Publishers set up successfully")

    def _setup_timers(self):
        """Set up ROS2 timers."""

        # Main control timer
        control_period = 1.0 / self.control_hz
        self.control_timer = self.create_timer(
            control_period,
            self.control_callback
        )

        # Diagnostics timer (5Hz)
        self.diagnostics_timer = self.create_timer(
            0.2,
            self.publish_diagnostics
        )

        self.get_logger().info(f"Timers set up successfully (control: {self.control_hz}Hz)")

    def odom_callback(self, msg: Odometry):
        """Handle odometry messages."""

        try:
            self.current_pose = msg.pose.pose

            # Extract velocity
            linear_vel = msg.twist.twist.linear
            self.current_velocity = np.sqrt(linear_vel.x**2 + linear_vel.y**2)

            # Extract heading angle
            orientation = msg.pose.pose.orientation
            _, _, self.current_yaw = euler_from_quaternion([
                orientation.x, orientation.y, orientation.z, orientation.w
            ])

            self.last_odom_time = time.time()

            if not self.first_odom_received:
                self.first_odom_received = True
                self.get_logger().info("First odometry message received")

        except Exception as e:
            self.get_logger().error(f"Error processing odometry: {e}")

    def trajectory_callback(self, msg: VehicleStateArray):
        """Handle reference trajectory messages."""

        try:
            if len(msg.states) == 0:
                self.get_logger().warning("Received empty trajectory")
                return

            # Convert VehicleStateArray to internal trajectory format
            self.reference_trajectory = []
            for state in msg.states:
                trajectory_point = {
                    'x': state.x,
                    'y': state.y,
                    'v': state.v,
                    'theta': state.theta,
                    'delta': state.delta
                }
                self.reference_trajectory.append(trajectory_point)

            self.last_trajectory_time = time.time()

            if self.debug_logging_enabled:
                self.get_logger().info(f"Received trajectory with {len(self.reference_trajectory)} points")

        except Exception as e:
            self.get_logger().error(f"Error processing trajectory: {e}")

    def status_callback(self, msg: Bool):
        """Handle path ready status messages."""
        self.path_ready = msg.data

        if self.debug_logging_enabled:
            self.get_logger().info(f"Path ready status: {self.path_ready}")

    def pose_estimate_callback(self, msg: PoseStamped):
        """Handle RViz pose estimate (for debugging)."""
        if self.debug_logging_enabled:
            self.get_logger().info("Received pose estimate from RViz")

    def get_current_state(self) -> np.ndarray:
        """Get current vehicle state as numpy array."""

        if self.current_pose is None:
            return np.zeros(4)

        return np.array([
            self.current_pose.position.x,
            self.current_pose.position.y,
            self.current_velocity,
            self.current_yaw
        ])

    def find_closest_reference_point(self, current_state: np.ndarray) -> int:
        """Find the closest point in the reference trajectory."""

        if not self.reference_trajectory:
            return 0

        min_distance = float('inf')
        closest_index = 0

        for i, point in enumerate(self.reference_trajectory):
            dx = current_state[0] - point['x']
            dy = current_state[1] - point['y']
            distance = np.sqrt(dx**2 + dy**2)

            if distance < min_distance:
                min_distance = distance
                closest_index = i

        return closest_index

    def get_reference_state(self, current_state: np.ndarray) -> np.ndarray:
        """Get reference state for LQR controller."""

        if not self.reference_trajectory:
            return current_state  # Return current state if no reference

        # Find closest reference point
        closest_index = self.find_closest_reference_point(current_state)

        # Use lookahead distance to find target point
        lookahead_points = max(1, int(self.lookahead_distance / self.dt))
        target_index = (closest_index + lookahead_points) % len(self.reference_trajectory)

        target_point = self.reference_trajectory[target_index]

        return np.array([
            target_point['x'],
            target_point['y'],
            target_point['v'],
            target_point['theta']
        ])

    def get_feedforward_control(self, reference_state: np.ndarray) -> np.ndarray:
        """Compute feedforward control from reference trajectory."""

        if not self.enable_feedforward or not self.reference_trajectory:
            return np.zeros(2)

        # Simple feedforward: assume reference acceleration = 0, steering from reference
        try:
            closest_index = self.find_closest_reference_point(reference_state)
            target_point = self.reference_trajectory[closest_index]

            # Use reference steering angle and zero acceleration as feedforward
            return np.array([0.0, target_point['delta']])

        except Exception:
            return np.zeros(2)

    def check_safety_conditions(self) -> bool:
        """Check if it's safe to send control commands."""

        if not self.enable_safety_checks:
            return True

        current_time = time.time()

        # Check if we have recent odometry
        if (self.last_odom_time is None or
                current_time - self.last_odom_time > self.safety_timeout):
            return False

        # Check if we have recent trajectory
        if (self.last_trajectory_time is None or
                current_time - self.last_trajectory_time > self.safety_timeout):
            return False

        # Check if path is ready
        if not self.path_ready:
            return False

        return True

    def control_callback(self):
        """Main control loop callback."""

        start_time = time.time()
        self.control_iteration_count += 1

        try:
            # Check safety conditions
            if not self.check_safety_conditions():
                self.publish_emergency_stop()
                return

            # Get current state
            current_state = self.get_current_state()

            # Validate current state
            if not self.kinematic_model.validate_state(current_state):
                self.get_logger().warning("Invalid current state, stopping")
                self.publish_emergency_stop()
                return

            # Get reference state
            reference_state = self.get_reference_state(current_state)

            # Get feedforward control
            feedforward_control = self.get_feedforward_control(reference_state)

            # Compute LQR control
            control = self.lqr_controller.compute_control(
                current_state,
                reference_state,
                feedforward_control
            )

            # Validate control output
            if not self.kinematic_model.validate_control(control, self.max_acceleration, self.max_steering_angle):
                self.get_logger().warning("Invalid control output, stopping")
                self.publish_emergency_stop()
                return

            # Publish control command
            self.publish_control_command(control)

            # Track performance
            if self.performance_logging_enabled:
                solve_time = time.time() - start_time
                self.control_loop_times.append(solve_time)

                # Keep history bounded
                if len(self.control_loop_times) > 1000:
                    self.control_loop_times = self.control_loop_times[-1000:]

            # Reset failure count on success
            self.consecutive_failures = 0
            self.last_successful_solve_time = time.time()
            self.control_active = True

            # Debug logging
            if (self.debug_logging_enabled and
                    self.control_iteration_count % self.log_frequency_divider == 0):
                state_error = np.linalg.norm(current_state - reference_state)
                self.get_logger().info(
                    f"Control iteration {self.control_iteration_count}: "
                    f"error={state_error:.3f}, control=[{control[0]:.3f}, {control[1]:.3f}]"
                )

        except Exception as e:
            self.consecutive_failures += 1
            self.get_logger().error(f"Control loop error: {e}")

            if self.consecutive_failures > 5:
                self.get_logger().error("Too many consecutive control failures, emergency stop")
                self.publish_emergency_stop()

    def publish_control_command(self, control: np.ndarray):
        """Publish Ackermann drive command."""

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        # Set control values
        msg.drive.speed = max(self.min_speed, min(self.max_speed, self.current_velocity + control[0] * self.dt))
        msg.drive.steering_angle = control[1]
        msg.drive.acceleration = control[0]

        self.control_publisher.publish(msg)

        # Publish state error for debugging
        if self.debug_logging_enabled:
            current_state = self.get_current_state()
            reference_state = self.get_reference_state(current_state)
            state_error = np.linalg.norm(current_state - reference_state)

            error_msg = Float32()
            error_msg.data = float(state_error)
            self.state_error_publisher.publish(error_msg)

    def publish_emergency_stop(self):
        """Publish emergency stop command."""

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.drive.speed = 0.0
        msg.drive.steering_angle = 0.0
        msg.drive.acceleration = -self.emergency_brake_threshold

        self.control_publisher.publish(msg)

        if not self.emergency_stop:
            self.emergency_stop = True
            self.emergency_stop_time = time.time()
            self.control_active = False
            self.get_logger().warning("🚨 Emergency stop activated")

    def publish_diagnostics(self):
        """Publish controller diagnostics."""

        try:
            diag_msg = DiagnosticArray()
            diag_msg.header.stamp = self.get_clock().now().to_msg()

            # Main controller status
            controller_status = DiagnosticStatus()
            controller_status.name = "lqr_controller"
            controller_status.hardware_id = "lqr_controller_node"

            # Determine overall status
            if self.control_active and not self.emergency_stop:
                controller_status.level = DiagnosticStatus.OK
                controller_status.message = "Controller active and healthy"
            elif self.emergency_stop:
                controller_status.level = DiagnosticStatus.ERROR
                controller_status.message = "Emergency stop active"
            else:
                controller_status.level = DiagnosticStatus.WARN
                controller_status.message = "Controller not active"

            # Add performance metrics
            if self.performance_logging_enabled and self.control_loop_times:
                controller_status.values.append(
                    KeyValue(key="avg_control_time", value=f"{np.mean(self.control_loop_times):.6f}")
                )
                controller_status.values.append(
                    KeyValue(key="max_control_time", value=f"{np.max(self.control_loop_times):.6f}")
                )

            # Add controller info
            controller_status.values.append(
                KeyValue(key="control_frequency", value=f"{self.control_hz}")
            )
            controller_status.values.append(
                KeyValue(key="consecutive_failures", value=f"{self.consecutive_failures}")
            )
            controller_status.values.append(
                KeyValue(key="path_ready", value=f"{self.path_ready}")
            )
            controller_status.values.append(
                KeyValue(key="reference_points", value=f"{len(self.reference_trajectory)}")
            )

            diag_msg.status.append(controller_status)
            self.diagnostics_publisher.publish(diag_msg)

        except Exception as e:
            self.get_logger().error(f"Error publishing diagnostics: {e}")


def main(args=None):
    """Main entry point for the LQR controller node."""

    rclpy.init(args=args)

    try:
        lqr_node = LQRNode()
        rclpy.spin(lqr_node)
    except KeyboardInterrupt:
        print("\nLQR Controller Node interrupted by user")
    except Exception as e:
        print(f"LQR Controller Node error: {e}")
        traceback.print_exc()
    finally:
        try:
            lqr_node.destroy_node()
        except BaseException:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()
