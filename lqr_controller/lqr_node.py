#!/usr/bin/env python3

"""
F1TENTH LQR Controller Node

This ROS2 node implements a Linear Quadratic Regulator (LQR) controller for trajectory tracking
in F1TENTH autonomous racing cars with enhanced features to prevent wobbling and improve curve handling.

Features:
- Adaptive lookahead distance based on velocity and curvature
- Curve detection and velocity adaptation
- Steering rate limiting to prevent oscillations
- Real-time diagnostics and monitoring

Author: Mohammed Azab <mohammed@azab.io>
License: MIT
Version: 2.0.0 (Enhanced Anti-Wobble)
"""

import rclpy
import numpy as np
import time
import traceback
from typing import Dict, List, Tuple
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
                import config as default_config
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
            
            self.position_weight = 5.0     
            self.velocity_weight = 1.0
            self.heading_weight = 8.0
            self.acceleration_weight = 0.2
            self.steering_weight = 2.0

            # Control Parameters
            self.control_hz = 20.0
            self.lookahead_distance = 1.0  
            self.enable_feedforward = True
            
            self.min_lookahead_distance = 0.5
            self.max_lookahead_distance = 2.5
            self.lookahead_time = 0.8
            
            self.enable_curve_detection = True
            self.curve_lookahead_points = 5
            self.max_curvature_threshold = 2.0
            self.curve_speed_factor = 0.7
            
            self.enable_steering_rate_limit = True
            self.max_steering_rate = 3.0
            
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
    3. Computes optimal control using LQR with adaptive lookahead
    4. Applies curve detection and velocity adaptation
    5. Implements steering rate limiting to prevent oscillations
    6. Publishes control commands and enhanced diagnostics
    
    Anti-Wobble Features:
    - Adaptive lookahead distance based on velocity and curvature
    - Curve detection and automatic speed reduction
    - Steering rate limiting to prevent oscillations
    - Enhanced LQR weight tuning for stability
    - Real-time performance monitoring
    """

    def __init__(self):
        super().__init__('lqr_controller_node')

        self._declare_parameters()
        self._load_parameters()
        self._initialize_enhanced_controllers()
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

        #  Anti-Wobble Parameters
        self.declare_parameter('min_lookahead_distance', default_config.min_lookahead_distance)
        self.declare_parameter('max_lookahead_distance', default_config.max_lookahead_distance)
        self.declare_parameter('lookahead_time', default_config.lookahead_time)

        # Curve Detection and Handling Parameters
        self.declare_parameter('enable_curve_detection', default_config.enable_curve_detection)
        self.declare_parameter('curve_lookahead_points', default_config.curve_lookahead_points)
        self.declare_parameter('max_curvature_threshold', default_config.max_curvature_threshold)
        self.declare_parameter('curve_speed_factor', default_config.curve_speed_factor)

        # Steering Rate Limiting Parameters (Critical for anti-wobble)
        self.declare_parameter('enable_steering_rate_limit', default_config.enable_steering_rate_limit)
        self.declare_parameter('max_steering_rate', default_config.max_steering_rate)

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
        self.base_lookahead_distance = self.get_parameter('lookahead_distance').value  # Renamed for clarity
        self.enable_feedforward = self.get_parameter('enable_feedforward').value

        #  Anti-Wobble Parameters
        self.min_lookahead_distance = self.get_parameter('min_lookahead_distance').value
        self.max_lookahead_distance = self.get_parameter('max_lookahead_distance').value
        self.lookahead_time = self.get_parameter('lookahead_time').value

        # Curve Detection and Handling Parameters
        self.enable_curve_detection = self.get_parameter('enable_curve_detection').value
        self.curve_lookahead_points = self.get_parameter('curve_lookahead_points').value
        self.max_curvature_threshold = self.get_parameter('max_curvature_threshold').value
        self.curve_speed_factor = self.get_parameter('curve_speed_factor').value

        # Steering Rate Limiting Parameters
        self.enable_steering_rate_limit = self.get_parameter('enable_steering_rate_limit').value
        self.max_steering_rate = self.get_parameter('max_steering_rate').value

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

    def _initialize_enhanced_controllers(self):
        """Initialize enhanced control components for anti-wobble features."""

        # Curve analyzer for detecting upcoming curves
        self.curve_analyzer = CurveAnalyzer(self.wheelbase)

        # Adaptive lookahead controller for dynamic lookahead distance
        self.adaptive_lookahead = AdaptiveLookaheadController(
            self.min_lookahead_distance,
            self.max_lookahead_distance,
            self.lookahead_time
        )

        # Steering rate limiter to prevent oscillations (critical for anti-wobble)
        self.steering_rate_limiter = SteeringRateLimiter(
            self.max_steering_rate, 
            self.dt
        ) if self.enable_steering_rate_limit else None

        self.get_logger().info("✅ Enhanced anti-wobble controllers initialized")
        self.get_logger().info(f"   - Curve detection: {'Enabled' if self.enable_curve_detection else 'Disabled'}")
        self.get_logger().info(f"   - Adaptive lookahead: {self.min_lookahead_distance:.1f}-{self.max_lookahead_distance:.1f}m")
        self.get_logger().info(f"   - Steering rate limit: {'Enabled' if self.enable_steering_rate_limit else 'Disabled'}")

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
        """Initialize enhanced node state variables."""

        # Vehicle state
        self.current_pose = None
        self.current_velocity = 0.0
        self.current_yaw = 0.0
        self.current_steering_angle = 0.0

        # Reference trajectory
        self.reference_trajectory = []
        self.path_ready = False

        # Enhanced state tracking for anti-wobble features
        self.current_curvature = 0.0
        self.current_lookahead_distance = self.base_lookahead_distance
        self.target_velocity = 0.0

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

        self.get_logger().info("Enhanced node state initialized with anti-wobble features")

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
        """Set up enhanced ROS2 publishers."""

        # Control command publisher
        self.control_publisher = self.create_publisher(
            AckermannDriveStamped,
            self.control_topic,
            self.qos_depth
        )

        # Diagnostics publisher (enhanced)
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

        # Enhanced diagnostic publishers for anti-wobble monitoring
        self.curvature_publisher = self.create_publisher(
            Float32,
            '/lqr_controller/curvature',
            self.qos_depth
        )

        self.lookahead_distance_publisher = self.create_publisher(
            Float32,
            '/lqr_controller/lookahead_distance',
            self.qos_depth
        )

        self.target_velocity_publisher = self.create_publisher(
            Float32,
            '/lqr_controller/target_velocity',
            self.qos_depth
        )

        self.get_logger().info("Enhanced publishers set up successfully")

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

    def get_enhanced_reference_state(self, current_state: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Get enhanced reference state with adaptive lookahead and curve analysis.
        
        Returns:
            reference_state: Target state vector
            analysis_info: Dictionary with curve analysis information
        """

        if not self.reference_trajectory:
            return current_state, {
                'curvature': 0.0, 
                'lookahead_distance': self.base_lookahead_distance,
                'target_velocity': 0.0,
                'closest_index': 0,
                'target_index': 0
            }

        # Find closest reference point
        closest_index = self.find_closest_reference_point(current_state)

        # Analyze upcoming curve if curve detection is enabled
        max_curvature = 0.0
        if self.enable_curve_detection:
            max_curvature, _ = self.curve_analyzer.analyze_upcoming_curve(
                self.reference_trajectory, 
                closest_index, 
                self.curve_lookahead_points
            )

        # Compute adaptive lookahead distance
        adaptive_lookahead_distance = self.adaptive_lookahead.compute_lookahead_distance(
            self.current_velocity, max_curvature
        )

        # Convert lookahead distance to points
        if len(self.reference_trajectory) > 1:
            # Estimate point spacing
            point_spacing = np.sqrt(
                (self.reference_trajectory[1]['x'] - self.reference_trajectory[0]['x'])**2 +
                (self.reference_trajectory[1]['y'] - self.reference_trajectory[0]['y'])**2
            )
            lookahead_points = max(1, int(adaptive_lookahead_distance / max(point_spacing, 0.01)))
        else:
            lookahead_points = 1

        # Find target point with adaptive lookahead
        target_index = (closest_index + lookahead_points) % len(self.reference_trajectory)
        target_point = self.reference_trajectory[target_index]

        # Adapt target velocity for curves
        target_velocity = target_point['v']
        if self.enable_curve_detection and max_curvature > 0.1:
            # Reduce velocity for high curvature sections
            target_velocity *= max(self.curve_speed_factor, 1.0 / (1.0 + max_curvature))

        reference_state = np.array([
            target_point['x'],
            target_point['y'],
            target_velocity,
            target_point['theta']
        ])

        # Store for diagnostics
        self.current_curvature = max_curvature
        self.current_lookahead_distance = adaptive_lookahead_distance
        self.target_velocity = target_velocity

        analysis_info = {
            'curvature': max_curvature,
            'lookahead_distance': adaptive_lookahead_distance,
            'target_velocity': target_velocity,
            'closest_index': closest_index,
            'target_index': target_index
        }

        return reference_state, analysis_info

    def get_reference_state(self, current_state: np.ndarray) -> np.ndarray:
        """Get reference state for LQR controller (backward compatibility)."""
        reference_state, _ = self.get_enhanced_reference_state(current_state)
        return reference_state

    def get_enhanced_feedforward_control(self, reference_state: np.ndarray, 
                                       analysis_info: Dict) -> np.ndarray:
        """Compute enhanced feedforward control with curve awareness."""

        if not self.enable_feedforward or not self.reference_trajectory:
            return np.zeros(2)

        try:
            target_index = analysis_info.get('target_index', 0)
            
            # Ensure target_index is valid
            if target_index >= len(self.reference_trajectory):
                target_index = len(self.reference_trajectory) - 1
            
            target_point = self.reference_trajectory[target_index]

            # Enhanced feedforward with velocity adaptation
            target_acceleration = 0.0
            curvature = analysis_info.get('curvature', 0.0)
            target_velocity = analysis_info.get('target_velocity', 0.0)
            
            if curvature > 0.1:
                # Decelerate for curves
                velocity_error = self.current_velocity - target_velocity
                target_acceleration = -2.0 * velocity_error  # Proportional speed control

            # Use reference steering angle with potential adjustment
            target_steering = target_point.get('delta', 0.0)

            return np.array([target_acceleration, target_steering])

        except Exception as e:
            if self.debug_logging_enabled:
                self.get_logger().warning(f"Enhanced feedforward control error: {e}")
            return np.zeros(2)

    def get_feedforward_control(self, reference_state: np.ndarray) -> np.ndarray:
        """Compute feedforward control from reference trajectory (backward compatibility)."""

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
        """Enhanced main control loop callback with anti-wobble features."""

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

            # Get enhanced reference state with curve analysis
            reference_state, analysis_info = self.get_enhanced_reference_state(current_state)

            # Get enhanced feedforward control
            feedforward_control = self.get_enhanced_feedforward_control(reference_state, analysis_info)

            # Compute LQR control
            control = self.lqr_controller.compute_control(
                current_state,
                reference_state,
                feedforward_control
            )

            # Apply steering rate limiting if enabled (CRITICAL for anti-wobble)
            if self.steering_rate_limiter is not None:
                control[1] = self.steering_rate_limiter.limit_steering_rate(control[1])

            # Validate control output
            if not self.kinematic_model.validate_control(control, self.max_acceleration, self.max_steering_angle):
                self.get_logger().warning("Invalid control output, stopping")
                self.publish_emergency_stop()
                return

            # Publish control command
            self.publish_control_command(control)

            # Publish enhanced diagnostics
            self.publish_enhanced_debug_info(analysis_info)

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

            # Enhanced debug logging
            if (self.debug_logging_enabled and
                    self.control_iteration_count % self.log_frequency_divider == 0):
                state_error = np.linalg.norm(current_state - reference_state)
                curvature = analysis_info.get('curvature', 0.0)
                lookahead = analysis_info.get('lookahead_distance', self.base_lookahead_distance)
                target_v = analysis_info.get('target_velocity', 0.0)
                self.get_logger().info(
                    f"Enhanced Control #{self.control_iteration_count}: "
                    f"error={state_error:.3f}, curvature={curvature:.3f}, "
                    f"lookahead={lookahead:.2f}m, "
                    f"target_v={target_v:.2f}m/s, "
                    f"control=[a:{control[0]:.3f}, δ:{control[1]:.3f}]"
                )

        except Exception as e:
            self.consecutive_failures += 1
            self.get_logger().error(f"Enhanced control loop error: {e}")

            if self.consecutive_failures > 5:
                self.get_logger().error("Too many consecutive control failures, emergency stop")
                self.publish_emergency_stop()

    def publish_enhanced_debug_info(self, analysis_info: Dict):
        """Publish enhanced debug information for monitoring."""

        try:
            # Publish curvature
            curvature_msg = Float32()
            curvature_msg.data = float(analysis_info.get('curvature', 0.0))
            self.curvature_publisher.publish(curvature_msg)

            # Publish lookahead distance
            lookahead_msg = Float32()
            lookahead_msg.data = float(analysis_info.get('lookahead_distance', self.base_lookahead_distance))
            self.lookahead_distance_publisher.publish(lookahead_msg)

            # Publish target velocity
            target_velocity_msg = Float32()
            target_velocity_msg.data = float(analysis_info.get('target_velocity', 0.0))
            self.target_velocity_publisher.publish(target_velocity_msg)
            
        except Exception as e:
            if self.debug_logging_enabled:
                self.get_logger().warning(f"Enhanced debug info publish error: {e}")
        self.target_velocity_publisher.publish(target_velocity_msg)

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
        """Publish enhanced controller diagnostics."""

        try:
            diag_msg = DiagnosticArray()
            diag_msg.header.stamp = self.get_clock().now().to_msg()

            # Main controller status
            controller_status = DiagnosticStatus()
            controller_status.name = "enhanced_lqr_controller"
            controller_status.hardware_id = "lqr_controller_node"

            # Determine overall status
            if self.control_active and not self.emergency_stop:
                controller_status.level = DiagnosticStatus.OK
                controller_status.message = "Enhanced controller active and healthy"
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

            # Add enhanced controller info
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

            # Enhanced anti-wobble metrics
            controller_status.values.append(
                KeyValue(key="current_velocity", value=f"{self.current_velocity:.2f}")
            )
            controller_status.values.append(
                KeyValue(key="current_curvature", value=f"{self.current_curvature:.3f}")
            )
            controller_status.values.append(
                KeyValue(key="lookahead_distance", value=f"{self.current_lookahead_distance:.2f}")
            )
            controller_status.values.append(
                KeyValue(key="target_velocity", value=f"{self.target_velocity:.2f}")
            )
            controller_status.values.append(
                KeyValue(key="curve_detection", value=f"{self.enable_curve_detection}")
            )
            controller_status.values.append(
                KeyValue(key="steering_rate_limit", value=f"{self.enable_steering_rate_limit}")
            )

            # LQR weight information for tuning
            controller_status.values.append(
                KeyValue(key="position_weight", value=f"{self.lqr_weights['position_weight']}")
            )
            controller_status.values.append(
                KeyValue(key="heading_weight", value=f"{self.lqr_weights['heading_weight']}")
            )
            controller_status.values.append(
                KeyValue(key="steering_weight", value=f"{self.lqr_weights['steering_weight']}")
            )

            diag_msg.status.append(controller_status)
            self.diagnostics_publisher.publish(diag_msg)

        except Exception as e:
            self.get_logger().error(f"Error publishing  diagnostics: {e}")


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



# Enhanced Classes for Anti-Wobble and Curve Handling

class CurveAnalyzer:
    """Analyze trajectory curvature and adapt control parameters."""
    
    def __init__(self, wheelbase: float):
        self.wheelbase = wheelbase
        self.curvature_history = []
        self.max_history = 20
    
    def compute_curvature_at_point(self, points: List[Dict], index: int) -> float:
        """Compute curvature at a specific trajectory point."""
        if index <= 0 or index >= len(points) - 1:
            return 0.0
        
        # Get three consecutive points
        p1 = np.array([points[index-1]['x'], points[index-1]['y']])
        p2 = np.array([points[index]['x'], points[index]['y']])
        p3 = np.array([points[index+1]['x'], points[index+1]['y']])
        
        # Compute curvature using three-point formula
        # κ = 2 * area / (|a| * |b| * |c|)
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)
        
        if a < 1e-6 or b < 1e-6 or c < 1e-6:
            return 0.0
        
        # Area of triangle using cross product
        area = 0.5 * abs(np.cross(p2 - p1, p3 - p1))
        
        curvature = 2 * area / (a * b * c)
        return curvature
    
    def analyze_upcoming_curve(self, trajectory: List[Dict], current_index: int, 
                             lookahead_points: int) -> Tuple[float, float]:
        """
        Analyze upcoming curve characteristics.
        
        Returns:
            max_curvature: Maximum curvature in lookahead window
            avg_curvature: Average curvature in lookahead window
        """
        if not trajectory or len(trajectory) < 3:
            return 0.0, 0.0
        
        curvatures = []
        end_index = min(current_index + lookahead_points, len(trajectory) - 1)
        
        for i in range(current_index, end_index):
            curvature = self.compute_curvature_at_point(trajectory, i)
            curvatures.append(curvature)
        
        if not curvatures:
            return 0.0, 0.0
        
        max_curvature = max(curvatures)
        avg_curvature = np.mean(curvatures)
        
        # Update history for smoothing
        self.curvature_history.append(max_curvature)
        if len(self.curvature_history) > self.max_history:
            self.curvature_history.pop(0)
        
        # Smooth curvature using recent history
        smoothed_max_curvature = np.mean(self.curvature_history[-5:])
        
        return smoothed_max_curvature, avg_curvature


class AdaptiveLookaheadController:
    """Compute adaptive lookahead distance based on velocity and curvature."""
    
    def __init__(self, min_distance: float, max_distance: float, lookahead_time: float):
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.lookahead_time = lookahead_time
    
    def compute_lookahead_distance(self, velocity: float, max_curvature: float) -> float:
        """
        Compute adaptive lookahead distance.
        
        Args:
            velocity: Current vehicle velocity [m/s]
            max_curvature: Maximum upcoming curvature [1/m]
        
        Returns:
            Optimal lookahead distance [m]
        """
        # Base lookahead from velocity and time
        time_based_lookahead = velocity * self.lookahead_time
        
        # Reduce lookahead for high curvature
        curvature_factor = 1.0 / (1.0 + 2.0 * max_curvature)
        
        # Combine factors
        adaptive_lookahead = time_based_lookahead * curvature_factor
        
        # Apply bounds
        adaptive_lookahead = np.clip(adaptive_lookahead, 
                                   self.min_distance, 
                                   self.max_distance)
        
        return adaptive_lookahead


class SteeringRateLimiter:
    """Limit steering rate to prevent oscillations."""
    
    def __init__(self, max_rate: float, dt: float):
        self.max_rate = max_rate  # rad/s
        self.dt = dt
        self.last_steering = 0.0
        self.last_time = None
    
    def limit_steering_rate(self, desired_steering: float) -> float:
        """Apply steering rate limiting."""
        current_time = time.time()
        
        if self.last_time is None:
            self.last_time = current_time
            self.last_steering = desired_steering
            return desired_steering
        
        # Compute maximum allowed change
        dt_actual = current_time - self.last_time
        max_change = self.max_rate * dt_actual
        
        # Limit the change
        steering_change = desired_steering - self.last_steering
        limited_change = np.clip(steering_change, -max_change, max_change)
        
        limited_steering = self.last_steering + limited_change
        
        # Update state
        self.last_steering = limited_steering
        self.last_time = current_time
        
        return limited_steering

