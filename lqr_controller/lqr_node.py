#!/usr/bin/env python3

"""
F1TENTH Fully Adaptive LQR Controller Node

This ROS2 node implements a fully adaptive Linear Quadratic Regulator (LQR) controller
that automatically adjusts its parameters based on vehicle state, trajectory characteristics,
and real-time performance metrics.

Author: Mohammed Azab <mohammed@azab.io>
License: MIT
Version: 3.0.0 (Fully Adaptive)
"""

import rclpy
import numpy as np
import time
import traceback
from typing import Dict, List, Tuple, Optional
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool, Float32
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from giu_f1t_interfaces.msg import VehicleStateArray
from tf_transformations import euler_from_quaternion
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

try:
    from .adaptive_lqr_controller import AdaptiveLQRController, AdaptiveParams
    from .kinematic_bicycle_model import KinematicBicycleModel
    from .safety_monitor import SafetyMonitor, SafetyParams
except ImportError:
    # Fallback for standalone execution
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    from adaptive_lqr_controller import AdaptiveLQRController, AdaptiveParams
    from kinematic_bicycle_model import KinematicBicycleModel
    from safety_monitor import SafetyMonitor, SafetyParams

# Import configuration defaults
try:
    import config
    CONFIG_AVAILABLE = True
    print("✅ Using config.py from", config.__file__)
except ImportError as e:
    CONFIG_AVAILABLE = False
    print(f"⚠️  Config not available: {e}, using defaults")

class AdaptiveLQRNode(Node):
    """
    ROS2 node implementing fully adaptive LQR controller for F1TENTH trajectory tracking.
    """

    def __init__(self):
        super().__init__('adaptive_lqr_controller_node')

        self._declare_parameters()
        self._load_parameters()
        self._initialize_adaptive_controller()
        self._initialize_safety_monitor()
        self._initialize_enhanced_controllers()
        self._initialize_state()
        self._setup_subscriptions()
        self._setup_publishers()
        self._setup_timers()

        self.get_logger().info("🚀 Fully Adaptive LQR Controller Node with Safety Monitor started!")

    def _declare_parameters(self):
        """Declare all ROS2 parameters including adaptive parameters."""
        
        # Vehicle Parameters
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('dt', 0.05)
        
        # Control Limits
        self.declare_parameter('max_acceleration', 5.0)
        self.declare_parameter('max_deceleration', 9.0)
        self.declare_parameter('max_steering_angle', 0.9)
        self.declare_parameter('min_speed', 0.1)
        self.declare_parameter('max_speed', 15.0)
        
        # Base LQR Weights
        self.declare_parameter('adaptive.base_position_weight', 5.0)
        self.declare_parameter('adaptive.base_velocity_weight', 1.0)
        self.declare_parameter('adaptive.base_heading_weight', 6.0)
        self.declare_parameter('adaptive.base_acceleration_weight', 0.3)
        self.declare_parameter('adaptive.base_steering_weight', 4.0)
        
        # Adaptation Parameters
        self.declare_parameter('adaptive.velocity_adaptation_factor', 0.5)
        self.declare_parameter('adaptive.curvature_adaptation_factor', 2.0)
        self.declare_parameter('adaptive.error_adaptation_factor', 1.5)
        self.declare_parameter('adaptive.performance_adaptation_factor', 0.3)
        
        self.declare_parameter('adaptive.min_weight_multiplier', 0.2)
        self.declare_parameter('adaptive.max_weight_multiplier', 5.0)
        self.declare_parameter('adaptive.adaptation_rate', 0.1)
        
        self.declare_parameter('adaptive.high_speed_threshold', 8.0)
        self.declare_parameter('adaptive.low_speed_threshold', 2.0)
        self.declare_parameter('adaptive.high_curvature_threshold', 1.0)
        self.declare_parameter('adaptive.moderate_curvature_threshold', 0.3)
        
        # Control Parameters
        self.declare_parameter('control_hz', 20.0)
        self.declare_parameter('lookahead_distance', 1.5)
        self.declare_parameter('enable_feedforward', True)
        
        # Anti-Wobble Parameters
        self.declare_parameter('min_lookahead_distance', 0.7)
        self.declare_parameter('max_lookahead_distance', 2.5)
        self.declare_parameter('lookahead_time', 0.8)
        self.declare_parameter('enable_steering_rate_limit', True)
        self.declare_parameter('max_steering_rate', 1.5)
        
        # Curve Detection
        self.declare_parameter('enable_curve_detection', True)
        self.declare_parameter('curve_lookahead_points', 5)
        self.declare_parameter('max_curvature_threshold', 1.0)
        self.declare_parameter('curve_speed_factor', 0.7)
        
        # Safety Parameters
        self.declare_parameter('enable_safety_checks', False)
        self.declare_parameter('safety_timeout', 1.0)
        self.declare_parameter('emergency_brake_threshold', 2.0)
        
        # Safety Monitor Parameters
        self.declare_parameter('safety.enable_safety_monitor', True)
        self.declare_parameter('safety.min_obstacle_distance', 1.5)
        self.declare_parameter('safety.emergency_brake_distance', 0.8)
        self.declare_parameter('safety.collision_check_angle', 60.0)
        self.declare_parameter('safety.max_lateral_acceleration', 3.0)
        self.declare_parameter('safety.max_angular_velocity', 2.0)
        self.declare_parameter('safety.wobble_time_threshold', 1.0)
        self.declare_parameter('safety.steering_oscillation_threshold', 0.3)
        self.declare_parameter('safety.safety_decel_rate', 2.0)
        self.declare_parameter('safety.emergency_decel_rate', 5.0)
        self.declare_parameter('safety.min_safe_speed', 1.0)
        
        # Topics
        self.declare_parameter('odom_topic', '/car_state/odom')
        self.declare_parameter('reference_topic', '/horizon_mapper/reference_trajectory')
        self.declare_parameter('status_topic', '/horizon_mapper/path_ready')
        self.declare_parameter('control_topic', '/drive')
        self.declare_parameter('pose_estimate_topic', '/initialpose')
        self.declare_parameter('lidar_topic', '/scan')
        
        # QoS and Logging
        self.declare_parameter('qos_depth', 10)
        self.declare_parameter('enable_logging', True)
        self.declare_parameter('debug_logging_enabled', False)
        self.declare_parameter('performance_logging_enabled', True)
        self.declare_parameter('log_frequency_divider', 10)

    def _load_parameters(self):
        """Load all parameters including adaptive configuration."""
        
        # Vehicle Parameters
        self.wheelbase = self.get_parameter('wheelbase').value
        self.dt = self.get_parameter('dt').value
        
        # Control Limits
        self.max_acceleration = self.get_parameter('max_acceleration').value
        self.max_deceleration = self.get_parameter('max_deceleration').value
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        self.min_speed = self.get_parameter('min_speed').value
        self.max_speed = self.get_parameter('max_speed').value
        
        # Load adaptive parameters
        self.adaptive_params = AdaptiveParams(
            base_position_weight=self.get_parameter('adaptive.base_position_weight').value,
            base_velocity_weight=self.get_parameter('adaptive.base_velocity_weight').value,
            base_heading_weight=self.get_parameter('adaptive.base_heading_weight').value,
            base_acceleration_weight=self.get_parameter('adaptive.base_acceleration_weight').value,
            base_steering_weight=self.get_parameter('adaptive.base_steering_weight').value,
            
            velocity_adaptation_factor=self.get_parameter('adaptive.velocity_adaptation_factor').value,
            curvature_adaptation_factor=self.get_parameter('adaptive.curvature_adaptation_factor').value,
            error_adaptation_factor=self.get_parameter('adaptive.error_adaptation_factor').value,
            performance_adaptation_factor=self.get_parameter('adaptive.performance_adaptation_factor').value,
            
            min_weight_multiplier=self.get_parameter('adaptive.min_weight_multiplier').value,
            max_weight_multiplier=self.get_parameter('adaptive.max_weight_multiplier').value,
            adaptation_rate=self.get_parameter('adaptive.adaptation_rate').value,
            
            high_speed_threshold=self.get_parameter('adaptive.high_speed_threshold').value,
            low_speed_threshold=self.get_parameter('adaptive.low_speed_threshold').value,
            high_curvature_threshold=self.get_parameter('adaptive.high_curvature_threshold').value,
            moderate_curvature_threshold=self.get_parameter('adaptive.moderate_curvature_threshold').value
        )
        
        # Control Parameters
        self.control_hz = self.get_parameter('control_hz').value
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.enable_feedforward = self.get_parameter('enable_feedforward').value
        
        # Anti-Wobble Parameters
        self.min_lookahead_distance = self.get_parameter('min_lookahead_distance').value
        self.max_lookahead_distance = self.get_parameter('max_lookahead_distance').value
        self.lookahead_time = self.get_parameter('lookahead_time').value
        self.enable_steering_rate_limit = self.get_parameter('enable_steering_rate_limit').value
        self.max_steering_rate = self.get_parameter('max_steering_rate').value
        
        # Curve Detection
        self.enable_curve_detection = self.get_parameter('enable_curve_detection').value
        self.curve_lookahead_points = self.get_parameter('curve_lookahead_points').value
        self.max_curvature_threshold = self.get_parameter('max_curvature_threshold').value
        self.curve_speed_factor = self.get_parameter('curve_speed_factor').value
        
        # Safety Parameters
        self.enable_safety_checks = self.get_parameter('enable_safety_checks').value
        self.safety_timeout = self.get_parameter('safety_timeout').value
        self.emergency_brake_threshold = self.get_parameter('emergency_brake_threshold').value
        
        # Load safety monitor parameters
        self.enable_safety_monitor = self.get_parameter('safety.enable_safety_monitor').value
        self.safety_params = SafetyParams(
            min_obstacle_distance=self.get_parameter('safety.min_obstacle_distance').value,
            emergency_brake_distance=self.get_parameter('safety.emergency_brake_distance').value,
            collision_check_angle=self.get_parameter('safety.collision_check_angle').value,
            max_lateral_acceleration=self.get_parameter('safety.max_lateral_acceleration').value,
            max_angular_velocity=self.get_parameter('safety.max_angular_velocity').value,
            wobble_time_threshold=self.get_parameter('safety.wobble_time_threshold').value,
            steering_oscillation_threshold=self.get_parameter('safety.steering_oscillation_threshold').value,
            safety_decel_rate=self.get_parameter('safety.safety_decel_rate').value,
            emergency_decel_rate=self.get_parameter('safety.emergency_decel_rate').value,
            min_safe_speed=self.get_parameter('safety.min_safe_speed').value
        )
        
        # Topics
        self.odom_topic = self.get_parameter('odom_topic').value
        self.reference_topic = self.get_parameter('reference_topic').value
        self.status_topic = self.get_parameter('status_topic').value
        self.control_topic = self.get_parameter('control_topic').value
        self.pose_estimate_topic = self.get_parameter('pose_estimate_topic').value
        self.lidar_topic = self.get_parameter('lidar_topic').value
        
        # QoS and Logging
        self.qos_depth = self.get_parameter('qos_depth').value
        self.enable_logging = self.get_parameter('enable_logging').value
        self.debug_logging_enabled = self.get_parameter('debug_logging_enabled').value
        self.performance_logging_enabled = self.get_parameter('performance_logging_enabled').value
        self.log_frequency_divider = self.get_parameter('log_frequency_divider').value

    def _initialize_adaptive_controller(self):
        """Initialize the fully adaptive LQR controller."""
        
        try:
            # Initialize adaptive LQR controller
            self.lqr_controller = AdaptiveLQRController(
                wheelbase=self.wheelbase,
                dt=self.dt,
                adaptive_params=self.adaptive_params,
                max_acceleration=self.max_acceleration,
                max_steering=self.max_steering_angle,
                enable_logging=self.enable_logging,
                logger=self.get_logger()
            )

            # Initialize kinematic model
            self.kinematic_model = KinematicBicycleModel(self.wheelbase, self.dt)

            self.get_logger().info("✅ Fully Adaptive LQR Controller initialized")
            self.get_logger().info(f"   - Base weights: pos={self.adaptive_params.base_position_weight}, "
                                 f"vel={self.adaptive_params.base_velocity_weight}, "
                                 f"head={self.adaptive_params.base_heading_weight}")
            self.get_logger().info(f"   - Adaptation rate: {self.adaptive_params.adaptation_rate}")
            self.get_logger().info(f"   - Speed thresholds: {self.adaptive_params.low_speed_threshold}-"
                                 f"{self.adaptive_params.high_speed_threshold} m/s")

        except Exception as e:
            self.get_logger().error(f"❌ Failed to initialize adaptive LQR controller: {e}")
            raise e

    def _initialize_safety_monitor(self):
        """Initialize the safety monitoring system."""
        
        if self.enable_safety_monitor:
            try:
                self.safety_monitor = SafetyMonitor(
                    safety_params=self.safety_params,
                    enable_logging=self.enable_logging,
                    logger=self.get_logger()
                )
                
                self.get_logger().info("✅ Safety Monitor initialized")
                self.get_logger().info(f"   - Collision avoidance: min_dist={self.safety_params.min_obstacle_distance}m")
                self.get_logger().info(f"   - Wobble detection: max_angular_vel={self.safety_params.max_angular_velocity} rad/s")
                self.get_logger().info(f"   - Emergency brake distance: {self.safety_params.emergency_brake_distance}m")
                
            except Exception as e:
                self.get_logger().error(f"❌ Failed to initialize safety monitor: {e}")
                self.enable_safety_monitor = False
                self.safety_monitor = None
        else:
            self.safety_monitor = None
            self.get_logger().info("⚠️  Safety Monitor disabled")

    def _initialize_enhanced_controllers(self):
        """Initialize enhanced control components."""
        
        # Simple curve analyzer implementation
        self.curve_analyzer = None
        self.adaptive_lookahead = None
        self.steering_rate_limiter = None
        
        if self.enable_curve_detection:
            self.curve_analyzer = SimpleCurveAnalyzer(
                lookahead_points=self.curve_lookahead_points,
                max_curvature_threshold=self.max_curvature_threshold
            )
        
        if self.enable_steering_rate_limit:
            self.steering_rate_limiter = SteeringRateLimiter(
                max_rate=self.max_steering_rate,
                dt=self.dt
            )
        
        self.get_logger().info("✅ Enhanced anti-wobble controllers initialized")
        self.get_logger().info(f"   - Curve detection: {'Enabled' if self.enable_curve_detection else 'Disabled'}")
        self.get_logger().info(f"   - Adaptive lookahead: {self.min_lookahead_distance}-{self.max_lookahead_distance}m")
        self.get_logger().info(f"   - Steering rate limit: {'Enabled' if self.enable_steering_rate_limit else 'Disabled'}")

    def _initialize_state(self):
        """Initialize node state variables."""
        
        # Vehicle state
        self.current_pose = None
        self.current_velocity = 0.0
        self.current_angular_velocity = 0.0
        self.current_heading = 0.0
        
        # Reference tracking
        self.reference_trajectory = []
        self.current_reference_index = 0
        self.path_ready = False
        
        # Control state
        self.control_active = False
        self.last_control_time = 0.0
        self.last_successful_solve_time = 0.0
        self.consecutive_failures = 0
        
        # Performance tracking
        self.control_iteration_count = 0
        self.control_loop_times = []
        
        # Enhanced state tracking
        self.current_curvature = 0.0
        self.current_lookahead_distance = self.lookahead_distance
        self.target_velocity = 0.0
        
        # Safety state
        self.last_control_command = np.zeros(2)
        
        self.get_logger().info("Enhanced node state initialized with anti-wobble features")

    def _setup_subscriptions(self):
        """Set up ROS2 subscriptions."""
        
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        
        # QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=self.qos_depth
        )
        
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=self.qos_depth
        )
        
        # Odometry subscription
        self.odom_subscription = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            sensor_qos
        )
        
        # Reference trajectory subscription
        self.reference_subscription = self.create_subscription(
            VehicleStateArray,
            self.reference_topic,
            self.reference_callback,
            reliable_qos
        )
        
        # Path status subscription
        self.status_subscription = self.create_subscription(
            Bool,
            self.status_topic,
            self.status_callback,
            reliable_qos
        )
        
        # Initial pose subscription (for reset)
        self.pose_subscription = self.create_subscription(
            PoseStamped,
            self.pose_estimate_topic,
            self.pose_estimate_callback,
            reliable_qos
        )
        
        # Lidar subscription for safety monitoring
        if self.enable_safety_monitor:
            self.lidar_subscription = self.create_subscription(
                LaserScan,
                self.lidar_topic,
                self.lidar_callback,
                sensor_qos
            )
            self.get_logger().info(f"✅ Subscribed to lidar topic: {self.lidar_topic}")
        
        self.get_logger().info("Subscriptions set up successfully")

    def _setup_publishers(self):
        """Set up ROS2 publishers including adaptive monitoring."""
        
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=self.qos_depth
        )
        
        # Control command publisher
        self.control_publisher = self.create_publisher(
            AckermannDriveStamped,
            self.control_topic,
            reliable_qos
        )
        
        # Diagnostics publisher
        self.diagnostics_publisher = self.create_publisher(
            DiagnosticArray,
            '/lqr_controller/diagnostics',
            reliable_qos
        )
        
        # Adaptive monitoring publishers
        self.adaptation_status_publisher = self.create_publisher(
            DiagnosticArray,
            '/adaptive_lqr/adaptation_status',
            reliable_qos
        )
        
        self.current_weights_publisher = self.create_publisher(
            Float32,
            '/adaptive_lqr/current_steering_weight',
            reliable_qos
        )
        
        self.performance_quality_publisher = self.create_publisher(
            Float32,
            '/adaptive_lqr/tracking_quality',
            reliable_qos
        )
        
        self.get_logger().info("Enhanced publishers set up successfully")

    def _setup_timers(self):
        """Set up periodic timers."""
        
        # Main control timer
        control_period = 1.0 / self.control_hz
        self.control_timer = self.create_timer(control_period, self.control_callback)
        
        # Diagnostics timer (1 Hz)
        self.diagnostics_timer = self.create_timer(1.0, self.publish_diagnostics)
        
        self.get_logger().info(f"Timers set up successfully (control: {self.control_hz}Hz)")

    def lidar_callback(self, msg: LaserScan):
        """Handle lidar scan messages for safety monitoring."""
        
        try:
            if self.safety_monitor:
                self.safety_monitor.update_lidar(msg)
                
        except Exception as e:
            if self.debug_logging_enabled:
                self.get_logger().warning(f"Error processing lidar data: {e}")

    def odom_callback(self, msg: Odometry):
        """Handle odometry messages."""
        
        try:
            self.current_pose = msg.pose.pose
            
            # Extract velocity information
            self.current_velocity = np.sqrt(
                msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2
            )
            self.current_angular_velocity = msg.twist.twist.angular.z
            
            # Extract heading from quaternion
            orientation = msg.pose.pose.orientation
            _, _, self.current_heading = euler_from_quaternion([
                orientation.x, orientation.y, orientation.z, orientation.w
            ])
            
            # Log first message
            if not hasattr(self, '_first_odom_received'):
                self._first_odom_received = True
                self.get_logger().info("First odometry message received")
                
        except Exception as e:
            self.get_logger().error(f"Error processing odometry: {e}")

    def reference_callback(self, msg: VehicleStateArray):
        """Handle reference trajectory messages."""
        
        try:
            self.reference_trajectory = []
            
            for state in msg.states:
                self.reference_trajectory.append({
                    'x': state.x,
                    'y': state.y,
                    'theta': state.theta,
                    'v': state.v
                })
            
            if self.debug_logging_enabled:
                self.get_logger().info(f"Received reference trajectory with {len(self.reference_trajectory)} points")
                
        except Exception as e:
            self.get_logger().error(f"Error processing reference trajectory: {e}")

    def status_callback(self, msg: Bool):
        """Handle path status messages."""
        self.path_ready = msg.data
        
        if self.debug_logging_enabled:
            self.get_logger().info(f"Path status: {'Ready' if self.path_ready else 'Not Ready'}")

    def pose_estimate_callback(self, msg: PoseStamped):
        """Handle initial pose estimate for reset."""
        if self.debug_logging_enabled:
            self.get_logger().info("Received pose estimate - resetting adaptive parameters")
        
        # Reset adaptive controller when pose is manually set
        self.lqr_controller.reset_adaptation()

    def check_safety_conditions(self) -> bool:
        """Check if it's safe to execute control."""
        
        if not self.path_ready:
            return False
        
        if self.current_pose is None:
            return False
        
        if len(self.reference_trajectory) == 0:
            return False
        
        if self.enable_safety_checks:
            current_time = time.time()
            if current_time - self.last_successful_solve_time > self.safety_timeout:
                return False
        
        return True

    def get_current_state(self) -> np.ndarray:
        """Get current vehicle state as numpy array."""
        
        if self.current_pose is None:
            return np.zeros(4)
        
        return np.array([
            self.current_pose.position.x,
            self.current_pose.position.y,
            self.current_velocity,
            self.current_heading
        ])

    def get_enhanced_reference_state(self, current_state: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Get enhanced reference state with curve analysis."""
        
        # Find closest reference point
        self.current_reference_index = self.find_closest_reference_point(current_state)
        
        if self.current_reference_index >= len(self.reference_trajectory):
            self.current_reference_index = len(self.reference_trajectory) - 1
        
        ref_point = self.reference_trajectory[self.current_reference_index]
        
        # Basic reference state
        reference_state = np.array([
            ref_point['x'],
            ref_point['y'], 
            ref_point['v'],
            ref_point['theta']
        ])
        
        # Analyze trajectory segment for adaptation
        analysis_info = {
            'curvature': 0.0,
            'complexity': 0.0,
            'lookahead_distance': self.lookahead_distance
        }
        
        if self.curve_analyzer:
            try:
                curve_info = self.curve_analyzer.analyze_segment(
                    self.reference_trajectory,
                    self.current_reference_index
                )
                analysis_info.update(curve_info)
                self.current_curvature = curve_info.get('curvature', 0.0)
            except Exception as e:
                if self.debug_logging_enabled:
                    self.get_logger().warning(f"Curve analysis failed: {e}")
        
        return reference_state, analysis_info

    def get_enhanced_feedforward_control(self, reference_state: np.ndarray, analysis_info: Dict) -> np.ndarray:
        """Get enhanced feedforward control with curve adaptation."""
        
        if not self.enable_feedforward:
            return np.zeros(2)
        
        # Basic feedforward
        target_velocity = reference_state[2]
        velocity_error = target_velocity - self.current_velocity
        
        # Simple velocity control
        feedforward_acceleration = 2.0 * velocity_error  # P-controller
        feedforward_acceleration = np.clip(
            feedforward_acceleration,
            -self.max_acceleration,
            self.max_acceleration
        )
        
        # Curve-adapted steering feedforward
        curvature = analysis_info.get('curvature', 0.0)
        feedforward_steering = curvature * 0.1  # Simple geometric relationship
        feedforward_steering = np.clip(
            feedforward_steering,
            -self.max_steering_angle,
            self.max_steering_angle
        )
        
        return np.array([feedforward_acceleration, feedforward_steering])

    def find_closest_reference_point(self, current_state: np.ndarray) -> int:
        """Find the index of the closest reference point."""
        
        if len(self.reference_trajectory) == 0:
            return 0
        
        min_distance = float('inf')
        closest_index = 0
        
        current_position = current_state[:2]
        
        for i, ref_point in enumerate(self.reference_trajectory):
            ref_position = np.array([ref_point['x'], ref_point['y']])
            distance = np.linalg.norm(current_position - ref_position)
            
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        return closest_index

    def control_callback(self):
        """Enhanced adaptive control loop with safety monitoring."""
        
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

            # Update safety monitor with current vehicle state
            if self.safety_monitor:
                self.safety_monitor.update_vehicle_state(
                    steering_angle=self.last_control_command[1],
                    angular_velocity=self.current_angular_velocity,
                    velocity=self.current_velocity
                )

            # Get enhanced reference state with curve analysis
            reference_state, analysis_info = self.get_enhanced_reference_state(current_state)

            # Get enhanced feedforward control
            feedforward_control = self.get_enhanced_feedforward_control(reference_state, analysis_info)

            # Find current trajectory index for adaptation
            current_index = self.find_closest_reference_point(current_state)

            # Compute adaptive LQR control with full trajectory information
            control = self.lqr_controller.compute_control(
                current_state,
                reference_state,
                feedforward_control,
                trajectory=self.reference_trajectory,
                current_index=current_index
            )

            # Apply steering rate limiting if enabled
            if self.steering_rate_limiter is not None:
                control[1] = self.steering_rate_limiter.limit_steering_rate(control[1])

            # Apply safety adjustments
            if self.safety_monitor:
                control = self.safety_monitor.get_safety_control_adjustment(
                    control, self.current_velocity
                )
                
                # Check for emergency stop
                safety_status = self.safety_monitor.get_safety_status()
                if safety_status['emergency_stop_required']:
                    self.publish_emergency_stop()
                    return

            # Validate control output
            if not self.kinematic_model.validate_control(control, self.max_acceleration, self.max_steering_angle):
                self.get_logger().warning("Invalid control output, stopping")
                self.publish_emergency_stop()
                return

            # Store last control command for safety monitoring
            self.last_control_command = control.copy()

            # Publish control command
            self.publish_control_command(control)

            # Publish enhanced diagnostics including adaptation and safety info
            self.publish_enhanced_debug_info(analysis_info)
            self.publish_adaptation_status()
            self.publish_safety_status()

            # Track performance
            if self.performance_logging_enabled:
                solve_time = time.time() - start_time
                self.control_loop_times.append(solve_time)

                if len(self.control_loop_times) > 1000:
                    self.control_loop_times = self.control_loop_times[-1000:]

            # Reset failure count on success
            self.consecutive_failures = 0
            self.last_successful_solve_time = time.time()
            self.control_active = True

            # Enhanced debug logging with adaptation and safety info
            if (self.debug_logging_enabled and
                    self.control_iteration_count % (self.log_frequency_divider * 2) == 0):
                
                adaptation_info = self.lqr_controller.get_adaptation_info()
                current_weights = adaptation_info['current_weights']
                performance_metrics = adaptation_info['performance_metrics']
                
                safety_info = ""
                if self.safety_monitor:
                    safety_status = self.safety_monitor.get_safety_status()
                    safety_info = f", safety=[active:{safety_status['safety_active']}, " \
                                f"obstacle:{safety_status['min_obstacle_distance']:.2f}m]"
                
                self.get_logger().info(
                    f"Adaptive Control #{self.control_iteration_count}: "
                    f"weights=[pos:{current_weights['position']:.2f}, "
                    f"head:{current_weights['heading']:.2f}, "
                    f"steer:{current_weights['steering']:.2f}], "
                    f"quality={performance_metrics.get('tracking_quality', 0.0):.3f}, "
                    f"control=[a:{control[0]:.3f}, δ:{control[1]:.3f}]"
                    f"{safety_info}"
                )

        except Exception as e:
            self.consecutive_failures += 1
            self.get_logger().error(f"Adaptive control loop error: {e}")

            if self.consecutive_failures > 5:
                self.get_logger().error("Too many consecutive control failures, emergency stop")
                self.publish_emergency_stop()

    def publish_control_command(self, control: np.ndarray):
        """Publish control command to the vehicle."""
        
        try:
            msg = AckermannDriveStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            
            msg.drive.acceleration = float(control[0])
            msg.drive.steering_angle = float(control[1])
            msg.drive.speed = float(self.current_velocity + control[0] * self.dt)
            
            self.control_publisher.publish(msg)
            self.last_control_time = time.time()
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish control command: {e}")

    def publish_emergency_stop(self):
        """Publish emergency stop command."""
        
        try:
            msg = AckermannDriveStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            
            msg.drive.acceleration = -self.emergency_brake_threshold
            msg.drive.steering_angle = 0.0
            msg.drive.speed = 0.0
            
            self.control_publisher.publish(msg)
            
            if self.debug_logging_enabled:
                self.get_logger().warning("Emergency stop published")
                
        except Exception as e:
            self.get_logger().error(f"Failed to publish emergency stop: {e}")

    def publish_safety_status(self):
        """Publish safety monitoring status."""
        
        try:
            if not self.safety_monitor:
                return
                
            safety_status = self.safety_monitor.get_safety_status()
            
            # Publish as diagnostic message
            diag_msg = DiagnosticArray()
            diag_msg.header.stamp = self.get_clock().now().to_msg()
            
            safety_diag = DiagnosticStatus()
            safety_diag.name = "adaptive_lqr_safety_monitor"
            safety_diag.hardware_id = "safety_monitor"
            
            if safety_status['emergency_stop_required']:
                safety_diag.level = DiagnosticStatus.ERROR
                safety_diag.message = "Emergency stop required"
            elif safety_status['safety_active']:
                safety_diag.level = DiagnosticStatus.WARN
                safety_diag.message = "Safety intervention active"
            else:
                safety_diag.level = DiagnosticStatus.OK
                safety_diag.message = "Safety monitoring normal"
            
            # Add safety metrics
            for key, value in safety_status.items():
                safety_diag.values.append(
                    KeyValue(key=f"safety_{key}", value=str(value))
                )
            
            diag_msg.status.append(safety_diag)
            self.diagnostics_publisher.publish(diag_msg)
            
        except Exception as e:
            if self.debug_logging_enabled:
                self.get_logger().warning(f"Failed to publish safety status: {e}")

    def publish_adaptation_status(self):
        """Publish current adaptation status and metrics."""
        
        try:
            adaptation_info = self.lqr_controller.get_adaptation_info()
            current_weights = adaptation_info['current_weights']
            performance_metrics = adaptation_info['performance_metrics']
            
            # Publish current steering weight
            steering_weight_msg = Float32()
            steering_weight_msg.data = float(current_weights['steering'])
            self.current_weights_publisher.publish(steering_weight_msg)
            
            # Publish tracking quality
            quality_msg = Float32()
            quality_msg.data = float(performance_metrics.get('tracking_quality', 0.0))
            self.performance_quality_publisher.publish(quality_msg)
            
            # Publish detailed adaptation status
            diag_msg = DiagnosticArray()
            diag_msg.header.stamp = self.get_clock().now().to_msg()
            
            adaptation_status = DiagnosticStatus()
            adaptation_status.name = "adaptive_lqr_adaptation"
            adaptation_status.hardware_id = "adaptive_lqr_controller"
            adaptation_status.level = DiagnosticStatus.OK
            adaptation_status.message = "Adaptation active"
            
            # Add current weights
            for weight_name, weight_value in current_weights.items():
                adaptation_status.values.append(
                    KeyValue(key=f"weight_{weight_name}", value=f"{weight_value:.3f}")
                )
            
            # Add performance metrics
            for metric_name, metric_value in performance_metrics.items():
                adaptation_status.values.append(
                    KeyValue(key=f"perf_{metric_name}", value=f"{metric_value:.4f}")
                )
            
            # Add adaptation count
            adaptation_status.values.append(
                KeyValue(key="adaptation_count", value=f"{adaptation_info['adaptation_count']}")
            )
            
            diag_msg.status.append(adaptation_status)
            self.adaptation_status_publisher.publish(diag_msg)
            
        except Exception as e:
            if self.debug_logging_enabled:
                self.get_logger().warning(f"Failed to publish adaptation status: {e}")

    def publish_enhanced_debug_info(self, analysis_info: Dict):
        """Publish enhanced debug information."""
        
        # This method can be used for additional debug publishing
        # Currently just placeholder
        pass

    def publish_diagnostics(self):
        """Publish enhanced diagnostics including adaptive information."""
        
        try:
            diag_msg = DiagnosticArray()
            diag_msg.header.stamp = self.get_clock().now().to_msg()
            
            # Main controller status
            controller_status = DiagnosticStatus()
            controller_status.name = "adaptive_lqr_controller"
            controller_status.hardware_id = "lqr_controller_node"
            
            if self.control_active:
                controller_status.level = DiagnosticStatus.OK
                controller_status.message = "Controller active and adaptive"
            else:
                controller_status.level = DiagnosticStatus.WARN
                controller_status.message = "Controller inactive"
            
            # Basic status
            controller_status.values.append(
                KeyValue(key="control_active", value=str(self.control_active))
            )
            controller_status.values.append(
                KeyValue(key="path_ready", value=str(self.path_ready))
            )
            controller_status.values.append(
                KeyValue(key="reference_points", value=str(len(self.reference_trajectory)))
            )
            controller_status.values.append(
                KeyValue(key="current_velocity", value=f"{self.current_velocity:.2f}")
            )
            controller_status.values.append(
                KeyValue(key="consecutive_failures", value=str(self.consecutive_failures))
            )
            
            # Performance metrics
            if self.performance_logging_enabled and self.control_loop_times:
                controller_status.values.append(
                    KeyValue(key="avg_control_time", value=f"{np.mean(self.control_loop_times):.6f}")
                )
                controller_status.values.append(
                    KeyValue(key="max_control_time", value=f"{np.max(self.control_loop_times):.6f}")
                )
            
            # Add adaptive controller information
            try:
                adaptation_info = self.lqr_controller.get_adaptation_info()
                current_weights = adaptation_info['current_weights']
                performance_metrics = adaptation_info['performance_metrics']
                
                # Add adaptive metrics
                controller_status.values.append(
                    KeyValue(key="adaptive_position_weight", value=f"{current_weights['position']:.3f}")
                )
                controller_status.values.append(
                    KeyValue(key="adaptive_heading_weight", value=f"{current_weights['heading']:.3f}")
                )
                controller_status.values.append(
                    KeyValue(key="adaptive_steering_weight", value=f"{current_weights['steering']:.3f}")
                )
                controller_status.values.append(
                    KeyValue(key="tracking_quality", value=f"{performance_metrics.get('tracking_quality', 0.0):.4f}")
                )
                controller_status.values.append(
                    KeyValue(key="error_trend", value=f"{performance_metrics.get('error_trend', 0.0):.4f}")
                )
                controller_status.values.append(
                    KeyValue(key="adaptation_count", value=f"{adaptation_info['adaptation_count']}")
                )
                
            except Exception as e:
                controller_status.values.append(
                    KeyValue(key="adaptation_error", value=str(e))
                )
            
            # Add safety monitor status
            if self.safety_monitor:
                try:
                    safety_status = self.safety_monitor.get_safety_status()
                    controller_status.values.append(
                        KeyValue(key="safety_active", value=str(safety_status['safety_active']))
                    )
                    controller_status.values.append(
                        KeyValue(key="min_obstacle_distance", value=f"{safety_status['min_obstacle_distance']:.2f}")
                    )
                    controller_status.values.append(
                        KeyValue(key="wobbling_detected", value=str(safety_status['wobbling_detected']))
                    )
                    controller_status.values.append(
                        KeyValue(key="safety_interventions", value=str(safety_status['safety_interventions']))
                    )
                except Exception as e:
                    controller_status.values.append(
                        KeyValue(key="safety_error", value=str(e))
                    )
            
            diag_msg.status.append(controller_status)
            self.diagnostics_publisher.publish(diag_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing diagnostics: {e}")


# Helper classes for enhanced features
class SimpleCurveAnalyzer:
    """Simple curve analysis for trajectory segments."""
    
    def __init__(self, lookahead_points: int = 5, max_curvature_threshold: float = 1.0):
        self.lookahead_points = lookahead_points
        self.max_curvature_threshold = max_curvature_threshold
    
    def analyze_segment(self, trajectory: List[Dict], current_index: int) -> Dict:
        """Analyze trajectory segment for curvature and complexity."""
        
        if len(trajectory) < 3 or current_index >= len(trajectory):
            return {'curvature': 0.0, 'complexity': 0.0}
        
        end_index = min(current_index + self.lookahead_points, len(trajectory))
        segment = trajectory[current_index:end_index]
        
        curvatures = []
        for i in range(1, len(segment) - 1):
            try:
                p1 = np.array([segment[i-1]['x'], segment[i-1]['y']])
                p2 = np.array([segment[i]['x'], segment[i]['y']])
                p3 = np.array([segment[i+1]['x'], segment[i+1]['y']])
                
                # Compute curvature using three points
                a = np.linalg.norm(p2 - p1)
                b = np.linalg.norm(p3 - p2)
                c = np.linalg.norm(p3 - p1)
                
                if a > 1e-6 and b > 1e-6 and c > 1e-6:
                    area = 0.5 * abs(np.cross(p2 - p1, p3 - p1))
                    curvature = 2 * area / (a * b * c)
                    curvatures.append(curvature)
            except:
                pass
        
        max_curvature = max(curvatures) if curvatures else 0.0
        complexity = np.std(curvatures) if len(curvatures) > 1 else 0.0
        
        return {
            'curvature': max_curvature,
            'complexity': complexity
        }


class SteeringRateLimiter:
    """Limit steering angle rate of change."""
    
    def __init__(self, max_rate: float, dt: float):
        self.max_rate = max_rate
        self.dt = dt
        self.last_steering = 0.0
    
    def limit_steering_rate(self, desired_steering: float) -> float:
        """Apply rate limiting to steering command."""
        
        max_change = self.max_rate * self.dt
        steering_change = desired_steering - self.last_steering
        
        if abs(steering_change) > max_change:
            limited_steering = self.last_steering + np.sign(steering_change) * max_change
        else:
            limited_steering = desired_steering
        
        self.last_steering = limited_steering
        return limited_steering


def main(args=None):
    """Main entry point for the fully adaptive LQR controller node."""
    
    rclpy.init(args=args)

    try:
        adaptive_lqr_node = AdaptiveLQRNode()
        rclpy.spin(adaptive_lqr_node)
    except KeyboardInterrupt:
        print("\nAdaptive LQR Controller Node interrupted by user")
    except Exception as e:
        print(f"Adaptive LQR Controller Node error: {e}")
        traceback.print_exc()
    finally:
        try:
            adaptive_lqr_node.destroy_node()
        except:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()

