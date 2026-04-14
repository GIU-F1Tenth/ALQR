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
import math
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
try:
    from tf_transformations import euler_from_quaternion
except ImportError:
    def euler_from_quaternion(quaternion, axes: str = 'sxyz'):
        """Fallback conversion: quaternion [x, y, z, w] -> roll, pitch, yaw."""
        del axes  # Fallback implementation supports only the default axis sequence.
        x, y, z, w = quaternion

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from dataclasses import dataclass
from collections import deque

try:
    from .adaptive_lqr_controller import AdaptiveLQRController, AdaptiveParams
    from .kinematic_bicycle_model import KinematicBicycleModel
except ImportError:
    # Fallback for standalone execution
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    from adaptive_lqr_controller import AdaptiveLQRController, AdaptiveParams
    from kinematic_bicycle_model import KinematicBicycleModel


@dataclass
class SafetyParams:
    """Safety monitoring parameters."""
    # Collision avoidance
    min_obstacle_distance: float = 1.5  # Minimum safe distance [m]
    emergency_brake_distance: float = 0.8  # Emergency brake distance [m]
    collision_check_angle: float = 60.0  # Angle range to check [degrees]
    
    # Wobble detection
    max_lateral_acceleration: float = 3.0  # Max lateral accel [m/s²]
    max_angular_velocity: float = 2.0  # Max angular velocity [rad/s]
    wobble_time_threshold: float = 1.0  # Time to consider wobbling [s]
    steering_oscillation_threshold: float = 0.3  # Steering oscillation limit [rad]
    
    # Deceleration parameters
    safety_decel_rate: float = 2.0  # Safety deceleration [m/s²]
    emergency_decel_rate: float = 5.0  # Emergency deceleration [m/s²]
    min_safe_speed: float = 1.0  # Minimum safe speed [m/s]
    
    # History tracking
    history_size: int = 50  # Number of samples to track
    update_rate: float = 20.0  # Update frequency [Hz]


class CollisionAvoidance:
    """Lidar-based collision avoidance system."""
    
    def __init__(self, safety_params: SafetyParams):
        self.params = safety_params
        self.last_scan = None
        self.obstacle_distances = []
        self.collision_imminent = False
        self.emergency_brake_active = False
        
    def update_lidar_data(self, scan: LaserScan):
        """Update with new lidar scan data."""
        self.last_scan = scan
        self._analyze_obstacles()
        
    def _analyze_obstacles(self):
        """Analyze lidar data for obstacles in the vehicle's path."""
        if self.last_scan is None:
            return
            
        # Convert angle range to indices
        angle_range = np.radians(self.params.collision_check_angle)
        total_angle = self.last_scan.angle_max - self.last_scan.angle_min
        indices_per_degree = len(self.last_scan.ranges) / total_angle
        
        # Check front sector
        center_index = len(self.last_scan.ranges) // 2
        check_indices = int(angle_range * indices_per_degree / 2)
        
        start_idx = max(0, center_index - check_indices)
        end_idx = min(len(self.last_scan.ranges), center_index + check_indices)
        
        # Find minimum distance in front sector
        front_distances = []
        for i in range(start_idx, end_idx):
            if (self.last_scan.range_min <= self.last_scan.ranges[i] <= self.last_scan.range_max):
                front_distances.append(self.last_scan.ranges[i])
        
        if front_distances:
            min_distance = min(front_distances)
            self.obstacle_distances.append(min_distance)
            
            # Maintain history
            if len(self.obstacle_distances) > self.params.history_size:
                self.obstacle_distances.pop(0)
            
            # Check for collision risk
            self.emergency_brake_active = min_distance < self.params.emergency_brake_distance
            self.collision_imminent = min_distance < self.params.min_obstacle_distance
    
    def get_safety_deceleration(self, current_speed: float) -> float:
        """Calculate required deceleration based on obstacle proximity."""
        if not self.obstacle_distances:
            return 0.0
            
        min_distance = min(self.obstacle_distances[-5:])  # Use recent minimum
        
        if self.emergency_brake_active:
            return self.params.emergency_decel_rate
        elif self.collision_imminent:
            # Proportional deceleration based on distance
            safety_factor = max(0.0, (self.params.min_obstacle_distance - min_distance) / 
                              self.params.min_obstacle_distance)
            return self.params.safety_decel_rate * safety_factor
        
        return 0.0
    
    def is_path_clear(self) -> bool:
        """Check if the path ahead is clear."""
        return not self.collision_imminent and not self.emergency_brake_active


class WobbleDetector:
    """Detect and mitigate vehicle wobbling behavior."""
    
    def __init__(self, safety_params: SafetyParams):
        self.params = safety_params
        self.steering_history = deque(maxlen=self.params.history_size)
        self.angular_velocity_history = deque(maxlen=self.params.history_size)
        self.lateral_accel_history = deque(maxlen=self.params.history_size)
        self.timestamps = deque(maxlen=self.params.history_size)
        
        self.wobbling_detected = False
        self.wobble_start_time = None
        
    def update_vehicle_state(self, steering_angle: float, angular_velocity: float, 
                           velocity: float, timestamp: float):
        """Update with current vehicle state."""
        
        # Calculate lateral acceleration
        lateral_accel = velocity * angular_velocity
        
        # Store history
        self.steering_history.append(steering_angle)
        self.angular_velocity_history.append(angular_velocity)
        self.lateral_accel_history.append(lateral_accel)
        self.timestamps.append(timestamp)
        
        # Detect wobbling
        self._detect_wobble()
    
    def _detect_wobble(self):
        """Detect wobbling behavior from vehicle state history."""
        if len(self.steering_history) < 10:  # Need minimum history
            return
            
        # Check for excessive steering oscillations
        steering_oscillation = self._calculate_oscillation(list(self.steering_history))
        
        # Check for excessive angular velocity
        max_angular_vel = max(abs(av) for av in list(self.angular_velocity_history)[-5:])
        
        # Check for excessive lateral acceleration
        max_lateral_accel = max(abs(la) for la in list(self.lateral_accel_history)[-5:])
        
        # Wobble conditions
        excessive_steering = steering_oscillation > self.params.steering_oscillation_threshold
        excessive_angular_vel = max_angular_vel > self.params.max_angular_velocity
        excessive_lateral_accel = max_lateral_accel > self.params.max_lateral_acceleration
        
        current_wobbling = excessive_steering or excessive_angular_vel or excessive_lateral_accel
        
        if current_wobbling and not self.wobbling_detected:
            self.wobble_start_time = self.timestamps[-1]
            self.wobbling_detected = True
        elif not current_wobbling:
            self.wobbling_detected = False
            self.wobble_start_time = None
    
    def _calculate_oscillation(self, data: List[float]) -> float:
        """Calculate oscillation magnitude in a signal."""
        if len(data) < 5:
            return 0.0
            
        # Calculate rate of change
        derivatives = []
        for i in range(1, len(data)):
            derivatives.append(abs(data[i] - data[i-1]))
        
        # Return average rate of change as oscillation measure
        return float(np.mean(derivatives)) if derivatives else 0.0
    
    def is_wobbling(self) -> bool:
        """Check if vehicle is currently wobbling."""
        if not self.wobbling_detected:
            return False
            
        # Check if wobbling for too long
        if self.wobble_start_time and self.timestamps:
            wobble_duration = self.timestamps[-1] - self.wobble_start_time
            return wobble_duration > self.params.wobble_time_threshold
            
        return False
    
    def get_wobble_deceleration(self) -> float:
        """Get recommended deceleration to reduce wobbling."""
        if self.is_wobbling():
            return self.params.safety_decel_rate * 0.5  # Gentle deceleration
        return 0.0


class SafetyMonitor:
    """Complete safety monitoring system for the adaptive LQR controller."""
    
    def __init__(self, safety_params: Optional[SafetyParams] = None, 
                 enable_logging: bool = True, logger: Optional[object] = None):
        """Initialize safety monitoring system."""
        
        self.params = safety_params or SafetyParams()
        self.enable_logging = enable_logging
        self.logger = logger
        
        # Initialize subsystems
        self.collision_avoidance = CollisionAvoidance(self.params)
        self.wobble_detector = WobbleDetector(self.params)
        
        # Safety state
        self.safety_active = False
        self.emergency_stop_required = False
        self.last_update_time = 0.0
        
        # Performance tracking
        self.safety_interventions = 0
        self.total_deceleration_time = 0.0
        
        self.log_info("Safety Monitor initialized")
    
    def log_info(self, message: str):
        """Log information message."""
        if self.logger:
            self.logger.info(f"[Safety] {message}")
        elif self.enable_logging:
            print(f"[Safety Monitor] {message}")
    
    def update_lidar(self, scan: LaserScan):
        """Update with new lidar scan."""
        self.collision_avoidance.update_lidar_data(scan)
        self._update_safety_state()
    
    def update_vehicle_state(self, steering_angle: float, angular_velocity: float,
                           velocity: float):
        """Update with current vehicle state."""
        current_time = time.time()
        self.wobble_detector.update_vehicle_state(
            steering_angle, angular_velocity, velocity, current_time
        )
        self.last_update_time = current_time
        self._update_safety_state()
    
    def _update_safety_state(self):
        """Update overall safety state."""
        # Check for emergency conditions
        path_blocked = not self.collision_avoidance.is_path_clear()
        wobbling = self.wobble_detector.is_wobbling()
        emergency_brake = self.collision_avoidance.emergency_brake_active
        
        self.emergency_stop_required = emergency_brake
        self.safety_active = path_blocked or wobbling
        
        if self.safety_active:
            self.safety_interventions += 1
    
    def get_safety_control_adjustment(self, current_control: np.ndarray,
                                    current_velocity: float) -> np.ndarray:
        """
        Get safety-adjusted control commands.
        
        Args:
            current_control: [acceleration, steering_angle]
            current_velocity: Current vehicle velocity
            
        Returns:
            Adjusted control commands with safety constraints
        """
        adjusted_control = current_control.copy()
        
        if self.emergency_stop_required:
            # Emergency stop
            adjusted_control[0] = -self.params.emergency_decel_rate
            adjusted_control[1] = 0.0  # Straight steering
            self.log_info("Emergency stop activated!")
            return adjusted_control
        
        # Calculate required safety deceleration
        collision_decel = self.collision_avoidance.get_safety_deceleration(current_velocity)
        wobble_decel = self.wobble_detector.get_wobble_deceleration()
        
        required_decel = max(collision_decel, wobble_decel)
        
        if required_decel > 0:
            # Apply safety deceleration
            adjusted_control[0] = min(adjusted_control[0], -required_decel)
            
            # Reduce steering aggressiveness if wobbling
            if self.wobble_detector.is_wobbling():
                adjusted_control[1] *= 0.7  # Reduce steering by 30%
                self.log_info(f"Wobble detected - reducing steering and speed")
            
            if collision_decel > 0:
                self.log_info(f"Obstacle detected - decelerating at {required_decel:.2f} m/s²")
        
        # Ensure minimum safe speed
        if current_velocity < self.params.min_safe_speed and adjusted_control[0] < 0:
            adjusted_control[0] = max(adjusted_control[0], -1.0)  # Gentle deceleration near stop
        
        return adjusted_control
    
    def get_safety_status(self) -> Dict:
        """Get current safety system status."""
        return {
            'safety_active': self.safety_active,
            'emergency_stop_required': self.emergency_stop_required,
            'collision_imminent': self.collision_avoidance.collision_imminent,
            'wobbling_detected': self.wobble_detector.is_wobbling(),
            'path_clear': self.collision_avoidance.is_path_clear(),
            'safety_interventions': self.safety_interventions,
            'min_obstacle_distance': min(self.collision_avoidance.obstacle_distances) 
                                   if self.collision_avoidance.obstacle_distances else float('inf')
        }
    
    def reset_safety_state(self):
        """Reset safety monitoring state."""
        self.safety_active = False
        self.emergency_stop_required = False
        self.safety_interventions = 0
        self.total_deceleration_time = 0.0
        
        # Clear histories
        self.wobble_detector.steering_history.clear()
        self.wobble_detector.angular_velocity_history.clear()
        self.wobble_detector.lateral_accel_history.clear()
        self.collision_avoidance.obstacle_distances.clear()
        
        self.log_info("Safety state reset")

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

        self.get_logger().info("LQR controller node started")

    def _declare_parameters(self):
        # Vehicle
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('dt', 0.05)

        # Speed limits
        self.declare_parameter('max_acceleration', 5.0)
        self.declare_parameter('max_steering_angle', 0.9)
        self.declare_parameter('min_speed', 0.1)
        self.declare_parameter('max_speed', 15.0)

        # LQR base weights
        self.declare_parameter('adaptive.base_position_weight', 5.0)
        self.declare_parameter('adaptive.base_velocity_weight', 1.0)
        self.declare_parameter('adaptive.base_heading_weight', 6.0)
        self.declare_parameter('adaptive.base_acceleration_weight', 0.3)
        self.declare_parameter('adaptive.base_steering_weight', 4.0)

        # Adaptation factors
        self.declare_parameter('adaptive.velocity_adaptation_factor', 0.5)
        self.declare_parameter('adaptive.curvature_adaptation_factor', 2.0)
        self.declare_parameter('adaptive.error_adaptation_factor', 1.5)
        self.declare_parameter('adaptive.performance_adaptation_factor', 0.3)
        self.declare_parameter('adaptive.min_weight_multiplier', 0.2)
        self.declare_parameter('adaptive.max_weight_multiplier', 5.0)
        self.declare_parameter('adaptive.adaptation_rate', 0.1)

        # Speed regime thresholds for adaptive behaviour
        self.declare_parameter('adaptive.high_speed_threshold', 8.0)
        self.declare_parameter('adaptive.low_speed_threshold', 2.0)
        self.declare_parameter('adaptive.high_curvature_threshold', 1.0)
        self.declare_parameter('adaptive.moderate_curvature_threshold', 0.3)

        # Control loop
        self.declare_parameter('control_hz', 20.0)
        self.declare_parameter('lookahead_distance', 1.5)
        self.declare_parameter('enable_feedforward', True)
        self.declare_parameter('speed_ramp_rate', 0.5)

        # Steering rate limiter
        self.declare_parameter('enable_steering_rate_limit', True)
        self.declare_parameter('max_steering_rate', 1.5)

        # Curve detection and speed reduction
        self.declare_parameter('enable_curve_detection', True)
        self.declare_parameter('curve_lookahead_points', 5)
        self.declare_parameter('max_curvature_threshold', 1.0)
        self.declare_parameter('curve_speed_factor', 0.7)

        # Solve-timeout safety gate
        self.declare_parameter('enable_safety_checks', False)
        self.declare_parameter('safety_timeout', 1.0)
        self.declare_parameter('emergency_brake_threshold', 2.0)

        # Safety monitor subsystem
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
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('reference_topic', '/horizon_mapper/reference_trajectory')
        self.declare_parameter('status_topic', '/horizon_mapper/path_ready')
        self.declare_parameter('control_topic', '/drive')
        self.declare_parameter('pose_estimate_topic', '/initialpose')
        self.declare_parameter('lidar_topic', '/scan')

        # Logging
        self.declare_parameter('qos_depth', 10)
        self.declare_parameter('debug', False)
        self.declare_parameter('log_frequency_divider', 10)

    def _load_parameters(self):
        # Vehicle
        self.wheelbase = self.get_parameter('wheelbase').value
        self.dt = self.get_parameter('dt').value

        # Speed limits
        self.max_acceleration = self.get_parameter('max_acceleration').value
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        self.min_speed = self.get_parameter('min_speed').value
        self.max_speed = self.get_parameter('max_speed').value

        # Build AdaptiveParams from config
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
            moderate_curvature_threshold=self.get_parameter('adaptive.moderate_curvature_threshold').value,
        )

        # Control loop
        self.control_hz = self.get_parameter('control_hz').value
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.enable_feedforward = self.get_parameter('enable_feedforward').value
        self.speed_ramp_rate = self.get_parameter('speed_ramp_rate').value

        # Steering rate limiter
        self.enable_steering_rate_limit = self.get_parameter('enable_steering_rate_limit').value
        self.max_steering_rate = self.get_parameter('max_steering_rate').value

        # Curve detection
        self.enable_curve_detection = self.get_parameter('enable_curve_detection').value
        self.curve_lookahead_points = self.get_parameter('curve_lookahead_points').value
        self.max_curvature_threshold = self.get_parameter('max_curvature_threshold').value
        self.curve_speed_factor = self.get_parameter('curve_speed_factor').value

        # Solve-timeout gate
        self.enable_safety_checks = self.get_parameter('enable_safety_checks').value
        self.safety_timeout = self.get_parameter('safety_timeout').value
        self.emergency_brake_threshold = self.get_parameter('emergency_brake_threshold').value

        # Safety monitor subsystem
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
            min_safe_speed=self.get_parameter('safety.min_safe_speed').value,
        )

        # Topics
        self.odom_topic = self.get_parameter('odom_topic').value
        self.reference_topic = self.get_parameter('reference_topic').value
        self.status_topic = self.get_parameter('status_topic').value
        self.control_topic = self.get_parameter('control_topic').value
        self.pose_estimate_topic = self.get_parameter('pose_estimate_topic').value
        self.lidar_topic = self.get_parameter('lidar_topic').value

        # Logging
        self.qos_depth = self.get_parameter('qos_depth').value
        self.debug = self.get_parameter('debug').value
        self.enable_logging = self.debug
        self.log_frequency_divider = self.get_parameter('log_frequency_divider').value

    def _initialize_adaptive_controller(self):
        try:
            self.lqr_controller = AdaptiveLQRController(
                wheelbase=self.wheelbase,
                dt=self.dt,
                adaptive_params=self.adaptive_params,
                max_acceleration=self.max_acceleration,
                max_steering=self.max_steering_angle,
                enable_logging=self.debug,
                logger=self.get_logger()
            )

            self.kinematic_model = KinematicBicycleModel(self.wheelbase, self.dt)

            self.get_logger().info(
                f"LQR controller ready | weights: pos={self.adaptive_params.base_position_weight} "
                f"vel={self.adaptive_params.base_velocity_weight} "
                f"head={self.adaptive_params.base_heading_weight} | "
                f"adapt_rate={self.adaptive_params.adaptation_rate} | "
                f"speed_range={self.adaptive_params.low_speed_threshold}-"
                f"{self.adaptive_params.high_speed_threshold} m/s"
            )

        except Exception as e:
            self.get_logger().error(f"LQR controller init failed: {e}")
            raise e

    def _initialize_safety_monitor(self):
        if self.enable_safety_monitor:
            try:
                self.safety_monitor = SafetyMonitor(
                    safety_params=self.safety_params,
                    enable_logging=self.debug,
                    logger=self.get_logger()
                )
                
                self.get_logger().info(
                    f"Safety monitor ready | min_dist={self.safety_params.min_obstacle_distance}m "
                    f"max_angular_vel={self.safety_params.max_angular_velocity} rad/s "
                    f"e-brake={self.safety_params.emergency_brake_distance}m"
                )
                
            except Exception as e:
                self.get_logger().error(f"Safety monitor init failed: {e}")
                self.enable_safety_monitor = False
                self.safety_monitor = None
        else:
            self.safety_monitor = None
            self.get_logger().info("Safety monitor disabled")

    def _initialize_enhanced_controllers(self):
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
        
        self.get_logger().info(
            f"Anti-wobble controllers ready | curve_detect={'on' if self.enable_curve_detection else 'off'} "
            f"lookahead={self.lookahead_distance}m "
            f"steer_rate_limit={'on' if self.enable_steering_rate_limit else 'off'}"
        )

    def _initialize_state(self):
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
        self._last_control_block_reason = None
        self._last_emergency_stop_reason = None
        self._last_emergency_stop_time = 0.0
        
        self.last_commanded_speed = 0.0

        self.get_logger().info("Node state initialized")

    def _get_runtime_snapshot(self) -> str:
        """Build a short summary of the controller runtime state."""
        pose_state = "set" if self.current_pose is not None else "none"
        reference_count = len(self.reference_trajectory)
        solve_age = "n/a"
        if self.last_successful_solve_time > 0.0:
            solve_age = f"{time.time() - self.last_successful_solve_time:.2f}s"

        return (
            f"path_ready={self.path_ready} ref_points={reference_count} "
            f"pose={pose_state} v={self.current_velocity:.2f}m/s "
            f"yaw={self.current_heading:.2f}rad solve_age={solve_age} "
            f"active={self.control_active} failures={self.consecutive_failures}"
        )

    def _log_control_block_reason(self, reason: str):
        """Log why the controller is not publishing drive commands."""
        if reason != self._last_control_block_reason:
            self._last_control_block_reason = reason
            self.get_logger().warning(f"Control blocked: {reason} | {self._get_runtime_snapshot()}")
        elif self.debug:
            self.get_logger().info(f"Control still blocked: {reason}")

    def _setup_subscriptions(self):
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

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
        
        self.odom_subscription = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            sensor_qos
        )
        
        self.reference_subscription = self.create_subscription(
            VehicleStateArray,
            self.reference_topic,
            self.reference_callback,
            reliable_qos
        )

        self.status_subscription = self.create_subscription(
            Bool,
            self.status_topic,
            self.status_callback,
            reliable_qos
        )

        self.pose_subscription = self.create_subscription(
            PoseStamped,
            self.pose_estimate_topic,
            self.pose_estimate_callback,
            reliable_qos
        )

        if self.enable_safety_monitor:
            self.lidar_subscription = self.create_subscription(
                LaserScan,
                self.lidar_topic,
                self.lidar_callback,
                sensor_qos
            )
            self.get_logger().info(f"Lidar: {self.lidar_topic}")
        
        self.get_logger().info("Subscriptions ready")

    def _setup_publishers(self):
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=self.qos_depth
        )

        self.control_publisher = self.create_publisher(
            AckermannDriveStamped,
            self.control_topic,
            reliable_qos
        )

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

        self.get_logger().info("Publishers ready")

    def _setup_timers(self):
        control_period = 1.0 / self.control_hz
        self.control_timer = self.create_timer(control_period, self.control_callback)
        self.diagnostics_timer = self.create_timer(1.0, self.publish_diagnostics)
        
        self.get_logger().info(f"Timers ready | control={self.control_hz}Hz")

    def lidar_callback(self, msg: LaserScan):
        """Handle lidar scan messages for safety monitoring."""
        
        try:
            if self.safety_monitor:
                self.safety_monitor.update_lidar(msg)
                
        except Exception as e:
            if self.debug:
                self.get_logger().warning(f"Lidar processing error: {e}")

    def odom_callback(self, msg: Odometry):
        """Handle odometry messages with robust velocity and pose extraction."""
        
        try:
            # Update pose
            self.current_pose = msg.pose.pose
            
            # Extract linear velocity (magnitude of velocity vector for 2D)
            linear_vel_x = msg.twist.twist.linear.x
            linear_vel_y = msg.twist.twist.linear.y
            self.current_velocity = np.sqrt(linear_vel_x**2 + linear_vel_y**2)
            
            # Extract angular velocity
            self.current_angular_velocity = msg.twist.twist.angular.z
            
            # Extract heading from quaternion
            orientation = msg.pose.pose.orientation
            _, _, self.current_heading = euler_from_quaternion([
                orientation.x, orientation.y, orientation.z, orientation.w
            ])
            
            # Log first message
            if not hasattr(self, '_first_odom_received'):
                self._first_odom_received = True
                self.get_logger().info(
                    f"Odometry: first message received | pose=({self.current_pose.position.x:.3f}, "
                    f"{self.current_pose.position.y:.3f}) v={self.current_velocity:.3f}m/s"
                )
                
        except Exception as e:
            self.get_logger().error(f"Odometry processing error: {e}")
            traceback.print_exc()

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
            
            if self.debug:
                self.get_logger().info(f"Trajectory: {len(self.reference_trajectory)} points")
                
        except Exception as e:
            self.get_logger().error(f"Trajectory processing error: {e}")

    def status_callback(self, msg: Bool):
        """Handle path status messages."""
        previous_state = self.path_ready
        self.path_ready = msg.data
        
        if self.path_ready != previous_state:
            self.get_logger().info(
                f"Path status changed: {'ready' if self.path_ready else 'not ready'} | {self._get_runtime_snapshot()}"
            )
        elif self.debug:
            self.get_logger().info(f"Path status received: {'ready' if self.path_ready else 'not ready'}")

    def pose_estimate_callback(self, msg: PoseStamped):
        """Handle initial pose estimate from RViz for pose initialization."""
        try:
            # Update current pose with RViz pose estimate
            self.current_pose = msg.pose
            
            # Extract heading from quaternion
            orientation = msg.pose.orientation
            _, _, self.current_heading = euler_from_quaternion([
                orientation.x, orientation.y, orientation.z, orientation.w
            ])
            
            # Reset velocity when pose is manually set
            self.current_velocity = 0.0
            self.current_angular_velocity = 0.0
            self.last_commanded_speed = 0.0
            
            # Reset trajectory tracking to find new starting point
            self.current_reference_index = 0
            
            # Reset adaptive controller for clean restart
            self.lqr_controller.reset_adaptation()
            
            self.get_logger().info(
                f"Pose updated from RViz | x={self.current_pose.position.x:.3f} "
                f"y={self.current_pose.position.y:.3f} theta={self.current_heading:.3f}rad"
            )
        except Exception as e:
            self.get_logger().error(f"Error processing pose estimate: {e}")

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
                if self.debug:
                    self.get_logger().warning(f"Curve analysis failed: {e}")
        
        return reference_state, analysis_info

    def get_enhanced_feedforward_control(self, reference_state: np.ndarray, analysis_info: Dict) -> np.ndarray:
        """Get enhanced feedforward control with curve adaptation."""
        
        if not self.enable_feedforward:
            return np.zeros(2)

        # Cap reference velocity to max_speed — this is what makes max_speed
        # actually limit the feedforward acceleration demand.
        target_velocity = min(reference_state[2], self.max_speed)

        # Reduce speed on curves when curve detection is active
        if self.enable_curve_detection:
            curvature = analysis_info.get('curvature', 0.0)
            if abs(curvature) > self.max_curvature_threshold * 0.5:
                target_velocity = target_velocity * self.curve_speed_factor

        velocity_error = target_velocity - self.current_velocity

        # P-controller feedforward acceleration
        feedforward_acceleration = 2.0 * velocity_error
        feedforward_acceleration = np.clip(
            feedforward_acceleration,
            -self.max_acceleration,
            self.max_acceleration
        )

        # Curve-adapted steering feedforward
        curvature = analysis_info.get('curvature', 0.0)
        feedforward_steering = curvature * 0.1
        feedforward_steering = np.clip(
            feedforward_steering,
            -self.max_steering_angle,
            self.max_steering_angle
        )

        return np.array([feedforward_acceleration, feedforward_steering])

    def find_closest_reference_point(self, current_state: np.ndarray) -> int:
        """Find the index of the closest reference point with sequential tracking.
        
        Uses a sequential search starting from the current reference index to avoid
        erratic jumps when the vehicle is stationary or moving slowly.
        """
        
        if len(self.reference_trajectory) == 0:
            return 0
        
        current_position = current_state[:2]
        current_index = self.current_reference_index if hasattr(self, 'current_reference_index') else 0
        
        # Search forward from current index to find lookahead point
        min_distance = float('inf')
        best_index = current_index
        
        # Search a window of points around current index
        search_start = max(0, current_index - 5)
        search_end = min(len(self.reference_trajectory), current_index + 15)
        
        for i in range(search_start, search_end):
            ref_point = self.reference_trajectory[i]
            ref_position = np.array([ref_point['x'], ref_point['y']])
            distance = np.linalg.norm(current_position - ref_position)
            
            if distance < min_distance:
                min_distance = distance
                best_index = i
        
        # If no good point found in window and we're very close to a point, advance
        if best_index == current_index and current_index < len(self.reference_trajectory) - 1:
            ref_point = self.reference_trajectory[current_index]
            ref_position = np.array([ref_point['x'], ref_point['y']])
            distance_to_current = np.linalg.norm(current_position - ref_position)
            
            # If we're close to current point, look ahead
            if distance_to_current < 0.5:  # Within 0.5m of current reference point
                best_index = min(current_index + 1, len(self.reference_trajectory) - 1)
        
        return best_index

    def control_callback(self):
        """Enhanced adaptive control loop with safety monitoring."""
        
        start_time = time.time()
        self.control_iteration_count += 1

        try:
            # Check safety conditions
            if not self.path_ready:
                self._log_control_block_reason("waiting for horizon mapper path_ready=true")
                self.publish_emergency_stop("path not ready")
                return

            if self.current_pose is None:
                self._log_control_block_reason("waiting for odometry/pose")
                self.publish_emergency_stop("no current pose")
                return

            if len(self.reference_trajectory) == 0:
                self._log_control_block_reason("reference trajectory is empty")
                self.publish_emergency_stop("empty reference trajectory")
                return

            if self.enable_safety_checks:
                current_time = time.time()
                solve_age = current_time - self.last_successful_solve_time
                if self.last_successful_solve_time > 0.0 and solve_age > self.safety_timeout:
                    self._log_control_block_reason(
                        f"control loop timed out after {solve_age:.2f}s without a successful solve"
                    )
                    self.publish_emergency_stop("control timeout")
                    return

            if not self.check_safety_conditions():
                self._log_control_block_reason("safety gate rejected the current state")
                self.publish_emergency_stop("safety gate rejected state")
                return

            current_state = self.get_current_state()

            if not self.kinematic_model.validate_state(current_state):
                self.get_logger().warning(
                    f"Invalid current state, stopping | state={current_state.tolist()} | {self._get_runtime_snapshot()}"
                )
                self.publish_emergency_stop("invalid current state")
                return

            if self.safety_monitor:
                self.safety_monitor.update_vehicle_state(
                    steering_angle=self.last_control_command[1],
                    angular_velocity=self.current_angular_velocity,
                    velocity=self.current_velocity
                )

            reference_state, analysis_info = self.get_enhanced_reference_state(current_state)
            feedforward_control = self.get_enhanced_feedforward_control(reference_state, analysis_info)
            current_index = self.find_closest_reference_point(current_state)

            control = self.lqr_controller.compute_control(
                current_state,
                reference_state,
                feedforward_control,
                trajectory=self.reference_trajectory,
                current_index=current_index
            )

            if self.steering_rate_limiter is not None:
                control[1] = self.steering_rate_limiter.limit_steering_rate(control[1])

            # Apply safety adjustments
            if self.safety_monitor:
                control = self.safety_monitor.get_safety_control_adjustment(
                    control, self.current_velocity
                )
                
                safety_status = self.safety_monitor.get_safety_status()
                if safety_status['emergency_stop_required']:
                    self.get_logger().warning(
                        "Emergency stop required by safety monitor | "
                        f"min_obstacle={safety_status['min_obstacle_distance']:.2f}m "
                        f"collision_imminent={safety_status['collision_imminent']} "
                        f"wobbling={safety_status['wobbling_detected']}"
                    )
                    self.publish_emergency_stop("safety monitor emergency stop")
                    return

            if not self.kinematic_model.validate_control(control, self.max_acceleration, self.max_steering_angle):
                self.get_logger().warning(
                    f"Invalid control output, stopping | accel={control[0]:.3f} steer={control[1]:.3f} | "
                    f"{self._get_runtime_snapshot()}"
                )
                self.publish_emergency_stop("invalid control output")
                return

            self.last_control_command = control.copy()
            self.publish_control_command(control, reference_state)
            self.publish_enhanced_debug_info(analysis_info)
            self.publish_adaptation_status()
            self.publish_safety_status()

            # Track solve timing — always collected, reported in /diagnostics
            solve_time = time.time() - start_time
            self.control_loop_times.append(solve_time)
            if len(self.control_loop_times) > 1000:
                self.control_loop_times = self.control_loop_times[-1000:]

            # Reset failure count on success
            self.consecutive_failures = 0
            self.last_successful_solve_time = time.time()
            self.control_active = True

            # Research-format step log — emitted every log_frequency_divider steps,
            # always on (not gated by debug mode).
            if self.control_iteration_count == 1 or self.control_iteration_count % self.log_frequency_divider == 0:
                adaptation_info = self.lqr_controller.get_adaptation_info()
                current_weights = adaptation_info['current_weights']
                perf = adaptation_info['performance_metrics']

                safety_str = ""
                if self.safety_monitor:
                    ss = self.safety_monitor.get_safety_status()
                    safety_str = (
                        f" | safety={'ACTIVE' if ss['safety_active'] else 'ok'}"
                        f" obs={ss['min_obstacle_distance']:.2f}m"
                    )

                self.get_logger().info(
                    f"[{self.control_iteration_count}] "
                    f"v={self.current_velocity:.2f}m/s v_ref={reference_state[2]:.2f}m/s "
                    f"a={control[0]:.3f} steer={control[1]:.3f} "
                    f"| ref_idx={current_index}/{len(self.reference_trajectory)} "
                    f"curv={analysis_info.get('curvature', 0.0):.3f} "
                    f"| w=pos:{current_weights['position']:.2f} "
                    f"head:{current_weights['heading']:.2f} "
                    f"steer:{current_weights['steering']:.2f} "
                    f"| quality={perf.get('tracking_quality', 0.0):.3f} "
                    f"solve={solve_time*1000:.1f}ms"
                    f"{safety_str}"
                )

        except Exception as e:
            self.consecutive_failures += 1
            self.get_logger().error(f"Control loop error: {e}")

            if self.consecutive_failures > 5:
                self.get_logger().error("5 consecutive control failures — emergency stop")
                self.publish_emergency_stop()

    def publish_control_command(self, control: np.ndarray, reference_state: np.ndarray = None):
        
        try:
            msg = AckermannDriveStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"

            msg.drive.acceleration = float(control[0])
            msg.drive.steering_angle = float(control[1])

            if reference_state is not None:
                target_velocity = min(reference_state[2], self.max_speed)
                step = self.speed_ramp_rate * self.dt
                if self.last_commanded_speed < target_velocity:
                    self.last_commanded_speed = min(
                        self.last_commanded_speed + step, target_velocity
                    )
                else:
                    self.last_commanded_speed = max(
                        self.last_commanded_speed - step, target_velocity
                    )
                msg.drive.speed = float(self.last_commanded_speed)
            else:
                self.last_commanded_speed = min(
                    self.last_commanded_speed + control[0] * self.dt, self.max_speed
                )
                msg.drive.speed = float(self.last_commanded_speed)

            msg.drive.speed = float(np.clip(msg.drive.speed, 0.0, self.max_speed))

            # Suppress positive acceleration when already at the speed limit so
            # firmware that uses the acceleration field also respects max_speed.
            if self.current_velocity >= self.max_speed:
                msg.drive.acceleration = float(min(msg.drive.acceleration, 0.0))
            
            self.control_publisher.publish(msg)
            self.last_control_time = time.time()

            self._last_control_block_reason = None

            if self.debug:
                self.get_logger().info(
                    f"Published drive command | accel={msg.drive.acceleration:.3f} "
                    f"steer={msg.drive.steering_angle:.3f} speed={msg.drive.speed:.3f} "
                    f"topic={self.control_topic}"
                )
            
        except Exception as e:
            self.get_logger().error(f"Control command publish failed: {e}")

    def publish_emergency_stop(self, reason: str = "unspecified"):
        """Publish emergency stop command."""
        
        try:
            msg = AckermannDriveStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            
            msg.drive.acceleration = -self.emergency_brake_threshold
            msg.drive.steering_angle = 0.0
            msg.drive.speed = 0.0
            
            self.control_publisher.publish(msg)
            self.control_active = False
            now = time.time()
            if (
                reason != self._last_emergency_stop_reason
                or (now - self._last_emergency_stop_time) > 1.0
            ):
                self._last_emergency_stop_reason = reason
                self._last_emergency_stop_time = now
                self.get_logger().warning(
                    f"Emergency stop sent | reason={reason} accel={msg.drive.acceleration:.3f} "
                    f"steer={msg.drive.steering_angle:.3f} speed={msg.drive.speed:.3f}"
                )
                
        except Exception as e:
            self.get_logger().error(f"Emergency stop publish failed: {e}")

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
            if self.debug:
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
            if self.debug:
                self.get_logger().warning(f"Failed to publish adaptation status: {e}")

    def publish_enhanced_debug_info(self, analysis_info: Dict):
        pass

    def publish_diagnostics(self):
        try:
            diag_msg = DiagnosticArray()
            diag_msg.header.stamp = self.get_clock().now().to_msg()

            controller_status = DiagnosticStatus()
            controller_status.name = "adaptive_lqr_controller"
            controller_status.hardware_id = "lqr_controller_node"
            controller_status.level = DiagnosticStatus.OK if self.control_active else DiagnosticStatus.WARN
            controller_status.message = "active" if self.control_active else "inactive"

            controller_status.values.append(KeyValue(key="control_active", value=str(self.control_active)))
            controller_status.values.append(KeyValue(key="path_ready", value=str(self.path_ready)))
            controller_status.values.append(KeyValue(key="reference_points", value=str(len(self.reference_trajectory))))
            controller_status.values.append(KeyValue(key="current_velocity", value=f"{self.current_velocity:.2f}"))
            controller_status.values.append(KeyValue(key="consecutive_failures", value=str(self.consecutive_failures)))

            if self.control_loop_times:
                controller_status.values.append(
                    KeyValue(key="avg_solve_ms", value=f"{np.mean(self.control_loop_times)*1000:.2f}")
                )
                controller_status.values.append(
                    KeyValue(key="max_solve_ms", value=f"{np.max(self.control_loop_times)*1000:.2f}")
                )

            try:
                adaptation_info = self.lqr_controller.get_adaptation_info()
                current_weights = adaptation_info['current_weights']
                performance_metrics = adaptation_info['performance_metrics']

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
            
            if self.safety_monitor:
                try:
                    ss = self.safety_monitor.get_safety_status()
                    controller_status.values.append(KeyValue(key="safety_active", value=str(ss['safety_active'])))
                    controller_status.values.append(KeyValue(key="min_obstacle_m", value=f"{ss['min_obstacle_distance']:.2f}"))
                    controller_status.values.append(KeyValue(key="wobbling", value=str(ss['wobbling_detected'])))
                    controller_status.values.append(KeyValue(key="safety_interventions", value=str(ss['safety_interventions'])))
                except Exception as e:
                    controller_status.values.append(KeyValue(key="safety_error", value=str(e)))

            diag_msg.status.append(controller_status)
            self.diagnostics_publisher.publish(diag_msg)

        except Exception as e:
            self.get_logger().error(f"Error publishing diagnostics: {e}")


class SimpleCurveAnalyzer:
    # Menger curvature estimate over a short lookahead window.
    def __init__(self, lookahead_points: int = 5, max_curvature_threshold: float = 1.0):
        self.lookahead_points = lookahead_points
        self.max_curvature_threshold = max_curvature_threshold

    def analyze_segment(self, trajectory: List[Dict], current_index: int) -> Dict:
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

                # Menger curvature: 2 * area / (|p1p2| * |p2p3| * |p1p3|)
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
    def __init__(self, max_rate: float, dt: float):
        self.max_rate = max_rate
        self.dt = dt
        self.last_steering = 0.0

    def limit_steering_rate(self, desired_steering: float) -> float:
        max_change = self.max_rate * self.dt
        steering_change = desired_steering - self.last_steering
        if abs(steering_change) > max_change:
            limited_steering = self.last_steering + np.sign(steering_change) * max_change
        else:
            limited_steering = desired_steering
        self.last_steering = limited_steering
        return limited_steering


def main(args=None):
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

