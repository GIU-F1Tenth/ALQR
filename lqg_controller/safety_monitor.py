#!/usr/bin/env python3

"""
Safety Monitor for Adaptive LQR Controller

This module implements safety features including:
- Lidar-based collision avoidance
- Wobble detection and mitigation
- Emergency braking systems

Author: Mohammed Azab <mohammed@azab.io>
License: MIT
Version: 1.0.0
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import time
from collections import deque
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


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
        return np.mean(derivatives) if derivatives else 0.0
    
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
                 enable_logging: bool = True, logger: Optional = None):
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