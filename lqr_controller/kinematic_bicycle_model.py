#!/usr/bin/env python3

"""
Kinematic Bicycle Model for LQR Controller

This module implements the kinematic bicycle model dynamics and linearization
for use with Linear Quadratic Regulator (LQR) control.

Author: Mohammed Azab <mohammed@azab.io>
License: MIT
Version: 1.0.0
"""

import numpy as np
from typing import Tuple, Optional
import sys
import os

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
                #print(f"✅ Using config.py from {abs_path}")
                config_imported = True
                break
            except ImportError as e:
                continue
        else:
            print(f"❌ Path does not exist: {abs_path}")
    
    if not config_imported:
        raise ImportError("Config module not found in any expected location")
        
except ImportError as e:
    # Fallback if config.py is not available
    print(f"Warning: {e}")
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


    default_config = DefaultConfig()

class KinematicBicycleModel:
    """
    Kinematic bicycle model for vehicle dynamics.

    State vector: [x, y, v, theta]
    - x: longitudinal position [m]
    - y: lateral position [m]
    - v: velocity [m/s]
    - theta: heading angle [rad]

    Control input: [a, delta]
    - a: acceleration [m/s²]
    - delta: steering angle [rad]
    """

    def __init__(self, wheelbase: float = None, dt: float = None):
        """
        Initialize the kinematic bicycle model.

        Args:
            wheelbase: Distance between front and rear axles [m]
            dt: Discrete time step [s]
        """
        self.L = wheelbase if wheelbase is not None else default_config.wheelbase
        self.dt = dt if dt is not None else default_config.dt

    def dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Compute the next state using kinematic bicycle model dynamics.

        Args:
            state: Current state [x, y, v, theta]
            control: Control input [a, delta]

        Returns:
            Next state [x_next, y_next, v_next, theta_next]
        """
        x, y, v, theta = state
        a, delta = control

        # State derivatives
        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        v_dot = a
        theta_dot = (v / self.L) * np.tan(delta)

        # Euler integration
        x_next = x + self.dt * x_dot
        y_next = y + self.dt * y_dot
        v_next = v + self.dt * v_dot
        theta_next = theta + self.dt * theta_dot

        return np.array([x_next, y_next, v_next, theta_next])

    def linearize(self, state: np.ndarray, control: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize the kinematic bicycle model around the given state and control.

        Args:
            state: Linearization point for state [x, y, v, theta]
            control: Linearization point for control [a, delta]

        Returns:
            A: State transition matrix (4x4)
            B: Control input matrix (4x2)
        """
        x, y, v, theta = state
        a, delta = control

        # Partial derivatives for A matrix (∂f/∂x)
        A = np.array([
            [1.0, 0.0, self.dt * np.cos(theta), -self.dt * v * np.sin(theta)],
            [0.0, 1.0, self.dt * np.sin(theta), self.dt * v * np.cos(theta)],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, self.dt * np.tan(delta) / self.L, 1.0]
        ])

        # Partial derivatives for B matrix (∂f/∂u)
        cos_delta_sq = np.cos(delta) ** 2
        B = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [self.dt, 0.0],
            [0.0, self.dt * v / (self.L * cos_delta_sq)]
        ])

        return A, B

    def compute_state_reference(self, trajectory_point: dict) -> np.ndarray:
        """
        Convert trajectory point to state reference vector.

        Args:
            trajectory_point: Dictionary with keys 'x', 'y', 'v', 'theta'

        Returns:
            State reference vector [x_ref, y_ref, v_ref, theta_ref]
        """
        return np.array([
            trajectory_point['x'],
            trajectory_point['y'],
            trajectory_point['v'],
            trajectory_point['theta']
        ])

    def validate_state(self, state: np.ndarray) -> bool:
        """
        Validate that the state vector is physically reasonable.

        Args:
            state: State vector [x, y, v, theta]

        Returns:
            True if state is valid, False otherwise
        """
        if len(state) != 4:
            return False

        x, y, v, theta = state

        # Check for NaN or infinite values
        if not np.all(np.isfinite(state)):
            return False

        # Check velocity bounds
        if v < -1.0 or v > default_config.max_speed:  # Allow small reverse velocity
            return False

        # Normalize heading angle to [-π, π]
        if abs(theta) > 10 * np.pi:  # Sanity check for reasonable heading
            return False

        return True

    def validate_control(self, control: np.ndarray, max_acceleration: float = None,
                         max_steering: float = None) -> bool:
        """
        Validate that the control input is within reasonable bounds.

        Args:
            control: Control vector [a, delta]
            max_acceleration: Maximum acceleration magnitude [m/s²]
            max_steering: Maximum steering angle magnitude [rad]

        Returns:
            True if control is valid, False otherwise
        """
        if len(control) != 2:
            return False

        a, delta = control

        # Check for NaN or infinite values
        if not np.all(np.isfinite(control)):
            return False

        max_accel = max_acceleration if max_acceleration is not None else max(default_config.max_acceleration, default_config.max_deceleration)
        max_steer = max_steering if max_steering is not None else default_config.max_steering_angle

        # Check acceleration bounds
        if abs(a) > max_accel:
            return False

        # Check steering bounds
        if abs(delta) > max_steer:
            return False

        return True

    def normalize_angle(self, angle: float) -> float:
        """
        Normalize angle to [-π, π] range.

        Args:
            angle: Input angle [rad]

        Returns:
            Normalized angle in [-π, π]
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def get_model_info(self) -> dict:
        """
        Get model parameters and information.

        Returns:
            Dictionary containing model information
        """
        return {
            'model_type': 'kinematic_bicycle',
            'wheelbase': self.L,
            'dt': self.dt,
            'state_size': 4,
            'control_size': 2,
            'state_names': ['x', 'y', 'v', 'theta'],
            'control_names': ['acceleration', 'steering_angle']
        }
