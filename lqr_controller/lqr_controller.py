#!/usr/bin/env python3

"""
LQR Controller Implementation

This module implements a Linear Quadratic Regulator (LQR) controller for
trajectory tracking in F1TENTH autonomous racing vehicles.

Author: Mohammed Azab <mohammed@azab.io>
License: MIT
Version: 1.0.0
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import scipy.linalg
import time
from .kinematic_bicycle_model import KinematicBicycleModel


class LQRController:
    """
    Linear Quadratic Regulator (LQR) controller for trajectory tracking.

    This controller:
    1. Linearizes the kinematic bicycle model around the current state
    2. Solves the discrete-time algebraic Riccati equation (DARE)
    3. Computes optimal control inputs using LQR feedback law
    """

    def __init__(self,
                 wheelbase: float = 0.33,
                 dt: float = 0.05,
                 Q: Optional[np.ndarray] = None,
                 R: Optional[np.ndarray] = None,
                 max_acceleration: float = 5.0,
                 max_steering: float = 0.5,
                 enable_logging: bool = True,
                 logger: Optional[Any] = None):
        """
        Initialize the LQR controller.

        Args:
            wheelbase: Vehicle wheelbase [m]
            dt: Control time step [s]
            Q: State cost matrix (4x4). If None, uses default weights
            R: Control cost matrix (2x2). If None, uses default weights
            max_acceleration: Maximum acceleration magnitude [m/s²]
            max_steering: Maximum steering angle magnitude [rad]
            enable_logging: Enable performance logging
            logger: ROS2 logger instance (optional)
        """
        self.model = KinematicBicycleModel(wheelbase, dt)
        self.dt = dt
        self.max_acceleration = max_acceleration
        self.max_steering = max_steering
        self.enable_logging = enable_logging
        self.logger = logger

        # Set up cost matrices
        if Q is None:
            # Default state cost matrix
            # Higher weights on position tracking, moderate on velocity and heading
            self.Q = np.diag([
                10.0,  # x position weight
                10.0,  # y position weight
                1.0,   # velocity weight
                5.0    # heading weight
            ])
        else:
            self.Q = Q

        if R is None:
            # Default control cost matrix
            # Moderate weight on acceleration, higher weight on steering (smoother steering)
            self.R = np.diag([
                0.1,   # acceleration weight
                1.0    # steering weight
            ])
        else:
            self.R = R

        # Performance tracking
        self.solve_times = []
        self.control_history = []
        self.state_error_history = []

        # LQR solution caching for efficiency
        self.last_linearization_state = None
        self.last_linearization_control = None
        self.cached_K = None
        self.cache_tolerance = 1e-3

        self.log_info("LQR Controller initialized successfully")

    def log_info(self, message: str):
        """Log information message."""
        if self.logger:
            self.logger.info(message)
        elif self.enable_logging:
            print(f"[LQR Controller] {message}")

    def log_warning(self, message: str):
        """Log warning message."""
        if self.logger:
            self.logger.warn(message)
        elif self.enable_logging:
            print(f"[LQR Controller WARNING] {message}")

    def log_error(self, message: str):
        """Log error message."""
        if self.logger:
            self.logger.error(message)
        elif self.enable_logging:
            print(f"[LQR Controller ERROR] {message}")

    def solve_dare(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Solve the discrete-time algebraic Riccati equation (DARE).

        Args:
            A: State transition matrix (4x4)
            B: Control input matrix (4x2)
            Q: State cost matrix (4x4)
            R: Control cost matrix (2x2)

        Returns:
            P: Solution to DARE (4x4)
        """
        try:
            # Solve discrete-time algebraic Riccati equation
            P = scipy.linalg.solve_discrete_are(A, B, Q, R)
            return P
        except Exception as e:
            self.log_error(f"Failed to solve DARE: {e}")
            # Return identity matrix as fallback
            return np.eye(A.shape[0])

    def compute_lqr_gain(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute the LQR feedback gain matrix.

        Args:
            A: State transition matrix (4x4)
            B: Control input matrix (4x2)

        Returns:
            K: LQR feedback gain matrix (2x4)
        """
        try:
            # Solve DARE to get P matrix
            P = self.solve_dare(A, B, self.Q, self.R)

            # Compute LQR gain: K = (R + B^T P B)^(-1) B^T P A
            BtPB = B.T @ P @ B
            BtPA = B.T @ P @ A

            # Add small regularization to avoid singular matrix
            reg_term = 1e-6 * np.eye(BtPB.shape[0])
            K = np.linalg.solve(self.R + BtPB + reg_term, BtPA)

            return K

        except Exception as e:
            self.log_error(f"Failed to compute LQR gain: {e}")
            # Return zero gain as fallback
            return np.zeros((2, 4))

    def should_recompute_gain(self, current_state: np.ndarray, current_control: np.ndarray) -> bool:
        """
        Check if LQR gain should be recomputed based on state/control changes.

        Args:
            current_state: Current vehicle state
            current_control: Current control input

        Returns:
            True if gain should be recomputed
        """
        if (self.last_linearization_state is None or
            self.last_linearization_control is None or
                self.cached_K is None):
            return True

        # Check if state or control have changed significantly
        state_diff = np.linalg.norm(current_state - self.last_linearization_state)
        control_diff = np.linalg.norm(current_control - self.last_linearization_control)

        return (state_diff > self.cache_tolerance or control_diff > self.cache_tolerance)

    def compute_control(self, current_state: np.ndarray, reference_state: np.ndarray,
                        feedforward_control: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute LQR control input for trajectory tracking.

        Args:
            current_state: Current vehicle state [x, y, v, theta]
            reference_state: Reference state [x_ref, y_ref, v_ref, theta_ref]
            feedforward_control: Optional feedforward control [a_ff, delta_ff]

        Returns:
            control: Optimal control input [acceleration, steering_angle]
        """
        start_time = time.time() if self.enable_logging else None

        try:
            # Validate inputs
            if not self.model.validate_state(current_state):
                self.log_warning("Invalid current state, using zero control")
                return np.zeros(2)

            if not self.model.validate_state(reference_state):
                self.log_warning("Invalid reference state, using zero control")
                return np.zeros(2)

            # Normalize heading angles to avoid large angle differences
            current_state_norm = current_state.copy()
            reference_state_norm = reference_state.copy()
            current_state_norm[3] = self.model.normalize_angle(current_state_norm[3])
            reference_state_norm[3] = self.model.normalize_angle(reference_state_norm[3])

            # Compute state error
            state_error = current_state_norm - reference_state_norm

            # Handle angle wrapping for heading error
            state_error[3] = self.model.normalize_angle(state_error[3])

            # Set up linearization point (typically at reference state)
            linearization_state = reference_state_norm
            linearization_control = feedforward_control if feedforward_control is not None else np.zeros(2)

            # Check if we need to recompute LQR gain
            if self.should_recompute_gain(linearization_state, linearization_control):
                # Linearize model around reference state
                A, B = self.model.linearize(linearization_state, linearization_control)

                # Compute LQR gain
                K = self.compute_lqr_gain(A, B)

                # Cache the results
                self.cached_K = K
                self.last_linearization_state = linearization_state.copy()
                self.last_linearization_control = linearization_control.copy()
            else:
                # Use cached gain
                K = self.cached_K

            # Compute feedback control: u = u_ff - K * (x - x_ref)
            feedback_control = -K @ state_error

            # Add feedforward control if provided
            if feedforward_control is not None:
                control = feedforward_control + feedback_control
            else:
                control = feedback_control

            # Apply control constraints
            control = self.apply_control_constraints(control)

            # Store for performance analysis
            if self.enable_logging:
                solve_time = time.time() - start_time
                self.solve_times.append(solve_time)
                self.control_history.append(control.copy())
                self.state_error_history.append(np.linalg.norm(state_error))

                # Keep history bounded
                max_history = 1000
                if len(self.solve_times) > max_history:
                    self.solve_times = self.solve_times[-max_history:]
                    self.control_history = self.control_history[-max_history:]
                    self.state_error_history = self.state_error_history[-max_history:]

            return control

        except Exception as e:
            self.log_error(f"Failed to compute LQR control: {e}")
            return np.zeros(2)

    def apply_control_constraints(self, control: np.ndarray) -> np.ndarray:
        """
        Apply control input constraints.

        Args:
            control: Unconstrained control input [acceleration, steering_angle]

        Returns:
            Constrained control input
        """
        constrained_control = control.copy()

        # Constrain acceleration
        constrained_control[0] = np.clip(constrained_control[0],
                                         -self.max_acceleration,
                                         self.max_acceleration)

        # Constrain steering angle
        constrained_control[1] = np.clip(constrained_control[1],
                                         -self.max_steering,
                                         self.max_steering)

        return constrained_control

    def update_cost_matrices(self, Q: np.ndarray, R: np.ndarray):
        """
        Update the cost matrices Q and R.

        Args:
            Q: New state cost matrix (4x4)
            R: New control cost matrix (2x2)
        """
        self.Q = Q
        self.R = R

        # Invalidate cached gain to force recomputation
        self.cached_K = None
        self.last_linearization_state = None
        self.last_linearization_control = None

        self.log_info("Cost matrices updated")

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get controller performance metrics.

        Returns:
            Dictionary containing performance metrics
        """
        if not self.solve_times:
            return {}

        return {
            'avg_solve_time': np.mean(self.solve_times),
            'max_solve_time': np.max(self.solve_times),
            'min_solve_time': np.min(self.solve_times),
            'avg_state_error': np.mean(self.state_error_history) if self.state_error_history else 0.0,
            'total_control_calls': len(self.solve_times)
        }

    def reset_performance_metrics(self):
        """Reset performance tracking metrics."""
        self.solve_times.clear()
        self.control_history.clear()
        self.state_error_history.clear()

    def get_controller_info(self) -> Dict[str, Any]:
        """
        Get controller configuration information.

        Returns:
            Dictionary containing controller information
        """
        return {
            'controller_type': 'LQR',
            'model_info': self.model.get_model_info(),
            'Q_matrix': self.Q.tolist(),
            'R_matrix': self.R.tolist(),
            'max_acceleration': self.max_acceleration,
            'max_steering': self.max_steering,
            'cache_tolerance': self.cache_tolerance,
            'has_cached_gain': self.cached_K is not None
        }
