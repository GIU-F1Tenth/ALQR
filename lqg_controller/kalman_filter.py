#!/usr/bin/env python3

"""
Extended Kalman Filter (EKF) Implementation for F1TENTH Vehicle State Estimation

This module implements an Extended Kalman Filter for estimating the full vehicle state
[x, y, v, theta] using noisy sensor measurements from LiDAR, IMU, and odometry.

The EKF handles:
- Nonlinear vehicle dynamics (kinematic bicycle model)
- Multiple sensor fusion (IMU angular velocity, odometry linear velocity, optional position updates)
- Adaptive noise covariance based on driving conditions

Author: Mohammed Azab <mohammed@azab.io>
License: MIT
Version: 1.0.0
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import time
from .kinematic_bicycle_model import KinematicBicycleModel


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for F1TENTH vehicle state estimation.

    State vector: [x, y, v, theta]
    - x, y: position in global frame [m]
    - v: linear velocity [m/s]
    - theta: heading angle [rad]

    Control input: [acceleration, steering_angle]

    Measurements:
    - IMU: angular velocity [rad/s]
    - Odometry: linear velocity [m/s]
    - Optional: Position updates from localization
    """

    def __init__(self,
                 wheelbase: float = 0.33,
                 dt: float = 0.02,  # 50 Hz
                 initial_state: Optional[np.ndarray] = None,
                 initial_covariance: Optional[np.ndarray] = None,
                 process_noise: Optional[np.ndarray] = None,
                 measurement_noise_imu: float = 0.1,
                 measurement_noise_odom: float = 0.05,
                 measurement_noise_position: float = 0.1,
                 enable_logging: bool = True,
                 logger: Optional[Any] = None):
        """
        Initialize the Extended Kalman Filter.

        Args:
            wheelbase: Vehicle wheelbase [m]
            dt: Filter time step [s]
            initial_state: Initial state estimate [x, y, v, theta]
            initial_covariance: Initial state covariance matrix (4x4)
            process_noise: Process noise covariance matrix (4x4)
            measurement_noise_imu: IMU angular velocity measurement noise variance
            measurement_noise_odom: Odometry velocity measurement noise variance
            measurement_noise_position: Position measurement noise variance
            enable_logging: Enable performance logging
            logger: ROS2 logger instance (optional)
        """

        self.model = KinematicBicycleModel(wheelbase, dt)
        self.dt = dt
        self.enable_logging = enable_logging
        self.logger = logger

        # State dimension
        self.n_states = 4  # [x, y, v, theta]

        # Initialize state estimate
        if initial_state is not None:
            self.x_hat = initial_state.copy()
        else:
            self.x_hat = np.zeros(4)  # [x, y, v, theta]

        # Initialize state covariance
        if initial_covariance is not None:
            self.P = initial_covariance.copy()
        else:
            # Conservative initial uncertainty
            self.P = np.diag([1.0, 1.0, 0.5, 0.1])  # [x, y, v, theta] uncertainties

        # Process noise covariance matrix Q
        if process_noise is not None:
            self.Q = process_noise.copy()
        else:
            # Default process noise - accounts for model uncertainties
            self.Q = np.diag([
                0.01,  # x process noise
                0.01,  # y process noise
                0.1,   # v process noise (velocity can change quickly)
                0.05   # theta process noise
            ])

        # Measurement noise variances
        self.R_imu = measurement_noise_imu**2  # Angular velocity measurement noise
        self.R_odom = measurement_noise_odom**2  # Linear velocity measurement noise
        self.R_position = measurement_noise_position**2  # Position measurement noise

        # Last control input for prediction
        self.last_control = np.zeros(2)

        # Performance tracking
        self.prediction_times = []
        self.update_times = []
        self.innovation_history = []

        # Filter health monitoring
        self.last_update_time = time.time()
        self.consecutive_prediction_only = 0
        self.max_prediction_only = 20  # Maximum predictions without measurements

        self.log_info("Extended Kalman Filter initialized successfully")

    def log_info(self, message: str):
        """Log information message."""
        if self.logger:
            self.logger.info(message)
        elif self.enable_logging:
            print(f"[EKF INFO] {message}")

    def log_warning(self, message: str):
        """Log warning message."""
        if self.logger:
            self.logger.warn(message)
        elif self.enable_logging:
            print(f"[EKF WARNING] {message}")

    def predict(self, control_input: np.ndarray) -> None:
        """
        EKF Prediction step using kinematic bicycle model.

        Args:
            control_input: Control input [acceleration, steering_angle]
        """
        start_time = time.time() if self.enable_logging else None

        try:
            # Store control input
            self.last_control = control_input.copy()

            # Predict next state using kinematic bicycle model
            x_pred = self.model.update_state(self.x_hat, control_input)

            # Compute Jacobian of process model (F matrix)
            F = self._compute_process_jacobian(self.x_hat, control_input)

            # Predict state covariance
            P_pred = F @ self.P @ F.T + self.Q

            # Update state and covariance
            self.x_hat = x_pred
            self.P = P_pred

            # Health monitoring
            self.consecutive_prediction_only += 1
            if self.consecutive_prediction_only > self.max_prediction_only:
                self.log_warning(f"Filter running on prediction only for {self.consecutive_prediction_only} steps")

            # Performance tracking
            if self.enable_logging and start_time:
                self.prediction_times.append(time.time() - start_time)

        except Exception as e:
            self.log_warning(f"Error in prediction step: {e}")
            # Keep previous state as fallback

    def update_imu(self, angular_velocity_measurement: float) -> None:
        """
        Update filter with IMU angular velocity measurement.

        Args:
            angular_velocity_measurement: Measured angular velocity [rad/s]
        """
        start_time = time.time() if self.enable_logging else None

        try:
            # Measurement model: h(x) = v * tan(delta) / wheelbase
            # where delta is the steering angle from last control input
            steering_angle = self.last_control[1] if len(self.last_control) > 1 else 0.0

            # Predicted angular velocity from current state
            if abs(steering_angle) < 1e-6:
                h_x = 0.0  # No turning
            else:
                h_x = self.x_hat[2] * np.tan(steering_angle) / self.model.wheelbase

            # Measurement residual (innovation)
            y = angular_velocity_measurement - h_x

            # Compute measurement Jacobian (H matrix)
            H = self._compute_imu_measurement_jacobian(steering_angle)

            # Innovation covariance
            S = H @ self.P @ H.T + self.R_imu

            # Kalman gain
            K = self.P @ H.T / S  # For scalar measurement

            # Update state estimate
            self.x_hat = self.x_hat + K * y

            # Update state covariance
            I_KH = np.eye(self.n_states) - np.outer(K, H)
            self.P = I_KH @ self.P

            # Reset consecutive prediction counter
            self.consecutive_prediction_only = 0
            self.last_update_time = time.time()

            # Store innovation for analysis
            if self.enable_logging:
                self.innovation_history.append(abs(y))
                if start_time:
                    self.update_times.append(time.time() - start_time)

        except Exception as e:
            self.log_warning(f"Error in IMU update: {e}")

    def update_odometry(self, velocity_measurement: float) -> None:
        """
        Update filter with odometry velocity measurement.

        Args:
            velocity_measurement: Measured linear velocity [m/s]
        """
        start_time = time.time() if self.enable_logging else None

        try:
            # Measurement model: h(x) = v (direct velocity measurement)
            h_x = self.x_hat[2]

            # Measurement residual (innovation)
            y = velocity_measurement - h_x

            # Measurement Jacobian (H matrix) - velocity is state[2]
            H = np.array([0.0, 0.0, 1.0, 0.0])

            # Innovation covariance
            S = H @ self.P @ H.T + self.R_odom

            # Kalman gain
            K = self.P @ H.T / S  # For scalar measurement

            # Update state estimate
            self.x_hat = self.x_hat + K * y

            # Update state covariance
            I_KH = np.eye(self.n_states) - np.outer(K, H)
            self.P = I_KH @ self.P

            # Reset consecutive prediction counter
            self.consecutive_prediction_only = 0
            self.last_update_time = time.time()

            # Store innovation for analysis
            if self.enable_logging:
                self.innovation_history.append(abs(y))
                if start_time:
                    self.update_times.append(time.time() - start_time)

        except Exception as e:
            self.log_warning(f"Error in odometry update: {e}")

    def update_position(self, position_measurement: np.ndarray) -> None:
        """
        Update filter with position measurement (e.g., from localization).

        Args:
            position_measurement: Measured position [x, y]
        """
        start_time = time.time() if self.enable_logging else None

        try:
            # Measurement model: h(x) = [x, y] (direct position measurement)
            h_x = self.x_hat[:2]

            # Measurement residual (innovation)
            y = position_measurement - h_x

            # Measurement Jacobian (H matrix) - position is states [0, 1]
            H = np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0]])

            # Innovation covariance
            R_pos = np.eye(2) * self.R_position
            S = H @ self.P @ H.T + R_pos

            # Kalman gain
            K = self.P @ H.T @ np.linalg.inv(S)

            # Update state estimate
            self.x_hat = self.x_hat + K @ y

            # Update state covariance
            I_KH = np.eye(self.n_states) - K @ H
            self.P = I_KH @ self.P

            # Reset consecutive prediction counter
            self.consecutive_prediction_only = 0
            self.last_update_time = time.time()

            # Store innovation for analysis
            if self.enable_logging:
                innovation_magnitude = np.linalg.norm(y)
                self.innovation_history.append(innovation_magnitude)
                if start_time:
                    self.update_times.append(time.time() - start_time)

        except Exception as e:
            self.log_warning(f"Error in position update: {e}")

    def _compute_process_jacobian(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of the process model (F matrix) for linearization.

        Args:
            state: Current state [x, y, v, theta]
            control: Control input [acceleration, steering_angle]

        Returns:
            F: Process model Jacobian (4x4)
        """
        x, y, v, theta = state
        a, delta = control

        # For kinematic bicycle model:
        # x_{k+1} = x_k + v_k * cos(theta_k) * dt
        # y_{k+1} = y_k + v_k * sin(theta_k) * dt
        # v_{k+1} = v_k + a_k * dt
        # theta_{k+1} = theta_k + (v_k * tan(delta_k) / wheelbase) * dt

        dt = self.dt
        L = self.model.wheelbase

        # Compute partial derivatives
        F = np.eye(4)

        # dx/dtheta = -v * sin(theta) * dt
        F[0, 3] = -v * np.sin(theta) * dt
        # dx/dv = cos(theta) * dt
        F[0, 2] = np.cos(theta) * dt

        # dy/dtheta = v * cos(theta) * dt
        F[1, 3] = v * np.cos(theta) * dt
        # dy/dv = sin(theta) * dt
        F[1, 2] = np.sin(theta) * dt

        # dv/dv = 1 (already in identity matrix)

        # dtheta/dv = tan(delta) / L * dt
        if abs(delta) < np.pi / 2 - 0.1:  # Avoid singularity
            F[3, 2] = np.tan(delta) / L * dt

        # dtheta/dtheta = 1 (already in identity matrix)

        return F

    def _compute_imu_measurement_jacobian(self, steering_angle: float) -> np.ndarray:
        """
        Compute Jacobian of IMU measurement model (H matrix).

        Args:
            steering_angle: Current steering angle [rad]

        Returns:
            H: Measurement model Jacobian (1x4)
        """
        # Measurement model: h(x) = v * tan(delta) / wheelbase
        L = self.model.wheelbase

        H = np.zeros(4)

        if abs(steering_angle) < np.pi / 2 - 0.1:  # Avoid singularity
            # dh/dv = tan(delta) / L
            H[2] = np.tan(steering_angle) / L

        return H

    def get_state_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current state estimate and covariance.

        Returns:
            state_estimate: Current state estimate [x, y, v, theta]
            covariance: Current state covariance matrix (4x4)
        """
        return self.x_hat.copy(), self.P.copy()

    def get_state_uncertainty(self) -> np.ndarray:
        """
        Get current state estimation uncertainties (standard deviations).

        Returns:
            uncertainties: Standard deviations [sigma_x, sigma_y, sigma_v, sigma_theta]
        """
        return np.sqrt(np.diag(self.P))

    def is_filter_healthy(self) -> bool:
        """
        Check if the filter is operating in a healthy state.

        Returns:
            healthy: True if filter is healthy, False otherwise
        """
        # Check if covariance is reasonable (not too large or too small)
        uncertainties = self.get_state_uncertainty()

        # Define reasonable bounds for uncertainties
        max_position_uncertainty = 5.0  # [m]
        max_velocity_uncertainty = 10.0  # [m/s]
        max_heading_uncertainty = np.pi  # [rad]

        min_uncertainty = 1e-6  # Avoid overconfidence

        healthy = True

        # Check position uncertainties
        if (uncertainties[0] > max_position_uncertainty or
            uncertainties[1] > max_position_uncertainty or
            uncertainties[0] < min_uncertainty or
                uncertainties[1] < min_uncertainty):
            healthy = False

        # Check velocity uncertainty
        if (uncertainties[2] > max_velocity_uncertainty or
                uncertainties[2] < min_uncertainty):
            healthy = False

        # Check heading uncertainty
        if (uncertainties[3] > max_heading_uncertainty or
                uncertainties[3] < min_uncertainty):
            healthy = False

        # Check if too many predictions without measurements
        if self.consecutive_prediction_only > self.max_prediction_only:
            healthy = False

        return healthy

    def reset_filter(self, initial_state: Optional[np.ndarray] = None,
                     initial_covariance: Optional[np.ndarray] = None) -> None:
        """
        Reset the filter to initial conditions.

        Args:
            initial_state: New initial state estimate
            initial_covariance: New initial covariance matrix
        """
        if initial_state is not None:
            self.x_hat = initial_state.copy()
        else:
            self.x_hat = np.zeros(4)

        if initial_covariance is not None:
            self.P = initial_covariance.copy()
        else:
            self.P = np.diag([1.0, 1.0, 0.5, 0.1])

        # Reset monitoring
        self.consecutive_prediction_only = 0
        self.last_update_time = time.time()

        # Clear performance history
        self.prediction_times.clear()
        self.update_times.clear()
        self.innovation_history.clear()

        self.log_info("Kalman Filter reset to initial conditions")

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get filter performance metrics.

        Returns:
            metrics: Dictionary of performance metrics
        """
        metrics = {}

        if self.prediction_times:
            metrics['avg_prediction_time'] = np.mean(self.prediction_times)
            metrics['max_prediction_time'] = np.max(self.prediction_times)

        if self.update_times:
            metrics['avg_update_time'] = np.mean(self.update_times)
            metrics['max_update_time'] = np.max(self.update_times)

        if self.innovation_history:
            metrics['avg_innovation'] = np.mean(self.innovation_history)
            metrics['max_innovation'] = np.max(self.innovation_history)

        metrics['consecutive_predictions'] = self.consecutive_prediction_only
        metrics['time_since_last_update'] = time.time() - self.last_update_time
        metrics['is_healthy'] = self.is_filter_healthy()

        uncertainties = self.get_state_uncertainty()
        metrics['position_uncertainty'] = np.linalg.norm(uncertainties[:2])
        metrics['velocity_uncertainty'] = uncertainties[2]
        metrics['heading_uncertainty'] = uncertainties[3]

        return metrics
