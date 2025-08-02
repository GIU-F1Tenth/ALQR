#!/usr/bin/env python3

"""
LQG Controller Implementation for F1TENTH Autonomous Racing

This module implements a Linear Quadratic Gaussian (LQG) controller that combines:
1. Extended Kalman Filter (EKF) for state estimation from noisy sensor data
2. Existing LQR controller for optimal control computation

The LQG controller provides optimal control under uncertainty by:
- Using the EKF to estimate the full vehicle state from noisy measurements
- Feeding the estimated state to the LQR controller for optimal control computation
- Maintaining separation principle: estimation and control can be designed independently

Author: Mohammed Azab <mohammed@azab.io>
License: MIT
Version: 1.0.0
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import time

from .kalman_filter import ExtendedKalmanFilter
from .adaptive_lqr_controller import AdaptiveLQRController


class LQGController:
    """
    Linear Quadratic Gaussian (LQG) controller combining EKF state estimation
    with LQR optimal control.

    The LQG controller follows the separation principle:
    1. EKF estimates the state from noisy sensor measurements
    2. LQR computes optimal control using the estimated state
    3. The combination is optimal under Gaussian noise assumptions
    """

    def __init__(self,
                 # Vehicle parameters
                 wheelbase: float = 0.33,
                 dt: float = 0.02,  # 50 Hz for real-time performance

                 # LQR parameters
                 Q: Optional[np.ndarray] = None,
                 R: Optional[np.ndarray] = None,
                 max_acceleration: float = 5.0,
                 max_steering: float = 0.5,

                 # EKF parameters
                 initial_state: Optional[np.ndarray] = None,
                 initial_covariance: Optional[np.ndarray] = None,
                 process_noise: Optional[np.ndarray] = None,
                 measurement_noise_imu: float = 0.1,
                 measurement_noise_odom: float = 0.05,
                 measurement_noise_position: float = 0.1,

                 # General parameters
                 enable_logging: bool = True,
                 logger: Optional[Any] = None):
        """
        Initialize the LQG controller.

        Args:
            wheelbase: Vehicle wheelbase [m]
            dt: Control time step [s]
            Q: LQR state cost matrix (4x4)
            R: LQR control cost matrix (2x2)
            max_acceleration: Maximum acceleration magnitude [m/s²]
            max_steering: Maximum steering angle magnitude [rad]
            initial_state: Initial state estimate for EKF [x, y, v, theta]
            initial_covariance: Initial state covariance matrix for EKF (4x4)
            process_noise: Process noise covariance matrix for EKF (4x4)
            measurement_noise_imu: IMU measurement noise standard deviation
            measurement_noise_odom: Odometry measurement noise standard deviation
            measurement_noise_position: Position measurement noise standard deviation
            enable_logging: Enable performance logging
            logger: ROS2 logger instance (optional)
        """

        self.dt = dt
        self.enable_logging = enable_logging
        self.logger = logger

        # Initialize Extended Kalman Filter for state estimation
        self.ekf = ExtendedKalmanFilter(
            wheelbase=wheelbase,
            dt=dt,
            initial_state=initial_state,
            initial_covariance=initial_covariance,
            process_noise=process_noise,
            measurement_noise_imu=measurement_noise_imu,
            measurement_noise_odom=measurement_noise_odom,
            measurement_noise_position=measurement_noise_position,
            enable_logging=enable_logging,
            logger=logger
        )

        # Initialize LQR controller for optimal control computation
        self.lqr = AdaptiveLQRController(
            wheelbase=wheelbase,
            dt=dt,
            Q=Q,
            R=R,
            max_acceleration=max_acceleration,
            max_steering=max_steering,
            enable_logging=enable_logging,
            logger=logger
        )

        # Performance tracking
        self.control_computation_times = []
        self.state_estimation_times = []
        self.total_computation_times = []

        # Control history for analysis
        self.estimated_states_history = []
        self.control_history = []
        self.reference_states_history = []

        # Filter health monitoring
        self.filter_health_history = []
        self.estimation_quality_threshold = 0.5  # Position uncertainty threshold [m]

        self.log_info("LQG Controller initialized successfully")

    def log_info(self, message: str):
        """Log information message."""
        if self.logger:
            self.logger.info(message)
        elif self.enable_logging:
            print(f"[LQG INFO] {message}")

    def log_warning(self, message: str):
        """Log warning message."""
        if self.logger:
            self.logger.warn(message)
        elif self.enable_logging:
            print(f"[LQG WARNING] {message}")

    def update_state_estimate(self, control_input: np.ndarray) -> np.ndarray:
        """
        Perform EKF prediction step with control input.

        Args:
            control_input: Control input [acceleration, steering_angle]

        Returns:
            estimated_state: Current state estimate [x, y, v, theta]
        """
        start_time = time.time() if self.enable_logging else None

        try:
            # EKF prediction step
            self.ekf.predict(control_input)

            # Get current state estimate
            estimated_state, _ = self.ekf.get_state_estimate()

            # Performance tracking
            if self.enable_logging and start_time:
                self.state_estimation_times.append(time.time() - start_time)

            return estimated_state

        except Exception as e:
            self.log_warning(f"Error in state estimation: {e}")
            # Return last known state as fallback
            estimated_state, _ = self.ekf.get_state_estimate()
            return estimated_state

    def update_measurement_imu(self, angular_velocity: float) -> None:
        """
        Update state estimate with IMU angular velocity measurement.

        Args:
            angular_velocity: Measured angular velocity [rad/s]
        """
        try:
            self.ekf.update_imu(angular_velocity)
        except Exception as e:
            self.log_warning(f"Error updating IMU measurement: {e}")

    def update_measurement_odometry(self, linear_velocity: float) -> None:
        """
        Update state estimate with odometry velocity measurement.

        Args:
            linear_velocity: Measured linear velocity [m/s]
        """
        try:
            self.ekf.update_odometry(linear_velocity)
        except Exception as e:
            self.log_warning(f"Error updating odometry measurement: {e}")

    def update_measurement_position(self, position: np.ndarray) -> None:
        """
        Update state estimate with position measurement (e.g., from localization).

        Args:
            position: Measured position [x, y]
        """
        try:
            self.ekf.update_position(position)
        except Exception as e:
            self.log_warning(f"Error updating position measurement: {e}")

    def compute_control(self,
                        reference_state: np.ndarray,
                        imu_angular_velocity: Optional[float] = None,
                        odom_velocity: Optional[float] = None,
                        position_measurement: Optional[np.ndarray] = None,
                        feedforward_control: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute LQG control input by combining state estimation and optimal control.

        This is the main method that implements the LQG control loop:
        1. Update state estimate with available sensor measurements
        2. Compute optimal control using estimated state and reference
        3. Use control input for next prediction step

        Args:
            reference_state: Reference state [x_ref, y_ref, v_ref, theta_ref]
            imu_angular_velocity: IMU angular velocity measurement [rad/s] (optional)
            odom_velocity: Odometry velocity measurement [m/s] (optional)
            position_measurement: Position measurement [x, y] (optional)
            feedforward_control: Feedforward control [a_ff, delta_ff] (optional)

        Returns:
            control_input: Optimal control input [acceleration, steering_angle]
            estimated_state: Current state estimate [x, y, v, theta]
        """
        total_start_time = time.time() if self.enable_logging else None

        try:
            # Update measurements if available
            if imu_angular_velocity is not None:
                self.update_measurement_imu(imu_angular_velocity)

            if odom_velocity is not None:
                self.update_measurement_odometry(odom_velocity)

            if position_measurement is not None:
                self.update_measurement_position(position_measurement)

            # Get current state estimate
            estimated_state, covariance = self.ekf.get_state_estimate()

            # Check estimation quality
            estimation_quality = self._assess_estimation_quality(covariance)

            # Compute optimal control using LQR with estimated state
            control_start_time = time.time() if self.enable_logging else None

            control_input = self.lqr.compute_control(
                current_state=estimated_state,
                reference_state=reference_state,
                feedforward_control=feedforward_control
            )

            if self.enable_logging and control_start_time:
                self.control_computation_times.append(time.time() - control_start_time)

            # Use computed control for next EKF prediction
            # (This will be called in the next cycle with the new control input)

            # Store data for analysis
            if self.enable_logging:
                self.estimated_states_history.append(estimated_state.copy())
                self.control_history.append(control_input.copy())
                self.reference_states_history.append(reference_state.copy())
                self.filter_health_history.append(self.ekf.is_filter_healthy())

                if total_start_time:
                    self.total_computation_times.append(time.time() - total_start_time)

            # Warn if estimation quality is poor
            if not estimation_quality:
                self.log_warning("Poor state estimation quality detected")

            return control_input, estimated_state

        except Exception as e:
            self.log_warning(f"Error in LQG control computation: {e}")
            # Return zero control as safe fallback
            estimated_state, _ = self.ekf.get_state_estimate()
            return np.zeros(2), estimated_state

    def _assess_estimation_quality(self, covariance: np.ndarray) -> bool:
        """
        Assess the quality of the current state estimation.

        Args:
            covariance: Current state covariance matrix

        Returns:
            quality_good: True if estimation quality is acceptable
        """
        # Check position uncertainty
        position_uncertainty = np.sqrt(covariance[0, 0] + covariance[1, 1])

        # Check if filter is healthy
        filter_healthy = self.ekf.is_filter_healthy()

        return (position_uncertainty < self.estimation_quality_threshold and
                filter_healthy)

    def get_state_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current state estimate and uncertainty.

        Returns:
            state_estimate: Current state estimate [x, y, v, theta]
            state_uncertainty: State estimation uncertainties (standard deviations)
        """
        state_estimate, covariance = self.ekf.get_state_estimate()
        state_uncertainty = self.ekf.get_state_uncertainty()
        return state_estimate, state_uncertainty

    def get_estimation_covariance(self) -> np.ndarray:
        """
        Get current state estimation covariance matrix.

        Returns:
            covariance: State covariance matrix (4x4)
        """
        _, covariance = self.ekf.get_state_estimate()
        return covariance

    def is_system_healthy(self) -> bool:
        """
        Check if the overall LQG system is healthy.

        Returns:
            healthy: True if both EKF and LQR are operating normally
        """
        ekf_healthy = self.ekf.is_filter_healthy()

        # Check LQR health (basic checks)
        lqr_healthy = True
        if hasattr(self.lqr, 'cached_K') and self.lqr.cached_K is not None:
            # Check if gain matrix is reasonable
            if np.any(np.isnan(self.lqr.cached_K)) or np.any(np.isinf(self.lqr.cached_K)):
                lqr_healthy = False

        return ekf_healthy and lqr_healthy

    def reset_filter(self, initial_state: Optional[np.ndarray] = None,
                     initial_covariance: Optional[np.ndarray] = None) -> None:
        """
        Reset the Kalman filter to initial conditions.

        Args:
            initial_state: New initial state estimate
            initial_covariance: New initial covariance matrix
        """
        self.ekf.reset_filter(initial_state, initial_covariance)

        # Clear LQG-specific history
        self.estimated_states_history.clear()
        self.control_history.clear()
        self.reference_states_history.clear()
        self.filter_health_history.clear()

        self.log_info("LQG Controller reset to initial conditions")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive LQG controller performance metrics.

        Returns:
            metrics: Dictionary of performance metrics
        """
        metrics = {}

        # EKF performance metrics
        ekf_metrics = self.ekf.get_performance_metrics()
        for key, value in ekf_metrics.items():
            metrics[f'ekf_{key}'] = value

        # LQR performance metrics
        if hasattr(self.lqr, 'get_performance_metrics'):
            lqr_metrics = self.lqr.get_performance_metrics()
            for key, value in lqr_metrics.items():
                metrics[f'lqr_{key}'] = value

        # LQG-specific metrics
        if self.control_computation_times:
            metrics['avg_control_computation_time'] = np.mean(self.control_computation_times)
            metrics['max_control_computation_time'] = np.max(self.control_computation_times)

        if self.state_estimation_times:
            metrics['avg_state_estimation_time'] = np.mean(self.state_estimation_times)
            metrics['max_state_estimation_time'] = np.max(self.state_estimation_times)

        if self.total_computation_times:
            metrics['avg_total_computation_time'] = np.mean(self.total_computation_times)
            metrics['max_total_computation_time'] = np.max(self.total_computation_times)

            # Check real-time performance (should be << dt)
            real_time_factor = np.max(self.total_computation_times) / self.dt
            metrics['real_time_factor'] = real_time_factor
            metrics['real_time_capable'] = real_time_factor < 0.1  # Use <10% of available time

        if self.filter_health_history:
            metrics['filter_health_percentage'] = np.mean(self.filter_health_history) * 100

        # Current system status
        metrics['system_healthy'] = self.is_system_healthy()

        # Estimation quality
        state_estimate, state_uncertainty = self.get_state_estimate()
        metrics['current_position_uncertainty'] = np.linalg.norm(state_uncertainty[:2])
        metrics['current_velocity_uncertainty'] = state_uncertainty[2]
        metrics['current_heading_uncertainty'] = state_uncertainty[3]

        return metrics

    def get_control_statistics(self) -> Dict[str, float]:
        """
        Get statistics about the control inputs generated.

        Returns:
            stats: Dictionary of control statistics
        """
        stats = {}

        if self.control_history:
            controls = np.array(self.control_history)

            # Acceleration statistics
            stats['avg_acceleration'] = np.mean(controls[:, 0])
            stats['std_acceleration'] = np.std(controls[:, 0])
            stats['max_acceleration'] = np.max(np.abs(controls[:, 0]))

            # Steering statistics
            stats['avg_steering'] = np.mean(controls[:, 1])
            stats['std_steering'] = np.std(controls[:, 1])
            stats['max_steering'] = np.max(np.abs(controls[:, 1]))

            # Control smoothness (rate of change)
            if len(controls) > 1:
                accel_rate = np.diff(controls[:, 0]) / self.dt
                steering_rate = np.diff(controls[:, 1]) / self.dt

                stats['avg_acceleration_rate'] = np.mean(np.abs(accel_rate))
                stats['max_acceleration_rate'] = np.max(np.abs(accel_rate))
                stats['avg_steering_rate'] = np.mean(np.abs(steering_rate))
                stats['max_steering_rate'] = np.max(np.abs(steering_rate))

        return stats
