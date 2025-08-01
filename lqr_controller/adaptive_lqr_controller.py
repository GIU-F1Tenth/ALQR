#!/usr/bin/env python3

"""
Fully Adaptive LQR Controller Implementation

This module implements a fully adaptive Linear Quadratic Regulator (LQR) controller
that dynamically adjusts its parameters based on:
- Vehicle velocity and acceleration
- Trajectory curvature and complexity
- Real-time performance metrics
- Environmental conditions

Author: Mohammed Azab <mohammed@azab.io>
License: MIT
Version: 3.0.0 (Fully Adaptive)
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import scipy.linalg
import time
from dataclasses import dataclass
from .kinematic_bicycle_model import KinematicBicycleModel


@dataclass
class AdaptiveParams:
    """Data class for adaptive parameter configuration."""
    # Base weights
    base_position_weight: float = 5.0
    base_velocity_weight: float = 1.0
    base_heading_weight: float = 6.0
    base_acceleration_weight: float = 0.3
    base_steering_weight: float = 4.0
    
    # Adaptation factors
    velocity_adaptation_factor: float = 0.5
    curvature_adaptation_factor: float = 2.0
    error_adaptation_factor: float = 1.5
    performance_adaptation_factor: float = 0.3
    
    # Adaptation limits
    min_weight_multiplier: float = 0.2
    max_weight_multiplier: float = 5.0
    adaptation_rate: float = 0.1
    
    # Performance thresholds
    good_performance_threshold: float = 0.1
    poor_performance_threshold: float = 0.5
    
    # Environmental adaptation
    high_speed_threshold: float = 8.0
    low_speed_threshold: float = 2.0
    high_curvature_threshold: float = 1.0
    moderate_curvature_threshold: float = 0.3


class PerformanceMonitor:
    """Monitor and analyze controller performance for adaptive tuning."""
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        
        # Performance metrics history
        self.lateral_errors = []
        self.heading_errors = []
        self.velocity_errors = []
        self.control_efforts = []
        self.solve_times = []
        
        # Sliding window statistics
        self.avg_lateral_error = 0.0
        self.avg_heading_error = 0.0
        self.avg_velocity_error = 0.0
        self.avg_control_effort = 0.0
        self.tracking_quality = 1.0
        
        # Performance trends
        self.error_trend = 0.0  # Positive = getting worse, Negative = improving
        self.stability_metric = 1.0
        
    def update(self, lateral_error: float, heading_error: float, 
               velocity_error: float, control: np.ndarray, solve_time: float):
        """Update performance metrics with new data."""
        
        # Add new measurements
        self.lateral_errors.append(abs(lateral_error))
        self.heading_errors.append(abs(heading_error))
        self.velocity_errors.append(abs(velocity_error))
        self.control_efforts.append(np.linalg.norm(control))
        self.solve_times.append(solve_time)
        
        # Maintain sliding window
        for metric_list in [self.lateral_errors, self.heading_errors, 
                           self.velocity_errors, self.control_efforts, self.solve_times]:
            if len(metric_list) > self.history_size:
                metric_list.pop(0)
        
        # Update statistics
        self._update_statistics()
        self._update_trends()
    
    def _update_statistics(self):
        """Update sliding window statistics."""
        if len(self.lateral_errors) > 5:
            self.avg_lateral_error = np.mean(self.lateral_errors[-20:])
            self.avg_heading_error = np.mean(self.heading_errors[-20:])
            self.avg_velocity_error = np.mean(self.velocity_errors[-20:])
            self.avg_control_effort = np.mean(self.control_efforts[-20:])
            
            # Compute overall tracking quality (lower is better)
            self.tracking_quality = (self.avg_lateral_error + 
                                   0.5 * self.avg_heading_error + 
                                   0.2 * self.avg_velocity_error)
    
    def _update_trends(self):
        """Update performance trend analysis."""
        if len(self.lateral_errors) > 20:
            # Compute error trend over recent history
            recent_errors = self.lateral_errors[-20:]
            old_errors = self.lateral_errors[-40:-20] if len(self.lateral_errors) >= 40 else recent_errors
            
            recent_avg = np.mean(recent_errors)
            old_avg = np.mean(old_errors)
            
            self.error_trend = (recent_avg - old_avg) / max(old_avg, 0.01)
            
            # Compute stability metric (lower variance = more stable)
            self.stability_metric = 1.0 / (1.0 + np.var(recent_errors))
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return {
            'tracking_quality': self.tracking_quality,
            'error_trend': self.error_trend,
            'stability_metric': self.stability_metric,
            'avg_lateral_error': self.avg_lateral_error,
            'avg_heading_error': self.avg_heading_error,
            'avg_control_effort': self.avg_control_effort
        }


class TrajectoryAnalyzer:
    """Analyze trajectory characteristics for adaptive parameter tuning."""
    
    def __init__(self):
        self.complexity_history = []
        self.curvature_history = []
        self.velocity_profile_history = []
        
    def analyze_trajectory_segment(self, trajectory: List[Dict], 
                                 current_index: int, lookahead_points: int = 10) -> Dict[str, float]:
        """Analyze upcoming trajectory segment characteristics."""
        
        if not trajectory or len(trajectory) < 3:
            return {
                'curvature': 0.0,
                'complexity': 0.0,
                'velocity_variation': 0.0,
                'acceleration_demand': 0.0
            }
        
        end_index = min(current_index + lookahead_points, len(trajectory))
        segment = trajectory[current_index:end_index]
        
        # Analyze curvature
        curvatures = self._compute_curvatures(segment)
        max_curvature = max(curvatures) if curvatures else 0.0
        avg_curvature = np.mean(curvatures) if curvatures else 0.0
        
        # Analyze velocity profile
        velocities = [point['v'] for point in segment]
        velocity_variation = np.std(velocities) if len(velocities) > 1 else 0.0
        
        # Analyze acceleration demands
        accelerations = []
        for i in range(1, len(segment)):
            dt = 0.1  # Assume constant time step
            dv = segment[i]['v'] - segment[i-1]['v']
            accelerations.append(abs(dv / dt))
        
        acceleration_demand = max(accelerations) if accelerations else 0.0
        
        # Compute trajectory complexity
        complexity = self._compute_trajectory_complexity(segment)
        
        return {
            'curvature': max_curvature,
            'avg_curvature': avg_curvature,
            'complexity': complexity,
            'velocity_variation': velocity_variation,
            'acceleration_demand': acceleration_demand
        }
    
    def _compute_curvatures(self, segment: List[Dict]) -> List[float]:
        """Compute curvature for trajectory segment."""
        curvatures = []
        
        for i in range(1, len(segment) - 1):
            p1 = np.array([segment[i-1]['x'], segment[i-1]['y']])
            p2 = np.array([segment[i]['x'], segment[i]['y']])
            p3 = np.array([segment[i+1]['x'], segment[i+1]['y']])
            
            # Three-point curvature formula
            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p3 - p2)
            c = np.linalg.norm(p3 - p1)
            
            if a > 1e-6 and b > 1e-6 and c > 1e-6:
                area = 0.5 * abs(np.cross(p2 - p1, p3 - p1))
                curvature = 2 * area / (a * b * c)
                curvatures.append(curvature)
        
        return curvatures
    
    def _compute_trajectory_complexity(self, segment: List[Dict]) -> float:
        """Compute trajectory complexity metric."""
        if len(segment) < 3:
            return 0.0
        
        # Compute heading changes
        heading_changes = []
        for i in range(1, len(segment)):
            dtheta = abs(segment[i]['theta'] - segment[i-1]['theta'])
            # Handle angle wrapping
            if dtheta > np.pi:
                dtheta = 2 * np.pi - dtheta
            heading_changes.append(dtheta)
        
        # Complexity is based on total heading variation
        complexity = sum(heading_changes) / len(heading_changes) if heading_changes else 0.0
        
        return complexity


class AdaptiveLQRController:
    """
    Fully Adaptive Linear Quadratic Regulator (LQR) controller.

    This controller automatically adapts its parameters based on:
    1. Vehicle state (velocity, acceleration)
    2. Trajectory characteristics (curvature, complexity)
    3. Real-time performance metrics
    4. Environmental conditions
    """

    def __init__(self,
                 wheelbase: float = 0.33,
                 dt: float = 0.05,
                 adaptive_params: Optional[AdaptiveParams] = None,
                 max_acceleration: float = 5.0,
                 max_steering: float = 0.5,
                 enable_logging: bool = True,
                 logger: Optional[Any] = None):
        """
        Initialize the adaptive LQR controller.

        Args:
            wheelbase: Vehicle wheelbase [m]
            dt: Control time step [s]
            adaptive_params: Adaptive parameter configuration
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

        # Adaptive parameters
        self.adaptive_params = adaptive_params or AdaptiveParams()
        
        # Initialize performance monitor and trajectory analyzer
        self.performance_monitor = PerformanceMonitor()
        self.trajectory_analyzer = TrajectoryAnalyzer()
        
        # Current adaptive weights
        self.current_weights = {
            'position': self.adaptive_params.base_position_weight,
            'velocity': self.adaptive_params.base_velocity_weight,
            'heading': self.adaptive_params.base_heading_weight,
            'acceleration': self.adaptive_params.base_acceleration_weight,
            'steering': self.adaptive_params.base_steering_weight
        }
        
        # Adaptation state
        self.last_adaptation_time = time.time()
        self.adaptation_interval = 0.1  # Adapt every 100ms
        
        # Current cost matrices (will be updated adaptively)
        self.Q = self._build_Q_matrix()
        self.R = self._build_R_matrix()
        
        # Cache for efficiency
        self.last_linearization_state = None
        self.last_linearization_control = None
        self.cached_K = None
        self.cache_tolerance = 1e-3
        
        # Performance tracking
        self.solve_times = []
        self.control_history = []
        self.adaptation_history = []

        self.log_info("Fully Adaptive LQR Controller initialized")

    def log_info(self, message: str):
        """Log information message."""
        if self.logger:
            self.logger.info(message)
        elif self.enable_logging:
            print(f"[Adaptive LQR] {message}")

    def _build_Q_matrix(self) -> np.ndarray:
        """Build state cost matrix from current weights."""
        return np.diag([
            self.current_weights['position'],  # x position
            self.current_weights['position'],  # y position
            self.current_weights['velocity'],  # velocity
            self.current_weights['heading']    # heading
        ])

    def _build_R_matrix(self) -> np.ndarray:
        """Build control cost matrix from current weights."""
        return np.diag([
            self.current_weights['acceleration'],  # acceleration
            self.current_weights['steering']       # steering
        ])

    def adapt_parameters(self, current_state: np.ndarray, reference_state: np.ndarray,
                        trajectory: List[Dict], current_index: int) -> bool:
        """
        Adapt controller parameters based on current conditions.
        
        Returns:
            True if parameters were updated, False otherwise
        """
        current_time = time.time()
        
        # Check if it's time to adapt
        if current_time - self.last_adaptation_time < self.adaptation_interval:
            return False
        
        self.last_adaptation_time = current_time
        
        # Analyze current trajectory segment
        trajectory_info = self.trajectory_analyzer.analyze_trajectory_segment(
            trajectory, current_index
        )
        
        # Get performance metrics
        performance_metrics = self.performance_monitor.get_performance_metrics()
        
        # Compute adaptation factors
        velocity_factor = self._compute_velocity_adaptation_factor(current_state[2])
        curvature_factor = self._compute_curvature_adaptation_factor(trajectory_info['curvature'])
        performance_factor = self._compute_performance_adaptation_factor(performance_metrics)
        complexity_factor = self._compute_complexity_adaptation_factor(trajectory_info['complexity'])
        
        # Store old weights for comparison
        old_weights = self.current_weights.copy()
        
        # Adapt each weight based on multiple factors
        self._adapt_position_weights(velocity_factor, curvature_factor, performance_factor)
        self._adapt_velocity_weights(velocity_factor, performance_factor, trajectory_info)
        self._adapt_heading_weights(curvature_factor, complexity_factor, performance_factor)
        self._adapt_control_weights(curvature_factor, performance_factor, trajectory_info)
        
        # Update cost matrices
        self.Q = self._build_Q_matrix()
        self.R = self._build_R_matrix()
        
        # Invalidate cache to force recomputation
        self.cached_K = None
        
        # Log adaptation if significant change occurred
        weight_change = max(
            abs(self.current_weights[key] - old_weights[key]) / old_weights[key]
            for key in old_weights.keys()
        )
        
        if weight_change > 0.05 and self.enable_logging:  # 5% threshold
            self.log_info(f"Adapted weights: {self.current_weights}")
            self.log_info(f"Factors - V:{velocity_factor:.2f}, C:{curvature_factor:.2f}, "
                         f"P:{performance_factor:.2f}, Cx:{complexity_factor:.2f}")
        
        # Store adaptation history
        self.adaptation_history.append({
            'time': current_time,
            'weights': self.current_weights.copy(),
            'factors': {
                'velocity': velocity_factor,
                'curvature': curvature_factor,
                'performance': performance_factor,
                'complexity': complexity_factor
            }
        })
        
        return True

    def _compute_velocity_adaptation_factor(self, velocity: float) -> float:
        """Compute adaptation factor based on vehicle velocity."""
        if velocity > self.adaptive_params.high_speed_threshold:
            # High speed: increase position tracking, reduce aggressiveness
            return 1.0 + self.adaptive_params.velocity_adaptation_factor
        elif velocity < self.adaptive_params.low_speed_threshold:
            # Low speed: more aggressive control
            return 1.0 - 0.3 * self.adaptive_params.velocity_adaptation_factor
        else:
            # Normal speed
            return 1.0

    def _compute_curvature_adaptation_factor(self, curvature: float) -> float:
        """Compute adaptation factor based on trajectory curvature."""
        if curvature > self.adaptive_params.high_curvature_threshold:
            # High curvature: prioritize heading tracking
            return 1.0 + self.adaptive_params.curvature_adaptation_factor
        elif curvature > self.adaptive_params.moderate_curvature_threshold:
            # Moderate curvature
            return 1.0 + 0.5 * self.adaptive_params.curvature_adaptation_factor
        else:
            # Low curvature: can be more aggressive
            return 1.0

    def _compute_performance_adaptation_factor(self, metrics: Dict[str, float]) -> float:
        """Compute adaptation factor based on performance metrics."""
        tracking_quality = metrics.get('tracking_quality', 0.0)
        error_trend = metrics.get('error_trend', 0.0)
        
        if tracking_quality > self.adaptive_params.poor_performance_threshold:
            # Poor performance: be more conservative
            return 1.0 + self.adaptive_params.performance_adaptation_factor
        elif tracking_quality < self.adaptive_params.good_performance_threshold:
            # Good performance: can be more aggressive
            return 1.0 - 0.5 * self.adaptive_params.performance_adaptation_factor
        else:
            # Average performance, consider trend
            if error_trend > 0.1:  # Getting worse
                return 1.0 + 0.3 * self.adaptive_params.performance_adaptation_factor
            elif error_trend < -0.1:  # Improving
                return 1.0 - 0.2 * self.adaptive_params.performance_adaptation_factor
            else:
                return 1.0

    def _compute_complexity_adaptation_factor(self, complexity: float) -> float:
        """Compute adaptation factor based on trajectory complexity."""
        if complexity > 1.0:  # High complexity
            return 1.0 + 0.5 * self.adaptive_params.curvature_adaptation_factor
        elif complexity > 0.5:  # Moderate complexity
            return 1.0 + 0.2 * self.adaptive_params.curvature_adaptation_factor
        else:
            return 1.0

    def _adapt_position_weights(self, velocity_factor: float, curvature_factor: float, 
                               performance_factor: float):
        """Adapt position tracking weights."""
        base_weight = self.adaptive_params.base_position_weight
        
        # Increase position weight for high speeds and poor performance
        adaptation = velocity_factor * performance_factor
        
        new_weight = base_weight * adaptation
        new_weight = np.clip(new_weight, 
                           base_weight * self.adaptive_params.min_weight_multiplier,
                           base_weight * self.adaptive_params.max_weight_multiplier)
        
        # Apply adaptation rate limiting
        current_weight = self.current_weights['position']
        self.current_weights['position'] = (
            (1 - self.adaptive_params.adaptation_rate) * current_weight +
            self.adaptive_params.adaptation_rate * new_weight
        )

    def _adapt_velocity_weights(self, velocity_factor: float, performance_factor: float,
                               trajectory_info: Dict[str, float]):
        """Adapt velocity tracking weights."""
        base_weight = self.adaptive_params.base_velocity_weight
        
        # Increase velocity weight for high acceleration demands
        acceleration_demand = trajectory_info.get('acceleration_demand', 0.0)
        acceleration_factor = 1.0 + 0.5 * min(acceleration_demand / 5.0, 1.0)
        
        adaptation = velocity_factor * performance_factor * acceleration_factor
        
        new_weight = base_weight * adaptation
        new_weight = np.clip(new_weight,
                           base_weight * self.adaptive_params.min_weight_multiplier,
                           base_weight * self.adaptive_params.max_weight_multiplier)
        
        current_weight = self.current_weights['velocity']
        self.current_weights['velocity'] = (
            (1 - self.adaptive_params.adaptation_rate) * current_weight +
            self.adaptive_params.adaptation_rate * new_weight
        )

    def _adapt_heading_weights(self, curvature_factor: float, complexity_factor: float,
                              performance_factor: float):
        """Adapt heading tracking weights."""
        base_weight = self.adaptive_params.base_heading_weight
        
        # Increase heading weight for curves and complex trajectories
        adaptation = curvature_factor * complexity_factor * performance_factor
        
        new_weight = base_weight * adaptation
        new_weight = np.clip(new_weight,
                           base_weight * self.adaptive_params.min_weight_multiplier,
                           base_weight * self.adaptive_params.max_weight_multiplier)
        
        current_weight = self.current_weights['heading']
        self.current_weights['heading'] = (
            (1 - self.adaptive_params.adaptation_rate) * current_weight +
            self.adaptive_params.adaptation_rate * new_weight
        )

    def _adapt_control_weights(self, curvature_factor: float, performance_factor: float,
                              trajectory_info: Dict[str, float]):
        """Adapt control effort weights."""
        # Adapt acceleration weight
        base_accel_weight = self.adaptive_params.base_acceleration_weight
        accel_adaptation = performance_factor
        
        new_accel_weight = base_accel_weight * accel_adaptation
        new_accel_weight = np.clip(new_accel_weight,
                                 base_accel_weight * self.adaptive_params.min_weight_multiplier,
                                 base_accel_weight * self.adaptive_params.max_weight_multiplier)
        
        current_accel_weight = self.current_weights['acceleration']
        self.current_weights['acceleration'] = (
            (1 - self.adaptive_params.adaptation_rate) * current_accel_weight +
            self.adaptive_params.adaptation_rate * new_accel_weight
        )
        
        # Adapt steering weight
        base_steering_weight = self.adaptive_params.base_steering_weight
        
        # Increase steering weight for high curvature (smoother steering)
        steering_adaptation = curvature_factor * performance_factor
        
        new_steering_weight = base_steering_weight * steering_adaptation
        new_steering_weight = np.clip(new_steering_weight,
                                    base_steering_weight * self.adaptive_params.min_weight_multiplier,
                                    base_steering_weight * self.adaptive_params.max_weight_multiplier)
        
        current_steering_weight = self.current_weights['steering']
        self.current_weights['steering'] = (
            (1 - self.adaptive_params.adaptation_rate) * current_steering_weight +
            self.adaptive_params.adaptation_rate * new_steering_weight
        )

    def compute_control(self, current_state: np.ndarray, reference_state: np.ndarray,
                       feedforward_control: Optional[np.ndarray] = None,
                       trajectory: Optional[List[Dict]] = None,
                       current_index: int = 0) -> np.ndarray:
        """
        Compute adaptive LQR control input for trajectory tracking.

        Args:
            current_state: Current vehicle state [x, y, v, theta]
            reference_state: Reference state [x_ref, y_ref, v_ref, theta_ref]
            feedforward_control: Optional feedforward control [a_ff, delta_ff]
            trajectory: Full trajectory for adaptation (optional)
            current_index: Current index in trajectory

        Returns:
            control: Optimal control input [acceleration, steering_angle]
        """
        start_time = time.time() if self.enable_logging else None

        try:
            # Validate inputs
            if not self.model.validate_state(current_state):
                self.log_info("Invalid current state, using zero control")
                return np.zeros(2)

            if not self.model.validate_state(reference_state):
                self.log_info("Invalid reference state, using zero control")
                return np.zeros(2)

            # Adaptive parameter tuning
            if trajectory is not None:
                self.adapt_parameters(current_state, reference_state, trajectory, current_index)

            # Normalize heading angles
            current_state_norm = current_state.copy()
            reference_state_norm = reference_state.copy()
            current_state_norm[3] = self.model.normalize_angle(current_state_norm[3])
            reference_state_norm[3] = self.model.normalize_angle(reference_state_norm[3])

            # Compute state error
            state_error = current_state_norm - reference_state_norm
            state_error[3] = self.model.normalize_angle(state_error[3])

            # Set up linearization point
            linearization_state = reference_state_norm
            linearization_control = feedforward_control if feedforward_control is not None else np.zeros(2)

            # Check if we need to recompute LQR gain
            if self.should_recompute_gain(linearization_state, linearization_control):
                # Linearize model
                A, B = self.model.linearize(linearization_state, linearization_control)

                # Compute LQR gain with current adaptive weights
                K = self.compute_lqr_gain(A, B)

                # Cache the results
                self.cached_K = K
                self.last_linearization_state = linearization_state.copy()
                self.last_linearization_control = linearization_control.copy()
            else:
                K = self.cached_K

            # Compute feedback control
            feedback_control = -K @ state_error

            # Add feedforward control if provided
            if feedforward_control is not None:
                control = feedforward_control + feedback_control
            else:
                control = feedback_control

            # Apply control constraints
            control = self.apply_control_constraints(control)

            # Update performance monitor
            solve_time = time.time() - start_time if start_time else 0.0
            self.performance_monitor.update(
                lateral_error=np.sqrt(state_error[0]**2 + state_error[1]**2),
                heading_error=abs(state_error[3]),
                velocity_error=abs(state_error[2]),
                control=control,
                solve_time=solve_time
            )

            # Store for performance analysis
            if self.enable_logging:
                self.solve_times.append(solve_time)
                self.control_history.append(control.copy())

                # Keep history bounded
                max_history = 1000
                if len(self.solve_times) > max_history:
                    self.solve_times = self.solve_times[-max_history:]
                    self.control_history = self.control_history[-max_history:]

            return control

        except Exception as e:
            self.log_info(f"Failed to compute adaptive LQR control: {e}")
            return np.zeros(2)

    def should_recompute_gain(self, current_state: np.ndarray, current_control: np.ndarray) -> bool:
        """Check if LQR gain should be recomputed."""
        if (self.last_linearization_state is None or
            self.last_linearization_control is None or
                self.cached_K is None):
            return True

        # Check if state or control have changed significantly
        state_diff = np.linalg.norm(current_state - self.last_linearization_state)
        control_diff = np.linalg.norm(current_control - self.last_linearization_control)

        return (state_diff > self.cache_tolerance or control_diff > self.cache_tolerance)

    def compute_lqr_gain(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute the LQR feedback gain matrix using current adaptive weights."""
        try:
            # Solve DARE with current adaptive Q and R matrices
            P = scipy.linalg.solve_discrete_are(A, B, self.Q, self.R)

            # Compute LQR gain
            BtPB = B.T @ P @ B
            BtPA = B.T @ P @ A

            # Add regularization
            reg_term = 1e-6 * np.eye(BtPB.shape[0])
            K = np.linalg.solve(self.R + BtPB + reg_term, BtPA)

            return K

        except Exception as e:
            self.log_info(f"Failed to compute adaptive LQR gain: {e}")
            return np.zeros((2, 4))

    def apply_control_constraints(self, control: np.ndarray) -> np.ndarray:
        """Apply control input constraints."""
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

    def get_adaptation_info(self) -> Dict[str, Any]:
        """Get current adaptation information."""
        return {
            'current_weights': self.current_weights.copy(),
            'performance_metrics': self.performance_monitor.get_performance_metrics(),
            'Q_matrix': self.Q.tolist(),
            'R_matrix': self.R.tolist(),
            'adaptation_count': len(self.adaptation_history)
        }

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get controller performance metrics."""
        base_metrics = {}
        
        if self.solve_times:
            base_metrics.update({
                'avg_solve_time': np.mean(self.solve_times),
                'max_solve_time': np.max(self.solve_times),
                'min_solve_time': np.min(self.solve_times),
                'total_control_calls': len(self.solve_times)
            })
        
        # Add adaptive performance metrics
        base_metrics.update(self.performance_monitor.get_performance_metrics())
        
        return base_metrics

    def reset_adaptation(self):
        """Reset adaptive parameters to base values."""
        self.current_weights = {
            'position': self.adaptive_params.base_position_weight,
            'velocity': self.adaptive_params.base_velocity_weight,
            'heading': self.adaptive_params.base_heading_weight,
            'acceleration': self.adaptive_params.base_acceleration_weight,
            'steering': self.adaptive_params.base_steering_weight
        }
        
        self.Q = self._build_Q_matrix()
        self.R = self._build_R_matrix()
        self.cached_K = None
        
        self.performance_monitor = PerformanceMonitor()
        self.adaptation_history.clear()
        
        self.log_info("Adaptive parameters reset to base values")