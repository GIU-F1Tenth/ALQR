#!/usr/bin/env python3

"""
Unit tests for LQR Controller

Tests the core functionality of the LQR controller implementation.

Author: Mohammed Azab <mohammed@azab.io>
License: MIT
"""

from lqr_controller.lqr_controller import LQRController
from lqr_controller.kinematic_bicycle_model import KinematicBicycleModel
import unittest
import numpy as np
import sys
import os

# Add the package to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestKinematicBicycleModel(unittest.TestCase):
    """Test cases for KinematicBicycleModel class."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = KinematicBicycleModel(wheelbase=0.33, dt=0.05)

    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.L, 0.33)
        self.assertEqual(self.model.dt, 0.05)

    def test_dynamics(self):
        """Test dynamics computation."""
        state = np.array([0.0, 0.0, 1.0, 0.0])  # [x, y, v, theta]
        control = np.array([0.0, 0.0])          # [a, delta]

        next_state = self.model.dynamics(state, control)

        # With zero acceleration and steering, vehicle should move forward
        self.assertAlmostEqual(next_state[0], 0.05, places=3)  # x += v*dt*cos(0)
        self.assertAlmostEqual(next_state[1], 0.0, places=3)   # y += v*dt*sin(0)
        self.assertAlmostEqual(next_state[2], 1.0, places=3)   # v unchanged
        self.assertAlmostEqual(next_state[3], 0.0, places=3)   # theta unchanged

    def test_linearization(self):
        """Test model linearization."""
        state = np.array([0.0, 0.0, 1.0, 0.0])
        control = np.array([0.0, 0.0])

        A, B = self.model.linearize(state, control)

        # Check matrix dimensions
        self.assertEqual(A.shape, (4, 4))
        self.assertEqual(B.shape, (4, 2))

        # Check some expected values
        self.assertAlmostEqual(A[0, 2], self.model.dt, places=3)  # dx/dv term
        self.assertAlmostEqual(B[2, 0], self.model.dt, places=3)  # dv/da term

    def test_state_validation(self):
        """Test state validation."""
        # Valid state
        valid_state = np.array([0.0, 0.0, 1.0, 0.0])
        self.assertTrue(self.model.validate_state(valid_state))

        # Invalid state (wrong size)
        invalid_state = np.array([0.0, 0.0, 1.0])
        self.assertFalse(self.model.validate_state(invalid_state))

        # Invalid state (NaN)
        nan_state = np.array([0.0, 0.0, np.nan, 0.0])
        self.assertFalse(self.model.validate_state(nan_state))

        # Invalid state (too high velocity)
        high_vel_state = np.array([0.0, 0.0, 20.0, 0.0])
        self.assertFalse(self.model.validate_state(high_vel_state))

    def test_control_validation(self):
        """Test control validation."""
        # Valid control
        valid_control = np.array([1.0, 0.1])
        self.assertTrue(self.model.validate_control(valid_control))

        # Invalid control (wrong size)
        invalid_control = np.array([1.0])
        self.assertFalse(self.model.validate_control(invalid_control))

        # Invalid control (too high acceleration)
        high_accel_control = np.array([10.0, 0.1])
        self.assertFalse(self.model.validate_control(high_accel_control))

        # Invalid control (too high steering)
        high_steer_control = np.array([1.0, 1.0])
        self.assertFalse(self.model.validate_control(high_steer_control))

    def test_angle_normalization(self):
        """Test angle normalization."""
        # Test various angles
        self.assertAlmostEqual(self.model.normalize_angle(0.0), 0.0)
        self.assertAlmostEqual(self.model.normalize_angle(np.pi), np.pi)
        self.assertAlmostEqual(self.model.normalize_angle(-np.pi), -np.pi)
        self.assertAlmostEqual(self.model.normalize_angle(2 * np.pi), 0.0, places=5)
        self.assertAlmostEqual(self.model.normalize_angle(3 * np.pi), np.pi, places=5)


class TestLQRController(unittest.TestCase):
    """Test cases for LQRController class."""

    def setUp(self):
        """Set up test fixtures."""
        self.controller = LQRController(
            wheelbase=0.33,
            dt=0.05,
            enable_logging=False  # Disable logging for tests
        )

    def test_initialization(self):
        """Test controller initialization."""
        self.assertEqual(self.controller.model.L, 0.33)
        self.assertEqual(self.controller.dt, 0.05)
        self.assertEqual(self.controller.Q.shape, (4, 4))
        self.assertEqual(self.controller.R.shape, (2, 2))

    def test_dare_solution(self):
        """Test DARE solution."""
        A = np.eye(4)
        B = np.random.rand(4, 2)
        Q = np.eye(4)
        R = np.eye(2)

        P = self.controller.solve_dare(A, B, Q, R)

        # P should be positive semi-definite
        self.assertEqual(P.shape, (4, 4))
        eigenvals = np.linalg.eigvals(P)
        self.assertTrue(np.all(eigenvals >= -1e-10))  # Allow small numerical errors

    def test_lqr_gain_computation(self):
        """Test LQR gain computation."""
        A = np.eye(4) + 0.1 * np.random.rand(4, 4)
        B = np.random.rand(4, 2)

        K = self.controller.compute_lqr_gain(A, B)

        # K should have correct dimensions
        self.assertEqual(K.shape, (2, 4))
        self.assertTrue(np.all(np.isfinite(K)))

    def test_control_computation(self):
        """Test control computation."""
        current_state = np.array([0.0, 0.0, 1.0, 0.0])
        reference_state = np.array([1.0, 0.0, 1.0, 0.0])

        control = self.controller.compute_control(current_state, reference_state)

        # Control should have correct dimensions
        self.assertEqual(control.shape, (2,))
        self.assertTrue(np.all(np.isfinite(control)))

        # Control should be within bounds
        self.assertLessEqual(abs(control[0]), self.controller.max_acceleration)
        self.assertLessEqual(abs(control[1]), self.controller.max_steering)

    def test_control_constraints(self):
        """Test control constraint application."""
        # Test large control inputs
        large_control = np.array([10.0, 2.0])
        constrained = self.controller.apply_control_constraints(large_control)

        self.assertLessEqual(abs(constrained[0]), self.controller.max_acceleration)
        self.assertLessEqual(abs(constrained[1]), self.controller.max_steering)

    def test_cost_matrix_update(self):
        """Test cost matrix updating."""
        new_Q = 2.0 * np.eye(4)
        new_R = 3.0 * np.eye(2)

        self.controller.update_cost_matrices(new_Q, new_R)

        np.testing.assert_array_equal(self.controller.Q, new_Q)
        np.testing.assert_array_equal(self.controller.R, new_R)
        self.assertIsNone(self.controller.cached_K)  # Cache should be invalidated

    def test_performance_metrics(self):
        """Test performance metrics collection."""
        # Run a few control computations
        current_state = np.array([0.0, 0.0, 1.0, 0.0])
        reference_state = np.array([1.0, 0.0, 1.0, 0.0])

        for _ in range(5):
            self.controller.compute_control(current_state, reference_state)

        metrics = self.controller.get_performance_metrics()

        if metrics:  # Only check if logging is enabled
            self.assertIn('avg_solve_time', metrics)
            self.assertIn('total_control_calls', metrics)
            self.assertEqual(metrics['total_control_calls'], 5)

    def test_controller_info(self):
        """Test controller information retrieval."""
        info = self.controller.get_controller_info()

        self.assertEqual(info['controller_type'], 'LQR')
        self.assertIn('model_info', info)
        self.assertIn('Q_matrix', info)
        self.assertIn('R_matrix', info)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = KinematicBicycleModel(wheelbase=0.33, dt=0.05)
        self.controller = LQRController(
            wheelbase=0.33,
            dt=0.05,
            enable_logging=False
        )

    def test_trajectory_tracking(self):
        """Test trajectory tracking scenario."""
        # Create a simple trajectory (straight line)
        trajectory_length = 10
        current_state = np.array([0.0, 0.0, 1.0, 0.0])

        for i in range(trajectory_length):
            # Reference state ahead of current
            reference_state = np.array([i * 0.1, 0.0, 1.0, 0.0])

            # Compute control
            control = self.controller.compute_control(current_state, reference_state)

            # Update state using model
            current_state = self.model.dynamics(current_state, control)

            # Verify state remains reasonable
            self.assertTrue(self.model.validate_state(current_state))
            self.assertTrue(self.model.validate_control(control))

    def test_circular_trajectory_tracking(self):
        """Test tracking a circular trajectory."""
        radius = 2.0
        angular_velocity = 0.5
        dt = 0.05

        current_state = np.array([radius, 0.0, 1.0, 0.0])

        for i in range(20):  # Simulate for 1 second
            t = i * dt

            # Circular reference trajectory
            ref_x = radius * np.cos(angular_velocity * t)
            ref_y = radius * np.sin(angular_velocity * t)
            ref_theta = angular_velocity * t + np.pi / 2
            ref_v = 1.0

            reference_state = np.array([ref_x, ref_y, ref_v, ref_theta])

            # Compute control
            control = self.controller.compute_control(current_state, reference_state)

            # Update state
            current_state = self.model.dynamics(current_state, control)

            # Verify state remains reasonable
            self.assertTrue(self.model.validate_state(current_state))


if __name__ == '__main__':
    unittest.main()
