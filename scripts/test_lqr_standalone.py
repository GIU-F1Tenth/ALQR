#!/usr/bin/env python3

"""
Test script for LQR Controller

Simple standalone test to verify LQR controller functionality without ROS2.

Author: Mohammed Azab <mohammed@azab.io>
License: MIT
"""

from lqr_controller.lqr_controller import LQRController
from lqr_controller.kinematic_bicycle_model import KinematicBicycleModel
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_straight_line_tracking():
    """Test LQR controller tracking a straight line."""

    print("Testing straight line tracking...")

    # Initialize model and controller
    model = KinematicBicycleModel(wheelbase=0.33, dt=0.05)
    controller = LQRController(
        wheelbase=0.33,
        dt=0.05,
        enable_logging=False
    )

    # Simulation parameters
    sim_time = 5.0
    dt = 0.05
    steps = int(sim_time / dt)

    # Initialize state
    current_state = np.array([0.0, 0.0, 1.0, 0.0])  # [x, y, v, theta]

    # Storage for results
    states = [current_state.copy()]
    controls = []

    # Run simulation
    for i in range(steps):
        # Reference state: move forward at constant velocity
        ref_x = (i + 10) * dt * 1.0  # 10 steps ahead
        reference_state = np.array([ref_x, 0.0, 1.0, 0.0])

        # Compute control
        control = controller.compute_control(current_state, reference_state)
        controls.append(control.copy())

        # Update state
        current_state = model.dynamics(current_state, control)
        states.append(current_state.copy())

    # Convert to arrays
    states = np.array(states)
    controls = np.array(controls)

    # Print results
    print(f"Final position: ({states[-1, 0]:.3f}, {states[-1, 1]:.3f})")
    print(f"Final velocity: {states[-1, 2]:.3f}")
    print(f"Final heading: {states[-1, 3]:.3f}")
    print(f"Average control: acceleration={np.mean(controls[:, 0]):.3f}, steering={np.mean(controls[:, 1]):.3f}")

    return states, controls


def test_circular_tracking():
    """Test LQR controller tracking a circular trajectory."""

    print("\nTesting circular trajectory tracking...")

    # Initialize model and controller
    model = KinematicBicycleModel(wheelbase=0.33, dt=0.05)
    controller = LQRController(
        wheelbase=0.33,
        dt=0.05,
        enable_logging=False
    )

    # Simulation parameters
    sim_time = 10.0
    dt = 0.05
    steps = int(sim_time / dt)

    # Circular trajectory parameters
    radius = 3.0
    angular_velocity = 0.3
    center_x, center_y = 3.0, 0.0

    # Initialize state on the circle
    current_state = np.array([center_x + radius, center_y, 1.0, np.pi / 2])

    # Storage for results
    states = [current_state.copy()]
    controls = []
    references = []

    # Run simulation
    for i in range(steps):
        t = i * dt

        # Circular reference trajectory (slightly ahead)
        lookahead_time = 0.5  # 0.5 seconds ahead
        ref_t = t + lookahead_time
        ref_x = center_x + radius * np.cos(angular_velocity * ref_t)
        ref_y = center_y + radius * np.sin(angular_velocity * ref_t)
        ref_theta = angular_velocity * ref_t + np.pi / 2
        ref_v = radius * angular_velocity  # Tangential velocity

        reference_state = np.array([ref_x, ref_y, ref_v, ref_theta])
        references.append(reference_state.copy())

        # Compute control
        control = controller.compute_control(current_state, reference_state)
        controls.append(control.copy())

        # Update state
        current_state = model.dynamics(current_state, control)
        states.append(current_state.copy())

    # Convert to arrays
    states = np.array(states)
    controls = np.array(controls)
    references = np.array(references)

    # Compute tracking error
    tracking_errors = []
    for i in range(len(states) - 1):
        error = np.linalg.norm(states[i, :2] - references[i, :2])
        tracking_errors.append(error)

    print(f"Average tracking error: {np.mean(tracking_errors):.3f} m")
    print(f"Max tracking error: {np.max(tracking_errors):.3f} m")
    print(f"Final position: ({states[-1, 0]:.3f}, {states[-1, 1]:.3f})")

    return states, controls, references, tracking_errors


def test_step_response():
    """Test LQR controller step response."""

    print("\nTesting step response...")

    # Initialize model and controller
    model = KinematicBicycleModel(wheelbase=0.33, dt=0.05)
    controller = LQRController(
        wheelbase=0.33,
        dt=0.05,
        enable_logging=False
    )

    # Simulation parameters
    sim_time = 3.0
    dt = 0.05
    steps = int(sim_time / dt)

    # Initialize state
    current_state = np.array([0.0, 0.0, 1.0, 0.0])

    # Reference: step input in x position
    reference_state = np.array([2.0, 0.0, 1.0, 0.0])

    # Storage for results
    states = [current_state.copy()]
    controls = []

    # Run simulation
    for i in range(steps):
        # Compute control
        control = controller.compute_control(current_state, reference_state)
        controls.append(control.copy())

        # Update state
        current_state = model.dynamics(current_state, control)
        states.append(current_state.copy())

    # Convert to arrays
    states = np.array(states)
    controls = np.array(controls)

    # Analyze settling time and overshoot
    final_position = reference_state[0]
    position_error = np.abs(states[:, 0] - final_position)
    settling_threshold = 0.05 * final_position  # 5% of final value

    # Find settling time
    settling_index = len(position_error) - 1
    for i in range(len(position_error) - 1, -1, -1):
        if position_error[i] > settling_threshold:
            settling_index = i + 1
            break

    settling_time = settling_index * dt
    overshoot = np.max(states[:, 0]) - final_position

    print(f"Settling time (5% criterion): {settling_time:.2f} s")
    print(f"Overshoot: {overshoot:.3f} m")
    print(f"Final position error: {np.abs(states[-1, 0] - final_position):.3f} m")

    return states, controls


def plot_results(states, controls, title="LQR Controller Test"):
    """Plot simulation results."""

    try:
        import matplotlib.pyplot as plt

        time = np.arange(len(states)) * 0.05

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(title)

        # Position trajectory
        axes[0, 0].plot(states[:, 0], states[:, 1], 'b-', linewidth=2, label='Actual')
        axes[0, 0].plot(states[0, 0], states[0, 1], 'go', markersize=8, label='Start')
        axes[0, 0].plot(states[-1, 0], states[-1, 1], 'ro', markersize=8, label='End')
        axes[0, 0].set_xlabel('X [m]')
        axes[0, 0].set_ylabel('Y [m]')
        axes[0, 0].set_title('Position Trajectory')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 0].axis('equal')

        # Velocity and heading
        axes[0, 1].plot(time[:-1], states[:-1, 2], 'b-', linewidth=2, label='Velocity')
        axes[0, 1].set_xlabel('Time [s]')
        axes[0, 1].set_ylabel('Velocity [m/s]')
        axes[0, 1].set_title('Velocity')
        axes[0, 1].grid(True)

        ax_twin = axes[0, 1].twinx()
        ax_twin.plot(time[:-1], states[:-1, 3], 'r-', linewidth=2, label='Heading')
        ax_twin.set_ylabel('Heading [rad]')

        # Control inputs
        control_time = np.arange(len(controls)) * 0.05
        axes[1, 0].plot(control_time, controls[:, 0], 'b-', linewidth=2)
        axes[1, 0].set_xlabel('Time [s]')
        axes[1, 0].set_ylabel('Acceleration [m/s²]')
        axes[1, 0].set_title('Control: Acceleration')
        axes[1, 0].grid(True)

        axes[1, 1].plot(control_time, controls[:, 1], 'r-', linewidth=2)
        axes[1, 1].set_xlabel('Time [s]')
        axes[1, 1].set_ylabel('Steering Angle [rad]')
        axes[1, 1].set_title('Control: Steering')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib not available, skipping plots")


def main():
    """Run all tests."""

    print("=" * 50)
    print("LQR Controller Test Suite")
    print("=" * 50)

    # Test 1: Straight line tracking
    states1, controls1 = test_straight_line_tracking()

    # Test 2: Circular trajectory tracking
    states2, controls2, refs2, errors2 = test_circular_tracking()

    # Test 3: Step response
    states3, controls3 = test_step_response()

    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("=" * 50)

    # Plot results if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        # Plot circular tracking
        plot_results(states2, controls2, "Circular Trajectory Tracking")

        # Plot step response
        plot_results(states3, controls3, "Step Response")

    except ImportError:
        print("Install matplotlib to see plots: pip install matplotlib")


if __name__ == '__main__':
    main()
