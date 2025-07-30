# LQR Controller

Linear Quadratic Regulator (LQR) controller package for F1TENTH autonomous racing systems.

## Overview

This package implements a Linear Quadratic Regulator (LQR) controller for trajectory tracking in F1TENTH autonomous racing cars. The controller subscribes to reference trajectories from the horizon_mapper package and computes optimal control inputs using LQR control theory.

## Features

- **Kinematic bicycle model linearization**: Linearizes the vehicle dynamics around the current state
- **LQR optimal control**: Computes control inputs [acceleration, steering_angle] using LQR
- **Trajectory tracking**: Processes reference trajectories into state vectors [x_ref, y_ref, theta_ref, v_ref]
- **ROS2 integration**: Full ROS2 Python implementation with parameter support
- **Modular design**: Clean, maintainable code structure with comprehensive error handling

## Dependencies

- ROS2 (Humble or newer)
- NumPy
- SciPy
- giu_f1t_interfaces
- ackermann_msgs
- geometry_msgs
- nav_msgs

## Usage

Launch the LQR controller node:

```bash
ros2 run lqr_controller lqr_node
```

## Configuration

The controller supports extensive parameter configuration. See `config/lqr_params.yaml` for all available parameters.

## Topics

### Subscribed Topics
- `/car_state/odom` (nav_msgs/Odometry): Vehicle odometry
- `/horizon_mapper/reference_trajectory` (giu_f1t_interfaces/VehicleStateArray): Reference trajectory
- `/horizon_mapper/path_ready` (std_msgs/Bool): Path status

### Published Topics
- `/drive` (ackermann_msgs/AckermannDriveStamped): Control commands
- `/lqr_controller/diagnostics` (diagnostic_msgs/DiagnosticArray): Controller diagnostics

## Author

Mohammed Azab <mohammed@azab.io>

## License

MIT License
