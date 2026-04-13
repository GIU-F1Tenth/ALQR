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

Build both ALQR components (LQR + Horizon Mapper) from the workspace root:

```bash
colcon build --base-paths src/control/alqr src/control/alqr/path_planner
```

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
- `/lqr_controller/state_error` (std_msgs/Float32): State tracking error

## Visualization

The package includes a comprehensive Python GUI visualizer for real-time monitoring and analysis:

### GUI Visualizer Features
- **Real-time trajectory tracking**: 2D plot showing vehicle path vs reference trajectory
- **Control input monitoring**: Time series plots of acceleration and steering commands
- **Performance metrics**: State error tracking, control frequency, and timing statistics
- **System diagnostics**: Controller status, safety states, and failure detection

### Quick Start (Standalone Demo)
```bash
cd scripts/
./launch_visualizer.sh standalone
```

### ROS2 Integration
```bash
# Terminal 1: Launch LQR controller
ros2 launch lqr_controller lqr_controller.launch.py

# Terminal 2: Launch visualizer
cd scripts/
./launch_visualizer.sh ros2
```

For detailed visualizer documentation, see `scripts/README.md`.

## Author

Mohammed Azab <mohammed@azab.io>

## License

MIT License
