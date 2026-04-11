# LQR Controller Documentation

## Overview

The LQR (Linear Quadratic Regulator) Controller is a sophisticated trajectory tracking controller for F1TENTH autonomous racing vehicles. It implements optimal control theory to compute control inputs that minimize a quadratic cost function while following reference trajectories.

## Theory

### Kinematic Bicycle Model

The controller uses a kinematic bicycle model for vehicle dynamics:

**State Vector:** `x = [x, y, v, θ]`
- `x`: longitudinal position [m]
- `y`: lateral position [m]  
- `v`: velocity [m/s]
- `θ`: heading angle [rad]

**Control Vector:** `u = [a, δ]`
- `a`: acceleration [m/s²]
- `δ`: steering angle [rad]

**Dynamics:**
```
ẋ = v * cos(θ)
ẏ = v * sin(θ)
v̇ = a
θ̇ = (v/L) * tan(δ)
```

Where `L` is the wheelbase.

### LQR Control

The LQR controller minimizes the quadratic cost function:

```
J = Σ[(x-x_ref)ᵀ Q (x-x_ref) + uᵀ R u]
```

Where:
- `Q`: State cost matrix (4×4)
- `R`: Control cost matrix (2×2)
- `x_ref`: Reference state trajectory

The optimal control law is:
```
u = u_ff - K(x - x_ref)
```

Where:
- `u_ff`: Feedforward control
- `K`: LQR gain matrix
- `(x - x_ref)`: State tracking error

## Architecture

### Core Components

1. **KinematicBicycleModel**: Implements vehicle dynamics and linearization
2. **LQRController**: Core LQR algorithm and control computation
3. **LQRNode**: ROS2 node for integration with the F1TENTH stack

### Message Flow

```
horizon_mapper → reference_trajectory → LQRNode
car_state/odom → current_state → LQRNode
LQRNode → control_commands → /drive
```

## Configuration

### Key Parameters

#### Vehicle Parameters
```yaml
wheelbase: 0.33          # Vehicle wheelbase [m]
dt: 0.05                 # Control time step [s]
```

#### Control Limits
```yaml
max_acceleration: 5.0    # Maximum acceleration [m/s²]
max_steering_angle: 0.5  # Maximum steering angle [rad]
min_speed: 0.1          # Minimum speed [m/s]
max_speed: 8.0          # Maximum speed [m/s]
```

#### LQR Cost Weights
```yaml
lqr_weights:
  position_weight: 10.0    # Position tracking importance
  velocity_weight: 1.0     # Velocity tracking importance  
  heading_weight: 5.0      # Heading tracking importance
  acceleration_weight: 0.1 # Acceleration effort penalty
  steering_weight: 1.0     # Steering effort penalty
```

### Tuning Guidelines

#### Position Tracking
- **Increase `position_weight`** for tighter trajectory following
- **Decrease** for smoother, less aggressive tracking

#### Heading Control
- **Increase `heading_weight`** for better orientation control
- **Decrease** if causing oscillations

#### Control Effort
- **Increase `acceleration_weight`** for smoother acceleration
- **Increase `steering_weight`** for smoother steering
- **Balance** with tracking performance

## Usage

### Basic Launch

```bash
# Launch LQR controller only
ros2 launch lqr_controller lqr_controller.launch.py

# Launch with horizon mapper
ros2 launch lqr_controller lqr_controller.launch.py

# Launch with custom config
ros2 launch lqr_controller lqr_controller.launch.py config_file:=/path/to/config.yaml
```

### Debug Mode

```bash
ros2 launch lqr_controller lqr_controller.launch.py debug:=true
```

### Simulation Mode

```bash
ros2 launch lqr_controller lqr_controller.launch.py use_sim_time:=true
```

## Topics

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/car_state/odom` | `nav_msgs/Odometry` | Vehicle odometry |
| `/horizon_mapper/reference_trajectory` | `giu_f1t_interfaces/VehicleStateArray` | Reference trajectory |
| `/horizon_mapper/path_ready` | `std_msgs/Bool` | Path status |
| `/initialpose` | `geometry_msgs/PoseStamped` | RViz pose estimate |

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/drive` | `ackermann_msgs/AckermannDriveStamped` | Control commands |
| `/lqr_controller/diagnostics` | `diagnostic_msgs/DiagnosticArray` | Controller diagnostics |
| `/lqr_controller/state_error` | `std_msgs/Float32` | Tracking error (debug) |

## Diagnostics

The controller publishes comprehensive diagnostics including:

- Controller status (OK/WARN/ERROR)
- Performance metrics (control loop timing)
- Safety status
- Trajectory information
- Failure counts

### Status Levels

- **OK**: Controller active and tracking
- **WARN**: Controller not active but ready
- **ERROR**: Emergency stop or critical failure

## Safety Features

### Safety Checks

1. **Data Freshness**: Monitors odometry and trajectory timestamps
2. **State Validation**: Checks for reasonable state values
3. **Control Validation**: Ensures control outputs are within bounds
4. **Emergency Stop**: Automatic stopping on failures

### Emergency Conditions

- No recent odometry data (> safety_timeout)
- No recent trajectory data (> safety_timeout)
- Invalid state or control values
- Path not ready
- Too many consecutive failures

## Performance

### Typical Performance

- **Control Frequency**: 20 Hz (configurable)
- **Computation Time**: < 1 ms per control cycle
- **Memory Usage**: ~10 MB
- **Tracking Accuracy**: < 10 cm RMS error at racing speeds

### Optimization Features

- **Gain Caching**: Avoids redundant LQR gain computation
- **State Validation**: Early exit for invalid inputs
- **Bounded History**: Prevents memory growth

## Troubleshooting

### Common Issues

#### Poor Tracking Performance
```
Symptoms: Large position errors, oscillations
Solutions: 
- Increase position_weight
- Decrease control_effort weights
- Check reference trajectory quality
- Verify vehicle parameters (wheelbase)
```

#### Oscillatory Behavior
```
Symptoms: Vehicle oscillating around reference
Solutions:
- Decrease heading_weight
- Increase steering_weight
- Reduce control frequency
- Check for sensor noise
```

#### Sluggish Response
```
Symptoms: Slow response to reference changes
Solutions:
- Increase position_weight and heading_weight
- Decrease control effort weights
- Increase lookahead_distance
- Enable feedforward control
```

#### Emergency Stops
```
Symptoms: Frequent emergency stops
Solutions:
- Check data rates and timing
- Increase safety_timeout
- Verify message connectivity
- Check vehicle state validity
```

### Debug Tools

#### Log Analysis
```bash
# Enable debug logging
ros2 param set /lqr_controller_node debug_logging_enabled true

# Monitor diagnostics
ros2 topic echo /lqr_controller/diagnostics

# Monitor state error
ros2 topic echo /lqr_controller/state_error
```

#### Performance Monitoring
```bash
# Check control frequency
ros2 topic hz /drive

# Monitor computation time
ros2 topic echo /lqr_controller/diagnostics --field status[0].values
```

## Integration with F1TENTH Stack

### Dependencies

The LQR controller integrates with:

1. **horizon_mapper**: Provides reference trajectories
2. **state_estimation**: Provides vehicle odometry
3. **vesc_driver**: Receives control commands

### Coordinate Frames

- **Input**: `map` frame (from odometry)
- **Output**: `base_link` frame (control commands)
- **Trajectory**: `map` frame (reference points)

### Message Compatibility

The controller uses standard F1TENTH message types:
- `ackermann_msgs/AckermannDriveStamped` for control
- `nav_msgs/Odometry` for state estimation
- `giu_f1t_interfaces/VehicleStateArray` for trajectories

## Testing

### Unit Tests

```bash
# Run all tests
colcon test --packages-select lqr_controller

# Run specific test
python3 -m pytest src/race_stack/lqr_controller/test/test_lqr_controller.py
```

### Standalone Testing

```bash
# Run standalone controller test
python3 src/race_stack/lqr_controller/scripts/test_lqr_standalone.py
```

### Simulation Testing

1. Launch F1TENTH simulator
2. Launch horizon_mapper
3. Launch LQR controller
4. Monitor performance in RViz

## Comparison with MPC

| Aspect | LQR | MPC |
|--------|-----|-----|
| **Computation** | Fast (analytical) | Slower (optimization) |
| **Horizons** | Infinite horizon | Finite horizon |
| **Constraints** | Hard to handle | Natural constraint handling |
| **Optimality** | Optimal for LTI systems | Optimal for constrained systems |
| **Tuning** | Simpler (Q, R matrices) | More complex (many parameters) |
| **Real-time** | Excellent | Good |
| **Robustness** | Good | Very good |

### When to Use LQR

- **High-frequency control** (>50 Hz)
- **Simple cost functions**
- **Unconstrained or soft-constrained problems**
- **Real-time critical applications**
- **Simpler tuning requirements**

### When to Use MPC

- **Hard constraints** on states/controls
- **Complex cost functions**
- **Preview of disturbances**
- **Multi-objective optimization**
- **Safety-critical applications**

## References

1. B. D. O. Anderson and J. B. Moore, "Optimal Control: Linear Quadratic Methods"
2. R. Rajamani, "Vehicle Dynamics and Control"
3. J. Kong et al., "Kinematic and dynamic vehicle models for autonomous driving control design"
