# LQR Parameter Tuning GUI

A comprehensive GUI system for real-time tuning of LQR controller parameters for F1TENTH autonomous racing vehicles.

## Features

### 🎛️ Real-Time Parameter Adjustment
- **LQR Cost Matrices**: Adjust Q and R matrix weights with sliders
- **Control Limits**: Tune acceleration, deceleration, and steering limits
- **Vehicle Parameters**: Modify wheelbase, time step, and lookahead distance
- **Anti-Wobble Settings**: Configure steering rate limiting and adaptive lookahead
- **Curve Detection**: Adjust curve detection and speed adaptation parameters
- **Safety Parameters**: Configure safety timeouts and emergency braking

### 📊 Real-Time Monitoring
- Live plots of lateral error, heading error, and velocity tracking
- Real-time display of control and state costs
- Performance metrics visualization
- Parameter effect visualization with simulated data

### 💾 Configuration Management
- Save/Load parameter configurations as YAML files
- Export parameters to Python config files
- Reset to default values
- Import/Export compatibility with existing config system

### 🔗 ROS2 Integration
- Real-time parameter updates to running LQR controller
- Live data monitoring from vehicle odometry
- Service-based parameter communication
- Compatible with simulation and physical vehicle

## Installation

### Prerequisites
```bash
# Install GUI dependencies
cd /home/mohammedazab/ws/src/race_stack/lqr_contoller/scripts
pip install -r requirements_gui.txt
```

### ROS2 Package Setup
Make sure your LQR controller package is properly built:
```bash
cd /home/mohammedazab/ws
colcon build --packages-select lqr_controller
source install/setup.bash
```

## Usage

### 1. Standalone GUI (No ROS2 Required)
Perfect for offline parameter tuning and configuration preparation:

```bash
cd /home/mohammedazab/ws/src/race_stack/lqr_contoller/scripts
python3 lqr_parameter_gui_standalone.py
```

**Features:**
- Full parameter adjustment interface
- Simulated performance visualization
- Configuration save/load functionality
- No ROS2 dependencies required

### 2. ROS2 Integrated GUI
For real-time tuning with running controller:

```bash
# Start the complete system
ros2 launch lqr_controller lqr_parameter_tuning.launch.py

# Or start components separately:
# Terminal 1: Start LQR controller
ros2 run lqr_controller lqr_node.py

# Terminal 2: Start parameter GUI
ros2 run lqr_controller lqr_parameter_gui.py
```

### 3. Quick Test with Standalone GUI
```bash
cd /home/mohammedazab/ws/src/race_stack/lqr_contoller
python3 scripts/lqr_parameter_gui_standalone.py
```

## GUI Interface Guide

### Parameter Sections

#### 🎯 LQR Cost Matrices
- **Q Matrix (State Cost)**:
  - `Position X/Y Weight`: Higher values penalize lateral deviations more
  - `Velocity Weight`: Controls velocity tracking importance
  - `Heading Weight`: Higher values improve heading tracking
  
- **R Matrix (Control Cost)**:
  - `Acceleration Weight`: Higher values create smoother acceleration
  - `Steering Weight`: Higher values reduce steering oscillations

#### ⚡ Control Limits
- `Max Acceleration/Deceleration`: Physical limits of the vehicle
- `Max Steering Angle`: Maximum wheel angle in radians
- `Min/Max Speed`: Operating speed range

#### 🚗 Vehicle Parameters
- `Wheelbase`: Distance between front and rear axles
- `Time Step`: Control loop period
- `Lookahead Distance`: Base lookahead for trajectory following

#### 🔧 Anti-Wobble Parameters
- `Enable Steering Rate Limit`: Prevents rapid steering changes
- `Max Steering Rate`: Maximum allowed steering velocity
- `Min/Max Lookahead Distance`: Adaptive lookahead bounds
- `Lookahead Time`: Time-based lookahead calculation

#### 🌀 Curve Detection
- `Enable Curve Detection`: Activates curve-aware control
- `Curve Lookahead Points`: Points to analyze for curvature
- `Max Curvature Threshold`: Threshold for curve detection
- `Curve Speed Factor`: Speed reduction in curves

### Real-Time Monitoring

The GUI provides live visualization of:
- **Lateral Error**: Cross-track error from reference trajectory
- **Heading Error**: Angular deviation from reference heading
- **Velocity Error**: Speed tracking error
- **Control Cost**: Current control effort cost
- **State Cost**: Current state tracking cost

### Configuration Management

#### Saving Configurations
1. Click "Save Config" button
2. Choose filename and location
3. Parameters saved in organized YAML format

#### Loading Configurations
1. Click "Load Config" button
2. Select YAML configuration file
3. Parameters automatically applied to GUI and controller

#### Exporting to Python
1. Click "Export to Python" button
2. Generates Python config file compatible with existing system

## Parameter Tuning Guidelines

### 🎯 For Better Trajectory Tracking
- **Increase Q matrix weights** (especially position and heading)
- **Decrease R matrix weights** (allow more aggressive control)
- **Adjust lookahead distance** based on speed and track complexity

### 🛡️ For Stability and Smoothness
- **Increase R matrix weights** (penalize control effort more)
- **Enable steering rate limiting**
- **Use adaptive lookahead parameters**
- **Enable curve detection**

### ⚡ For Performance Tuning
- **Start with low gains and gradually increase**
- **Monitor real-time plots for oscillations**
- **Test at different speeds and track sections**
- **Save good configurations for different scenarios**

### 🔧 Common Parameter Ranges

| Parameter | Conservative | Balanced | Aggressive |
|-----------|--------------|----------|------------|
| Position Weight | 1-5 | 5-15 | 15-50 |
| Heading Weight | 1-3 | 3-8 | 8-20 |
| Steering Weight | 5-10 | 2-5 | 0.5-2 |
| Acceleration Weight | 1-5 | 0.1-1 | 0.01-0.1 |
| Max Steering Rate | 0.5-1 | 1-2 | 2-5 |

## Troubleshooting

### GUI Won't Start
```bash
# Check Python dependencies
python3 -c "import tkinter, numpy, matplotlib, yaml"

# Check config import
cd /home/mohammedazab/ws/src/race_stack/lqr_contoller
python3 -c "import sys; sys.path.append('config'); import config; print('Config OK')"
```

### ROS2 Connection Issues
```bash
# Check if LQR controller is running
ros2 node list | grep lqr

# Check parameter service
ros2 service list | grep lqr

# Test parameter communication
ros2 param list /lqr_controller
```

### Parameter Updates Not Working
1. Verify ROS2 connection status in GUI
2. Check that LQR controller node is running
3. Ensure parameter names match between GUI and controller
4. Check ROS2 logs for error messages

## Development and Customization

### Adding New Parameters
1. Add parameter to `_load_default_parameters()` in GUI
2. Add slider/control in appropriate section
3. Update save/load functions to handle new parameter
4. Ensure LQR controller accepts the new parameter

### Customizing Plots
Modify `_setup_plots()` and `_update_plots()` methods to:
- Add new monitoring variables
- Change plot layouts
- Add different visualization types

### ROS2 Integration
The GUI communicates with ROS2 through:
- Parameter services for real-time updates
- Topic subscriptions for monitoring data
- Custom message types for complex data

## Files Structure

```
scripts/
├── lqr_parameter_gui.py              # Full ROS2 integrated GUI
├── lqr_parameter_gui_standalone.py   # Standalone GUI for testing
├── requirements_gui.txt              # Python dependencies
└── README_GUI.md                     # This documentation

launch/
└── lqr_parameter_tuning.launch.py   # Launch file for complete system

config/
├── config.py                        # Python configuration
├── lqr_params.yaml                  # YAML configuration
└── saved_configs/                   # Directory for saved configurations
```

## Tips for Effective Tuning

1. **Start Conservative**: Begin with higher R matrix weights for stability
2. **Incremental Changes**: Make small adjustments and observe effects
3. **Save Frequently**: Save good configurations before experimenting
4. **Test Scenarios**: Tune for different track sections (straights, curves, chicanes)
5. **Monitor Plots**: Watch for oscillations or instability in real-time plots
6. **Use Simulation First**: Test parameters in simulation before physical vehicle

## License

MIT License - See LICENSE file for details.

## Author

Mohammed Azab <mohammed@azab.io>

## Contributing

Feel free to submit issues, suggestions, or pull requests to improve the GUI system.
