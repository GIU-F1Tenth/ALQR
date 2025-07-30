# LQR Controller Visualizer

This directory contains Python GUI visualization tools for the LQR (Linear Quadratic Regulator) controller used in F1TENTH autonomous racing.

## Overview

The visualizer provides real-time monitoring and analysis of the LQR controller performance, including:

- **Vehicle trajectory tracking** - Real-time 2D plot showing vehicle path vs reference trajectory
- **Control input monitoring** - Time series plots of acceleration and steering commands
- **Performance metrics** - State error tracking, control frequency, and timing statistics
- **System diagnostics** - Controller status, safety states, and failure detection

## Files

### `lqr_visualizer.py`
Full-featured visualizer that connects to ROS2 topics to display live data from the LQR controller node.

**Features:**
- Subscribes to vehicle odometry, control commands, and diagnostics
- Real-time plotting with 4 main tabs
- Status indicators and performance metrics
- Requires active ROS2 environment

### `lqr_visualizer_standalone.py`
Standalone demonstration version that generates synthetic data for testing and development.

**Features:**
- No ROS2 dependencies required
- Generates realistic synthetic vehicle dynamics
- Figure-8 reference trajectory
- Perfect for development and demonstration

### `requirements.txt`
Python package dependencies for the visualizer tools.

## Installation

### For Standalone Version (No ROS2)

1. Install Python dependencies:
```bash
pip install matplotlib numpy
```

2. Run the standalone visualizer:
```bash
python3 lqr_visualizer_standalone.py
```

### For ROS2 Version

1. Ensure ROS2 is installed and sourced:
```bash
source /opt/ros/humble/setup.bash  # or your ROS2 distribution
```

2. Install Python dependencies:
```bash
pip install matplotlib numpy
# ROS2 Python packages should already be available
```

3. Build the workspace:
```bash
cd /path/to/your/workspace
colcon build --packages-select lqr_controller
source install/setup.bash
```

4. Run the LQR controller node:
```bash
ros2 launch lqr_controller lqr_controller.launch.py
```

5. In another terminal, run the visualizer:
```bash
python3 lqr_visualizer.py
```

## GUI Layout

The visualizer features a tabbed interface with the following sections:

### 1. Controller Status Panel (Top)
- Real-time status indicators (Controller Active, Path Ready, Emergency Stop)
- Current vehicle state values (position, velocity, steering)
- Current control commands and state error

### 2. Trajectory Tab
- 2D plot showing vehicle path and reference trajectory
- Current vehicle position with orientation arrow
- Real-time updates as vehicle moves

### 3. Control Inputs Tab
- Time series plots of acceleration and steering commands
- Helps analyze control behavior and stability
- Shows control limits and saturation

### 4. Performance Tab
- State error magnitude over time
- Velocity tracking comparison (actual vs reference)
- Performance trend analysis

### 5. Diagnostics Tab
- Detailed text display of all system metrics
- Control timing statistics
- Failure counts and status information
- Data history and update timestamps

## Usage Tips

### Monitoring Controller Performance
- Watch the **state error** in the Performance tab to ensure good tracking
- Check **control frequency** in diagnostics to verify real-time performance
- Monitor **consecutive failures** for stability assessment

### Analyzing Trajectory Tracking
- Compare the red vehicle path with blue reference trajectory
- Look for consistent offset or oscillations indicating tuning issues
- Check that vehicle orientation arrow aligns with trajectory direction

### Debugging Control Issues
- Use Control Inputs tab to see if commands are saturating (hitting limits)
- Watch for excessive oscillation in steering commands
- Monitor acceleration commands for smoothness

### Safety Monitoring
- Emergency stop indicator should normally show "Normal" (green)
- Path ready should be "Ready" (green) when controller is active
- Controller status should show "Active" (green) during normal operation

## Configuration

### Customizing Update Rate
The GUI updates at 10 Hz by default. Modify the timer in `update_plots()`:
```python
self.root.after(100, self.update_plots)  # 100ms = 10 Hz
```

### Adjusting History Length
Change the maximum number of stored data points:
```python
self.max_history = 1000  # Increase for longer history
```

### Modifying Plot Appearance
Update matplotlib styling in the individual plot functions:
- `update_trajectory_plot()` - Trajectory colors and styles
- `update_control_plots()` - Control input plot formatting
- `update_performance_plots()` - Performance metric visualization

## Troubleshooting

### Common Issues

**"ROS2 Python libraries not found"**
- Ensure ROS2 is properly installed and sourced
- Try the standalone version for testing without ROS2

**GUI appears but no data**
- Check that LQR controller node is running
- Verify topic names match between controller and visualizer
- Use `ros2 topic list` to check available topics

**Plots not updating**
- Check ROS2 connection and topic publishing rates
- Verify callback functions are receiving data
- Check console for error messages

**Performance Issues**
- Reduce update frequency if GUI is slow
- Decrease history length for better performance
- Close other heavy applications

### Debug Mode

Enable debug output by modifying the data callback:
```python
def data_callback(self, data_type: str, data):
    print(f"Received {data_type}: {data}")  # Add this line
    # ... rest of function
```

## Development

### Adding New Visualizations

1. Create new tab in `setup_gui()`:
```python
def setup_new_tab(self):
    new_frame = ttk.Frame(self.notebook)
    self.notebook.add(new_frame, text="New Tab")
    # Add your widgets
```

2. Add update function:
```python
def update_new_plot(self):
    # Your plotting code here
    pass
```

3. Call update function in `update_plots()`:
```python
def update_plots(self):
    # ... existing code
    self.update_new_plot()
```

### Custom Data Sources

To use different data sources, modify the `data_callback()` function to handle your specific data format and create appropriate data generation or subscription logic.

## License

MIT License - See main project LICENSE file for details.

## Author

Mohammed Azab <mohammed@azab.io>

## Version History

- v1.0.0 - Initial release with full ROS2 integration and standalone demo
