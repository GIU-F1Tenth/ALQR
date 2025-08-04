# LQR Parameter Tuning GUI - Quick Start Guide

## 🚀 Quick Start

### Test Dependencies
```bash
cd /home/mohammedazab/ws/src/race_stack/lqr_contoller/scripts
python3 launch_gui.py --test
```

### Launch Standalone GUI (Recommended for Testing)
```bash
cd /home/mohammedazab/ws/src/race_stack/lqr_contoller/scripts
python3 launch_gui.py
```

### Launch ROS2 Integrated GUI
```bash
cd /home/mohammedazab/ws/src/race_stack/lqr_contoller/scripts
python3 launch_gui.py --ros2
```

## 📁 Files Created

```
scripts/
├── lqr_parameter_gui.py              # Full ROS2 integrated GUI
├── lqr_parameter_gui_standalone.py   # Standalone GUI for testing
├── launch_gui.py                     # Easy launcher script
├── test_gui_components.py            # Component testing
├── requirements_gui.txt              # Python dependencies
└── README_GUI.md                     # Detailed documentation

launch/
└── lqr_parameter_tuning.launch.py   # ROS2 launch file
```

## 🎛️ GUI Features

### Real-Time Parameter Adjustment
- **LQR Q Matrix**: Position, velocity, and heading weights
- **LQR R Matrix**: Acceleration and steering weights  
- **Control Limits**: Max acceleration, deceleration, steering
- **Vehicle Parameters**: Wheelbase, time step, lookahead
- **Anti-Wobble**: Steering rate limiting, adaptive lookahead
- **Curve Detection**: Curvature-based speed adaptation
- **Safety**: Timeout and emergency brake settings

### Live Monitoring (Standalone Mode)
- Real-time plots of tracking errors
- Control and state cost visualization
- Parameter effect demonstration
- Simulated performance metrics

### Configuration Management
- Save/Load YAML configurations
- Export to Python config files
- Reset to default values
- Parameter validation

## 🔧 Parameter Tuning Tips

### For Better Tracking
- **Increase Q weights** (position, heading)
- **Decrease R weights** (allow more control effort)
- **Adjust lookahead distance**

### For Stability
- **Increase R weights** (smoother control)
- **Enable steering rate limiting**
- **Use adaptive lookahead**

### Common Ranges
| Parameter | Conservative | Balanced | Aggressive |
|-----------|--------------|----------|------------|
| Position Weight | 1-5 | 5-15 | 15-50 |
| Heading Weight | 1-3 | 3-8 | 8-20 |
| Steering Weight | 5-10 | 2-5 | 0.5-2 |
| Acceleration Weight | 1-5 | 0.1-1 | 0.01-0.1 |

## 📊 Current Configuration Integration

The GUI automatically loads parameters from your existing `config.py`:

```python
# Your current config values are used as defaults:
wheelbase = 0.33
dt = 0.05
max_acceleration = 5.0
max_deceleration = 9.0
max_steering_angle = 0.9
max_speed = 15.0
position_weight = 5.0
velocity_weight = 1.0
heading_weight = 6.0
acceleration_weight = 0.3
steering_weight = 4.0
# ... and all other parameters
```

## 🔗 Integration with Your LQR Controller

### Modified Files
- `kinematic_bicycle_model.py`: Now uses config.py values
- `config.py`: Fixed Python syntax for proper importing

### ROS2 Integration (When Ready)
The GUI can communicate with your LQR controller via:
- Parameter services for real-time updates
- Topic subscriptions for monitoring data
- Custom messages for complex data exchange

## 🛟 Troubleshooting

### GUI Won't Start
```bash
# Check dependencies
python3 launch_gui.py --test

# Install missing packages
pip install numpy matplotlib PyYAML
```

### Config Import Issues
The GUI includes fallback default values if config.py is not accessible.

### ROS2 Connection Issues
Start with standalone mode first, then try ROS2 integration once your controller is running.

## 🎯 Next Steps

1. **Test the standalone GUI**: `python3 launch_gui.py`
2. **Experiment with parameters** using the sliders
3. **Save good configurations** for different scenarios
4. **Integrate with ROS2** when your controller is ready
5. **Customize the GUI** for your specific needs

## 📞 Support

The GUI system is designed to be:
- **Self-contained**: Works without ROS2 for testing
- **Extensible**: Easy to add new parameters
- **Robust**: Handles missing dependencies gracefully
- **User-friendly**: Intuitive interface with real-time feedback

For questions or customization needs, refer to the detailed documentation in `README_GUI.md`.

---

**Ready to tune your LQR controller? Start with:**
```bash
cd /home/mohammedazab/ws/src/race_stack/lqr_contoller/scripts
python3 launch_gui.py
```
