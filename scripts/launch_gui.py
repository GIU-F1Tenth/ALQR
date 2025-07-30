#!/usr/bin/env python3

"""
LQR Parameter GUI Launcher

Simple launcher script for the LQR parameter tuning GUI.
Provides options for standalone or ROS2 integrated mode.

Usage:
    python3 launch_gui.py              # Standalone mode
    python3 launch_gui.py --ros2       # ROS2 integrated mode
    python3 launch_gui.py --test       # Test dependencies only
"""

import sys
import os
import argparse

def test_dependencies():
    """Test if all required dependencies are available."""
    print("Testing GUI dependencies...")
    
    missing_deps = []
    
    try:
        import tkinter
        print("✓ tkinter available")
    except ImportError:
        missing_deps.append("tkinter")
        print("✗ tkinter not available")
    
    try:
        import numpy
        print("✓ numpy available")
    except ImportError:
        missing_deps.append("numpy")
        print("✗ numpy not available")
    
    try:
        import yaml
        print("✓ PyYAML available")
    except ImportError:
        missing_deps.append("PyYAML")
        print("✗ PyYAML not available")
    
    try:
        import matplotlib
        print("✓ matplotlib available")
    except ImportError:
        missing_deps.append("matplotlib")
        print("✗ matplotlib not available")
    
    # Test config
    try:
        import importlib.util
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.py')
        spec = importlib.util.spec_from_file_location("lqr_config", config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        print("✓ config module available")
    except Exception as e:
        print(f"✗ config module issue: {e}")
        # Config is not critical, so don't add to missing_deps
    
    if missing_deps:
        print(f"\n✗ Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        return False
    else:
        print("\n✓ All dependencies satisfied!")
        return True

def launch_standalone():
    """Launch standalone GUI."""
    print("Launching LQR Parameter GUI (Standalone Mode)...")
    
    gui_script = os.path.join(os.path.dirname(__file__), 'lqr_parameter_gui_standalone.py')
    
    if not os.path.exists(gui_script):
        print(f"Error: GUI script not found at {gui_script}")
        return False
    
    try:
        # Import and run the GUI
        sys.path.insert(0, os.path.dirname(__file__))
        from lqr_parameter_gui_standalone import SimpleLQRParameterGUI
        
        import tkinter as tk
        root = tk.Tk()
        app = SimpleLQRParameterGUI(root)
        
        print("GUI launched successfully!")
        print("Close the GUI window to exit.")
        
        root.mainloop()
        return True
        
    except Exception as e:
        print(f"Error launching GUI: {e}")
        import traceback
        traceback.print_exc()
        return False

def launch_ros2():
    """Launch ROS2 integrated GUI."""
    print("Launching LQR Parameter GUI (ROS2 Mode)...")
    
    try:
        import rclpy
        print("✓ ROS2 available")
    except ImportError:
        print("✗ ROS2 not available. Use standalone mode instead.")
        return False
    
    gui_script = os.path.join(os.path.dirname(__file__), 'lqr_parameter_gui.py')
    
    if not os.path.exists(gui_script):
        print(f"Error: ROS2 GUI script not found at {gui_script}")
        return False
    
    try:
        # Import and run the ROS2 GUI
        sys.path.insert(0, os.path.dirname(__file__))
        from lqr_parameter_gui import LQRParameterGUI
        
        import tkinter as tk
        root = tk.Tk()
        app = LQRParameterGUI(root)
        
        print("ROS2 GUI launched successfully!")
        print("Close the GUI window to exit.")
        
        root.mainloop()
        return True
        
    except Exception as e:
        print(f"Error launching ROS2 GUI: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Launch LQR Parameter Tuning GUI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--ros2', 
        action='store_true',
        help='Launch ROS2 integrated GUI (default: standalone)'
    )
    
    parser.add_argument(
        '--test', 
        action='store_true',
        help='Test dependencies only'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LQR Parameter Tuning GUI Launcher")
    print("=" * 60)
    
    # Test dependencies first
    if not test_dependencies():
        print("\n✗ Dependency check failed!")
        sys.exit(1)
    
    if args.test:
        print("\n✓ Dependency test completed successfully!")
        return
    
    # Launch appropriate GUI
    success = False
    
    if args.ros2:
        success = launch_ros2()
    else:
        success = launch_standalone()
    
    if success:
        print("\n✓ GUI session completed successfully!")
    else:
        print("\n✗ GUI launch failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
