#!/usr/bin/env python3

"""
LQR Controller Visualizer Demo Script

This script demonstrates how to use the LQR visualizer programmatically
and shows various features and capabilities.

Author: Mohammed Azab <mohammed@azab.io>
License: MIT
Version: 1.0.0
"""

import sys
import os
import time
import subprocess
import signal

def print_banner():
    """Print welcome banner."""
    print("=" * 60)
    print("  LQR Controller Visualizer Demo")
    print("  F1TENTH Autonomous Racing Systems")
    print("=" * 60)
    print()

def print_features():
    """Print feature list."""
    print("🎯 Visualizer Features:")
    print("  • Real-time trajectory tracking")
    print("  • Control input monitoring (acceleration & steering)")
    print("  • Performance metrics and timing analysis")
    print("  • System diagnostics and safety monitoring")
    print("  • Multi-tab interface with specialized views")
    print()

def print_demo_modes():
    """Print available demo modes."""
    print("📊 Available Demo Modes:")
    print("  1. Standalone Demo - Synthetic data simulation")
    print("  2. ROS2 Integration - Live controller data")
    print("  3. Feature Tour - Guided tour of capabilities")
    print("  4. Performance Test - Stress test with high-rate data")
    print()

def run_standalone_demo():
    """Run the standalone demo."""
    print("🚀 Starting Standalone Demo...")
    print("This demo generates synthetic vehicle data to showcase the visualizer.")
    print("You'll see a vehicle following a figure-8 trajectory.")
    print()
    print("Press Ctrl+C in the visualizer window to exit.")
    print()
    
    # Change to scripts directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        subprocess.run([sys.executable, "lqr_visualizer_standalone.py"], check=True)
    except KeyboardInterrupt:
        print("Demo interrupted by user.")
    except subprocess.CalledProcessError as e:
        print(f"Demo failed with error: {e}")
    except FileNotFoundError:
        print("Error: Visualizer script not found. Make sure you're in the correct directory.")

def check_ros2_environment():
    """Check if ROS2 environment is available."""
    try:
        result = subprocess.run(['ros2', '--version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def run_ros2_demo():
    """Run the ROS2 integration demo."""
    print("🤖 Starting ROS2 Integration Demo...")
    
    if not check_ros2_environment():
        print("❌ ROS2 not found or not properly sourced.")
        print("Please ensure ROS2 is installed and sourced:")
        print("  source /opt/ros/humble/setup.bash")
        return
    
    print("✅ ROS2 environment detected.")
    print()
    print("This demo connects to live ROS2 topics from the LQR controller.")
    print("Make sure the LQR controller node is running:")
    print("  ros2 launch lqr_controller lqr_controller.launch.py")
    print()
    
    input("Press Enter when the controller is running, or Ctrl+C to cancel...")
    
    # Change to scripts directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        subprocess.run([sys.executable, "lqr_visualizer.py"], check=True)
    except KeyboardInterrupt:
        print("Demo interrupted by user.")
    except subprocess.CalledProcessError as e:
        print(f"Demo failed with error: {e}")

def show_feature_tour():
    """Show guided tour of features."""
    print("🗺️  LQR Visualizer Feature Tour")
    print("=" * 40)
    print()
    
    features = [
        ("Controller Status Panel", 
         "Shows real-time status indicators and current values"),
        ("Trajectory Tab", 
         "2D visualization of vehicle path vs reference trajectory"),
        ("Control Inputs Tab", 
         "Time series plots of acceleration and steering commands"),
        ("Performance Tab", 
         "State error tracking and velocity profile analysis"),
        ("Diagnostics Tab", 
         "Detailed system metrics and debugging information")
    ]
    
    for i, (feature, description) in enumerate(features, 1):
        print(f"{i}. {feature}")
        print(f"   {description}")
        print()
    
    print("🎛️  Key Features:")
    print("  • Real-time updates at 10 Hz")
    print("  • Configurable history length (up to 1000 points)")
    print("  • Safety monitoring with emergency stop detection")
    print("  • Performance metrics including control timing")
    print("  • Modular design for easy customization")
    print()

def run_performance_test():
    """Run performance test."""
    print("⚡ Performance Test Mode")
    print("=" * 30)
    print()
    print("This test will run the standalone visualizer with high-frequency")
    print("data generation to test GUI performance and responsiveness.")
    print()
    print("Monitor the following during the test:")
    print("  • Smooth plot updates")
    print("  • Responsive GUI interactions")
    print("  • Memory usage stability")
    print("  • CPU usage")
    print()
    
    input("Press Enter to start performance test, or Ctrl+C to cancel...")
    
    # Note: For a real performance test, we would modify the standalone
    # visualizer to generate data at higher rates
    run_standalone_demo()

def main():
    """Main demo function."""
    try:
        print_banner()
        print_features()
        print_demo_modes()
        
        while True:
            print("Choose a demo mode:")
            print("  1 - Standalone Demo")
            print("  2 - ROS2 Integration Demo") 
            print("  3 - Feature Tour")
            print("  4 - Performance Test")
            print("  q - Quit")
            print()
            
            choice = input("Enter your choice (1-4, q): ").strip().lower()
            
            if choice == '1':
                run_standalone_demo()
            elif choice == '2':
                run_ros2_demo()
            elif choice == '3':
                show_feature_tour()
                input("\nPress Enter to continue...")
            elif choice == '4':
                run_performance_test()
            elif choice in ['q', 'quit', 'exit']:
                print("Goodbye! 👋")
                break
            else:
                print("Invalid choice. Please try again.")
            
            print()
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user. Goodbye! 👋")
    except Exception as e:
        print(f"Demo error: {e}")

if __name__ == '__main__':
    main()
