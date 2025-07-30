#!/bin/bash

# LQR Controller Visualizer Launcher
# This script provides an easy way to launch the LQR visualizer tools

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

show_help() {
    cat << EOF
LQR Controller Visualizer Launcher

Usage: $0 [OPTIONS] [MODE]

MODES:
    standalone    Launch standalone visualizer with synthetic data (default)
    ros2         Launch ROS2-connected visualizer (requires active ROS2 environment)
    install      Install Python dependencies
    test         Test dependencies and run basic checks

OPTIONS:
    -h, --help   Show this help message

EXAMPLES:
    $0                    # Launch standalone visualizer
    $0 standalone         # Same as above
    $0 ros2              # Launch ROS2 visualizer
    $0 install           # Install dependencies
    $0 test              # Test installation

EOF
}

install_dependencies() {
    echo "Installing LQR Visualizer dependencies..."
    
    # Check if pip is available
    if ! command -v pip3 &> /dev/null; then
        echo "Error: pip3 not found. Please install pip3 first."
        exit 1
    fi
    
    # Install basic dependencies
    echo "Installing matplotlib and numpy..."
    pip3 install matplotlib numpy
    
    echo "Dependencies installed successfully!"
    echo "Note: For ROS2 mode, ensure ROS2 is properly installed and sourced."
}

test_installation() {
    echo "Testing LQR Visualizer installation..."
    
    # Test Python packages
    echo "Checking Python packages..."
    python3 -c "import matplotlib, numpy, tkinter; print('✓ Core packages available')" || {
        echo "✗ Missing required packages. Run '$0 install' to install them."
        exit 1
    }
    
    # Test scripts exist
    echo "Checking script files..."
    if [[ -f "$SCRIPT_DIR/lqr_visualizer_standalone.py" ]]; then
        echo "✓ Standalone visualizer script found"
    else
        echo "✗ Standalone script not found"
        exit 1
    fi
    
    if [[ -f "$SCRIPT_DIR/lqr_visualizer.py" ]]; then
        echo "✓ ROS2 visualizer script found"
    else
        echo "✗ ROS2 script not found"
        exit 1
    fi
    
    # Test standalone script syntax
    echo "Testing script syntax..."
    python3 -m py_compile "$SCRIPT_DIR/lqr_visualizer_standalone.py" && echo "✓ Standalone script syntax OK"
    python3 -m py_compile "$SCRIPT_DIR/lqr_visualizer.py" && echo "✓ ROS2 script syntax OK"
    
    echo "All tests passed! ✓"
}

launch_standalone() {
    echo "Launching LQR Controller Visualizer (Standalone Mode)..."
    echo "This will show synthetic data for demonstration purposes."
    echo "Press Ctrl+C to exit."
    echo ""
    
    cd "$SCRIPT_DIR"
    python3 lqr_visualizer_standalone.py
}

launch_ros2() {
    echo "Launching LQR Controller Visualizer (ROS2 Mode)..."
    echo "Make sure the LQR controller node is running."
    echo "Press Ctrl+C to exit."
    echo ""
    
    # Check if ROS2 is sourced
    if [[ -z "${ROS_DISTRO}" ]]; then
        echo "Warning: ROS_DISTRO not set. Make sure ROS2 is sourced:"
        echo "  source /opt/ros/humble/setup.bash"
        echo ""
    fi
    
    cd "$SCRIPT_DIR"
    python3 lqr_visualizer.py
}

# Parse command line arguments
MODE="standalone"  # default mode

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        standalone|ros2|install|test)
            MODE="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use '$0 --help' for usage information."
            exit 1
            ;;
    esac
done

# Execute the requested mode
case $MODE in
    standalone)
        launch_standalone
        ;;
    ros2)
        launch_ros2
        ;;
    install)
        install_dependencies
        ;;
    test)
        test_installation
        ;;
    *)
        echo "Invalid mode: $MODE"
        show_help
        exit 1
        ;;
esac
