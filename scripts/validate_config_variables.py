#!/usr/bin/env python3

"""
Configuration Variable Matching Validator

This script creates a comprehensive report of variable matching between
config.py and lqr_node.py expectations.
"""

import sys
import os

# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
config_path = os.path.join(project_root, 'config')

# Add config to path
sys.path.insert(0, config_path)

def validate_config():
    """Validate configuration variables."""
    
    results = []
    results.append("=" * 60)
    results.append("LQR CONTROLLER CONFIGURATION VALIDATION")
    results.append("=" * 60)
    results.append("")
    
    try:
        import config
        results.append("✅ SUCCESS: Config module imported successfully")
        results.append(f"   Config path: {config_path}")
        results.append("")
        
        # Define expected variables
        expected_vars = {
            # Vehicle Parameters
            'wheelbase': (float, "Vehicle wheelbase in meters"),
            'dt': (float, "Control time step in seconds"),
            
            # Control Limits
            'max_acceleration': (float, "Maximum acceleration in m/s²"),
            'max_deceleration': (float, "Maximum deceleration in m/s²"),
            'max_steering_angle': (float, "Maximum steering angle in radians"),
            'min_speed': (float, "Minimum speed in m/s"),
            'max_speed': (float, "Maximum speed in m/s"),
            
            # LQR Cost Function Weights
            'position_weight': (float, "Position error weight"),
            'velocity_weight': (float, "Velocity error weight"),
            'heading_weight': (float, "Heading error weight"),
            'acceleration_weight': (float, "Acceleration control weight"),
            'steering_weight': (float, "Steering control weight"),
            
            # Control Parameters
            'control_hz': (float, "Control frequency in Hz"),
            'lookahead_distance': (float, "Lookahead distance in meters"),
            'enable_feedforward': (bool, "Enable feedforward control"),
            
            # Safety Parameters
            'enable_safety_checks': (bool, "Enable safety checking"),
            'safety_timeout': (float, "Safety timeout in seconds"),
            'emergency_brake_threshold': (float, "Emergency brake threshold"),
            
            # ROS2 Topics
            'odom_topic': (str, "Odometry topic name"),
            'reference_topic': (str, "Reference trajectory topic"),
            'status_topic': (str, "Path status topic"),
            'control_topic': (str, "Control command topic"),
            'pose_estimate_topic': (str, "Pose estimate topic"),
            
            # Quality of Service
            'qos_depth': (int, "ROS2 QoS queue depth"),
            
            # Logging and Debug
            'enable_logging': (bool, "Enable logging"),
            'debug_logging_enabled': (bool, "Enable debug logging"),
            'performance_logging_enabled': (bool, "Enable performance logging"),
            'log_frequency_divider': (int, "Log frequency divider")
        }
        
        results.append("VARIABLE VALIDATION RESULTS:")
        results.append("-" * 40)
        
        missing_vars = []
        type_mismatches = []
        successful_vars = []
        
        for var_name, (expected_type, description) in expected_vars.items():
            if hasattr(config, var_name):
                actual_value = getattr(config, var_name)
                actual_type = type(actual_value)
                
                if isinstance(actual_value, expected_type):
                    results.append(f"✅ {var_name:<25} = {actual_value} ({actual_type.__name__})")
                    successful_vars.append(var_name)
                else:
                    results.append(f"⚠️  {var_name:<25} = {actual_value} (type mismatch: got {actual_type.__name__}, expected {expected_type.__name__})")
                    type_mismatches.append((var_name, actual_type, expected_type))
            else:
                results.append(f"❌ {var_name:<25} = MISSING")
                missing_vars.append(var_name)
        
        results.append("")
        results.append("SUMMARY:")
        results.append("-" * 20)
        results.append(f"✅ Successful variables: {len(successful_vars)}")
        results.append(f"⚠️  Type mismatches: {len(type_mismatches)}")
        results.append(f"❌ Missing variables: {len(missing_vars)}")
        results.append("")
        
        if missing_vars:
            results.append("MISSING VARIABLES:")
            for var in missing_vars:
                expected_type, description = expected_vars[var]
                results.append(f"  - {var} ({expected_type.__name__}): {description}")
            results.append("")
        
        if type_mismatches:
            results.append("TYPE MISMATCHES:")
            for var, actual, expected in type_mismatches:
                results.append(f"  - {var}: got {actual.__name__}, expected {expected.__name__}")
            results.append("")
        
        # Test ConfigWrapper creation
        results.append("TESTING CONFIGWRAPPER:")
        results.append("-" * 25)
        
        class ConfigWrapper:
            def __init__(self, config_module):
                # Core variables test
                self.wheelbase = getattr(config_module, 'wheelbase', 0.33)
                self.max_acceleration = getattr(config_module, 'max_acceleration', 5.0)
                self.position_weight = getattr(config_module, 'position_weight', 10.0)
                self.velocity_weight = getattr(config_module, 'velocity_weight', 1.0)
                self.odom_topic = getattr(config_module, 'odom_topic', "/car_state/odom")
        
        try:
            wrapper = ConfigWrapper(config)
            results.append("✅ ConfigWrapper created successfully")
            results.append(f"   wheelbase: {wrapper.wheelbase}")
            results.append(f"   max_acceleration: {wrapper.max_acceleration}")
            results.append(f"   position_weight: {wrapper.position_weight}")
            results.append(f"   velocity_weight: {wrapper.velocity_weight}")
            results.append(f"   odom_topic: {wrapper.odom_topic}")
        except Exception as e:
            results.append(f"❌ ConfigWrapper creation failed: {e}")
        
        results.append("")
        
        if len(missing_vars) == 0 and len(type_mismatches) == 0:
            results.append("🎉 ALL TESTS PASSED! Configuration is fully compatible.")
        else:
            results.append("⚠️  Some issues found. Please review the missing variables and type mismatches above.")
            
    except ImportError as e:
        results.append(f"❌ CRITICAL ERROR: Failed to import config module")
        results.append(f"   Error: {e}")
        results.append(f"   Config path attempted: {config_path}")
        results.append(f"   Path exists: {os.path.exists(config_path)}")
        if os.path.exists(config_path):
            results.append(f"   Contents: {os.listdir(config_path)}")
    except Exception as e:
        results.append(f"❌ UNEXPECTED ERROR: {e}")
        import traceback
        results.append(traceback.format_exc())
    
    return results

def main():
    """Main function to run validation and save results."""
    
    results = validate_config()
    
    # Write results to file
    output_file = os.path.join(os.path.dirname(__file__), 'config_validation_report.txt')
    with open(output_file, 'w') as f:
        f.write('\n'.join(results))
    
    # Also print to console
    for line in results:
        print(line)
    
    print(f"\nFull report saved to: {output_file}")

if __name__ == '__main__':
    main()
