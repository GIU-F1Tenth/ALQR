# LQR Controller Configuration

# Vehicle Parameters
wheelbase= 0.33
dt= 0.05

# Control Limits
max_acceleration= 5.0
max_deceleration= 5.0
max_steering_angle= 0.5
min_speed= 0.1
max_speed= 8.0

# LQR Cost Function Weights
# Higher weights penalize deviations more heavily
#lqr_weights=
position_weight= 10.0
velocity_weight= 1.0
heading_weight= 5.0
acceleration_weight= 0.1
steering_weight= 1.0

# Control Parameters
control_hz= 20.0
lookahead_distance= 0.5
enable_feedforward= True

# Safety Parameters
enable_safety_checks= True
safety_timeout= 1.0
emergency_brake_threshold= 2.0

# ROS2 Topics
odom_topic= "/car_state/odom"
reference_topic= "/horizon_mapper/reference_trajectory"
status_topic= "/horizon_mapper/path_ready"
control_topic= "/drive"
pose_estimate_topic= "/initialpose"

# Quality of Service
qos_depth= 10  # ROS2 QoS queue depth

# Logging and Debug
enable_logging= True
debug_logging_enabled= False
performance_logging_enabled= True
log_frequency_divider= 10
