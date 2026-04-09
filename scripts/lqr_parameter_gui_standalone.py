#!/usr/bin/env python3

"""
Standalone LQR Parameter Tuning GUI

A simplified version of the LQR parameter tuning GUI that works without ROS2
for testing parameter configurations offline.

Author: Mohammed Azab <mohammed@azab.io>
License: MIT
Version: 1.0.0
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import threading
import time
import sys
import os

# Import config for default values
def _load_config():
    """Load configuration module safely."""
    try:
        import importlib.util
        # Use YAML configuration instead of config.py
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'lqr_params.yaml')
        spec = importlib.util.spec_from_file_location("lqr_config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module
    except Exception:
        return None

config = _load_config()
CONFIG_AVAILABLE = config is not None


class SimpleLQRParameterGUI:
    """Simplified GUI for LQR parameter tuning without ROS2 dependency."""
    
    def __init__(self, master):
        self.master = master
        self.master.title("LQR Parameter Tuning (Standalone) - F1TENTH")
        self.master.geometry("1400x900")
        self.master.configure(bg='#f0f0f0')
        
        # Parameter storage
        self.parameters = self._load_default_parameters()
        self.parameter_widgets = {}
        
        # Simulation data for demonstration
        self.simulation_running = False
        self.simulation_data = {
            'time': [],
            'lateral_error': [],
            'heading_error': [],
            'velocity_error': [],
            'control_cost': [],
            'state_cost': []
        }
        
        # Create GUI
        self._create_widgets()
        self._setup_plots()
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.simulation_thread.start()
    
    def _load_default_parameters(self):
        """Load default parameters from config or use hardcoded values."""
        if CONFIG_AVAILABLE:
            return {
                # Vehicle Parameters
                'wheelbase': config.wheelbase,
                'dt': config.dt,
                
                # Control Limits
                'max_acceleration': config.max_acceleration,
                'max_deceleration': config.max_deceleration,
                'max_steering_angle': config.max_steering_angle,
                'min_speed': config.min_speed,
                'max_speed': config.max_speed,
                
                # LQR Q Matrix Weights (State Cost)
                'q_position_x': config.position_weight,
                'q_position_y': config.position_weight,
                'q_velocity': config.velocity_weight,
                'q_heading': config.heading_weight,
                
                # LQR R Matrix Weights (Control Cost)
                'r_acceleration': config.acceleration_weight,
                'r_steering': config.steering_weight,
                
                # Control Parameters
                'control_hz': config.control_hz,
                'lookahead_distance': config.lookahead_distance,
                'enable_feedforward': config.enable_feedforward,
                
                # Anti-Wobble Parameters
                'min_lookahead_distance': config.min_lookahead_distance,
                'max_lookahead_distance': config.max_lookahead_distance,
                'lookahead_time': config.lookahead_time,
                'enable_steering_rate_limit': config.enable_steering_rate_limit,
                'max_steering_rate': config.max_steering_rate,
                
                # Curve Detection
                'enable_curve_detection': config.enable_curve_detection,
                'curve_lookahead_points': config.curve_lookahead_points,
                'max_curvature_threshold': config.max_curvature_threshold,
                'curve_speed_factor': config.curve_speed_factor,
                
                # Safety Parameters
                'enable_safety_checks': config.enable_safety_checks,
                'safety_timeout': config.safety_timeout,
                'emergency_brake_threshold': config.emergency_brake_threshold,
            }
        else:
            # Hardcoded defaults
            return {
                'wheelbase': 0.33, 'dt': 0.05, 'max_acceleration': 5.0,
                'max_deceleration': 9.0, 'max_steering_angle': 0.9,
                'min_speed': 0.1, 'max_speed': 15.0, 'q_position_x': 5.0,
                'q_position_y': 5.0, 'q_velocity': 1.0, 'q_heading': 6.0,
                'r_acceleration': 0.3, 'r_steering': 4.0, 'control_hz': 20.0,
                'lookahead_distance': 1.5, 'enable_feedforward': True,
                'min_lookahead_distance': 0.7, 'max_lookahead_distance': 2.5,
                'lookahead_time': 0.8, 'enable_steering_rate_limit': True,
                'max_steering_rate': 1.5, 'enable_curve_detection': True,
                'curve_lookahead_points': 5, 'max_curvature_threshold': 1.0,
                'curve_speed_factor': 0.7, 'enable_safety_checks': False,
                'safety_timeout': 1.0, 'emergency_brake_threshold': 2.0,
            }
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        # Create main paned window
        main_paned = ttk.PanedWindow(self.master, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for parameters
        left_frame = ttk.Frame(main_paned, width=600)
        main_paned.add(left_frame, weight=1)
        
        # Right panel for plots
        right_frame = ttk.Frame(main_paned, width=800)
        main_paned.add(right_frame, weight=1)
        
        # Create parameter controls in left panel
        self._create_left_panel(left_frame)
        
        # Create plots in right panel
        self._create_right_panel(right_frame)
    
    def _create_left_panel(self, parent):
        """Create the left panel with parameter controls."""
        # Create scrollable frame
        canvas = tk.Canvas(parent, bg='#f0f0f0')
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Control buttons
        self._create_control_buttons(scrollable_frame)
        
        # Parameter sections
        self._create_lqr_section(scrollable_frame)
        self._create_control_limits_section(scrollable_frame)
        self._create_vehicle_section(scrollable_frame)
        self._create_anti_wobble_section(scrollable_frame)
        self._create_curve_detection_section(scrollable_frame)
        self._create_safety_section(scrollable_frame)
        
        # Current matrices display
        self._create_matrices_display(scrollable_frame)
        
        # Bind mousewheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    def _create_right_panel(self, parent):
        """Create the right panel with plots and simulation controls."""
        # Simulation controls
        control_frame = ttk.LabelFrame(parent, text="Simulation Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=5)
        
        self.sim_button = ttk.Button(
            control_frame,
            text="Start Simulation",
            command=self._toggle_simulation
        )
        self.sim_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame,
            text="Clear Data",
            command=self._clear_simulation_data
        ).pack(side=tk.LEFT, padx=5)
        
        # Simulation status
        self.sim_status_label = tk.Label(
            control_frame,
            text="Simulation: Stopped",
            fg="red",
            font=("Arial", 10, "bold")
        )
        self.sim_status_label.pack(side=tk.RIGHT)
        
        # Plot frame
        self.plot_frame = ttk.Frame(parent)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def _create_control_buttons(self, parent):
        """Create control buttons."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Save Config", command=self._save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Config", command=self._load_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset Defaults", command=self._reset_defaults).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export to Python", command=self._export_to_python).pack(side=tk.RIGHT, padx=5)
    
    def _create_lqr_section(self, parent):
        """Create LQR weights section."""
        lqr_frame = ttk.LabelFrame(parent, text="LQR Cost Matrices", padding=10)
        lqr_frame.pack(fill=tk.X, pady=5)
        
        # Q Matrix
        q_frame = ttk.LabelFrame(lqr_frame, text="Q Matrix - State Cost Weights", padding=5)
        q_frame.pack(fill=tk.X, pady=5)
        
        q_params = [
            ('q_position_x', 'Position X Weight', 0.1, 50.0, 0.1),
            ('q_position_y', 'Position Y Weight', 0.1, 50.0, 0.1),
            ('q_velocity', 'Velocity Weight', 0.1, 10.0, 0.1),
            ('q_heading', 'Heading Weight', 0.1, 20.0, 0.1),
        ]
        
        for param, label, min_val, max_val, resolution in q_params:
            self._create_parameter_slider(q_frame, param, label, min_val, max_val, resolution)
        
        # R Matrix
        r_frame = ttk.LabelFrame(lqr_frame, text="R Matrix - Control Cost Weights", padding=5)
        r_frame.pack(fill=tk.X, pady=5)
        
        r_params = [
            ('r_acceleration', 'Acceleration Weight', 0.01, 5.0, 0.01),
            ('r_steering', 'Steering Weight', 0.1, 10.0, 0.1),
        ]
        
        for param, label, min_val, max_val, resolution in r_params:
            self._create_parameter_slider(r_frame, param, label, min_val, max_val, resolution)
    
    def _create_control_limits_section(self, parent):
        """Create control limits section."""
        limits_frame = ttk.LabelFrame(parent, text="Control Limits", padding=10)
        limits_frame.pack(fill=tk.X, pady=5)
        
        limits_params = [
            ('max_acceleration', 'Max Acceleration (m/s²)', 1.0, 15.0, 0.1),
            ('max_deceleration', 'Max Deceleration (m/s²)', 1.0, 15.0, 0.1),
            ('max_steering_angle', 'Max Steering Angle (rad)', 0.1, 1.5, 0.01),
            ('max_speed', 'Max Speed (m/s)', 1.0, 25.0, 0.1),
        ]
        
        for param, label, min_val, max_val, resolution in limits_params:
            self._create_parameter_slider(limits_frame, param, label, min_val, max_val, resolution)
    
    def _create_vehicle_section(self, parent):
        """Create vehicle parameters section."""
        vehicle_frame = ttk.LabelFrame(parent, text="Vehicle Parameters", padding=10)
        vehicle_frame.pack(fill=tk.X, pady=5)
        
        vehicle_params = [
            ('wheelbase', 'Wheelbase (m)', 0.1, 1.0, 0.01),
            ('dt', 'Time Step (s)', 0.01, 0.2, 0.001),
            ('lookahead_distance', 'Lookahead Distance (m)', 0.1, 5.0, 0.1),
        ]
        
        for param, label, min_val, max_val, resolution in vehicle_params:
            self._create_parameter_slider(vehicle_frame, param, label, min_val, max_val, resolution)
    
    def _create_anti_wobble_section(self, parent):
        """Create anti-wobble section."""
        wobble_frame = ttk.LabelFrame(parent, text="Anti-Wobble Parameters", padding=10)
        wobble_frame.pack(fill=tk.X, pady=5)
        
        wobble_params = [
            ('max_steering_rate', 'Max Steering Rate (rad/s)', 0.1, 5.0, 0.1),
            ('min_lookahead_distance', 'Min Lookahead (m)', 0.1, 2.0, 0.1),
            ('max_lookahead_distance', 'Max Lookahead (m)', 1.0, 5.0, 0.1),
        ]
        
        for param, label, min_val, max_val, resolution in wobble_params:
            self._create_parameter_slider(wobble_frame, param, label, min_val, max_val, resolution)
    
    def _create_curve_detection_section(self, parent):
        """Create curve detection section."""
        curve_frame = ttk.LabelFrame(parent, text="Curve Detection", padding=10)
        curve_frame.pack(fill=tk.X, pady=5)
        
        curve_params = [
            ('max_curvature_threshold', 'Curvature Threshold', 0.1, 5.0, 0.1),
            ('curve_speed_factor', 'Speed Factor', 0.1, 1.0, 0.01),
        ]
        
        for param, label, min_val, max_val, resolution in curve_params:
            self._create_parameter_slider(curve_frame, param, label, min_val, max_val, resolution)
    
    def _create_safety_section(self, parent):
        """Create safety section."""
        safety_frame = ttk.LabelFrame(parent, text="Safety Parameters", padding=10)
        safety_frame.pack(fill=tk.X, pady=5)
        
        safety_params = [
            ('safety_timeout', 'Safety Timeout (s)', 0.1, 5.0, 0.1),
            ('emergency_brake_threshold', 'Emergency Brake (m/s²)', 1.0, 10.0, 0.1),
        ]
        
        for param, label, min_val, max_val, resolution in safety_params:
            self._create_parameter_slider(safety_frame, param, label, min_val, max_val, resolution)
    
    def _create_matrices_display(self, parent):
        """Create current matrices display."""
        matrix_frame = ttk.LabelFrame(parent, text="Current LQR Matrices", padding=10)
        matrix_frame.pack(fill=tk.X, pady=5)
        
        # Q Matrix display
        q_display_frame = ttk.Frame(matrix_frame)
        q_display_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(q_display_frame, text="Q Matrix:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.q_matrix_text = tk.Text(q_display_frame, height=5, width=50, font=("Courier", 9))
        self.q_matrix_text.pack(fill=tk.X)
        
        # R Matrix display
        r_display_frame = ttk.Frame(matrix_frame)
        r_display_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(r_display_frame, text="R Matrix:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.r_matrix_text = tk.Text(r_display_frame, height=3, width=50, font=("Courier", 9))
        self.r_matrix_text.pack(fill=tk.X)
        
        # Update matrices initially
        self._update_matrix_display()
    
    def _create_parameter_slider(self, parent, param_name, label, min_val, max_val, resolution):
        """Create parameter slider."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(frame, text=f"{label}:", width=25).pack(side=tk.LEFT)
        
        value_var = tk.DoubleVar(value=self.parameters.get(param_name, min_val))
        value_label = tk.Label(frame, textvariable=value_var, width=8, relief=tk.SUNKEN)
        value_label.pack(side=tk.RIGHT, padx=5)
        
        slider = tk.Scale(
            frame, from_=min_val, to=max_val, resolution=resolution,
            orient=tk.HORIZONTAL, variable=value_var,
            command=lambda val, param=param_name: self._on_parameter_change(param, float(val))
        )
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        self.parameter_widgets[param_name] = {'slider': slider, 'value_var': value_var}
    
    def _setup_plots(self):
        """Setup matplotlib plots."""
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('LQR Controller Performance Monitoring', fontsize=14)
        
        # Configure subplots
        self.ax1.set_title('Lateral Error')
        self.ax1.set_ylabel('Error (m)')
        self.ax1.grid(True)
        
        self.ax2.set_title('Heading Error')
        self.ax2.set_ylabel('Error (rad)')
        self.ax2.grid(True)
        
        self.ax3.set_title('Control Cost')
        self.ax3.set_ylabel('Cost')
        self.ax3.set_xlabel('Time (s)')
        self.ax3.grid(True)
        
        self.ax4.set_title('State Cost')
        self.ax4.set_ylabel('Cost')
        self.ax4.set_xlabel('Time (s)')
        self.ax4.grid(True)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize empty plots
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=2)
        self.line2, = self.ax2.plot([], [], 'r-', linewidth=2)
        self.line3, = self.ax3.plot([], [], 'g-', linewidth=2)
        self.line4, = self.ax4.plot([], [], 'm-', linewidth=2)
        
        # Animation
        self.anim = animation.FuncAnimation(
            self.fig, self._update_plots, interval=100, blit=False
        )
    
    def _on_parameter_change(self, param_name, value):
        """Handle parameter changes."""
        self.parameters[param_name] = value
        self._update_matrix_display()
    
    def _update_matrix_display(self):
        """Update the matrix display."""
        # Q Matrix
        Q = np.diag([
            self.parameters['q_position_x'],
            self.parameters['q_position_y'], 
            self.parameters['q_velocity'],
            self.parameters['q_heading']
        ])
        
        # R Matrix
        R = np.diag([
            self.parameters['r_acceleration'],
            self.parameters['r_steering']
        ])
        
        # Update text widgets
        self.q_matrix_text.delete(1.0, tk.END)
        self.q_matrix_text.insert(1.0, f"Q = \n{Q}")
        
        self.r_matrix_text.delete(1.0, tk.END)
        self.r_matrix_text.insert(1.0, f"R = \n{R}")
    
    def _toggle_simulation(self):
        """Toggle simulation."""
        self.simulation_running = not self.simulation_running
        
        if self.simulation_running:
            self.sim_button.config(text="Stop Simulation")
            self.sim_status_label.config(text="Simulation: Running", fg="green")
        else:
            self.sim_button.config(text="Start Simulation")
            self.sim_status_label.config(text="Simulation: Stopped", fg="red")
    
    def _clear_simulation_data(self):
        """Clear simulation data."""
        for key in self.simulation_data:
            self.simulation_data[key].clear()
    
    def _simulation_loop(self):
        """Simulation loop for generating demo data."""
        t = 0
        while True:
            if self.simulation_running:
                # Generate simulated data based on current parameters
                # This would be replaced with real data from the LQR controller
                
                # Simple simulation that shows the effect of parameter changes
                q_sum = (self.parameters['q_position_x'] + self.parameters['q_position_y'] + 
                        self.parameters['q_velocity'] + self.parameters['q_heading'])
                r_sum = self.parameters['r_acceleration'] + self.parameters['r_steering']
                
                # Simulate tracking errors (lower Q weights = higher errors)
                lateral_error = 0.1 * np.sin(t * 0.5) / (q_sum / 20.0) + 0.02 * np.random.randn()
                heading_error = 0.05 * np.cos(t * 0.3) / (self.parameters['q_heading'] / 6.0) + 0.01 * np.random.randn()
                velocity_error = 0.2 * np.sin(t * 0.2) / (self.parameters['q_velocity'] / 1.0) + 0.05 * np.random.randn()
                
                # Simulate costs
                control_cost = r_sum * (0.5 + 0.3 * np.sin(t * 0.8))
                state_cost = q_sum * (lateral_error**2 + heading_error**2 + velocity_error**2)
                
                # Store data
                self.simulation_data['time'].append(t)
                self.simulation_data['lateral_error'].append(lateral_error)
                self.simulation_data['heading_error'].append(heading_error)
                self.simulation_data['velocity_error'].append(velocity_error)
                self.simulation_data['control_cost'].append(control_cost)
                self.simulation_data['state_cost'].append(state_cost)
                
                # Keep only last 100 points
                for key in self.simulation_data:
                    if len(self.simulation_data[key]) > 100:
                        self.simulation_data[key].pop(0)
                
                t += self.parameters['dt']
            
            time.sleep(0.1)
    
    def _update_plots(self, frame):
        """Update plots with simulation data."""
        if not self.simulation_data['time']:
            return
        
        time_data = self.simulation_data['time']
        
        # Update each plot
        self.line1.set_data(time_data, self.simulation_data['lateral_error'])
        self.line2.set_data(time_data, self.simulation_data['heading_error'])
        self.line3.set_data(time_data, self.simulation_data['control_cost'])
        self.line4.set_data(time_data, self.simulation_data['state_cost'])
        
        # Update axis limits
        if len(time_data) > 1:
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.relim()
                ax.autoscale_view()
        
        self.canvas.draw()
    
    def _save_config(self):
        """Save configuration to file."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    yaml.dump(self.parameters, f, default_flow_style=False)
                messagebox.showinfo("Success", f"Configuration saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")
    
    def _load_config(self):
        """Load configuration from file."""
        filename = filedialog.askopenfilename(
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    loaded_params = yaml.safe_load(f)
                
                self.parameters.update(loaded_params)
                self._update_gui_from_parameters()
                messagebox.showinfo("Success", f"Configuration loaded from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load: {e}")
    
    def _reset_defaults(self):
        """Reset to default parameters."""
        if messagebox.askyesno("Confirm", "Reset all parameters to defaults?"):
            self.parameters = self._load_default_parameters()
            self._update_gui_from_parameters()
    
    def _update_gui_from_parameters(self):
        """Update GUI from current parameters."""
        for param_name, widgets in self.parameter_widgets.items():
            if param_name in self.parameters:
                widgets['value_var'].set(self.parameters[param_name])
        self._update_matrix_display()
    
    def _export_to_python(self):
        """Export current parameters to Python config file."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".py",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write("# LQR Controller Configuration\n")
                    f.write("# Generated from LQR Parameter GUI\n\n")
                    
                    for param, value in self.parameters.items():
                        if isinstance(value, bool):
                            f.write(f"{param} = {value}\n")
                        elif isinstance(value, (int, float)):
                            f.write(f"{param} = {value}\n")
                        else:
                            f.write(f"{param} = '{value}'\n")
                
                messagebox.showinfo("Success", f"Python config exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {e}")


def main():
    """Main entry point."""
    root = tk.Tk()
    app = SimpleLQRParameterGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted")


if __name__ == "__main__":
    main()
