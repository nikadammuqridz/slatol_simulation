import pybullet as p
import pybullet_data
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import threading
import queue

class SLATOLSimulator:
    """Main simulation class with interactive GUI"""
    
    def __init__(self, gui=True):
        self.gui = gui
        self.sim_time = 0
        self.dt = 1/240  # Simulation timestep
        
        # Physics client
        if gui:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)
        
        # Configure simulation
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(self.dt, physicsClientId=self.client_id)
        p.setRealTimeSimulation(0)
        
        # Create ground
        self.ground_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        
        # Robot and controller
        self.robot = None
        self.controller = None
        
        # Simulation state
        self.is_running = False
        self.crash_detected = False
        self.data_queue = queue.Queue()
        
        # GUI elements
        self.fig = None
        self.ax = None
        self.pitch_line = None
        
        # Parameters
        self.leg_mass_ratio = 0.15  # μ
        self.wind_ratio = 0.0  # η
        self.target_omega = 15.0  # ω_n [rad/s]
    
    def initialize_robot(self):
        """Initialize or reinitialize robot with current parameters"""
        if self.robot:
            # Remove old robot
            p.removeBody(self.robot.body_id, physicsClientId=self.client_id)
        
        # Create new robot with current parameters
        self.robot = SLATOLRobot(self.client_id, self.leg_mass_ratio)
        self.controller = SLATOLController(self.robot, self.target_omega)
        
        # Reset simulation time
        self.sim_time = 0
        self.crash_detected = False
    
    def run_simulation_step(self):
        """Run a single simulation step"""
        if not self.is_running or self.crash_detected:
            return
        
        # Get robot state
        robot_state = self.robot.get_state()
        
        # Calculate wind force: F_wind = η * m_total * g
        wind_force = self.wind_ratio * self.robot.total_mass * 9.81
        
        # Apply wind disturbance (if any)
        if abs(wind_force) > 0.001:
            self.robot.apply_disturbance(wind_force, 0, 0.1)
        
        # Compute control
        tau_hip, tau_knee = self.controller.compute_control(
            self.dt, robot_state, wind_force
        )
        
        # Apply control
        self.robot.apply_joint_torques(tau_hip, tau_knee)
        
        # Step simulation
        p.stepSimulation(physicsClientId=self.client_id)
        
        # Check for crash (body pitch > 45°)
        if abs(robot_state['body_pitch']) > np.pi/4:  # 45 degrees
            self.crash_detected = True
            print(f"CRASH DETECTED at t={self.sim_time:.2f}s")
            # Color robot red
            p.changeVisualShape(
                self.robot.body_id, -1,
                rgbaColor=[1, 0, 0, 1],
                physicsClientId=self.client_id
            )
        
        # Collect data for plotting
        self.data_queue.put({
            'time': self.sim_time,
            'pitch': robot_state['body_pitch'],
            'state': self.controller.state,
            'wind_ratio': self.wind_ratio,
            'mass_ratio': self.leg_mass_ratio,
            'crashed': self.crash_detected
        })
        
        self.sim_time += self.dt
    
    def run_simulation(self, duration=10.0):
        """Run simulation for specified duration"""
        self.is_running = True
        start_time = time.time()
        
        while self.is_running and time.time() - start_time < duration:
            self.run_simulation_step()
            
            # Small delay for GUI
            if self.gui:
                time.sleep(self.dt)
        
        self.is_running = False
    
    def create_gui(self):
        """Create interactive GUI for parameter tuning"""
        self.fig, self.ax = plt.subplots(2, 2, figsize=(12, 8))
        plt.subplots_adjust(left=0.1, bottom=0.3, right=0.95, top=0.95)
        
        # Pitch plot
        self.ax[0, 0].set_title('Body Pitch Angle')
        self.ax[0, 0].set_xlabel('Time [s]')
        self.ax[0, 0].set_ylabel('Pitch [rad]')
        self.ax[0, 0].grid(True)
        self.ax[0, 0].set_ylim(-np.pi/2, np.pi/2)
        self.pitch_line, = self.ax[0, 0].plot([], [], 'b-', linewidth=2)
        
        # State indicator
        self.ax[0, 1].set_title('Controller State')
        self.ax[0, 1].set_xlabel('Time [s]')
        self.ax[0, 1].set_ylabel('State')
        self.ax[0, 1].grid(True)
        self.ax[0, 1].set_ylim(0, 4)
        self.state_line, = self.ax[0, 1].plot([], [], 'g-', linewidth=2)
        
        # Parameter history
        self.ax[1, 0].set_title('Mass Ratio (μ) History')
        self.ax[1, 0].set_xlabel('Time [s]')
        self.ax[1, 0].set_ylabel('μ')
        self.ax[1, 0].grid(True)
        self.ax[1, 0].set_ylim(0.05, 0.4)
        self.mass_line, = self.ax[1, 0].plot([], [], 'r-', linewidth=2)
        
        # Wind ratio history
        self.ax[1, 1].set_title('Wind Ratio (η) History')
        self.ax[1, 1].set_xlabel('Time [s]')
        self.ax[1, 1].set_ylabel('η')
        self.ax[1, 1].grid(True)
        self.ax[1, 1].set_ylim(0, 0.5)
        self.wind_line, = self.ax[1, 1].plot([], [], 'orange', linewidth=2)
        
        # Create sliders
        slider_ax1 = plt.axes([0.1, 0.15, 0.65, 0.03])
        slider_ax2 = plt.axes([0.1, 0.10, 0.65, 0.03])
        slider_ax3 = plt.axes([0.1, 0.05, 0.65, 0.03])
        
        self.slider_mass = Slider(
            slider_ax1, 'Leg Mass Ratio (μ)', 0.05, 0.40, 
            valinit=self.leg_mass_ratio, valstep=0.01
        )
        self.slider_wind = Slider(
            slider_ax2, 'Wind Ratio (η)', 0.0, 0.5, 
            valinit=self.wind_ratio, valstep=0.01
        )
        self.slider_omega = Slider(
            slider_ax3, 'Natural Freq (ω)', 5.0, 30.0,
            valinit=self.target_omega, valstep=1.0
        )
        
        # Create buttons
        button_ax1 = plt.axes([0.8, 0.15, 0.1, 0.05])
        button_ax2 = plt.axes([0.8, 0.08, 0.1, 0.05])
        button_ax3 = plt.axes([0.8, 0.01, 0.1, 0.05])
        
        self.button_reset = Button(button_ax1, 'Reset & Jump')
        self.button_run = Button(button_ax2, 'Run Simulation')
        self.button_stop = Button(button_ax3, 'Stop')
        
        # Connect callbacks
        self.slider_mass.on_changed(self.update_mass_ratio)
        self.slider_wind.on_changed(self.update_wind_ratio)
        self.slider_omega.on_changed(self.update_omega)
        
        self.button_reset.on_clicked(self.reset_simulation)
        self.button_run.on_clicked(self.start_simulation)
        self.button_stop.on_clicked(self.stop_simulation)
        
        # Data storage
        self.time_data = []
        self.pitch_data = []
        self.state_data = []
        self.mass_data = []
        self.wind_data = []
        
        # Start data processing thread
        self.data_thread = threading.Thread(target=self.process_data)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        # Start GUI update thread
        self.gui_thread = threading.Thread(target=self.update_gui)
        self.gui_thread.daemon = True
        self.gui_thread.start()
        
        plt.show()
    
    def update_mass_ratio(self, val):
        """Update leg mass ratio"""
        self.leg_mass_ratio = val
        self.robot.update_mass_distribution(val)
        self.controller.update_adaptive_gains()
    
    def update_wind_ratio(self, val):
        """Update wind ratio"""
        self.wind_ratio = val
    
    def update_omega(self, val):
        """Update natural frequency"""
        self.target_omega = val
        self.controller.target_omega = val
        self.controller.update_adaptive_gains()
    
    def reset_simulation(self, event):
        """Reset simulation"""
        self.is_running = False
        time.sleep(0.1)
        
        # Clear data
        self.time_data.clear()
        self.pitch_data.clear()
        self.state_data.clear()
        self.mass_data.clear()
        self.wind_data.clear()
        
        # Reinitialize robot
        self.initialize_robot()
        
        print("Simulation reset")
    
    def start_simulation(self, event):
        """Start simulation"""
        if not self.is_running:
            self.is_running = True
            print("Starting simulation...")
            
            # Run simulation in background thread
            sim_thread = threading.Thread(
                target=self.run_simulation, 
                args=(30.0,)  # 30 second simulation
            )
            sim_thread.daemon = True
            sim_thread.start()
    
    def stop_simulation(self, event):
        """Stop simulation"""
        self.is_running = False
        print("Simulation stopped")
    
    def process_data(self):
        """Process incoming data from simulation"""
        while True:
            try:
                data = self.data_queue.get(timeout=0.1)
                
                self.time_data.append(data['time'])
                self.pitch_data.append(data['pitch'])
                
                # Convert state to numeric for plotting
                state_map = {
                    "STANCE_COMPRESSION": 1,
                    "STANCE_THRUST": 2,
                    "FLIGHT_SWING": 3,
                    "FLIGHT_LANDING": 4
                }
                self.state_data.append(state_map.get(data['state'], 0))
                
                self.mass_data.append(data['mass_ratio'])
                self.wind_data.append(data['wind_ratio'])
                
            except queue.Empty:
                continue
    
    def update_gui(self):
        """Update GUI plots"""
        while True:
            if self.fig and len(self.time_data) > 0:
                # Update plots
                self.pitch_line.set_data(self.time_data, self.pitch_data)
                self.state_line.set_data(self.time_data, self.state_data)
                self.mass_line.set_data(self.time_data, self.mass_data)
                self.wind_line.set_data(self.time_data, self.wind_data)
                
                # Update axes limits
                if len(self.time_data) > 0:
                    self.ax[0, 0].set_xlim(0, max(self.time_data))
                    self.ax[0, 1].set_xlim(0, max(self.time_data))
                    self.ax[1, 0].set_xlim(0, max(self.time_data))
                    self.ax[1, 1].set_xlim(0, max(self.time_data))
                
                # Redraw
                self.fig.canvas.draw_idle()
            
            time.sleep(0.1)
    
    def close(self):
        """Cleanup"""
        self.is_running = False
        if self.gui:
            plt.close(self.fig)
        p.disconnect(physicsClientId=self.client_id)