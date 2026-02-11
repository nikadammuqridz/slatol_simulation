import pybullet as p
import pybullet_data
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import threading
import queue

class EnhancedSLATOLSimulator(SLATOLSimulator):
    """Enhanced simulator with auto-tuning visualization"""
    
    def create_gui(self):
        """Create advanced GUI with auto-tuning visualization"""
        self.fig = plt.figure(figsize=(16, 10))
        
        # Main plots
        gs = self.fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Plot 1: Body pitch
        self.ax1 = self.fig.add_subplot(gs[0, :2])
        self.ax1.set_title('Body Pitch Angle', fontsize=12, fontweight='bold')
        self.ax1.set_xlabel('Time [s]')
        self.ax1.set_ylabel('Pitch [rad]')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_ylim(-np.pi/2, np.pi/2)
        self.pitch_line, = self.ax1.plot([], [], 'b-', linewidth=2, label='Pitch')
        self.pitch_target_line, = self.ax1.plot([], [], 'r--', linewidth=1, label='Target (0)')
        self.ax1.legend()
        
        # Plot 2: Bandwidth tracking
        self.ax2 = self.fig.add_subplot(gs[0, 2:])
        self.ax2.set_title('Controller Bandwidth Tracking', fontsize=12, fontweight='bold')
        self.ax2.set_xlabel('Time [s]')
        self.ax2.set_ylabel('ω [rad/s]')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_ylim(0, 30)
        self.bandwidth_target_line, = self.ax2.plot([], [], 'k--', linewidth=2, label='Target ω_n')
        self.bandwidth_hip_line, = self.ax2.plot([], [], 'g-', linewidth=2, label='Hip ω_n')
        self.bandwidth_knee_line, = self.ax2.plot([], [], 'b-', linewidth=2, label='Knee ω_n')
        self.ax2.legend()
        
        # Plot 3: Gain evolution
        self.ax3 = self.fig.add_subplot(gs[1, :2])
        self.ax3.set_title('Auto-Tuned Gain Evolution', fontsize=12, fontweight='bold')
        self.ax3.set_xlabel('Leg Mass Ratio (μ)')
        self.ax3.set_ylabel('Gain Value')
        self.ax3.grid(True, alpha=0.3)
        self.ax3.set_xlim(0.05, 0.40)
        self.kp_hip_line, = self.ax3.plot([], [], 'g-', linewidth=2, label='Kp_hip')
        self.kp_knee_line, = self.ax3.plot([], [], 'b-', linewidth=2, label='Kp_knee')
        self.kd_hip_line, = self.ax3.plot([], [], 'g--', linewidth=1, label='Kd_hip')
        self.kd_knee_line, = self.ax3.plot([], [], 'b--', linewidth=1, label='Kd_knee')
        self.ax3.legend()
        
        # Plot 4: Theory vs Practice
        self.ax4 = self.fig.add_subplot(gs[1, 2:])
        self.ax4.set_title('Theory vs Practice: Kp ∝ μ', fontsize=12, fontweight='bold')
        self.ax4.set_xlabel('Leg Mass Ratio (μ)')
        self.ax4.set_ylabel('Kp / (m_total·L²·ω_n²/3)')
        self.ax4.grid(True, alpha=0.3)
        self.theory_line, = self.ax4.plot([], [], 'k-', linewidth=2, label='Theory: Kp ∝ μ')
        self.practice_line, = self.ax4.plot([], [], 'ro', markersize=6, label='Practice')
        self.ax4.legend()
        
        # Plot 5: Stability region (live update)
        self.ax5 = self.fig.add_subplot(gs[2, :2])
        self.ax5.set_title('Live Stability Region', fontsize=12, fontweight='bold')
        self.ax5.set_xlabel('Leg Mass Ratio (μ)')
        self.ax5.set_ylabel('Max Wind Ratio (η)')
        self.ax5.grid(True, alpha=0.3)
        self.ax5.set_xlim(0.05, 0.40)
        self.ax5.set_ylim(0, 0.5)
        self.stability_scatter = self.ax5.scatter([], [], c=[], s=50, alpha=0.6, 
                                                  cmap='RdYlGn', vmin=0, vmax=1)
        self.stability_boundary, = self.ax5.plot([], [], 'k-', linewidth=2)
        
        # Plot 6: Performance metrics
        self.ax6 = self.fig.add_subplot(gs[2, 2:])
        self.ax6.set_title('Performance Metrics', fontsize=12, fontweight='bold')
        self.ax6.axis('off')
        self.metrics_text = self.ax6.text(0.1, 0.9, '', transform=self.ax6.transAxes,
                                         fontsize=10, verticalalignment='top',
                                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Control panel
        control_ax = self.fig.add_axes([0.05, 0.02, 0.9, 0.12])
        control_ax.axis('off')
        
        # Sliders
        slider_y = 0.7
        slider_height = 0.2
        slider_width = 0.2
        
        # Leg mass ratio slider
        self.slider_mass_ax = self.fig.add_axes([0.05, slider_y, slider_width, slider_height])
        self.slider_mass = Slider(self.slider_mass_ax, 'μ (Leg Mass Ratio)', 
                                  0.05, 0.40, valinit=self.leg_mass_ratio, valstep=0.01)
        
        # Wind ratio slider
        self.slider_wind_ax = self.fig.add_axes([0.30, slider_y, slider_width, slider_height])
        self.slider_wind = Slider(self.slider_wind_ax, 'η (Wind Ratio)', 
                                  0.0, 0.5, valinit=self.wind_ratio, valstep=0.01)
        
        # Target omega slider
        self.slider_omega_ax = self.fig.add_axes([0.55, slider_y, slider_width, slider_height])
        self.slider_omega = Slider(self.slider_omega_ax, 'ω_n (Target Bandwidth)', 
                                   5.0, 30.0, valinit=self.target_omega, valstep=1.0)
        
        # Gain mode selector
        self.radio_ax = self.fig.add_axes([0.80, slider_y, 0.15, slider_height])
        self.radio = RadioButtons(self.radio_ax, ['Auto-Tuning', 'Fixed Gains'])
        
        # Buttons
        button_y = 0.35
        button_height = 0.1
        
        self.button_reset_ax = self.fig.add_axes([0.05, button_y, 0.1, button_height])
        self.button_run_ax = self.fig.add_axes([0.20, button_y, 0.1, button_height])
        self.button_stop_ax = self.fig.add_axes([0.35, button_y, 0.1, button_height])
        self.button_sweep_ax = self.fig.add_axes([0.50, button_y, 0.15, button_height])
        self.button_export_ax = self.fig.add_axes([0.70, button_y, 0.15, button_height])
        
        self.button_reset = Button(self.button_reset_ax, 'Reset')
        self.button_run = Button(self.button_run_ax, 'Run Test')
        self.button_stop = Button(self.button_stop_ax, 'Stop')
        self.button_sweep = Button(self.button_sweep_ax, 'Quick Sweep')
        self.button_export = Button(self.button_export_ax, 'Export Data')
        
        # Connect callbacks
        self.slider_mass.on_changed(self.update_mass_ratio)
        self.slider_wind.on_changed(self.update_wind_ratio)
        self.slider_omega.on_changed(self.update_omega)
        self.radio.on_clicked(self.toggle_gain_mode)
        
        self.button_reset.on_clicked(self.reset_simulation)
        self.button_run.on_clicked(self.run_test)
        self.button_stop.on_clicked(self.stop_simulation)
        self.button_sweep.on_clicked(self.run_quick_sweep)
        self.button_export.on_clicked(self.export_data)
        
        # Data storage
        self.time_data = []
        self.pitch_data = []
        self.bandwidth_data = []
        self.gain_data = []
        self.stability_data = []
        
        # Performance metrics
        self.use_auto_tuning = True
        self.test_history = []
        
        # Start update threads
        self.start_update_threads()
        
        plt.show()
    
    def toggle_gain_mode(self, label):
        """Toggle between auto-tuning and fixed gains"""
        self.use_auto_tuning = (label == 'Auto-Tuning')
        
        if self.controller:
            if self.use_auto_tuning:
                self.controller.update_adaptive_gains()
                print("Switched to AUTO-TUNING mode")
            else:
                # Fixed gains (typical values from literature)
                self.controller.Kp_hip = 1000.0
                self.controller.Kd_hip = 50.0
                self.controller.Kp_knee = 500.0
                self.controller.Kd_knee = 25.0
                print("Switched to FIXED GAINS mode")
    
    def run_simulation_step(self):
        """Enhanced simulation step with performance monitoring"""
        if not self.is_running or self.crash_detected:
            return
        
        # Get robot state
        robot_state = self.robot.get_state()
        
        # Calculate wind force
        wind_force = self.wind_ratio * self.robot.total_mass * 9.81
        
        # Apply wind
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
        
        # Get performance metrics
        metrics = self.controller.get_performance_metrics()
        
        # Check for crash
        if abs(robot_state['body_pitch']) > np.pi/4:
            self.crash_detected = True
            p.changeVisualShape(self.robot.body_id, -1,
                              rgbaColor=[1, 0, 0, 1],
                              physicsClientId=self.client_id)
        
        # Store data
        self.time_data.append(self.sim_time)
        self.pitch_data.append(robot_state['body_pitch'])
        
        if metrics:
            self.bandwidth_data.append({
                'time': self.sim_time,
                'target': metrics['target_bandwidth'],
                'hip': metrics['achieved_bandwidth_hip'],
                'knee': metrics['achieved_bandwidth_knee']
            })
            
            self.gain_data.append({
                'time': self.sim_time,
                'mu': self.leg_mass_ratio,
                'Kp_hip': metrics['current_gains']['Kp_hip'],
                'Kp_knee': metrics['current_gains']['Kp_knee'],
                'Kd_hip': metrics['current_gains']['Kd_hip'],
                'Kd_knee': metrics['current_gains']['Kd_knee']
            })
        
        self.sim_time += self.dt
    
    def update_gui(self):
        """Update GUI with real-time data"""
        while True:
            try:
                if self.fig and len(self.time_data) > 0:
                    # Update pitch plot
                    self.pitch_line.set_data(self.time_data, self.pitch_data)
                    if len(self.time_data) > 0:
                        self.ax1.set_xlim(0, max(self.time_data))
                    
                    # Update bandwidth plot
                    if self.bandwidth_data:
                        times = [d['time'] for d in self.bandwidth_data]
                        target = [d['target'] for d in self.bandwidth_data]
                        hip = [d['hip'] for d in self.bandwidth_data]
                        knee = [d['knee'] for d in self.bandwidth_data]
                        
                        self.bandwidth_target_line.set_data(times, target)
                        self.bandwidth_hip_line.set_data(times, hip)
                        self.bandwidth_knee_line.set_data(times, knee)
                        
                        if len(times) > 0:
                            self.ax2.set_xlim(0, max(times))
                    
                    # Update gain evolution
                    if self.gain_data:
                        mus = [d['mu'] for d in self.gain_data]
                        kp_hip = [d['Kp_hip'] for d in self.gain_data]
                        kp_knee = [d['Kp_knee'] for d in self.gain_data]
                        kd_hip = [d['Kd_hip'] for d in self.gain_data]
                        kd_knee = [d['Kd_knee'] for d in self.gain_data]
                        
                        self.kp_hip_line.set_data(mus, kp_hip)
                        self.kp_knee_line.set_data(mus, kp_knee)
                        self.kd_hip_line.set_data(mus, kd_hip)
                        self.kd_knee_line.set_data(mus, kd_knee)
                        
                        if len(mus) > 0:
                            self.ax3.set_xlim(min(mus), max(mus))
                            self.ax3.set_ylim(0, max(max(kp_hip), max(kp_knee)) * 1.1)
                    
                    # Update theory vs practice
                    if self.gain_data and len(self.gain_data) > 1:
                        mus = [d['mu'] for d in self.gain_data]
                        kp_hip = [d['Kp_hip'] for d in self.gain_data]
                        
                        # Theory: Kp = C·μ
                        C = (1/3) * self.robot.total_mass * self.robot.leg_length**2 * self.target_omega**2
                        theory_kp = [C * mu for mu in mus]
                        
                        self.theory_line.set_data(mus, theory_kp)
                        self.practice_line.set_data(mus, kp_hip)
                        
                        if len(mus) > 0:
                            self.ax4.set_xlim(min(mus), max(mus))
                            y_max = max(max(theory_kp), max(kp_hip)) * 1.1
                            self.ax4.set_ylim(0, y_max if y_max > 0 else 1)
                    
                    # Update metrics text
                    if self.controller and hasattr(self.controller, 'get_performance_metrics'):
                        metrics = self.controller.get_performance_metrics()
                        if metrics:
                            text = (f"Mode: {'AUTO-TUNING' if self.use_auto_tuning else 'FIXED'}\n"
                                   f"Target ω_n: {metrics['target_bandwidth']:.1f} rad/s\n"
                                   f"Hip ω_n: {metrics['achieved_bandwidth_hip']:.1f} rad/s\n"
                                   f"Knee ω_n: {metrics['achieved_bandwidth_knee']:.1f} rad/s\n"
                                   f"μ: {self.leg_mass_ratio:.3f}\n"
                                   f"η: {self.wind_ratio:.3f}\n"
                                   f"Kp_hip: {metrics['current_gains']['Kp_hip']:.1f}\n"
                                   f"Status: {'CRASHED' if self.crash_detected else 'STABLE'}")
                            self.metrics_text.set_text(text)
                    
                    # Redraw
                    self.fig.canvas.draw_idle()
                
                time.sleep(0.05)
                
            except Exception as e:
                print(f"GUI update error: {e}")
                time.sleep(0.1)
    
    def run_quick_sweep(self, event):
        """Run a quick parameter sweep for demonstration"""
        print("\n=== QUICK SWEEP STARTED ===")
        
        original_mu = self.leg_mass_ratio
        original_eta = self.wind_ratio
        
        test_points = [
            (0.1, 0.1), (0.1, 0.3),
            (0.2, 0.1), (0.2, 0.2),
            (0.3, 0.1), (0.3, 0.15),
            (0.4, 0.05), (0.4, 0.1)
        ]
        
        results = []
        
        for mu, eta in test_points:
            # Set parameters
            self.leg_mass_ratio = mu
            self.wind_ratio = eta
            self.robot.update_mass_distribution(mu)
            self.controller.update_adaptive_gains()
            
            # Run brief test
            self.reset_simulation(None)
            self.is_running = True
            test_duration = 2.0
            start_time = time.time()
            
            while time.time() - start_time < test_duration:
                self.run_simulation_step()
                time.sleep(self.dt)
            
            self.is_running = False
            
            # Record result
            results.append({
                'mu': mu,
                'eta': eta,
                'stable': not self.crash_detected,
                'max_pitch': max(abs(np.array(self.pitch_data))) if self.pitch_data else 0
            })
            
            print(f"μ={mu:.2f}, η={eta:.2f}: {'STABLE' if not self.crash_detected else 'UNSTABLE'}")
        
        # Restore original
        self.leg_mass_ratio = original_mu
        self.wind_ratio = original_eta
        self.robot.update_mass_distribution(original_mu)
        self.controller.update_adaptive_gains()
        self.reset_simulation(None)
        
        # Plot results
        self.plot_sweep_results(results)
        
        print("=== QUICK SWEEP COMPLETE ===")
    
    def plot_sweep_results(self, results):
        """Plot sweep results"""
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Stability region
        stable_mus = [r['mu'] for r in results if r['stable']]
        stable_etas = [r['eta'] for r in results if r['stable']]
        unstable_mus = [r['mu'] for r in results if not r['stable']]
        unstable_etas = [r['eta'] for r in results if not r['stable']]
        
        ax[0].scatter(stable_mus, stable_etas, c='green', s=100, label='Stable', alpha=0.7)
        ax[0].scatter(unstable_mus, unstable_etas, c='red', s=100, label='Unstable', alpha=0.7)
        ax[0].set_xlabel('Leg Mass Ratio (μ)')
        ax[0].set_ylabel('Wind Ratio (η)')
        ax[0].set_title('Stability Region - Auto-Tuning')
        ax[0].grid(True, alpha=0.3)
        ax[0].legend()
        
        # Max pitch vs μ
        for r in results:
            color = 'green' if r['stable'] else 'red'
            ax[1].scatter(r['mu'], r['max_pitch'], c=color, s=100, alpha=0.7)
        
        ax[1].set_xlabel('Leg Mass Ratio (μ)')
        ax[1].set_ylabel('Max Body Pitch [rad]')
        ax[1].set_title('Performance Degradation with μ')
        ax[1].grid(True, alpha=0.3)
        ax[1].axhline(y=np.pi/4, color='r', linestyle='--', alpha=0.5, label='Crash threshold')
        ax[1].legend()
        
        plt.tight_layout()
        plt.show()