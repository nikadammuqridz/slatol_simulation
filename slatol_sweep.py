import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures
import pickle

class StabilitySweep:
    """Automated parameter sweep for stability region mapping"""
    
    def __init__(self):
        self.results = []
        
        # Parameter ranges (from Eq. 3.12 analysis)
        self.mass_ratios = np.linspace(0.05, 0.40, 15)  # μ
        self.wind_ratios = np.linspace(0.0, 0.5, 20)    # η
        
        # Simulation parameters
        self.sim_duration = 5.0  # Simulation time per test [s]
        self.crash_threshold = np.pi/4  # 45 degrees
    
    def run_single_test(self, mu, eta):
        """
        Run a single stability test
        
        Returns:
            success: True if stable, False if crashed
            metrics: Dict of performance metrics
        """
        import pybullet as p
        import pybullet_data
        
        # Setup simulation
        client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=client_id)
        p.setTimeStep(1/240, physicsClientId=client_id)
        
        # Create ground
        p.loadURDF("plane.urdf", physicsClientId=client_id)
        
        # Import robot and controller
        from slatol_robot import SLATOLRobot
        from slatol_controller import SLATOLController
        
        # Create robot
        robot = SLATOLRobot(client_id, mu)
        controller = SLATOLController(robot, target_omega=15.0)
        
        # Run simulation
        max_pitch = 0
        max_roll = 0
        sim_time = 0
        dt = 1/240
        crashed = False
        
        for step in range(int(self.sim_duration / dt)):
            # Get state
            state = robot.get_state()
            
            # Calculate wind force
            wind_force = eta * robot.total_mass * 9.81
            
            # Apply wind
            if abs(wind_force) > 0.001:
                robot.apply_disturbance(wind_force, 0, dt)
            
            # Compute control
            tau_hip, tau_knee = controller.compute_control(dt, state, wind_force)
            
            # Apply control
            robot.apply_joint_torques(tau_hip, tau_knee)
            
            # Step simulation
            p.stepSimulation(physicsClientId=client_id)
            
            # Check for crash
            pitch = abs(state['body_pitch'])
            max_pitch = max(max_pitch, pitch)
            
            if pitch > self.crash_threshold:
                crashed = True
                break
            
            sim_time += dt
        
        # Cleanup
        p.disconnect(physicsClientId=client_id)
        
        # Calculate metrics
        metrics = {
            'mu': mu,
            'eta': eta,
            'stable': not crashed,
            'max_pitch': max_pitch,
            'sim_time': sim_time,
            'total_mass': robot.total_mass,
            'leg_inertia': robot.leg_inertia
        }
        
        return not crashed, metrics
    
    def run_sweep(self, num_workers=4):
        """Run parameter sweep using parallel processing"""
        self.results = []
        
        # Generate all parameter combinations
        test_cases = []
        for mu in self.mass_ratios:
            for eta in self.wind_ratios:
                test_cases.append((mu, eta))
        
        print(f"Running {len(test_cases)} stability tests...")
        
        # Run tests in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for mu, eta in test_cases:
                future = executor.submit(self.run_single_test, mu, eta)
                futures.append((mu, eta, future))
            
            # Collect results
            for mu, eta, future in tqdm(fruits, total=len(test_cases)):
                try:
                    stable, metrics = future.result(timeout=10)
                    self.results.append(metrics)
                except Exception as e:
                    print(f"Error for μ={mu}, η={eta}: {e}")
        
        return self.results
    
    def analyze_results(self):
        """Analyze sweep results and generate stability region"""
        if not self.results:
            print("No results to analyze. Run sweep first.")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Calculate stability boundary
        stability_map = {}
        for mu in self.mass_ratios:
            # Find maximum η for which system is stable at this μ
            mu_results = df[df['mu'] == mu]
            stable_etas = mu_results[mu_results['stable'] == True]['eta']
            
            if len(stable_etas) > 0:
                max_stable_eta = max(stable_etas)
            else:
                max_stable_eta = 0
            
            stability_map[mu] = max_stable_eta
        
        return stability_map
    
    def plot_stability_region(self, save_path=None):
        """Plot the stability sufficiency region"""
        stability_map = self.analyze_results()
        
        if not stability_map:
            print("No stability data to plot")
            return
        
        # Extract data for plotting
        mus = sorted(stability_map.keys())
        max_etas = [stability_map[mu] for mu in mus]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Stability boundary
        ax1.plot(mus, max_etas, 'b-', linewidth=2, label='Stability Boundary')
        ax1.fill_between(mus, max_etas, 0, alpha=0.3, color='green', label='Stable Region')
        ax1.fill_between(mus, max_etas, 0.5, alpha=0.3, color='red', label='Unstable Region')
        
        ax1.set_xlabel('Leg Mass Ratio (μ)', fontsize=12)
        ax1.set_ylabel('Maximum Wind Ratio (η)', fontsize=12)
        ax1.set_title('Stability Sufficiency Region', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(0.05, 0.40)
        ax1.set_ylim(0, 0.5)
        
        # Plot 2: Individual test results
        scatter_mus = [r['mu'] for r in self.results]
        scatter_etas = [r['eta'] for r in self.results]
        scatter_stable = [r['stable'] for r in self.results]
        
        colors = ['green' if s else 'red' for s in scatter_stable]
        ax2.scatter(scatter_mus, scatter_etas, c=colors, alpha=0.6, s=30)
        
        # Overlay boundary
        ax2.plot(mus, max_etas, 'k-', linewidth=2, label='Boundary')
        
        ax2.set_xlabel('Leg Mass Ratio (μ)', fontsize=12)
        ax2.set_ylabel('Wind Ratio (η)', fontsize=12)
        ax2.set_title('Individual Test Results', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlim(0.05, 0.40)
        ax2.set_ylim(0, 0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
        # Print quantitative analysis
        print("\n=== STABILITY ANALYSIS ===")
        print(f"Total tests: {len(self.results)}")
        print(f"Stable tests: {sum([1 for r in self.results if r['stable']])}")
        print(f"Unstable tests: {sum([1 for r in self.results if not r['stable']])}")
        
        # Theoretical boundary from Eq. 3.12
        print("\nTheoretical boundary (τ_max = 20 Nm):")
        mus_theory = np.linspace(0.05, 0.40, 10)
        tau_max = 20  # Assumed max motor torque [Nm]
        d_cp = 0.3  # Center of pressure distance [m]
        L_leg = 0.7  # Leg length [m]
        theta_dd_swing = 50  # Leg swing acceleration [rad/s²]
        
        # η_max = (τ_max - μ*m_total*L_leg²*θ_dd) / (m_total*g*d_cp)
        for mu in [0.1, 0.2, 0.3, 0.4]:
            m_total = 5 / (1 - mu)
            inertial_load = mu * m_total * L_leg**2 * theta_dd_swing
            eta_max = (tau_max - inertial_load) / (m_total * 9.81 * d_cp)
            print(f"  μ={mu:.2f}: η_max ≈ {max(0, eta_max):.3f}")
    
    def save_results(self, filename='stability_sweep_results.pkl'):
        """Save results to file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Results saved to {filename}")
    
    def load_results(self, filename='stability_sweep_results.pkl'):
        """Load results from file"""
        with open(filename, 'rb') as f:
            self.results = pickle.load(f)
        print(f"Loaded {len(self.results)} results from {filename}")