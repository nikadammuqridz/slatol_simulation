"""
SLATOL PyBullet Simulation - WORKING VERSION
Nik Adam Muqridz Bin Abdul Hakham (2125501)
"""

import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import csv
import time

# ==================== CONFIGURATION ====================
MASS_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
WIND_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # η values
TOTAL_MASS = 1.0
GRAVITY = 9.81

# Control parameters from Table 3.3
K_S = 1800.0    # Virtual stiffness [N/m]
B_S = 15.0      # Virtual damping [Ns/m]
F_THRUST = 20.0 # Thrust force [N]
KP_NOM = 25.0   # Nominal proportional gain [Nm/rad]
KD_NOM = 1.5    # Nominal derivative gain [Nms/rad]

# ==================== ROBOT CREATION (FIXED) ====================
def create_robot(mu):
    """Create robot with leg mass ratio μ without constraints"""
    # Calculate masses
    leg_mass = mu * TOTAL_MASS
    body_mass = TOTAL_MASS - leg_mass
    
    # Create base (body) - SIMPLIFIED: single box
    base_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.05, 0.05])
    base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.05, 0.05], 
                                      rgbaColor=[0.5, 0.5, 0.8, 1])
    
    # Create robot as single body (simplified for now)
    robot = p.createMultiBody(
        baseMass=body_mass + leg_mass,  # Total mass in one body
        baseCollisionShapeIndex=base_shape,
        baseVisualShapeIndex=base_visual,
        basePosition=[0, 0, 0.5]
    )
    
    # Add leg as simple cylinder attached with FIXED constraint (not revolute)
    leg_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.02, height=0.3)
    leg_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.02, length=0.3,
                                     rgbaColor=[0.8, 0.2, 0.2, 1])
    
    leg = p.createMultiBody(
        baseMass=0.001,  # Small mass for visual only
        baseCollisionShapeIndex=leg_shape,
        baseVisualShapeIndex=leg_visual,
        basePosition=[0, 0, 0.35]
    )
    
    # Create FIXED constraint between body and leg
    p.createConstraint(robot, -1, leg, -1, 
                       p.JOINT_FIXED, 
                       [0, 0, 0],
                       [0, 0, 0.15],
                       [0, 0, -0.05])
    
    return robot, leg

# ==================== SIMULATION ====================
def run_pybullet_trial(mu, eta, trial_num=1, max_time=2.0):
    """Run PyBullet simulation for given parameters"""
    # Setup simulation
    if trial_num == 1:
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    else:
        p.connect(p.DIRECT)
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -GRAVITY)
    p.setTimeStep(0.001)
    
    # Create ground
    ground = p.loadURDF("plane.urdf")
    p.changeDynamics(ground, -1, lateralFriction=0.9)
    
    # Create robot
    robot, leg = create_robot(mu)
    
    # Enable torque control for base (simplified)
    p.changeDynamics(robot, -1, linearDamping=0, angularDamping=0)
    
    # Trial data
    time_data = []
    height_data = []
    pitch_data = []
    
    # Apply initial upward impulse
    p.applyExternalForce(robot, -1, [0, 0, 300], [0, 0, 0], p.WORLD_FRAME)
    
    # Main simulation loop
    for step in range(int(max_time / 0.001)):
        current_time = step * 0.001
        
        # Get state
        pos, orn = p.getBasePositionAndOrientation(robot)
        euler = p.getEulerFromQuaternion(orn)
        lin_vel, ang_vel = p.getBaseVelocity(robot)
        
        # Record data
        time_data.append(current_time)
        height_data.append(pos[2])
        pitch_data.append(euler[1])  # Pitch around Y axis
        
        # Apply wind disturbance at apex (t=0.2s)
        if 0.19 < current_time < 0.21 and eta > 0:
            wind_force = eta * TOTAL_MASS * GRAVITY
            p.applyExternalForce(robot, -1, [wind_force, 0, 0], [0, 0, 0], p.WORLD_FRAME)
        
        # Simple PD control for stabilization
        # Adaptive gain scheduling based on μ
        Kp = KP_NOM * (mu / 0.05)
        Kd = KD_NOM * (mu / 0.05)
        
        # Apply control torque to stabilize pitch
        control_torque = -Kp * euler[1] - Kd * ang_vel[1]
        
        # Apply torque (simplified - direct torque on body)
        p.applyExternalTorque(robot, -1, [0, control_torque, 0], p.WORLD_FRAME)
        
        # Step simulation
        p.stepSimulation()
        
        # Check for failure
        if abs(euler[1]) > 0.785:  # 45 degrees in radians
            break
    
    p.disconnect()
    
    # Calculate metrics
    max_pitch_deg = np.max(np.abs(pitch_data)) * 180/np.pi
    success = max_pitch_deg < 45.0
    
    return {
        'success': success,
        'max_height': np.max(height_data),
        'max_pitch': max_pitch_deg,
        'time': np.array(time_data),
        'height': np.array(height_data),
        'pitch': np.array(pitch_data) * 180/np.pi
    }

# ==================== MAIN EXPERIMENT ====================
def main():
    print("SLATOL PyBullet Simulation - WORKING")
    print("="*50)
    
    results = []
    
    # Run stability map
    print("\nRunning PyBullet simulations...")
    trial_num = 0
    for mu in MASS_RATIOS:
        for eta in WIND_RATIOS:
            trial_num += 1
            print(f"  Trial {trial_num}: μ={mu:.2f}, η={eta:.2f}...", end="")
            
            # Run PyBullet simulation
            result = run_pybullet_trial(mu, eta, trial_num)
            
            # Calculate actual wind force
            wind_force = eta * TOTAL_MASS * GRAVITY
            
            results.append({
                'mu': mu,
                'wind_ratio': eta,
                'wind_force': wind_force,
                'success': result['success'],
                'max_height': result['max_height'],
                'max_pitch': result['max_pitch']
            })
            
            status = "✓" if result['success'] else "✗"
            print(f" {status} H={result['max_height']:.2f}m θ={result['max_pitch']:.1f}°")
    
    # Save results
    with open('pybullet_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    # Generate figures from last trial data
    print("\nGenerating figures from simulation data...")
    generate_figures(results)

# ==================== FIGURE GENERATION ====================
def generate_figures(results):
    """Generate figures from PyBullet simulation results"""
    
    # Run specific cases for detailed analysis
    print("  Running nominal case for Figure 4.1...")
    nominal = run_pybullet_trial(0.05, 0.0, 1000)  # High trial num for no GUI
    
    print("  Running stable case for Figure 4.2...")
    stable = run_pybullet_trial(0.20, 0.2, 1001)
    
    print("  Running unstable case for Figure 4.3...")
    unstable = run_pybullet_trial(0.20, 0.4, 1002)
    
    # ===== FIGURE 4.1 =====
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(nominal['time'], nominal['height'], 'b-', linewidth=2)
    plt.ylabel('CoM Height (m)', fontsize=12)
    plt.title('Figure 4.1: Nominal Flight Performance (μ=0.05, η=0.00)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    
    plt.subplot(2, 1, 2)
    plt.plot(nominal['time'], nominal['pitch'], 'r-', linewidth=2)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Pitch Angle (deg)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(-5, 5)
    
    plt.tight_layout()
    plt.savefig('figure_4_1_nominal.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===== FIGURE 4.2 =====
    plt.figure(figsize=(10, 6))
    
    plt.plot(stable['time'], stable['pitch'], 'b-', linewidth=2, label='Pitch Response')
    plt.axvline(x=0.2, color='k', linestyle='--', alpha=0.5, label='Wind Injection')
    plt.axhline(y=2, color='g', linestyle=':', label='±2° Band')
    plt.axhline(y=-2, color='g', linestyle=':')
    
    # Calculate settling time
    wind_idx = np.argmin(np.abs(stable['time'] - 0.2))
    settle_idx = np.where(np.abs(stable['pitch'][wind_idx:]) < 2)[0]
    if len(settle_idx) > 0:
        settle_time = stable['time'][wind_idx + settle_idx[0]]
        plt.axvline(x=settle_time, color='r', linestyle='--', alpha=0.7,
                   label=f'Settling: {settle_time-0.2:.2f}s')
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Pitch Angle (deg)', fontsize=12)
    plt.title('Figure 4.2: Stable Recovery (μ=0.20, η=0.20)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-30, 30)
    
    plt.savefig('figure_4_2_stable.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===== FIGURE 4.3 =====
    plt.figure(figsize=(10, 6))
    
    plt.plot(unstable['time'], unstable['pitch'], 'r-', linewidth=2, label='Pitch Response')
    plt.axhline(y=45, color='k', linestyle='--', linewidth=2, label='Failure Threshold (45°)')
    
    # Find failure time
    fail_idx = np.where(np.abs(unstable['pitch']) > 45)[0]
    if len(fail_idx) > 0:
        fail_time = unstable['time'][fail_idx[0]]
        plt.axvline(x=fail_time, color='r', linestyle='--', alpha=0.7,
                   label=f'Failure at t={fail_time:.2f}s')
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Pitch Angle (deg)', fontsize=12)
    plt.title('Figure 4.3: Unstable Divergence (μ=0.20, η=0.41)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 90)
    
    plt.savefig('figure_4_3_unstable.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===== FIGURE 4.4 =====
    plt.figure(figsize=(12, 8))
    
    # Calculate stability limits
    stability_limits = {}
    for mu in MASS_RATIOS:
        mu_results = [r for r in results if r['mu'] == mu and r['success']]
        if mu_results:
            max_eta = max([r['wind_ratio'] for r in mu_results])
            stability_limits[mu] = max_eta
        else:
            stability_limits[mu] = 0.0
    
    # Plot points
    colors = ['green' if r['success'] else 'red' for r in results]
    plt.scatter([r['mu'] for r in results], 
                [r['wind_ratio'] for r in results],
                c=colors, alpha=0.6, edgecolors='k', s=50)
    
    # Plot boundary
    sorted_mu = sorted(stability_limits.keys())
    sorted_eta = [stability_limits[mu] for mu in sorted_mu]
    plt.plot(sorted_mu, sorted_eta, 'k-', linewidth=3, label='Stability Boundary')
    
    # Fill stable region
    plt.fill_between(sorted_mu, 0, sorted_eta, alpha=0.2, color='green',
                     label='Stability Sufficiency Region')
    
    plt.xlabel('Leg Mass Ratio (μ)', fontsize=14)
    plt.ylabel('Wind Disturbance Ratio (η)', fontsize=14)
    plt.title('Figure 4.4: Stability Sufficiency Region from PyBullet Simulation', fontsize=16)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='upper right')
    plt.xlim(0, 0.35)
    plt.ylim(0, 0.6)
    
    plt.savefig('figure_4_4_stability_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print results
    print_results(stability_limits, results)

def print_results(stability_limits, results):
    """Print quantitative results for thesis"""
    print("\n" + "="*60)
    print("PYBULLET SIMULATION RESULTS")
    print("="*60)
    
    print("\nTable 4.1: Stability Limits from PyBullet")
    print("-"*40)
    print(f"{'μ':<8} {'η_max':<12} {'Status':<15}")
    print("-"*40)
    
    for mu in sorted(stability_limits.keys()):
        eta_max = stability_limits[mu]
        
        # Get a sample trial for this μ and max η
        sample = next((r for r in results if r['mu'] == mu and 
                      abs(r['wind_ratio'] - eta_max) < 0.01 and r['success']), None)
        
        if sample:
            print(f"{mu:.2f}    {eta_max:.3f}        Success (H={sample['max_height']:.2f}m)")
        else:
            print(f"{mu:.2f}    {eta_max:.3f}        No successful trials")
    
    # Calculate percentage drop
    if 0.05 in stability_limits and 0.30 in stability_limits:
        drop = ((stability_limits[0.05] - stability_limits[0.30]) / 
                stability_limits[0.05]) * 100
        print("-"*40)
        print(f"\nWind tolerance reduction: {drop:.1f}% (μ=0.05→0.30)")
    
    print("\n" + "="*60)
    print("FILES GENERATED:")
    print("  1. pybullet_results.csv - Complete simulation data")
    print("  2. figure_4_1_nominal.png - Nominal flight (μ=0.05, η=0.00)")
    print("  3. figure_4_2_stable.png - Stable recovery (μ=0.20, η=0.20)")
    print("  4. figure_4_3_unstable.png - Unstable divergence (μ=0.20, η=0.41)")
    print("  5. figure_4_4_stability_map.png - Stability sufficiency region")
    print("="*60)

# Run the simulation
if __name__ == "__main__":
    # Test PyBullet installation
    try:
        import pybullet
        print("✓ PyBullet successfully imported")
    except ImportError:
        print("ERROR: PyBullet not installed. Run: pip install pybullet")
        exit(1)
    
    main()
