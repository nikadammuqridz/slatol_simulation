#!/usr/bin/env python3
"""
SLATOL PyBullet Simulation - CORRECTED VERSION
- 2-DOF leg (hip & knee revolute joints)
- Virtual Model Control (VMC) for stance
- Angular Momentum Control (AMC) for flight attitude
- Adaptive gain scheduling based on leg mass ratio μ
- Interactive debug GUI mode (single trial)
- Automated sweep mode
- Fixed Figure 4.3 (same style as 4.2)
"""

import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import sys

# ==================== CONFIGURATION ====================
TOTAL_MASS = 1.0          # kg
GRAVITY = 9.81           # m/s²
L1 = 0.15                # thigh length (m)
L2 = 0.15                # shank length (m)
BASE_HALF_EXTENTS = [0.1, 0.05, 0.05]   # x,y,z
BASE_HEIGHT = 2 * BASE_HALF_EXTENTS[2]  # 0.1 m

# Control parameters (from Table 3.3)
K_S = 1800.0            # Virtual stiffness [N/m]
B_S = 15.0             # Virtual damping [Ns/m]
F_THRUST = 20.0        # Thrust force [N]
KP_NOM = 25.0          # Nominal proportional gain [Nm/rad]
KD_NOM = 1.5           # Nominal derivative gain [Nms/rad]
K_RAIBERT = 0.04       # Velocity gain for foot placement [s]

# Stability criteria
PITCH_LIMIT_DEG = 45.0
HEIGHT_MIN = 0.15      # m (too low = crash)

# ==================== ROBOT CREATION (2-DOF LEG) ====================
def create_robot(mu):
    """
    Builds a 2-DOF single-legged robot in PyBullet.
    Returns: (robot_id, joint_indices)
    joint_indices = {'hip': int, 'knee': int}
    """
    p.resetSimulation()
    p.setGravity(0, 0, -GRAVITY)

    # ---- Mass distribution ----
    leg_mass = mu * TOTAL_MASS
    body_mass = TOTAL_MASS - leg_mass
    # Assign all leg mass to thigh, shank negligible (for simplicity)
    thigh_mass = leg_mass
    shank_mass = 0.001

    # ---- Base (body) ----
    base_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=BASE_HALF_EXTENTS)
    base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=BASE_HALF_EXTENTS,
                                      rgbaColor=[0.5, 0.5, 0.8, 1])
    base_pos = [0, 0, 0.5]   # initial height ~0.5 m
    base_orn = p.getQuaternionFromEuler([0, 0, 0])

    robot = p.createMultiBody(
        baseMass=body_mass,
        baseCollisionShapeIndex=base_collision,
        baseVisualShapeIndex=base_visual,
        basePosition=base_pos,
        baseOrientation=base_orn
    )

    # ---- Thigh (link 1) ----
    # Collision: cylinder oriented along Z, radius 0.02, height L1
    thigh_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.02, height=L1)
    thigh_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.02, length=L1,
                                       rgbaColor=[0.8, 0.2, 0.2, 1])
    # Joint position: at bottom of base (local frame of base)
    hip_pos = [0, 0, -BASE_HALF_EXTENTS[2]]   # (0,0,-0.05) relative to base CoM
    # Thigh's CoM is at [0,0,-L1/2] in its own local frame (joint at top)
    thigh_pos = [0, 0, -L1/2]   # relative to joint frame

    hip_joint = p.createConstraint(
        robot, -1, -1, -1,
        p.JOINT_REVOLUTE,
        [0, 1, 0],          # axis around Y
        hip_pos,
        thigh_pos,
        [0, 0, 0],         # parent frame position (already set)
        parentFramePosition=hip_pos,
        childFramePosition=thigh_pos,
        parentFrameOrientation=p.getQuaternionFromEuler([0,0,0]),
        childFrameOrientation=p.getQuaternionFromEuler([0,0,0])
    )
    # Set thigh mass
    p.changeDynamics(hip_joint, -1, mass=thigh_mass)
    # Disable motor so we can apply torques later
    p.setJointMotorControl2(robot, hip_joint, p.VELOCITY_CONTROL, force=0)

    # ---- Shank (link 2) ----
    shank_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.02, height=L2)
    shank_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.02, length=L2,
                                       rgbaColor=[0.2, 0.8, 0.2, 1])
    # Knee joint attaches to bottom of thigh
    # In thigh's local frame, bottom is at [0,0,-L1]
    knee_pos_parent = [0, 0, -L1]   # relative to thigh joint frame
    # Shank's CoM is at [0,0,-L2/2] in its own frame
    shank_pos = [0, 0, -L2/2]

    knee_joint = p.createConstraint(
        robot, hip_joint, -1, -1,
        p.JOINT_REVOLUTE,
        [0, 1, 0],
        knee_pos_parent,
        shank_pos,
        [0, 0, 0],
        parentFramePosition=knee_pos_parent,
        childFramePosition=shank_pos
    )
    p.changeDynamics(knee_joint, -1, mass=shank_mass)
    p.setJointMotorControl2(robot, knee_joint, p.VELOCITY_CONTROL, force=0)

    # Store joint indices for easy access
    joint_dict = {'hip': hip_joint, 'knee': knee_joint}
    return robot, joint_dict

# ==================== KINEMATICS & JACOBIAN ====================
def forward_kinematics(theta1, theta2):
    """Compute foot position (x,z) relative to hip joint frame."""
    x = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    z = -L1 * np.cos(theta1) - L2 * np.cos(theta1 + theta2)
    return x, z

def jacobian(theta1, theta2):
    """Jacobian matrix J (2x2) mapping joint velocities to foot Cartesian velocity."""
    J11 = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    J12 = L2 * np.cos(theta1 + theta2)
    J21 = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    J22 = L2 * np.sin(theta1 + theta2)
    return np.array([[J11, J12], [J21, J22]])

def inverse_kinematics(x_des, z_des, theta1_guess=0, theta2_guess=-np.pi/2):
    """Analytical IK for a 2R planar leg."""
    r = np.sqrt(x_des**2 + z_des**2)
    if r > L1 + L2 or r < abs(L1 - L2):
        return None   # unreachable
    cos_theta2 = (r**2 - L1**2 - L2**2) / (2 * L1 * L2)
    theta2 = -np.arccos(np.clip(cos_theta2, -1, 1))
    # Two solutions; we pick the one with knee bent outward (negative)
    alpha = np.arctan2(x_des, -z_des)   # angle of foot vector from vertical
    beta = np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
    theta1 = alpha - beta
    return theta1, theta2

# ==================== SIMULATION (SINGLE TRIAL) ====================
def run_pybullet_trial(mu, eta, use_gui=False, max_time=2.0):
    """Run one hopping trial. Returns dict with data and success flag."""
    if use_gui:
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=0,
                                     cameraPitch=-40, cameraTargetPosition=[0,0,0.5])
    else:
        p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -GRAVITY)
    p.setTimeStep(0.001)
    p.setRealTimeSimulation(0)

    # Ground
    plane = p.loadURDF("plane.urdf")
    p.changeDynamics(plane, -1, lateralFriction=0.9)

    # Robot
    robot, joints = create_robot(mu)

    # State machine
    state = "FLIGHT"   # start in flight (initial impulse)
    stance_start_time = 0
    wind_applied = False
    wind_timer = 0.0

    # Data logging
    time_data = []
    height_data = []
    pitch_data = []
    foot_force_data = []
    leg_state_data = []   # (r, phi, r_dot)

    # Initial upward impulse (to start hop)
    p.applyExternalForce(robot, -1, [0, 0, 300], [0, 0, 0], p.WORLD_FRAME)

    # Adaptive gains (Eq 3.11)
    Kp_amc = KP_NOM * (mu / 0.05)
    Kd_amc = KD_NOM * (mu / 0.05)

    # Main loop
    for step in range(int(max_time / 0.001)):
        t = step * 0.001

        # --- Get state ---
        pos, orn = p.getBasePositionAndOrientation(robot)
        euler = p.getEulerFromQuaternion(orn)
        theta_body = euler[1]   # pitch (Y axis)
        lin_vel, ang_vel = p.getBaseVelocity(robot)
        theta_dot_body = ang_vel[1]

        # Get joint states
        hip_state = p.getJointState(robot, joints['hip'])
        knee_state = p.getJointState(robot, joints['knee'])
        theta1 = hip_state[0]
        theta2 = knee_state[0]
        dtheta1 = hip_state[1]
        dtheta2 = knee_state[1]

        # Foot position in world frame? For contact detection we use ground reaction force.
        # Simpler: check if foot is below ground.
        foot_x, foot_z = forward_kinematics(theta1, theta2)
        # foot position in world frame:
        foot_world = p.multiplyTransforms(pos, orn, [foot_x, 0, foot_z], [0,0,0,1])
        foot_z_world = foot_world[0][2]

        # Ground contact force (simplified: normal force at foot)
        contact = False
        force_norm = 0
        if foot_z_world <= 0.01:   # near ground
            contact = True
            # Estimate normal force from penetration (very simple)
            penetration = -foot_z_world
            force_norm = 1000 * penetration   # crude stiffness
        foot_force_data.append(force_norm)

        # Virtual leg coordinates (Eq 3.3)
        r = np.sqrt(foot_x**2 + foot_z**2)
        phi = np.arctan2(foot_x, -foot_z)   # angle from vertical
        # r_dot: time derivative
        J = jacobian(theta1, theta2)
        foot_vel = J @ np.array([dtheta1, dtheta2])
        r_dot = (foot_x * foot_vel[0] + foot_z * foot_vel[1]) / (r + 1e-6)

        leg_state = (r, phi, r_dot)

        # --- State machine (Table 3.2) ---
        if contact and state == "FLIGHT":
            state = "COMPRESSION"
            stance_start_time = t
        elif state == "COMPRESSION" and r_dot >= 0:
            state = "THRUST"
        elif state == "THRUST" and not contact:
            state = "FLIGHT"
        elif state in ["COMPRESSION", "THRUST"] and not contact:
            state = "FLIGHT"
        elif state == "FLIGHT" and t > stance_start_time + 0.05 and foot_z_world < 0.05:
            # Prepare landing: switch to LANDING mode (foot placement)
            state = "LANDING"

        # --- Control actions ---
        tau_hip = 0
        tau_knee = 0

        if state in ["COMPRESSION", "THRUST"]:
            # VMC: Virtual Model Control (Eq 3.8, 3.9)
            r0 = L1 + L2   # full leg extension
            Fr = K_S * (r0 - r) - B_S * r_dot
            if state == "THRUST":
                Fr += F_THRUST
            # Project to Cartesian force
            Fx = Fr * np.sin(phi)
            Fz = -Fr * np.cos(phi)
            # Jacobian transpose to joint torques
            Jt = J.T
            tau_joint = Jt @ np.array([Fx, Fz])
            tau_hip, tau_knee = tau_joint[0], tau_joint[1]

        elif state == "FLIGHT":
            # AMC: Angular Momentum Control (Eq 3.10)
            tau_hip = -Kp_amc * theta_body - Kd_amc * theta_dot_body
            tau_knee = 0   # keep knee fixed during flight

        elif state == "LANDING":
            # Foot placement (Raibert heuristic, Eq 3.11)
            x_foot_des = (lin_vel[0] * 0.02) + K_RAIBERT * (lin_vel[0] - 0.0)   # desired speed = 0
            z_foot_des = -0.1   # aim slightly below ground to ensure contact
            angles = inverse_kinematics(x_foot_des, z_foot_des, theta1, theta2)
            if angles is not None:
                theta1_des, theta2_des = angles
                # Simple PD controller to reach target angles
                kp_joint = 50
                kd_joint = 2
                tau_hip = kp_joint * (theta1_des - theta1) - kd_joint * dtheta1
                tau_knee = kp_joint * (theta2_des - theta2) - kd_joint * dtheta2

        # Apply torques (clamp to reasonable limits)
        max_torque = 5.0   # motor saturation
        tau_hip = np.clip(tau_hip, -max_torque, max_torque)
        tau_knee = np.clip(tau_knee, -max_torque, max_torque)

        p.setJointMotorControl2(robot, joints['hip'], p.TORQUE_CONTROL, force=tau_hip)
        p.setJointMotorControl2(robot, joints['knee'], p.TORQUE_CONTROL, force=tau_knee)

        # --- Wind disturbance (at apex, Section 3.4.1) ---
        if not wind_applied and state == "FLIGHT" and lin_vel[2] < 0 and eta > 0:
            # Apex detected (vertical velocity becomes negative)
            wind_force = eta * TOTAL_MASS * GRAVITY
            p.applyExternalForce(robot, -1, [wind_force, 0, 0], [0, 0, 0], p.WORLD_FRAME)
            wind_applied = True
            wind_timer = t

        # --- Record data ---
        time_data.append(t)
        height_data.append(pos[2])
        pitch_data.append(theta_body * 180/np.pi)
        leg_state_data.append(leg_state)

        # --- Step simulation ---
        p.stepSimulation()

        # --- Failure check (Eq 3.13) ---
        if abs(theta_body) > np.radians(PITCH_LIMIT_DEG) or pos[2] < HEIGHT_MIN:
            break

    p.disconnect()

    # --- Process data ---
    max_pitch = np.max(np.abs(pitch_data))
    success = max_pitch < PITCH_LIMIT_DEG and max(height_data) > 0.3

    return {
        'success': success,
        'max_height': np.max(height_data),
        'max_pitch': max_pitch,
        'time': np.array(time_data),
        'height': np.array(height_data),
        'pitch': np.array(pitch_data),
        'wind_applied': wind_applied,
        'wind_time': wind_timer if wind_applied else None
    }

# ==================== SWEEP MODE ====================
def sweep_mode():
    """Run full parameter sweep and generate figures."""
    MASS_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    WIND_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    results = []
    print("\n=== Starting Sweep Mode ===")
    trial_num = 0
    for mu in MASS_RATIOS:
        for eta in WIND_RATIOS:
            trial_num += 1
            print(f"  Trial {trial_num}: μ={mu:.2f}, η={eta:.2f} ...", end='', flush=True)
            res = run_pybullet_trial(mu, eta, use_gui=False, max_time=2.0)
            results.append({
                'mu': mu,
                'wind_ratio': eta,
                'wind_force': eta * TOTAL_MASS * GRAVITY,
                'success': res['success'],
                'max_height': res['max_height'],
                'max_pitch': res['max_pitch']
            })
            status = "✓" if res['success'] else "✗"
            print(f" {status}  H={res['max_height']:.2f}m  θ={res['max_pitch']:.1f}°")

    # Save CSV
    with open('pybullet_results_corrected.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Generate figures
    generate_figures(results)

def generate_figures(results):
    """Create Figures 4.1–4.4 from thesis."""
    print("\n=== Generating Figures ===")

    # Run specific cases for figures
    print("  Running nominal (μ=0.05, η=0.0) for Fig 4.1 ...")
    nominal = run_pybullet_trial(0.05, 0.0, use_gui=False, max_time=2.0)

    print("  Running stable (μ=0.20, η=0.2) for Fig 4.2 ...")
    stable = run_pybullet_trial(0.20, 0.2, use_gui=False, max_time=2.0)

    print("  Running unstable (μ=0.20, η=0.41) for Fig 4.3 ...")
    unstable = run_pybullet_trial(0.20, 0.41, use_gui=False, max_time=2.0)

    # ----- Figure 4.1: Nominal flight -----
    plt.figure(figsize=(10, 8))
    plt.subplot(2,1,1)
    plt.plot(nominal['time'], nominal['height'], 'b-', lw=2)
    plt.ylabel('CoM Height (m)')
    plt.title('Figure 4.1: Nominal Flight Performance (μ=0.05, η=0.00)')
    plt.grid(alpha=0.3)
    plt.ylim(0,1.0)

    plt.subplot(2,1,2)
    plt.plot(nominal['time'], nominal['pitch'], 'r-', lw=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch Angle (deg)')
    plt.grid(alpha=0.3)
    plt.ylim(-5,5)
    plt.tight_layout()
    plt.savefig('figure_4_1_nominal.png', dpi=300)
    plt.close()

    # ----- Figure 4.2: Stable recovery -----
    plt.figure(figsize=(10,6))
    plt.plot(stable['time'], stable['pitch'], 'b-', lw=2, label='Pitch Response')
    plt.axvline(x=stable['wind_time'], color='k', ls='--', alpha=0.5, label='Wind Injection')
    plt.axhline(y=2, color='g', ls=':', label='±2° Band')
    plt.axhline(y=-2, color='g', ls=':')
    # Settling time (first time within ±2° after wind)
    if stable['wind_time']:
        idx_start = np.argmin(np.abs(stable['time'] - stable['wind_time']))
        in_band = np.where(np.abs(stable['pitch'][idx_start:]) < 2)[0]
        if len(in_band) > 0:
            t_settle = stable['time'][idx_start + in_band[0]]
            plt.axvline(x=t_settle, color='r', ls='--', alpha=0.7,
                        label=f'Settling: {t_settle-stable["wind_time"]:.2f}s')
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch Angle (deg)')
    plt.title('Figure 4.2: Stable Recovery (μ=0.20, η=0.20)')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.ylim(-30,30)
    plt.savefig('figure_4_2_stable.png', dpi=300)
    plt.close()

    # ----- Figure 4.3: Unstable divergence (SAME STYLE as 4.2) -----
    plt.figure(figsize=(10,6))
    plt.plot(unstable['time'], unstable['pitch'], 'r-', lw=2, label='Pitch Response')
    plt.axhline(y=45, color='k', ls='--', lw=2, label='Failure Threshold (45°)')
    if unstable['wind_time']:
        plt.axvline(x=unstable['wind_time'], color='k', ls='--', alpha=0.5)
    # Find failure time
    fail_idx = np.where(np.abs(unstable['pitch']) > 45)[0]
    if len(fail_idx) > 0:
        t_fail = unstable['time'][fail_idx[0]]
        plt.axvline(x=t_fail, color='r', ls='--', alpha=0.7,
                    label=f'Failure at t={t_fail:.2f}s')
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch Angle (deg)')
    plt.title('Figure 4.3: Unstable Divergence (μ=0.20, η=0.41)')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.ylim(-10,90)
    plt.savefig('figure_4_3_unstable.png', dpi=300)
    plt.close()

    # ----- Figure 4.4: Stability Sufficiency Region -----
    plt.figure(figsize=(12,8))
    # Extract data from results
    mus = [r['mu'] for r in results]
    etas = [r['wind_ratio'] for r in results]
    colors = ['green' if r['success'] else 'red' for r in results]
    plt.scatter(mus, etas, c=colors, alpha=0.6, edgecolors='k', s=50)

    # Compute stability boundary (max eta per mu)
    boundary = {}
    for mu in sorted(set(mus)):
        mu_res = [r for r in results if r['mu'] == mu and r['success']]
        boundary[mu] = max([r['wind_ratio'] for r in mu_res]) if mu_res else 0.0
    mu_vals = sorted(boundary.keys())
    eta_max = [boundary[mu] for mu in mu_vals]
    plt.plot(mu_vals, eta_max, 'k-', lw=3, label='Stability Boundary')
    plt.fill_between(mu_vals, 0, eta_max, alpha=0.2, color='green',
                     label='Stability Sufficiency Region')

    plt.xlabel('Leg Mass Ratio (μ)')
    plt.ylabel('Wind Disturbance Ratio (η)')
    plt.title('Figure 4.4: Stability Sufficiency Region from PyBullet')
    plt.grid(alpha=0.3, ls='--')
    plt.legend(loc='upper right')
    plt.xlim(0,0.35)
    plt.ylim(0,0.6)
    plt.savefig('figure_4_4_stability_map.png', dpi=300)
    plt.close()

    print("  Figures saved.")

# ==================== DEBUG GUI MODE ====================
def debug_mode():
    """Interactive single trial with GUI, custom μ and η."""
    print("\n=== Debug GUI Mode ===")
    try:
        mu = float(input("Enter leg mass ratio μ (0.05–0.30): "))
        eta = float(input("Enter wind ratio η (0.0–0.5): "))
    except:
        print("Invalid input. Using default μ=0.20, η=0.20")
        mu, eta = 0.20, 0.20

    print(f"Running trial with μ={mu:.2f}, η={eta:.2f} (GUI)...")
    res = run_pybullet_trial(mu, eta, use_gui=True, max_time=3.0)

    # Plot after simulation closes
    plt.figure(figsize=(12,8))
    plt.subplot(2,1,1)
    plt.plot(res['time'], res['height'], 'b-')
    plt.ylabel('Height (m)')
    plt.title(f'Debug Trial: μ={mu}, η={eta} - {"SUCCESS" if res["success"] else "FAILURE"}')
    plt.grid(alpha=0.3)

    plt.subplot(2,1,2)
    plt.plot(res['time'], res['pitch'], 'r-')
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (deg)')
    plt.axhline(y=45, color='k', ls='--')
    plt.axhline(y=-45, color='k', ls='--')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('debug_trial.png')
    plt.show()

# ==================== MAIN ====================
def main():
    print("\n" + "="*60)
    print(" SLATOL PyBullet Simulation - CORRECTED VERSION")
    print("="*60)
    print("Select mode:")
    print("  1. Sweep mode (run full parameter sweep)")
    print("  2. Debug GUI mode (single trial with visualization)")
    print("  3. Exit")

    choice = input("Enter choice (1/2/3): ").strip()
    if choice == '1':
        sweep_mode()
    elif choice == '2':
        debug_mode()
    else:
        print("Exiting.")
        sys.exit(0)

if __name__ == "__main__":
    try:
        import pybullet
    except ImportError:
        print("ERROR: PyBullet not installed. Run: pip install pybullet")
        sys.exit(1)
    main()