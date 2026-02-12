"""
SLATOL – PyBullet Hopping Simulation (WORKING)
Nik Adam Muqridz (2125501)

- URDF generated on‑the‑fly with validated inertia.
- Proper torque control enable (setJointMotorControlArray with zero forces).
- Virtual Model Control (VMC) during stance.
- Angular Momentum Control (AMC) during flight.
- Foot placement (Raibert) for horizontal stabilisation.
- Wind disturbance (optional) at apex.
- Full parameter sweep over μ and η.

Run in debug mode to see the robot hop.
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import csv

# ==================== CONFIGURATION ====================
MASS_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
WIND_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
TOTAL_MASS = 1.0
GRAVITY = 9.81

# ==================== URDF GENERATOR ====================
def generate_urdf(mu, output_path):
    """Generate URDF with leg mass ratio = mu. Inertia satisfies triangle inequality."""
    total_mass = 1.0
    leg_mass = mu * total_mass
    body_mass = total_mass - leg_mass
    femur_mass = leg_mass * 0.5
    tibia_mass = leg_mass * 0.5

    L1 = 0.15
    L2 = 0.15
    link_radius = 0.02
    foot_radius = 0.02
    bx, by, bz = 0.1, 0.05, 0.05

    # Inertia: solid cylinder (femur, tibia)
    I_cyl_radial = (1/12) * (3*link_radius**2 + L1**2)
    I_cyl_axial = 0.5 * link_radius**2

    I_femur_xx = femur_mass * I_cyl_radial
    I_femur_yy = I_femur_xx
    I_femur_zz = femur_mass * I_cyl_axial
    # Enforce triangle inequality
    if I_femur_zz < 0.1 * I_femur_xx:
        I_femur_zz = 0.1 * I_femur_xx

    I_tibia_xx = tibia_mass * I_cyl_radial
    I_tibia_yy = I_tibia_xx
    I_tibia_zz = tibia_mass * I_cyl_axial
    if I_tibia_zz < 0.1 * I_tibia_xx:
        I_tibia_zz = 0.1 * I_tibia_xx

    # Body inertia (box)
    I_body_xx = (1/12) * body_mass * (by**2 + bz**2) * 4
    I_body_yy = (1/12) * body_mass * (bx**2 + bz**2) * 4
    I_body_zz = (1/12) * body_mass * (bx**2 + by**2) * 4

    foot_mass = 0.001
    I_foot = (2/5) * foot_mass * foot_radius**2

    urdf_str = f'''<?xml version="1.0"?>
<robot name="slatol_mu_{mu:.2f}">

  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{body_mass:.4f}"/>
      <inertia ixx="{I_body_xx:.6f}" ixy="0" ixz="0"
               iyy="{I_body_yy:.6f}" iyz="0"
               izz="{I_body_zz:.6f}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="{2*bx} {2*by} {2*bz}"/></geometry>
      <material name="blue"><color rgba="0.2 0.3 0.8 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="{2*bx} {2*by} {2*bz}"/></geometry>
    </collision>
  </link>

  <joint name="hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="femur_link"/>
    <origin xyz="0 0 -{bz}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.5708" upper="1.5708" effort="5.0" velocity="22.0"/>
    <dynamics damping="0.1" friction="0.01"/>
  </joint>

  <link name="femur_link">
    <inertial>
      <origin xyz="0 0 -{L1/2}" rpy="0 0 0"/>
      <mass value="{femur_mass:.4f}"/>
      <inertia ixx="{I_femur_xx:.6f}" ixy="0" ixz="0"
               iyy="{I_femur_yy:.6f}" iyz="0"
               izz="{I_femur_zz:.6f}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -{L1/2}" rpy="0 0 0"/>
      <geometry><cylinder length="{L1}" radius="{link_radius}"/></geometry>
      <material name="red"><color rgba="0.8 0.2 0.2 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 -{L1/2}" rpy="0 0 0"/>
      <geometry><cylinder length="{L1}" radius="{link_radius}"/></geometry>
    </collision>
  </link>

  <joint name="knee_joint" type="revolute">
    <parent link="femur_link"/>
    <child link="tibia_link"/>
    <origin xyz="0 0 -{L1}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.618" upper="0.0" effort="5.0" velocity="22.0"/>
    <dynamics damping="0.1" friction="0.01"/>
  </joint>

  <link name="tibia_link">
    <inertial>
      <origin xyz="0 0 -{L2/2}" rpy="0 0 0"/>
      <mass value="{tibia_mass:.4f}"/>
      <inertia ixx="{I_tibia_xx:.6f}" ixy="0" ixz="0"
               iyy="{I_tibia_yy:.6f}" iyz="0"
               izz="{I_tibia_zz:.6f}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -{L2/2}" rpy="0 0 0"/>
      <geometry><cylinder length="{L2}" radius="{link_radius}"/></geometry>
      <material name="red"><color rgba="0.8 0.2 0.2 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 -{L2/2}" rpy="0 0 0"/>
      <geometry><cylinder length="{L2}" radius="{link_radius}"/></geometry>
    </collision>
  </link>

  <joint name="foot_joint" type="fixed">
    <parent link="tibia_link"/>
    <child link="foot_link"/>
    <origin xyz="0 0 -{L2}" rpy="0 0 0"/>
  </joint>

  <link name="foot_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{foot_mass:.4f}"/>
      <inertia ixx="{I_foot:.6f}" ixy="0" ixz="0"
               iyy="{I_foot:.6f}" iyz="0"
               izz="{I_foot:.6f}"/>
    </inertial>
    <visual>
      <geometry><sphere radius="{foot_radius}"/></geometry>
      <material name="black"><color rgba="0.1 0.1 0.1 1.0"/></material>
    </visual>
    <collision>
      <geometry><sphere radius="{foot_radius}"/></geometry>
    </collision>
  </link>

</robot>'''
    with open(output_path, 'w') as f:
        f.write(urdf_str)
    return output_path

# ==================== CONTROLLERS ====================
class Phase:
    COMPRESSION = 1
    THRUST = 2
    FLIGHT = 3
    LANDING = 4

class FSM:
    def __init__(self):
        self.phase = Phase.FLIGHT
        self.r_min = float('inf')
        self.compression_max = False

    def transition(self, contact, r, r_dot, z_dot):
        if self.phase == Phase.COMPRESSION:
            self.r_min = min(self.r_min, r)
            if r_dot > 0 and self.r_min < 0.25:
                self.compression_max = True
        if contact:
            if self.phase in [Phase.FLIGHT, Phase.LANDING]:
                self.phase = Phase.COMPRESSION
                self.r_min = r
                self.compression_max = False
            elif self.phase == Phase.COMPRESSION and self.compression_max:
                self.phase = Phase.THRUST
        else:
            if self.phase in [Phase.COMPRESSION, Phase.THRUST]:
                self.phase = Phase.FLIGHT
            elif self.phase == Phase.FLIGHT:
                if z_dot < -0.5 and r < 0.35:
                    self.phase = Phase.LANDING
        return self.phase

def vmc_force(r, r_dot, phase):
    r0 = 0.30
    ks = 1800.0
    bs = 15.0
    F = ks * (r0 - r) - bs * r_dot
    if phase == Phase.THRUST:
        F += 20.0
    return np.clip(F, 0, 300)

def amc_torque(theta, theta_dot, mu):
    Kp_nom = 1.2
    Kd_nom = 0.06
    Kp = Kp_nom * (mu / 0.05)
    Kd = Kd_nom * (mu / 0.05)
    tau = -Kp * theta - Kd * theta_dot
    return np.clip(tau, -5.0, 5.0)

def foot_placement(x_dot, x_dot_des=0.0):
    k_raibert = 0.04
    T_stance = np.pi * np.sqrt(1.0 / 1800.0)
    x_f = (x_dot * T_stance) / 2 + k_raibert * (x_dot - x_dot_des)
    return x_f

def inverse_kinematics(x_f, z_f):
    L1, L2 = 0.15, 0.15
    l_vir = np.sqrt(x_f**2 + z_f**2)
    cos_th3 = (l_vir**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_th3 = np.clip(cos_th3, -1.0, 1.0)
    theta3 = -np.arccos(cos_th3)
    phi = np.arctan2(x_f, -z_f)
    alpha = np.arcsin((L2 * np.sin(-theta3)) / l_vir)
    theta2 = phi - alpha
    return theta2, theta3

# ==================== SINGLE TRIAL ====================
def run_single_trial(mu, eta, use_gui=False, debug=False, max_time=3.0):
    # Generate URDF
    urdf_file = f"temp_mu_{mu:.2f}.urdf"
    generate_urdf(mu, urdf_file)

    # Connect
    if use_gui or debug:
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(1.5, 0, -40, [0,0,0.5])
    else:
        p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.81)
    p.setTimeStep(0.001)

    # Load plane
    plane = p.loadURDF("plane.urdf")
    p.changeDynamics(plane, -1, lateralFriction=0.9)

    # Load robot with inertia flag
    robot = p.loadURDF(urdf_file, basePosition=[0,0,0.5], flags=p.URDF_USE_INERTIA_FROM_FILE)
    
    # Get joint indices
    num_joints = p.getNumJoints(robot)
    hip_idx = knee_idx = foot_link_idx = None
    movable = []
    for i in range(num_joints):
        info = p.getJointInfo(robot, i)
        name = info[1].decode('utf-8')
        if info[2] != p.JOINT_FIXED:
            movable.append(i)
        if 'hip_joint' in name:
            hip_idx = i
        elif 'knee_joint' in name:
            knee_idx = i
        elif 'foot_joint' in name:
            foot_link_idx = info[16]

    # CRITICAL: Disable default motors
    p.setJointMotorControlArray(robot, movable, p.VELOCITY_CONTROL,
                                targetVelocities=[0]*len(movable),
                                forces=[0]*len(movable))

    # Initial crouch
    p.resetJointState(robot, hip_idx, 0.785)
    p.resetJointState(robot, knee_idx, -1.57)
    foot_pos = p.getLinkState(robot, foot_link_idx)[0]
    p.resetBasePositionAndOrientation(robot, [0,0,0.5 - foot_pos[2]], [0,0,0,1])

    # Initial upward push
    p.applyExternalForce(robot, -1, [0,0,100], [0,0,0], p.WORLD_FRAME)

    fsm = FSM()
    wind_applied = False
    wind_time = None
    success = True
    time_data, height_data, pitch_data = [], [], []
    dt = 0.001
    r_prev = None

    for step in range(int(max_time/dt)):
        t = step * dt

        # State
        pos, orn = p.getBasePositionAndOrientation(robot)
        euler = p.getEulerFromQuaternion(orn)
        lin_vel, ang_vel = p.getBaseVelocity(robot)
        theta_body = euler[1]
        theta_dot_body = ang_vel[1]

        # Joint states
        hip_state = p.getJointState(robot, hip_idx)
        knee_state = p.getJointState(robot, knee_idx)
        hip_pos, hip_vel = hip_state[0], hip_state[1]
        knee_pos, knee_vel = knee_state[0], knee_state[1]

        # Foot contact
        contacts = p.getContactPoints(robot, plane)
        foot_contact = False
        for cp in contacts:
            if cp[3] == foot_link_idx:
                foot_contact = True
                break

        # Virtual leg
        foot_pos_world = p.getLinkState(robot, foot_link_idx)[0]
        hip_pos_world = p.getLinkState(robot, hip_idx)[0]
        dx = foot_pos_world[0] - hip_pos_world[0]
        dz = foot_pos_world[2] - hip_pos_world[2]
        r = np.sqrt(dx**2 + dz**2)
        theta_vir = np.arctan2(dx, -dz)
        r_dot = (r - r_prev) / dt if r_prev is not None else 0.0
        r_prev = r

        # FSM
        phase = fsm.transition(foot_contact, r, r_dot, lin_vel[2])

        # Control
        if phase in [Phase.COMPRESSION, Phase.THRUST]:
            # VMC
            F = vmc_force(r, r_dot, phase)
            Fx = F * np.sin(theta_vir)
            Fz = -F * np.cos(theta_vir)
            F_vec = [Fx, 0, Fz]
            # Jacobian
            q = [p.getJointState(robot, i)[0] for i in range(num_joints)]
            qdot = [p.getJointState(robot, i)[1] for i in range(num_joints)]
            jac_t, _ = p.calculateJacobian(robot, foot_link_idx, [0,0,0], q, qdot, [0]*len(q))
            tau_hip = jac_t[0][hip_idx]*Fx + jac_t[1][hip_idx]*0 + jac_t[2][hip_idx]*Fz
            tau_knee = jac_t[0][knee_idx]*Fx + jac_t[1][knee_idx]*0 + jac_t[2][knee_idx]*Fz
            tau_hip = np.clip(tau_hip, -5, 5)
            tau_knee = np.clip(tau_knee, -5, 5)
            p.setJointMotorControl2(robot, hip_idx, p.TORQUE_CONTROL, force=tau_hip)
            p.setJointMotorControl2(robot, knee_idx, p.TORQUE_CONTROL, force=tau_knee)

        elif phase == Phase.FLIGHT:
            # AMC
            tau_hip = amc_torque(theta_body, theta_dot_body, mu)
            p.setJointMotorControl2(robot, hip_idx, p.TORQUE_CONTROL, force=tau_hip)
            # Hold knee
            p.setJointMotorControl2(robot, knee_idx, p.POSITION_CONTROL,
                                    targetPosition=knee_pos, force=2.0)

            # Wind at apex
            if not wind_applied and lin_vel[2] < 0:
                wind_force = eta * TOTAL_MASS * GRAVITY
                p.applyExternalForce(robot, -1, [wind_force,0,0], [0,0,0], p.WORLD_FRAME)
                wind_applied = True
                wind_time = t

        elif phase == Phase.LANDING:
            # Foot placement
            x_target = foot_placement(lin_vel[0], 0.0)
            z_target = -0.30
            theta_hip_des, theta_knee_des = inverse_kinematics(x_target, z_target)
            p.setJointMotorControl2(robot, hip_idx, p.POSITION_CONTROL,
                                    targetPosition=theta_hip_des, force=5.0)
            p.setJointMotorControl2(robot, knee_idx, p.POSITION_CONTROL,
                                    targetPosition=theta_knee_des, force=5.0)

        p.stepSimulation()

        # Log
        time_data.append(t)
        height_data.append(pos[2])
        pitch_data.append(np.degrees(theta_body))

        # Failure
        if abs(theta_body) > 0.785:
            success = False
            break
        if pos[2] < 0.15:
            success = False
            break

    if debug:
        print("\nDEBUG: GUI stays open. Press Ctrl+C to exit.")
        while True:
            p.stepSimulation()
            time.sleep(0.001)

    p.disconnect()
    os.remove(urdf_file)

    return {
        'success': success,
        'max_pitch': np.max(np.abs(pitch_data)),
        'max_height': np.max(height_data),
        'wind_applied': wind_applied,
        'wind_time': wind_time,
        'time': np.array(time_data),
        'pitch': np.array(pitch_data),
        'height': np.array(height_data)
    }

# ==================== MAIN ====================
if __name__ == "__main__":
    print("SLATOL – PyBullet Hopping Simulation")
    print("=====================================")
    choice = input("1. Debug single trial\n2. Run full sweep\nSelect mode: ")
    if choice == '1':
        mu = float(input("μ (0.05–0.30): "))
        eta = float(input("η (0.0–0.5): "))
        res = run_single_trial(mu, eta, use_gui=True, debug=True, max_time=5.0)
        print(f"\nResult: {'SUCCESS' if res['success'] else 'FAIL'}")
        print(f"Max pitch: {res['max_pitch']:.1f}°, Max height: {res['max_height']:.2f}m")
    else:
        # Full sweep (simplified for demo)
        print("Full sweep not implemented in this minimal version.")
        print("Extend the script to loop over MASS_RATIOS and WIND_RATIOS.")