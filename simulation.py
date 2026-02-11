"""
Single trial simulation for SLATOL
Nik Adam Muqridz (2125501)
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
from urdf_generator import generate_urdf
from controllers import Phase, SLATOL_FSM, compute_vmc_torques, compute_amc_torque
from controllers import compute_foot_placement, inverse_kinematics

def run_single_trial(mu, eta, trial_id, use_gui=False, max_time=3.0):
    """
    Returns dict with success, max_pitch, max_height, settling_time, max_overshoot,
    and time series for plotting.
    """
    # --- Generate URDF for this mu ---
    urdf_filename = f"slatol_mu_{mu:.2f}.urdf"
    generate_urdf(mu, output_path=urdf_filename)

    # --- Connect to PyBullet ---
    if use_gui:
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0,
                                     cameraPitch=-40, cameraTargetPosition=[0,0,0.5])
    else:
        p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(0.001)
    p.setRealTimeSimulation(0)

    # --- Load ground and robot ---
    plane = p.loadURDF("plane.urdf")
    p.changeDynamics(plane, -1, lateralFriction=0.9)
    robot = p.loadURDF(urdf_filename, basePosition=[0, 0, 0.5])

    # Get joint indices by name (assuming order in URDF)
    num_joints = p.getNumJoints(robot)
    hip_idx = None
    knee_idx = None
    foot_link_idx = None
    for i in range(num_joints):
        info = p.getJointInfo(robot, i)
        name = info[1].decode('utf-8')
        if 'hip_joint' in name:
            hip_idx = i
        elif 'knee_joint' in name:
            knee_idx = i
        elif 'foot_joint' in name:   # the child link index
            foot_link_idx = info[16]   # link index of child
    # fallback if not found (by position)
    if hip_idx is None:
        hip_idx = 1
        knee_idx = 2
        foot_link_idx = 4   # typical after body (0), femur (1), tibia (2), foot (3)? careful

    # Set initial crouched posture (Q4): hip=45°, knee=-90°
    p.resetJointState(robot, hip_idx, 0.785)
    p.resetJointState(robot, knee_idx, -1.57)

    # Enable torque control for both joints
    p.setJointMotorControl2(robot, hip_idx, p.VELOCITY_CONTROL, force=0)
    p.setJointMotorControl2(robot, knee_idx, p.VELOCITY_CONTROL, force=0)

    # Give initial upward push to start hopping
    p.applyExternalForce(robot, -1, [0, 0, 200], [0,0,0], p.WORLD_FRAME)

    # --- Data logging ---
    time_data = []
    height_data = []
    pitch_data = []
    joint_pos_data = []
    joint_vel_data = []
    joint_torque_data = []
    foot_contact_data = []

    fsm = SLATOL_FSM()
    wind_applied = False
    wind_start_time = None
    success = True
    failure_cause = ""

    # For r_dot estimation
    r_prev = None
    dt = 0.001

    # Simulation loop
    for step in range(int(max_time / dt)):
        t = step * dt

        # --- State estimation ---
        pos, orn = p.getBasePositionAndOrientation(robot)
        euler = p.getEulerFromQuaternion(orn)
        lin_vel, ang_vel = p.getBaseVelocity(robot)

        # Joint states
        hip_state = p.getJointState(robot, hip_idx)
        knee_state = p.getJointState(robot, knee_idx)
        hip_pos, hip_vel = hip_state[0], hip_state[1]
        knee_pos, knee_vel = knee_state[0], knee_state[1]

        # Foot contact detection
        contact_points = p.getContactPoints(robot, plane)
        foot_contact = False
        contact_force = 0.0
        for cp in contact_points:
            if cp[3] == foot_link_idx:   # link index A
                foot_contact = True
                contact_force += cp[9]   # normal force
        foot_contact_flag = foot_contact and contact_force > 5.0

        # Virtual leg length and angle (hip to foot)
        foot_pos = p.getLinkState(robot, foot_link_idx)[0]
        hip_pos_world = p.getLinkState(robot, hip_idx)[0]   # joint origin in world
        dx = foot_pos[0] - hip_pos_world[0]
        dz = foot_pos[2] - hip_pos_world[2]
        r = np.sqrt(dx**2 + dz**2)
        theta_vir = np.arctan2(dx, -dz)   # Eq 3.3

        # r_dot (finite difference)
        if r_prev is not None:
            r_dot = (r - r_prev) / dt
        else:
            r_dot = 0.0
        r_prev = r

        # --- FSM transition ---
        phase = fsm.transition(foot_contact_flag, r, r_dot, lin_vel[2])

        # --- Control selection ---
        if phase in [Phase.COMPRESSION, Phase.THRUST]:
            # Stance: VMC
            tau_hip, tau_knee = compute_vmc_torques(
                robot, hip_idx, knee_idx, foot_link_idx,
                phase, r, r_dot, theta_vir, mu
            )
            p.setJointMotorControl2(robot, hip_idx, p.TORQUE_CONTROL, force=tau_hip)
            p.setJointMotorControl2(robot, knee_idx, p.TORQUE_CONTROL, force=tau_knee)

        elif phase == Phase.FLIGHT:
            # Flight: AMC on hip, position servo on knee (Q8)
            tau_hip = compute_amc_torque(euler[1], ang_vel[1], mu)
            p.setJointMotorControl2(robot, hip_idx, p.TORQUE_CONTROL, force=tau_hip)
            # Hold knee at current position (prevent flailing)
            p.setJointMotorControl2(robot, knee_idx, p.POSITION_CONTROL,
                                    targetPosition=knee_pos,
                                    positionGain=0.5, velocityGain=1.0,
                                    force=2.0, maxVelocity=5.0)

            # --- Wind disturbance at apex (Q11) ---
            if not wind_applied and phase == Phase.FLIGHT and lin_vel[2] < 0:
                wind_force = eta * 1.0 * 9.81   # η * m_total * g
                p.applyExternalForce(robot, -1, [wind_force, 0, 0],
                                     [0,0,0], p.WORLD_FRAME)
                wind_applied = True
                wind_start_time = t

        elif phase == Phase.LANDING:
            # Landing preparation: foot placement (Q10)
            x_target = compute_foot_placement(lin_vel[0], 0.0)
            z_target = -0.30   # approximate foot height at landing (below hip)
            theta_hip_des, theta_knee_des = inverse_kinematics(x_target, z_target)

            p.setJointMotorControl2(robot, hip_idx, p.POSITION_CONTROL,
                                    targetPosition=theta_hip_des,
                                    positionGain=0.5, velocityGain=1.0,
                                    force=5.0, maxVelocity=10.0)
            p.setJointMotorControl2(robot, knee_idx, p.POSITION_CONTROL,
                                    targetPosition=theta_knee_des,
                                    positionGain=0.5, velocityGain=1.0,
                                    force=5.0, maxVelocity=10.0)

        # --- Step simulation ---
        p.stepSimulation()

        # --- Log data ---
        time_data.append(t)
        height_data.append(pos[2])
        pitch_data.append(euler[1] * 180/np.pi)
        joint_pos_data.append([hip_pos, knee_pos])
        joint_vel_data.append([hip_vel, knee_vel])
        # applied joint torques (last element of joint state)
        hip_torque_applied = p.getJointState(robot, hip_idx)[3]
        knee_torque_applied = p.getJointState(robot, knee_idx)[3]
        joint_torque_data.append([hip_torque_applied, knee_torque_applied])
        foot_contact_data.append(1 if foot_contact_flag else 0)

        # --- Failure check (Q12) ---
        if abs(euler[1]) > 0.785:   # 45 deg
            success = False
            failure_cause = "pitch_exceeded"
            break
        if pos[2] < 0.15:           # 15 cm
            success = False
            failure_cause = "height_crash"
            break

    p.disconnect()

    # --- Compute quality metrics (Q13) ---
    settling_time = None
    max_overshoot = None
    if success and wind_applied:
        t_arr = np.array(time_data)
        pitch_arr = np.array(pitch_data)
        wind_idx = np.argmin(np.abs(t_arr - wind_start_time))
        post_pitch = pitch_arr[wind_idx:]
        # Max overshoot (absolute) after wind
        max_overshoot = np.max(np.abs(post_pitch))
        # Settling time to ±2 deg
        within_band = np.abs(post_pitch) < 2.0
        if np.any(within_band):
            settle_idx = np.where(within_band)[0][0]
            settling_time = t_arr[wind_idx + settle_idx] - wind_start_time

    # Return results
    return {
        'mu': mu,
        'eta': eta,
        'success': success,
        'failure_cause': failure_cause,
        'max_height': np.max(height_data) if height_data else 0,
        'max_pitch': np.max(np.abs(pitch_data)) if pitch_data else 0,
        'settling_time': settling_time,
        'max_overshoot': max_overshoot,
        'wind_applied': wind_applied,
        'time': np.array(time_data),
        'height': np.array(height_data),
        'pitch': np.array(pitch_data),
        'joint_pos': np.array(joint_pos_data),
        'joint_vel': np.array(joint_vel_data),
        'joint_torque': np.array(joint_torque_data),
        'foot_contact': np.array(foot_contact_data)
    }