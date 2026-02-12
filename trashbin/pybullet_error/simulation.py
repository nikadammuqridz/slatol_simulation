"""
Single trial simulation for SLATOL
FIXED: 
- URDF_USE_INERTIA_FROM_FILE flag
- Proper torque control enable (zero forces)
- Extract DoF-length vectors for Jacobian
- Correct DoF indexing
Nik Adam Muqridz (2125501)
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
from urdf_generator import generate_urdf
from controllers import Phase, SLATOL_FSM, compute_vmc_torques, compute_amc_torque
from controllers import compute_foot_placement, inverse_kinematics

def run_single_trial(mu, eta, trial_id, use_gui=False, debug=False, max_time=3.0):
    """
    Returns dict with results.
    If debug=True, GUI stays open after simulation for inspection.
    """
    # --- Generate URDF for this mu ---
    urdf_filename = f"slatol_mu_{mu:.2f}.urdf"
    generate_urdf(mu, output_path=urdf_filename)

    # --- Connect to PyBullet ---
    if use_gui or debug:
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

    # --- Load ground ---
    plane = p.loadURDF("plane.urdf")
    p.changeDynamics(plane, -1, lateralFriction=0.9)

    
        # --- CRITICAL FIX: Completely disable default motors ---
    # Step 1: Set both joints to VELOCITY_CONTROL with zero force AND zero target velocity
    p.setJointMotorControlArray(
        robot,
        [hip_idx, knee_idx],
        p.VELOCITY_CONTROL,
        targetVelocities=[0.0, 0.0],
        forces=[0.0, 0.0]
    )

    # Step 2: Explicitly set maximum force to 0 (extra safety)
    p.setJointMotorControlArray(
        robot,
        [hip_idx, knee_idx],
        p.VELOCITY_CONTROL,
        forces=[0.0, 0.0]
    )

    # Step 3: For TORQUE_CONTROL, also set maxVelocity to 0 to prevent any residual PD action
    p.setJointMotorControlArray(
        robot,
        [hip_idx, knee_idx],
        p.TORQUE_CONTROL,
        forces=[0.0, 0.0]
    )
    robot = p.loadURDF(
        urdf_filename,
        basePosition=[0, 0, 0.5],
        flags=p.URDF_USE_INERTIA_FROM_FILE   # This was missing!
    )
        # --- DIAGNOSTIC: Verify inertia was loaded ---
    print("\n🔍 INERTIA DIAGNOSTIC:")
    femur_dyn = p.getDynamicsInfo(robot, hip_idx)  # Dynamics info for link attached to hip joint
    tibia_dyn = p.getDynamicsInfo(robot, knee_idx)  # Dynamics info for link attached to knee joint
    print(f"  Femur mass: {femur_dyn[0]:.4f} kg")  # Should be ~ μ*0.5
    print(f"  Femur inertia: {femur_dyn[2]}")      # Should NOT be all zeros
    print(f"  Tibia mass: {tibia_dyn[0]:.4f} kg")
    print(f"  Tibia inertia: {tibia_dyn[2]}")
    print("-" * 50)

    # --- Get joint info and identify movable joints ---
    num_joints = p.getNumJoints(robot)
    hip_idx = None
    knee_idx = None
    foot_link_idx = None
    movable_joint_indices = []
    for i in range(num_joints):
        info = p.getJointInfo(robot, i)
        name = info[1].decode('utf-8')
        joint_type = info[2]
        if joint_type != p.JOINT_FIXED:
            movable_joint_indices.append(i)
        if 'hip_joint' in name:
            hip_idx = i
        elif 'knee_joint' in name:
            knee_idx = i
        elif 'foot_joint' in name:
            foot_link_idx = info[16]   # child link index

    if hip_idx is None or knee_idx is None or foot_link_idx is None:
        raise ValueError("Could not find required joints/links in URDF.")

    # --- Create mapping from original joint index to DoF index ---
    joint_idx_to_dof = {idx: dof_idx for dof_idx, idx in enumerate(movable_joint_indices)}
    hip_dof_idx = joint_idx_to_dof[hip_idx]
    knee_dof_idx = joint_idx_to_dof[knee_idx]

    print(f"Movable joints (DoF): {movable_joint_indices}")
    print(f"Hip: orig idx {hip_idx} -> DoF idx {hip_dof_idx}")
    print(f"Knee: orig idx {knee_idx} -> DoF idx {knee_dof_idx}")

    # --- CRITICAL FIX 2: Properly disable velocity control to enable torque control ---
    # Set both joints to VELOCITY_CONTROL with zero forces
    p.setJointMotorControlArray(
        robot,
        [hip_idx, knee_idx],
        p.VELOCITY_CONTROL,
        forces=[0.0, 0.0]
    )

    # --- Start with robot on ground, leg compressed ---
    # Crouched: hip=45°, knee=-90°
    p.resetJointState(robot, hip_idx, 0.785)   # 45°
    p.resetJointState(robot, knee_idx, -1.57)  # -90°
    # Move base so foot touches ground (z=0)
    foot_pos = p.getLinkState(robot, foot_link_idx)[0]
    p.resetBasePositionAndOrientation(robot, [0, 0, 0.5 - foot_pos[2]], [0,0,0,1])

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

    # Give initial upward push (reduced from 200N to 120N)
    p.applyExternalForce(robot, -1, [0, 0, 120], [0,0,0], p.WORLD_FRAME)

    # Simulation loop
    for step in range(int(max_time / dt)):
        t = step * dt

        # --- State estimation ---
        pos, orn = p.getBasePositionAndOrientation(robot)
        euler = p.getEulerFromQuaternion(orn)
        lin_vel, ang_vel = p.getBaseVelocity(robot)

        # --- Joint states (ALL joints) ---
        joint_states = [p.getJointState(robot, i) for i in range(num_joints)]

        # --- Extract DoF-length vectors for movable joints ---
        q_dof = [joint_states[i][0] for i in movable_joint_indices]
        qdot_dof = [joint_states[i][1] for i in movable_joint_indices]

        # Specific joint states for hip/knee (using original indices)
        hip_state = joint_states[hip_idx]
        knee_state = joint_states[knee_idx]
        hip_pos, hip_vel = hip_state[0], hip_state[1]
        knee_pos, knee_vel = knee_state[0], knee_state[1]

        # --- Foot contact detection ---
        contact_points = p.getContactPoints(robot, plane)
        foot_contact = False
        contact_force = 0.0
        for cp in contact_points:
            if cp[3] == foot_link_idx:
                foot_contact = True
                contact_force += cp[9]
        foot_contact_flag = foot_contact and contact_force > 5.0

        # --- Virtual leg length and angle (hip to foot) ---
        foot_pos_world = p.getLinkState(robot, foot_link_idx)[0]
        hip_pos_world = p.getLinkState(robot, hip_idx)[0]
        dx = foot_pos_world[0] - hip_pos_world[0]
        dz = foot_pos_world[2] - hip_pos_world[2]
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
                robot, foot_link_idx,
                phase, r, r_dot, theta_vir,
                q_dof, qdot_dof,           # DoF-length vectors
                hip_dof_idx, knee_dof_idx   # DoF indices
            )
            p.setJointMotorControl2(robot, hip_idx, p.TORQUE_CONTROL, force=tau_hip)
            p.setJointMotorControl2(robot, knee_idx, p.TORQUE_CONTROL, force=tau_knee)

        elif phase == Phase.FLIGHT:
            # Flight: AMC on hip, position servo on knee
            tau_hip = compute_amc_torque(euler[1], ang_vel[1], mu)
            p.setJointMotorControl2(robot, hip_idx, p.TORQUE_CONTROL, force=tau_hip)
            # Hold knee at current position
            p.setJointMotorControl2(robot, knee_idx, p.POSITION_CONTROL,
                                    targetPosition=knee_pos,
                                    positionGain=0.5, velocityGain=1.0,
                                    force=2.0, maxVelocity=5.0)

            # --- Wind disturbance at apex ---
            if not wind_applied and phase == Phase.FLIGHT and lin_vel[2] < 0:
                wind_force = eta * 1.0 * 9.81
                p.applyExternalForce(robot, -1, [wind_force, 0, 0],
                                     [0,0,0], p.WORLD_FRAME)
                wind_applied = True
                wind_start_time = t

        elif phase == Phase.LANDING:
            # Landing preparation: foot placement
            x_target = compute_foot_placement(lin_vel[0], 0.0)
            z_target = -0.30   # approximate foot height at landing
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
        hip_torque_applied = p.getJointState(robot, hip_idx)[3]
        knee_torque_applied = p.getJointState(robot, knee_idx)[3]
        joint_torque_data.append([hip_torque_applied, knee_torque_applied])
        foot_contact_data.append(1 if foot_contact_flag else 0)

        # --- Failure check ---
        if abs(euler[1]) > 0.785:   # 45 deg
            success = False
            failure_cause = "pitch_exceeded"
            break
        if pos[2] < 0.15:           # 15 cm
            success = False
            failure_cause = "height_crash"
            break

    # --- Keep GUI open in debug mode ---
    if debug:
        print("\n🔍 DEBUG MODE: GUI stays open. Press Ctrl+C or close window to exit.")
        try:
            while True:
                p.stepSimulation()
                time.sleep(0.001)
        except KeyboardInterrupt:
            pass

    p.disconnect()

    # --- Compute quality metrics ---
    settling_time = None
    max_overshoot = None
    if success and wind_applied:
        t_arr = np.array(time_data)
        pitch_arr = np.array(pitch_data)
        wind_idx = np.argmin(np.abs(t_arr - wind_start_time))
        post_pitch = pitch_arr[wind_idx:]
        max_overshoot = np.max(np.abs(post_pitch))
        within_band = np.abs(post_pitch) < 2.0
        if np.any(within_band):
            settle_idx = np.where(within_band)[0][0]
            settling_time = t_arr[wind_idx + settle_idx] - wind_start_time

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
        'wind_time': wind_start_time,
        'time': np.array(time_data),
        'height': np.array(height_data),
        'pitch': np.array(pitch_data),
        'joint_pos': np.array(joint_pos_data),
        'joint_vel': np.array(joint_vel_data),
        'joint_torque': np.array(joint_torque_data),
        'foot_contact': np.array(foot_contact_data)
    }
       