import pybullet as p
import pybullet_data
import time
import math
import os
import csv
import sys  # Added for interactive menu
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. DYNAMIC URDF GENERATION
# ==========================================
def generate_urdf(mu, filename="temp_slatol.urdf"):
    """Generates a SLATOL URDF dynamically based on the Leg Mass Ratio (mu)."""
    total_mass = 1.0
    mass_body = total_mass * (1.0 - mu)
    mass_link = (total_mass * mu) / 2.0  # Split evenly between Femur and Tibia
    
    urdf_content = f"""<?xml version="1.0"?>
<robot name="slatol">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass_body}"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.2 0.1 0.1"/></geometry>
      <material name="blue"><color rgba="0.2 0.2 0.8 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.2 0.1 0.1"/></geometry>
    </collision>
  </link>

  <link name="femur">
    <inertial>
      <origin xyz="0 0 -0.075" rpy="0 0 0"/> <mass value="{mass_link}"/>
      <inertia ixx="0.0005" ixy="0" ixz="0" iyy="0.0005" iyz="0" izz="0.0001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.075" rpy="0 0 0"/>
      <geometry><cylinder radius="0.02" length="0.15"/></geometry>
      <material name="gray"><color rgba="0.5 0.5 0.5 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.075" rpy="0 0 0"/>
      <geometry><cylinder radius="0.02" length="0.15"/></geometry>
    </collision>
  </link>

  <link name="tibia">
    <inertial>
      <origin xyz="0 0 -0.075" rpy="0 0 0"/> <mass value="{mass_link}"/>
      <inertia ixx="0.0005" ixy="0" ixz="0" iyy="0.0005" iyz="0" izz="0.0001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.075" rpy="0 0 0"/>
      <geometry><cylinder radius="0.02" length="0.15"/></geometry>
      <material name="red"><color rgba="0.8 0.2 0.2 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.075" rpy="0 0 0"/>
      <geometry><cylinder radius="0.02" length="0.15"/></geometry>
    </collision>
  </link>

  <joint name="hip_joint" type="continuous">
    <parent link="base_link"/>
    <child link="femur"/>
    <origin xyz="0 0 -0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <joint name="knee_joint" type="revolute">
    <parent link="femur"/>
    <child link="tibia"/>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.61" upper="0" effort="5.0" velocity="22.0"/> <dynamics damping="0.1" friction="0.0"/>
  </joint>
</robot>
"""
    with open(filename, "w") as f:
        f.write(urdf_content)
    return filename

# ==========================================
# 2. HYBRID REACTIVE CONTROLLER
# ==========================================
class SLATOL_Controller:
    def __init__(self, mu):
        self.mu = mu
        self.L1 = 0.15
        self.L2 = 0.15
        self.max_torque = 5.0
        
        # VMC Parameters (Stance)
        self.k_s = 2000.0   # Stiff spring
        self.b_s = 10.0     # Low damping to preserve energy
        self.r0 = 0.25      # Target Virtual Leg Length
        self.f_thrust = 45.0 # Strong thrust for liftoff
        
        # AMC Adaptive Gains (Flight)
        self.kp_nom = 25.0
        self.kd_nom = 1.5
        self.kp_amc = self.kp_nom * (self.mu / 0.05)
        self.kd_amc = self.kd_nom * (self.mu / 0.05)
        
        # Target Tuck Angle
        self.knee_tuck_rad = -math.radians(150)

    def compute_kinematics(self, theta1, theta2, dtheta1, dtheta2):
        """Forward Kinematics and Jacobian"""
        # Cartesian position relative to hip
        x = self.L1 * math.sin(theta1) + self.L2 * math.sin(theta1 + theta2)
        z = -self.L1 * math.cos(theta1) - self.L2 * math.cos(theta1 + theta2)
        r = math.sqrt(x**2 + z**2)
        
        # Analytical Jacobian components
        dx_dth1 = self.L1 * math.cos(theta1) + self.L2 * math.cos(theta1 + theta2)
        dx_dth2 = self.L2 * math.cos(theta1 + theta2)
        dz_dth1 = self.L1 * math.sin(theta1) + self.L2 * math.sin(theta1 + theta2)
        dz_dth2 = self.L2 * math.sin(theta1 + theta2)
        
        # Radial velocity
        vx = dx_dth1 * dtheta1 + dx_dth2 * dtheta2
        vz = dz_dth1 * dtheta1 + dz_dth2 * dtheta2
        r_dot = (x * vx + z * vz) / r if r != 0 else 0
        
        return x, z, r, r_dot, dx_dth1, dx_dth2, dz_dth1, dz_dth2

    def get_torques(self, is_stance, is_thrust, theta1, theta2, dtheta1, dtheta2, pitch, dpitch):
        x, z, r, r_dot, dx_dth1, dx_dth2, dz_dth1, dz_dth2 = self.compute_kinematics(theta1, theta2, dtheta1, dtheta2)
        
        if is_stance:
            # 1. Stance Phase (VMC)
            # Spring force + Damping
            f_radial = self.k_s * (self.r0 - r) - self.b_s * r_dot
            
            # Thrust Injection (Only when extending)
            if is_thrust:
                f_radial += self.f_thrust
                
            # Convert Polar Force to Cartesian
            # Angle of virtual leg
            phi = math.atan2(x, -z)
            fx = f_radial * math.sin(phi)
            fz = -f_radial * math.cos(phi)
            
            # Jacobian Transpose to Joint Torques
            tau_hip = fx * dx_dth1 + fz * dz_dth1
            tau_knee = fx * dx_dth2 + fz * dz_dth2
            
        else:
            # 2. Flight Phase (AMC + Tuck)
            # Hip: Active Attitude Control
            tau_hip = -self.kp_amc * pitch - self.kd_amc * dpitch
            
            # Knee: Passive Tuck (Position Servo)
            kp_tuck = 10.0
            kd_tuck = 0.5
            tau_knee = -kp_tuck * (theta2 - self.knee_tuck_rad) - kd_tuck * dtheta2

        # Actuator Saturation
        tau_hip = np.clip(tau_hip, -self.max_torque, self.max_torque)
        tau_knee = np.clip(tau_knee, -self.max_torque, self.max_torque)
        
        return tau_hip, tau_knee

# ==========================================
# 3. SIMULATION EXECUTION
# ==========================================
def run_simulation(mu, eta, gui_mode=False):
    if gui_mode:
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    else:
        p.connect(p.DIRECT)
        
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1/1000.0)
    
    # Environment
    plane = p.loadURDF("plane.urdf")
    p.changeDynamics(plane, -1, lateralFriction=0.9)
    
    # Load Robot - SPAWN HIGH TO FORCE DROP & JUMP
    urdf_file = generate_urdf(mu)
    start_pos = [0, 0, 0.6]  # FIXED: Height 0.6m ensures drop impact
    robot = p.loadURDF(urdf_file, start_pos, useFixedBase=False)
    
    # Enable Force Sensors on Knee (Tibia)
    p.enableJointForceTorqueSensor(robot, 1, enableSensor=1)
    
    # Initial Pose (Slightly bent to avoid singularity)
    p.resetJointState(robot, 0, 0.5)  # Hip
    p.resetJointState(robot, 1, -1.0) # Knee
    
    # Turn off default position control (Allow torque control)
    p.setJointMotorControl2(robot, 0, p.VELOCITY_CONTROL, force=0)
    p.setJointMotorControl2(robot, 1, p.VELOCITY_CONTROL, force=0)
    
    controller = SLATOL_Controller(mu)
    
    wind_applied = False
    log_t, log_z, log_pitch = [], [], []
    max_pitch = 0.0
    
    # Simulation Loop
    for i in range(2000): # 2.0 seconds
        t = i / 1000.0
        
        # 1. Get State
        pos, ori = p.getBasePositionAndOrientation(robot)
        lin_vel, ang_vel = p.getBaseVelocity(robot)
        pitch = p.getEulerFromQuaternion(ori)[1]
        dpitch = ang_vel[1]
        
        joint_states = p.getJointStates(robot, [0, 1])
        th1, dth1 = joint_states[0][0], joint_states[0][1]
        th2, dth2 = joint_states[1][0], joint_states[1][1]
        
        # 2. Enforce 2D Planar Constraint (Lock Y, Roll, Yaw)
        p.resetBaseVelocity(robot, [lin_vel[0], 0, lin_vel[2]], [0, dpitch, 0])
        curr_quat = p.getQuaternionFromEuler([0, pitch, 0])
        p.resetBasePositionAndOrientation(robot, [pos[0], 0, pos[2]], curr_quat)

        # 3. State Machine Logic
        pts = p.getContactPoints(robot, plane)
        is_stance = len(pts) > 0
        
        # Thrust is active if we are in stance AND extending leg
        _, _, r, r_dot, _, _, _, _ = controller.compute_kinematics(th1, th2, dth1, dth2)
        is_thrust = is_stance and (r_dot >= 0)
        
        # 4. Apply Wind Disturbance at Apex
        if not is_stance and lin_vel[2] < 0 and not wind_applied:
            f_wind = eta * 1.0 * 9.81
            p.applyExternalForce(robot, -1, [f_wind, 0, 0], pos, p.WORLD_FRAME)
            wind_applied = True
            if gui_mode:
                print(f"[INFO] Wind Gust Applied: {f_wind:.2f}N")
            
        # 5. Get Torques from Controller
        tau_hip, tau_knee = controller.get_torques(is_stance, is_thrust, th1, th2, dth1, dth2, pitch, dpitch)
        
        p.setJointMotorControl2(robot, 0, p.TORQUE_CONTROL, force=tau_hip)
        p.setJointMotorControl2(robot, 1, p.TORQUE_CONTROL, force=tau_knee)
        
        p.stepSimulation()
        
        if gui_mode:
            time.sleep(1/1000.0)
            
        # Logging
        log_t.append(t)
        log_z.append(pos[2])
        log_pitch.append(math.degrees(pitch))
        if abs(math.degrees(pitch)) > abs(max_pitch):
            max_pitch = math.degrees(pitch)
            
        # Early Failure Check
        if pos[2] < 0.05 or abs(math.degrees(pitch)) > 45:
            if gui_mode:
                print("[INFO] Robot Crashed/Tumbled!")
            p.disconnect()
            return False, log_t, log_z, log_pitch
            
    p.disconnect()
    return True, log_t, log_z, log_pitch

# ==========================================
# 4. INTERACTIVE MENU (VS CODE FRIENDLY)
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*40)
    print("   SLATOL SIMULATION LAUNCHER")
    print("="*40)
    print("1. Single Simulation with GUI (Demo)")
    print("2. Batch Processing (Generate Map)")
    print("-" * 40)
    
    choice = input("Select option (1 or 2): ").strip()
    
    if choice == '1':
        print("[INFO] Running single nominal hop in GUI mode...")
        success, t, z, pitch = run_simulation(mu=0.05, eta=0.0, gui_mode=True)
        
        # Plot Single Run
        plt.figure(figsize=(10, 6))
        plt.subplot(2,1,1)
        plt.plot(t, z, label="CoM Height", color='b')
        plt.axhline(y=0.6, color='g', linestyle='--', label="Spawn Height")
        plt.ylabel("Height (m)")
        plt.title("SLATOL Jump Trajectory")
        plt.legend()
        plt.grid()
        
        plt.subplot(2,1,2)
        plt.plot(t, pitch, label="Pitch Angle", color='r')
        plt.ylabel("Pitch (deg)")
        plt.xlabel("Time (s)")
        plt.grid()
        plt.tight_layout()
        plt.show()

    elif choice == '2':
        print("[INFO] Running Stability Search (Batch Mode)...")
        mu_range = np.arange(0.05, 0.35, 0.05)
        eta_range = np.arange(0.0, 0.55, 0.05)
        
        max_eta_per_mu = []

        with open("stability_map.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Mu", "Eta", "Stable"])
            
            for mu in mu_range:
                max_eta = 0.0
                print(f"Testing Leg Mass Ratio: {mu:.2f}...", end=" ")
                for eta in eta_range:
                    stable, _, _, _ = run_simulation(mu, eta, gui_mode=False)
                    writer.writerow([round(mu,2), round(eta,2), stable])
                    
                    if stable:
                        max_eta = eta
                    else:
                        break # Optimization: If fail, higher winds will also fail
                
                max_eta_per_mu.append(max_eta)
                print(f"Max Wind: {max_eta:.2f}")

        # Plot Stability Map
        plt.figure(figsize=(8, 6))
        plt.plot(mu_range, max_eta_per_mu, 'k-o', linewidth=2, label="Stability Boundary")
        plt.fill_between(mu_range, max_eta_per_mu, 0, color='lightgreen', alpha=0.5, label="Sufficiency Region")
        plt.fill_between(mu_range, max_eta_per_mu, 0.55, color='lightcoral', alpha=0.5, label="Failure Region")
        
        plt.title("Stability Sufficiency Map")
        plt.xlabel("Leg Mass Ratio ($\mu$)")
        plt.ylabel("Wind Disturbance Ratio ($\eta$)")
        plt.grid(True, linestyle='--')
        plt.legend()
        plt.savefig("stability_map.png")
        print("\n[INFO] Batch complete. Saved 'stability_map.csv' and 'stability_map.png'.")

    else:
        print("[ERROR] Invalid selection. Exiting.")