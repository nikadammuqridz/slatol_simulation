import numpy as np
import math

class SLATOLController:
    """Finite State Machine controller for SLATOL robot"""
    
    def __init__(self, robot, target_omega=15.0):
        """
        Initialize controller with adaptive gain scheduling
        
        Args:
            robot: SLATOLRobot instance
            target_omega: Target natural frequency [rad/s]
        """
        self.robot = robot
        self.target_omega = target_omega
        
        # Control states
        self.state = "STANCE_COMPRESSION"
        self.phase_timer = 0
        self.jump_counter = 0
        
        # SLIP parameters
        self.l0 = robot.l1 + robot.l2  # Natural leg length
        self.k_spring = 1000.0  # Virtual spring stiffness [N/m]
        self.b_damping = 50.0   # Virtual damping [Ns/m]
        
        # Raibert heuristic parameters
        self.T_stance = 0.2  # Stance duration [s]
        self.K_raibert = 0.1  # Raibert gain
        
        # Foot placement
        self.foot_target_x = 0
        self.foot_target_z = -self.l0
        
        # Energy compensation (from animal-inspired model)
        self.alpha_energy = 2800  # From Huang & Zhang Table 3
        
        # State machine timing
        self.timing = {
            "STANCE_COMPRESSION": 0.1,
            "STANCE_THRUST": 0.1,
            "FLIGHT_SWING": 0.3,
            "FLIGHT_LANDING": 0.1
        }
        
        # Adaptive gains storage
        self.Kp_hip = 0
        self.Kd_hip = 0
        self.Kp_knee = 0
        self.Kd_knee = 0
        
        # Update gains based on current leg inertia
        self.update_adaptive_gains()
    
    def update_adaptive_gains(self):
        """
        Adaptive gain scheduling based on leg inertia
        Eq. 3.13: Kp = ω_n² * I_leg(μ)
        """
        I_leg = self.robot.leg_inertia
        
        # Position gains (scaled with inertia)
        self.Kp_hip = self.target_omega**2 * I_leg
        self.Kp_knee = self.target_omega**2 * I_leg * 0.5  # Knee has less inertia
        
        # Damping gains (critical damping)
        zeta = 1.0  # Critical damping ratio
        self.Kd_hip = 2 * zeta * self.target_omega * I_leg
        self.Kd_knee = 2 * zeta * self.target_omega * I_leg * 0.5
    
    def state_transition(self, dt, robot_state):
        """
        Finite State Machine transition logic
        Based on Huang & Zhang (2023) Table 2
        """
        self.phase_timer += dt
        
        # Check for state transitions
        if self.state == "STANCE_COMPRESSION":
            if self.phase_timer >= self.timing["STANCE_COMPRESSION"]:
                self.state = "STANCE_THRUST"
                self.phase_timer = 0
        
        elif self.state == "STANCE_THRUST":
            if self.phase_timer >= self.timing["STANCE_THRUST"]:
                self.state = "FLIGHT_SWING"
                self.phase_timer = 0
                self.jump_counter += 1
        
        elif self.state == "FLIGHT_SWING":
            if self.phase_timer >= self.timing["FLIGHT_SWING"]:
                self.state = "FLIGHT_LANDING"
                self.phase_timer = 0
        
        elif self.state == "FLIGHT_LANDING":
            if self.phase_timer >= self.timing["FLIGHT_LANDING"]:
                self.state = "STANCE_COMPRESSION"
                self.phase_timer = 0
        
        return self.state
    
    def compute_control(self, dt, robot_state, wind_force=0):
        """
        Main control computation based on current state
        
        Returns:
            tau_hip, tau_knee: Joint torques [Nm]
        """
        # Update state
        current_state = self.state_transition(dt, robot_state)
        
        # Get current foot position
        q_hip, q_knee = robot_state['joint_positions']
        foot_pos = self.robot.forward_kinematics(q_hip, q_knee)
        
        # State-specific control
        if "STANCE" in current_state:
            tau_hip, tau_knee = self.stance_control(
                robot_state, 
                current_state,
                wind_force
            )
        
        else:  # FLIGHT states
            tau_hip, tau_knee = self.flight_control(
                robot_state,
                current_state
            )
        
        return tau_hip, tau_knee
    
    def stance_control(self, robot_state, phase, wind_force):
        """
        Stance phase control: SLIP model with energy compensation
        Based on Huang & Zhang (2023) Eq. 3.8
        """
        q_hip, q_knee = robot_state['joint_positions']
        qd_hip, qd_knee = robot_state['joint_velocities']
        
        # Compute virtual leg state
        foot_pos = self.robot.forward_kinematics(q_hip, q_knee)
        l_vir = np.linalg.norm(foot_pos)
        l_dot = self.compute_virtual_leg_velocity(q_hip, q_knee, qd_hip, qd_knee)
        
        # Virtual spring force (SLIP model)
        F_vir = self.k_spring * (self.l0 - l_vir) - self.b_damping * l_dot
        
        # Energy compensation during thrust phase (animal-inspired)
        if phase == "STANCE_THRUST":
            vx = robot_state['body_velocity'][0]
            # Log-based energy addition from cheetah model
            if abs(vx) > 0.1:
                F_add = self.alpha_energy * np.log(abs(vx) + 1)
                F_vir += F_add
        
        # Add wind disturbance compensation
        F_vir += wind_force * 0.5  # Empirical compensation factor
        
        # Convert to foot force vector (along virtual leg)
        if l_vir > 0.01:
            F_foot = F_vir * foot_pos / l_vir
        else:
            F_foot = np.array([0, 0, F_vir])
        
        # Jacobian transpose to get joint torques
        J = self.robot.compute_jacobian(q_hip, q_knee)
        F_foot_xz = np.array([F_foot[0], F_foot[2]])  # x-z components only
        
        tau_jacobian = J.T @ F_foot_xz
        
        # Body attitude stabilization (PD on pitch)
        body_pitch = robot_state['body_pitch']
        body_pitch_rate = robot_state['body_pitch_rate']
        
        tau_pitch = -self.Kp_hip * body_pitch - self.Kd_hip * body_pitch_rate
        
        # Combine torques
        tau_hip = tau_jacobian[0] + tau_pitch
        tau_knee = tau_jacobian[1]
        
        return tau_hip, tau_knee
    
    def flight_control(self, robot_state, phase):
        """
        Flight phase control: Foot placement and attitude correction
        Based on Raibert heuristic and angular momentum control
        """
        q_hip, q_knee = robot_state['joint_positions']
        qd_hip, qd_knee = robot_state['joint_velocities']
        
        if phase == "FLIGHT_SWING":
            # Swing leg to clear obstacle
            # Bezier curve trajectory planning
            t_norm = self.phase_timer / self.timing["FLIGHT_SWING"]
            
            # Fifth-order Bezier curve for smooth trajectory
            h_foot = 0.3  # Ground clearance [m]
            x_target = self.compute_raibert_placement(robot_state)
            
            # Bezier interpolation
            x_swing = self.l0 * np.sin(np.pi * t_norm) * 0.5
            z_swing = -self.l0 + h_foot * np.sin(np.pi * t_norm)
            
            # Desired foot position
            x_des = x_target + x_swing
            z_des = z_swing
        
        else:  # FLIGHT_LANDING
            # Prepare for landing using Raibert heuristic
            x_target = self.compute_raibert_placement(robot_state)
            
            # Smooth landing trajectory
            t_norm = self.phase_timer / self.timing["FLIGHT_LANDING"]
            x_des = x_target * t_norm
            z_des = -self.l0 * (1 - t_norm)  # Approach ground smoothly
        
        # Inverse kinematics for desired foot position
        q_hip_des, q_knee_des = self.robot.inverse_kinematics(x_des, z_des)
        
        # PD control to track desired joint angles
        tau_hip = self.Kp_hip * (q_hip_des - q_hip) - self.Kd_hip * qd_hip
        tau_knee = self.Kp_knee * (q_knee_des - q_knee) - self.Kd_knee * qd_knee
        
        # Angular momentum control for body attitude
        body_pitch = robot_state['body_pitch']
        body_pitch_rate = robot_state['body_pitch_rate']
        
        # Eq. 3.9: τ_hip = -K_amc * θ_dot (angular momentum control)
        K_amc = self.robot.leg_inertia * 10  # Angular momentum gain
        tau_amc = -K_amc * body_pitch_rate
        
        tau_hip += tau_amc
        
        return tau_hip, tau_knee
    
    def compute_raibert_placement(self, robot_state):
        """
        Raibert heuristic for foot placement
        Eq. 3.10: x_land = (v_x * T_s)/2 + K_raibert * (v_x - v_des)
        """
        vx = robot_state['body_velocity'][0]
        vx_des = 0  # Hopping in place
        
        x_land = (vx * self.T_stance) / 2 + self.K_raibert * (vx - vx_des)
        
        # Limit to kinematic reach
        max_reach = self.l1 + self.l2
        x_land = np.clip(x_land, -max_reach * 0.8, max_reach * 0.8)
        
        return x_land
    
    def compute_virtual_leg_velocity(self, q_hip, q_knee, qd_hip, qd_knee):
        """Compute time derivative of virtual leg length"""
        # Analytic derivative of l = sqrt(x² + z²)
        x = self.l1 * np.sin(q_hip) + self.l2 * np.sin(q_hip + q_knee)
        z = -self.l1 * np.cos(q_hip) - self.l2 * np.cos(q_hip + q_knee)
        
        x_dot = (self.l1 * np.cos(q_hip) + self.l2 * np.cos(q_hip + q_knee)) * qd_hip + \
                self.l2 * np.cos(q_hip + q_knee) * qd_knee
        
        z_dot = (self.l1 * np.sin(q_hip) + self.l2 * np.sin(q_hip + q_knee)) * qd_hip + \
                self.l2 * np.sin(q_hip + q_knee) * qd_knee
        
        l = np.sqrt(x**2 + z**2)
        if l > 0.001:
            l_dot = (x * x_dot + z * z_dot) / l
        else:
            l_dot = 0
        
        return l_dot
    
    def reset(self):
        """Reset controller state"""
        self.state = "STANCE_COMPRESSION"
        self.phase_timer = 0
        self.jump_counter = 0
        self.update_adaptive_gains()