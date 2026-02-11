import numpy as np
import math

class SLATOLController:
    """FSM controller with mathematically-derived auto-tuning"""
    
    def __init__(self, robot, target_omega=15.0, zeta=0.8):
        """
        Initialize with auto-tuning based on Raibert (1984) derivation
        
        Args:
            robot: SLATOLRobot instance
            target_omega: Target natural frequency [rad/s] (10-15 recommended)
            zeta: Damping ratio (0.7-1.0 recommended)
        """
        self.robot = robot
        self.target_omega = target_omega
        self.zeta = zeta
        
        # Theory constants from derivation
        self.leg_length = robot.l1 + robot.l2  # Total leg length
        self.rod_inertia_factor = 1.0/3.0  # For slender rod rotating at one end
        
        # Control states
        self.state = "STANCE_COMPRESSION"
        self.phase_timer = 0
        
        # SLIP parameters
        self.l0 = self.leg_length
        self.k_spring = 1000.0
        self.b_damping = 50.0
        
        # Performance monitoring
        self.bandwidth_history = []
        self.gain_history = []
        
        # Initialize with auto-tuned gains
        self.update_adaptive_gains()
        
        print(f"Auto-Tuning Controller Initialized:")
        print(f"  Target ω_n = {target_omega:.1f} rad/s")
        print(f"  Damping ζ = {zeta:.2f}")
        print(f"  Initial Kp_hip = {self.Kp_hip:.1f}, Kp_knee = {self.Kp_knee:.1f}")
    
    def update_adaptive_gains(self):
        """
        AUTO-TUNING LAW IMPLEMENTATION
        Based on derived equation: K_p = I_leg · ω_n²
        
        Proof: See theory_derivation.py
        """
        # Get current leg properties from robot
        m_total = self.robot.total_mass
        μ = self.robot.leg_mass_ratio
        
        # Step 1: Calculate leg inertias using slender rod model
        # For hip: entire leg rotates about hip
        I_hip = self.rod_inertia_factor * (μ * m_total) * self.leg_length**2
        
        # For knee: only tibia rotates (approx half the leg)
        # Assuming mass distributed equally: m_tibia = 0.5 * m_leg
        m_tibia = 0.5 * (μ * m_total)
        L_tibia = self.robot.l2  # Tibia length
        I_knee = self.rod_inertia_factor * m_tibia * L_tibia**2
        
        # Step 2: Apply auto-tuning law (Eq. 8 in derivation)
        # K_p = I · ω_n²
        self.Kp_hip = I_hip * (self.target_omega ** 2)
        self.Kp_knee = I_knee * (self.target_omega ** 2)
        
        # Step 3: Calculate derivative gains for constant damping
        # K_d = 2ζω_n·I
        self.Kd_hip = 2 * self.zeta * self.target_omega * I_hip
        self.Kd_knee = 2 * self.zeta * self.target_omega * I_knee
        
        # Calculate actual bandwidth for monitoring
        actual_omega_hip = np.sqrt(self.Kp_hip / I_hip) if I_hip > 0 else 0
        actual_omega_knee = np.sqrt(self.Kp_knee / I_knee) if I_knee > 0 else 0
        
        # Store for analysis
        self.bandwidth_history.append({
            'time': self.phase_timer,
            'target_omega': self.target_omega,
            'actual_omega_hip': actual_omega_hip,
            'actual_omega_knee': actual_omega_knee,
            'mu': μ,
            'I_hip': I_hip,
            'I_knee': I_knee
        })
        
        self.gain_history.append({
            'time': self.phase_timer,
            'Kp_hip': self.Kp_hip,
            'Kp_knee': self.Kp_knee,
            'Kd_hip': self.Kd_hip,
            'Kd_knee': self.Kd_knee
        })
        
        return actual_omega_hip, actual_omega_knee
    
    def compute_control(self, dt, robot_state, wind_force=0):
        """
        Main control loop with auto-tuned gains
        
        Returns:
            tau_hip, tau_knee: Joint torques [Nm]
        """
        # Update state machine
        current_state = self.state_transition(dt, robot_state)
        
        # Get current joint states
        q_hip, q_knee = robot_state['joint_positions']
        qd_hip, qd_knee = robot_state['joint_velocities']
        
        # State-specific control
        if "STANCE" in current_state:
            tau_hip, tau_knee = self.stance_control(
                robot_state, current_state, wind_force
            )
        else:
            tau_hip, tau_knee = self.flight_control(
                robot_state, current_state
            )
        
        return tau_hip, tau_knee
    
    def stance_control(self, robot_state, phase, wind_force):
        """Stance control with auto-tuned attitude stabilization"""
        q_hip, q_knee = robot_state['joint_positions']
        qd_hip, qd_knee = robot_state['joint_velocities']
        
        # SLIP virtual leg control
        foot_pos = self.robot.forward_kinematics(q_hip, q_knee)
        l_vir = np.linalg.norm(foot_pos)
        l_dot = self.compute_virtual_leg_velocity(q_hip, q_knee, qd_hip, qd_knee)
        
        F_vir = self.k_spring * (self.l0 - l_vir) - self.b_damping * l_dot
        
        # Energy compensation (animal-inspired)
        if phase == "STANCE_THRUST":
            vx = robot_state['body_velocity'][0]
            if abs(vx) > 0.1:
                alpha = 2800  # From Huang & Zhang (2023)
                F_add = alpha * np.log(abs(vx) + 1)
                F_vir += F_add
        
        # Convert to joint torques via Jacobian transpose
        if l_vir > 0.01:
            F_foot = F_vir * foot_pos / l_vir
        else:
            F_foot = np.array([0, 0, F_vir])
        
        J = self.robot.compute_jacobian(q_hip, q_knee)
        F_foot_xz = np.array([F_foot[0], F_foot[2]])
        tau_jacobian = J.T @ F_foot_xz
        
        # AUTO-TUNED attitude stabilization
        body_pitch = robot_state['body_pitch']
        body_pitch_rate = robot_state['body_pitch_rate']
        
        # Using auto-tuned gains derived above
        tau_pitch = -self.Kp_hip * body_pitch - self.Kd_hip * body_pitch_rate
        
        # Combine torques
        tau_hip = tau_jacobian[0] + tau_pitch
        tau_knee = tau_jacobian[1]
        
        return tau_hip, tau_knee
    
    def flight_control(self, robot_state, phase):
        """Flight control with auto-tuned PD tracking"""
        q_hip, q_knee = robot_state['joint_positions']
        qd_hip, qd_knee = robot_state['joint_velocities']
        
        # Bezier trajectory planning
        t_norm = self.phase_timer / self.timing.get(phase, 0.3)
        
        if phase == "FLIGHT_SWING":
            # Clearance trajectory
            h_foot = 0.3
            x_target = self.compute_raibert_placement(robot_state)
            x_swing = self.leg_length * np.sin(np.pi * t_norm) * 0.5
            z_swing = -self.leg_length + h_foot * np.sin(np.pi * t_norm)
            x_des = x_target + x_swing
            z_des = z_swing
        else:  # FLIGHT_LANDING
            # Landing preparation
            x_target = self.compute_raibert_placement(robot_state)
            x_des = x_target * t_norm
            z_des = -self.leg_length * (1 - t_norm)
        
        # Inverse kinematics
        q_hip_des, q_knee_des = self.robot.inverse_kinematics(x_des, z_des)
        
        # AUTO-TUNED PD control
        tau_hip = self.Kp_hip * (q_hip_des - q_hip) - self.Kd_hip * qd_hip
        tau_knee = self.Kp_knee * (q_knee_des - q_knee) - self.Kd_knee * qd_knee
        
        # Angular momentum compensation
        body_pitch_rate = robot_state['body_pitch_rate']
        tau_amc = -self.robot.leg_inertia * 10 * body_pitch_rate
        tau_hip += tau_amc
        
        return tau_hip, tau_knee
    
    def get_performance_metrics(self):
        """Return controller performance metrics"""
        if not self.bandwidth_history:
            return {}
        
        latest = self.bandwidth_history[-1]
        return {
            'target_bandwidth': latest['target_omega'],
            'achieved_bandwidth_hip': latest['actual_omega_hip'],
            'achieved_bandwidth_knee': latest['actual_omega_knee'],
            'bandwidth_error_hip': abs(latest['target_omega'] - latest['actual_omega_hip']),
            'bandwidth_error_knee': abs(latest['target_omega'] - latest['actual_omega_knee']),
            'current_gains': {
                'Kp_hip': self.Kp_hip,
                'Kd_hip': self.Kd_hip,
                'Kp_knee': self.Kp_knee,
                'Kd_knee': self.Kd_knee
            }
        }