"""
Control algorithms for SLATOL
FSM, VMC, AMC, Foot Placement, Inverse Kinematics
FIXED: Jacobian uses DoF-length vectors and DoF indices
Nik Adam Muqridz (2125501)
"""

import numpy as np
import pybullet as p
from enum import Enum

class Phase(Enum):
    COMPRESSION = 1
    THRUST = 2
    FLIGHT = 3
    LANDING = 4

class SLATOL_FSM:
    """Finite State Machine (Table 3.2)"""
    def __init__(self):
        self.phase = Phase.FLIGHT
        self.compression_max = False
        self.r_min = float('inf')
        self.apex_triggered = False

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

# ----------------------------------------------------------------------
# Virtual Model Control (VMC) – stance phase
# Eqs 3.8, 3.9
def compute_vmc_torques(robot, foot_link_idx,
                        phase, r, r_dot, theta_vir,
                        q_dof, qdot_dof,          # DoF-length vectors
                        hip_dof_idx, knee_dof_idx):
    """
    q_dof, qdot_dof: positions/velocities of ONLY movable joints.
    hip_dof_idx, knee_dof_idx: column indices in Jacobian.
    """
    # Virtual spring parameters (Table 3.3)
    r0 = 0.30
    ks = 1800.0
    bs = 15.0
    F_r = ks * (r0 - r) - bs * r_dot

    if phase == Phase.THRUST:
        F_r += 20.0

    F_r = np.clip(F_r, 0, 300)

    # Project to Cartesian (hip frame)
    Fx = F_r * np.sin(theta_vir)
    Fz = -F_r * np.cos(theta_vir)
    F_vect = [Fx, 0, Fz]

    # --- Calculate Jacobian with DoF-length vectors ---
    jac_trans, jac_rot = p.calculateJacobian(
        robot,
        foot_link_idx,
        [0, 0, 0],
        q_dof,
        qdot_dof,
        [0.0] * len(q_dof)   # dummy accelerations
    )

    # jac_trans is tuple of 3 arrays (3 x numDoF)
    # Extract columns using DoF indices
    jac_hip = np.array([jac_trans[0][hip_dof_idx],
                        jac_trans[1][hip_dof_idx],
                        jac_trans[2][hip_dof_idx]])
    jac_knee = np.array([jac_trans[0][knee_dof_idx],
                         jac_trans[1][knee_dof_idx],
                         jac_trans[2][knee_dof_idx]])

    tau_hip = np.dot(jac_hip, F_vect)
    tau_knee = np.dot(jac_knee, F_vect)

    # Torque limits (5.0 Nm)
    tau_hip = np.clip(tau_hip, -5.0, 5.0)
    tau_knee = np.clip(tau_knee, -5.0, 5.0)

    return tau_hip, tau_knee

# ----------------------------------------------------------------------
# Angular Momentum Control (AMC) – flight phase
def compute_amc_torque(body_pitch, body_pitch_rate, mu):
    Kp_nom = 25.0
    Kd_nom = 1.5
    Kp = Kp_nom * (mu / 0.05)
    Kd = Kd_nom * (mu / 0.05)
    tau = -Kp * body_pitch - Kd * body_pitch_rate
    return np.clip(tau, -5.0, 5.0)

# ----------------------------------------------------------------------
# Foot placement (Raibert heuristic) – Eq 3.11
def compute_foot_placement(x_dot, x_dot_des=0.0):
    k_raibert = 0.04
    k = 1800.0
    m = 1.0
    T_stance = np.pi * np.sqrt(m / k)   # ~0.074 s
    x_f = (x_dot * T_stance) / 2 + k_raibert * (x_dot - x_dot_des)
    return x_f

# ----------------------------------------------------------------------
# Inverse Kinematics – Eq 3.4
def inverse_kinematics(x_f, z_f):
    L1 = 0.15
    L2 = 0.15
    l_vir = np.sqrt(x_f**2 + z_f**2)

    cos_th3 = (l_vir**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_th3 = np.clip(cos_th3, -1.0, 1.0)
    theta3 = -np.arccos(cos_th3)

    phi = np.arctan2(x_f, -z_f)
    alpha = np.arcsin((L2 * np.sin(-theta3)) / l_vir)
    theta2 = phi - alpha
    return theta2, theta3