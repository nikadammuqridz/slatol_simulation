import pybullet as p
import pybullet_data
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

class SLATOLRobot:
    """2-DOF single-legged robot with SLIP dynamics and variable mass ratio"""
    
    def __init__(self, client_id, leg_mass_ratio=0.15):
        """
        Initialize robot with configurable mass distribution
        
        Args:
            client_id: PyBullet physics client ID
            leg_mass_ratio (μ): m_leg / m_total (0.05-0.40)
        """
        self.client_id = client_id
        self.leg_mass_ratio = leg_mass_ratio
        
        # Fixed parameters (from Huang & Zhang, 2023)
        self.body_mass = 5.0  # kg
        self.l1 = 0.35  # Femur length [m]
        self.l2 = 0.35  # Tibia length [m]
        
        # Calculate total mass and leg mass from ratio
        self.update_mass_distribution(leg_mass_ratio)
        
        # Joint limits
        self.hip_limit = (-np.pi/2, np.pi/2)
        self.knee_limit = (0, np.pi)
        
        # Robot IDs
        self.body_id = None
        self.hip_joint_idx = None
        self.knee_joint_idx = None
        self.foot_link_idx = None
        
        # State variables
        self.joint_positions = np.zeros(2)
        self.joint_velocities = np.zeros(2)
        self.body_position = np.zeros(3)
        self.body_orientation = np.array([0, 0, 0, 1])
        
        # Build the robot
        self.build_robot()
    
    def update_mass_distribution(self, leg_mass_ratio):
        """Update mass distribution based on leg mass ratio μ"""
        self.leg_mass_ratio = leg_mass_ratio
        
        # Total mass: m_total = m_body / (1 - μ)
        self.total_mass = self.body_mass / (1 - leg_mass_ratio)
        self.leg_mass = leg_mass_ratio * self.total_mass
        
        # Distribute leg mass equally between femur and tibia
        self.femur_mass = self.leg_mass / 2
        self.tibia_mass = self.leg_mass / 2
        
        # Leg inertia (point mass approximation at center)
        self.leg_inertia = self.leg_mass * ((self.l1 + self.l2)/2)**2
    
    def build_robot(self):
        """Create the robot in PyBullet using URDF"""
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load floating base (body)
        self.body_id = p.loadURDF(
            "cube_small.urdf",
            basePosition=[0, 0, 0.5],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.client_id
        )
        
        # Change body mass and color
        p.changeVisualShape(self.body_id, -1, rgbaColor=[0.3, 0.3, 0.8, 1])
        p.changeDynamics(
            self.body_id, -1,
            mass=self.body_mass,
            lateralFriction=0.9
        )
        
        # Create hip joint (revolute)
        hip_joint = p.createConstraint(
            self.body_id, -1, -1, -1,
            p.JOINT_REVOLUTE,
            [0, 1, 0],  # Axis for y-rotation (pitch in sagittal plane)
            [0, 0, -0.25],  # Position relative to body
            [0, 0, 0],
            parentFrameOrientation=[0, 0, 0, 1],
            physicsClientId=self.client_id
        )
        
        # Create femur link
        femur_collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=0.02,
            height=self.l1,
            physicsClientId=self.client_id
        )
        femur_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.02,
            length=self.l1,
            rgbaColor=[0.8, 0.2, 0.2, 1],
            physicsClientId=self.client_id
        )
        
        femur_id = p.createMultiBody(
            baseMass=self.femur_mass,
            baseCollisionShapeIndex=femur_collision,
            baseVisualShapeIndex=femur_visual,
            basePosition=[0, 0, -self.l1/2],
            baseOrientation=p.getQuaternionFromEuler([np.pi/2, 0, 0]),
            physicsClientId=self.client_id
        )
        
        # Connect femur to body via hip joint
        p.changeConstraint(
            hip_joint,
            jointChildBodyUniqueId=femur_id,
            parentFramePosition=[0, 0, -0.25],
            childFramePosition=[0, 0, self.l1/2],
            maxForce=500,
            physicsClientId=self.client_id
        )
        
        # Create knee joint
        knee_joint = p.createConstraint(
            femur_id, -1, -1, -1,
            p.JOINT_REVOLUTE,
            [0, 1, 0],
            [0, 0, -self.l1/2],
            [0, 0, 0],
            parentFrameOrientation=[0, 0, 0, 1],
            physicsClientId=self.client_id
        )
        
        # Create tibia link
        tibia_collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=0.02,
            height=self.l2,
            physicsClientId=self.client_id
        )
        tibia_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.02,
            length=self.l2,
            rgbaColor=[0.2, 0.8, 0.2, 1],
            physicsClientId=self.client_id
        )
        
        tibia_id = p.createMultiBody(
            baseMass=self.tibia_mass,
            baseCollisionShapeIndex=tibia_collision,
            baseVisualShapeIndex=tibia_visual,
            basePosition=[0, 0, -self.l2/2],
            baseOrientation=p.getQuaternionFromEuler([np.pi/2, 0, 0]),
            physicsClientId=self.client_id
        )
        
        # Connect tibia to femur via knee joint
        p.changeConstraint(
            knee_joint,
            jointChildBodyUniqueId=tibia_id,
            parentFramePosition=[0, 0, -self.l1/2],
            childFramePosition=[0, 0, self.l2/2],
            maxForce=500,
            physicsClientId=self.client_id
        )
        
        # Create foot (massless sphere)
        foot_collision = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=0.03,
            physicsClientId=self.client_id
        )
        foot_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.03,
            rgbaColor=[0.8, 0.8, 0.2, 1],
            physicsClientId=self.client_id
        )
        
        foot_id = p.createMultiBody(
            baseMass=0.001,  # Near massless
            baseCollisionShapeIndex=foot_collision,
            baseVisualShapeIndex=foot_visual,
            basePosition=[0, 0, -self.l2/2],
            physicsClientId=self.client_id
        )
        
        # Fix foot to tibia
        p.createConstraint(
            tibia_id, -1, foot_id, -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, -self.l2/2],
            [0, 0, 0],
            physicsClientId=self.client_id
        )
        
        # Store IDs
        self.hip_joint_idx = hip_joint
        self.knee_joint_idx = knee_joint
        self.foot_link_idx = foot_id
        
        # Enable joint motors for control
        p.setJointMotorControl2(
            self.body_id, -1,
            p.VELOCITY_CONTROL,
            targetVelocity=0,
            force=0,
            physicsClientId=self.client_id
        )
    
    def forward_kinematics(self, q_hip, q_knee):
        """
        Forward kinematics: joint angles to foot position
        Based on Huang & Zhang (2023) Eq. 3.1-3.2
        """
        x_foot = self.l1 * np.sin(q_hip) + self.l2 * np.sin(q_hip + q_knee)
        z_foot = -self.l1 * np.cos(q_hip) - self.l2 * np.cos(q_hip + q_knee)
        
        return np.array([x_foot, 0, z_foot])
    
    def inverse_kinematics(self, x_des, z_des):
        """
        Inverse kinematics: desired foot position to joint angles
        Based on Huang & Zhang (2023) Eq. 3.5
        """
        # Virtual leg length
        l_vir = np.sqrt(x_des**2 + z_des**2)
        
        # Law of Cosines for knee angle
        cos_q_knee = (l_vir**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        cos_q_knee = np.clip(cos_q_knee, -1, 1)  # Numerical safety
        q_knee = np.arccos(cos_q_knee)
        
        # Hip angle
        alpha = np.arctan2(x_des, -z_des)
        beta = np.arcsin((self.l2 * np.sin(q_knee)) / l_vir)
        q_hip = alpha - beta
        
        return q_hip, q_knee
    
    def compute_jacobian(self, q_hip, q_knee):
        """
        Compute Jacobian matrix J = ∂x_foot/∂q
        Based on Huang & Zhang (2023) Eq. 3.7
        """
        J = np.array([
            [self.l1 * np.cos(q_hip) + self.l2 * np.cos(q_hip + q_knee), 
             self.l2 * np.cos(q_hip + q_knee)],
            [0, 0],  # No y-motion in sagittal plane
            [self.l1 * np.sin(q_hip) + self.l2 * np.sin(q_hip + q_knee), 
             self.l2 * np.sin(q_hip + q_knee)]
        ])
        
        # Return only x-z components (2x2)
        return J[[0, 2], :]
    
    def get_state(self):
        """Get current robot state"""
        # Body state
        body_state = p.getBasePositionAndOrientation(
            self.body_id, 
            physicsClientId=self.client_id
        )
        body_vel = p.getBaseVelocity(self.body_id, physicsClientId=self.client_id)
        
        # Joint states (approximated from constraints)
        self.body_position = np.array(body_state[0])
        self.body_orientation = np.array(body_state[1])
        
        # Extract pitch from quaternion
        euler = p.getEulerFromQuaternion(self.body_orientation)
        body_pitch = euler[1]  # y-axis rotation
        
        # For simulation, we'll use simplified joint state
        # In reality, you'd read from actual joint encoders
        
        return {
            'body_position': self.body_position,
            'body_velocity': np.array(body_vel[0]),
            'body_pitch': body_pitch,
            'body_pitch_rate': body_vel[1][1],  # y-axis angular velocity
            'joint_positions': self.joint_positions,
            'joint_velocities': self.joint_velocities
        }
    
    def apply_joint_torques(self, tau_hip, tau_knee):
        """
        Apply torques to joints using PyBullet constraint motors
        """
        # Apply hip torque (acts on femur relative to body)
        p.changeConstraint(
            self.hip_joint_idx,
            maxForce=abs(tau_hip),
            gear=1,
            erp=0.8,
            physicsClientId=self.client_id
        )
        
        # Apply knee torque (acts on tibia relative to femur)
        p.changeConstraint(
            self.knee_joint_idx,
            maxForce=abs(tau_knee),
            gear=1,
            erp=0.8,
            physicsClientId=self.client_id
        )
    
    def apply_disturbance(self, force_x, force_z, duration=0.1):
        """
        Apply external disturbance (simulating wind gust)
        """
        p.applyExternalForce(
            self.body_id, -1,
            forceObj=[force_x, 0, force_z],
            posObj=self.body_position,
            flags=p.WORLD_FRAME,
            physicsClientId=self.client_id
        )
    
    def reset(self, position=[0, 0, 0.5], orientation=[0, 0, 0, 1]):
        """Reset robot to initial state"""
        p.resetBasePositionAndOrientation(
            self.body_id,
            position,
            orientation,
            physicsClientId=self.client_id
        )
        p.resetBaseVelocity(
            self.body_id,
            [0, 0, 0],
            [0, 0, 0],
            physicsClientId=self.client_id
        )
        
        # Reset joint positions
        self.joint_positions = np.array([0, np.pi/2])  # Standing pose
        self.joint_velocities = np.zeros(2)