"""
URDF Generator for SLATOL Robot
Nik Adam Muqridz (2125501)
Implements Table 3.1 and your clarifications Q1, Q3, Q4, Q15.
"""

import numpy as np
import os

def generate_urdf(mu, output_path=None):
    """
    Generate SLATOL URDF with leg mass ratio = mu.
    Returns URDF string; if output_path is given, writes file.
    """
    total_mass = 1.0
    leg_mass = mu * total_mass
    body_mass = total_mass - leg_mass
    
    # Mass split: equal between femur and tibia (Q1)
    femur_mass = leg_mass * 0.5
    tibia_mass = leg_mass * 0.5
    
    # Geometry (all dimensions in meters)
    L1 = 0.15          # Femur length
    L2 = 0.15          # Tibia length
    link_radius = 0.02
    foot_radius = 0.02
    
    # Body half extents (box)
    bx, by, bz = 0.1, 0.05, 0.05
    
    # ------ Inertia: solid cylinder (femur, tibia) ------
    # Cylinder axis along Z in link frame (after placement)
    I_cyl_xx = (1/12) * (3*link_radius**2 + L1**2)   # per kg
    I_cyl_zz = 0.5 * link_radius**2                  # per kg
    
    # Femur inertia (at CoM, cylinder axis Z)
    I_femur_xx = femur_mass * I_cyl_xx
    I_femur_yy = I_femur_xx      # axisymmetric
    I_femur_zz = femur_mass * I_cyl_zz
    
    # Tibia inertia (same geometry)
    I_tibia_xx = tibia_mass * I_cyl_xx
    I_tibia_yy = I_tibia_xx
    I_tibia_zz = tibia_mass * I_cyl_zz
    
    # Body inertia (box, at CoM)
    I_body_xx = (1/12) * body_mass * (by**2 + bz**2) * 4   # half extents -> full dims
    I_body_yy = (1/12) * body_mass * (bx**2 + bz**2) * 4
    I_body_zz = (1/12) * body_mass * (bx**2 + by**2) * 4
    
    # Foot sphere inertia (point mass approx)
    foot_mass = 0.001   # very light, only for collision
    I_foot = (2/5) * foot_mass * foot_radius**2
    
    # ---------- URDF template ----------
    urdf_str = f'''<?xml version="1.0"?>
<robot name="slatol_mu_{mu:.2f}">

  <!-- BODY -->
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
      <geometry>
        <box size="{2*bx} {2*by} {2*bz}"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.3 0.8 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="{2*bx} {2*by} {2*bz}"/>
      </geometry>
    </collision>
  </link>

  <!-- HIP JOINT (revolute, axis Y) -->
  <joint name="hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="femur_link"/>
    <origin xyz="0 0 -{bz}" rpy="0 0 0"/>   <!-- hip at bottom center of body -->
    <axis xyz="0 1 0"/>
    <limit lower="-1.5708" upper="1.5708" effort="5.0" velocity="22.0"/>
    <dynamics damping="0.1" friction="0.01"/>
  </joint>

  <!-- FEMUR LINK -->
  <link name="femur_link">
    <inertial>
      <!-- CoM is half way along the link (negative Z in link frame) -->
      <origin xyz="0 0 -{L1/2}" rpy="0 0 0"/>
      <mass value="{femur_mass:.4f}"/>
      <inertia ixx="{I_femur_xx:.6f}" ixy="0" ixz="0"
               iyy="{I_femur_yy:.6f}" iyz="0"
               izz="{I_femur_zz:.6f}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -{L1/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder length="{L1}" radius="{link_radius}"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0.2 0.2 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -{L1/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder length="{L1}" radius="{link_radius}"/>
      </geometry>
    </collision>
  </link>

  <!-- KNEE JOINT (revolute, axis Y) -->
  <joint name="knee_joint" type="revolute">
    <parent link="femur_link"/>
    <child link="tibia_link"/>
    <origin xyz="0 0 -{L1}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.618" upper="0.0" effort="5.0" velocity="22.0"/>
    <dynamics damping="0.1" friction="0.01"/>
  </joint>

  <!-- TIBIA LINK -->
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
      <geometry>
        <cylinder length="{L2}" radius="{link_radius}"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0.2 0.2 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -{L2/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder length="{L2}" radius="{link_radius}"/>
      </geometry>
    </collision>
  </link>

  <!-- FIXED JOINT TO FOOT (small sphere for contact) -->
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
      <geometry>
        <sphere radius="{foot_radius}"/>
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="{foot_radius}"/>
      </geometry>
    </collision>
  </link>

</robot>
'''
    if output_path:
        with open(output_path, 'w') as f:
            f.write(urdf_str)
        return output_path
    return urdf_str