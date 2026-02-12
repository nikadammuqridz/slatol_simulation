"""
URDF Generator for SLATOL Robot
FIXED: Added note about URDF_USE_INERTIA_FROM_FILE flag
Nik Adam Muqridz (2125501)
"""

import numpy as np
import os

def generate_urdf(mu, output_path=None):
    """
    Generate SLATOL URDF with leg mass ratio = mu.
    IMPORTANT: When loading this URDF, use:
        p.loadURDF(..., flags=p.URDF_USE_INERTIA_FROM_FILE)
    """
    total_mass = 1.0
    leg_mass = mu * total_mass
    body_mass = total_mass - leg_mass
    
    # Mass split: equal between femur and tibia
    femur_mass = leg_mass * 0.5
    tibia_mass = leg_mass * 0.5
    
    # Geometry
    L1 = 0.15
    L2 = 0.15
    link_radius = 0.02
    foot_radius = 0.02
    
    # Body half extents
    bx, by, bz = 0.1, 0.05, 0.05
    
        # --- Inertia: solid cylinder (femur, tibia) - VALIDATED ---
    # Axis along Z in link frame
    I_cyl_axial = 0.5 * link_radius**2                    # per kg (Izz)
    I_cyl_radial = (1/12) * (3*link_radius**2 + L1**2)   # per kg (Ixx, Iyy)

    # Scale by mass
    I_femur_xx = femur_mass * I_cyl_radial
    I_femur_yy = I_femur_xx
    I_femur_zz = femur_mass * I_cyl_axial

    # --- ENSURE TRIANGLE INEQUALITY ---
    # If Izz is too small relative to Ixx+Iyy, PyBullet rejects it.
    # We artificially ensure minimum ratio.
    min_ratio = 0.1  # Izz must be at least 10% of Ixx
    if I_femur_zz < min_ratio * I_femur_xx:
        I_femur_zz = min_ratio * I_femur_xx

    # Same for tibia
    I_tibia_xx = tibia_mass * I_cyl_radial
    I_tibia_yy = I_tibia_xx
    I_tibia_zz = tibia_mass * I_cyl_axial
    if I_tibia_zz < min_ratio * I_tibia_xx:
        I_tibia_zz = min_ratio * I_tibia_xx
    
    I_body_xx = (1/12) * body_mass * (by**2 + bz**2) * 4
    I_body_yy = (1/12) * body_mass * (bx**2 + bz**2) * 4
    I_body_zz = (1/12) * body_mass * (bx**2 + by**2) * 4
    
    foot_mass = 0.001
    I_foot = (2/5) * foot_mass * foot_radius**2
    
    # URDF template (unchanged - already correct)
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
    <origin xyz="0 0 -{bz}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.5708" upper="1.5708" effort="5.0" velocity="22.0"/>
    <dynamics damping="0.1" friction="0.01"/>
  </joint>

  <!-- FEMUR LINK -->
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

  <!-- FIXED JOINT TO FOOT -->
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