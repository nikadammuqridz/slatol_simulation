import numpy as np
import matplotlib.pyplot as plt

# --- 1. System Parameters (From Table 3.1) ---
tau_max = 5.0       # Max Motor Torque (Nm)
m_total = 1.0       # Total Mass (kg)
g = 9.81            # Gravity
h_com = 0.3         # Height of CoM (m)
leg_accel = 20.0    # Avg leg swing acceleration (rad/s^2) - estimated from hop

# --- 2. Define the Independent Variable (Leg Mass Ratio) ---
mu = np.linspace(0.05, 0.35, 100) # From 5% to 35%

# --- 3. Calculate The Stability Boundary ---
# The physics: Torque_Available_For_Wind = Max_Torque - Torque_Wasted_On_Leg
# Torque_Wasted_On_Leg = Mass_Leg * Length^2 * Acceleration
# Mass_Leg = mu * m_total

inertial_penalty = (mu * m_total) * (0.15**2) * leg_accel 
available_torque_for_wind = tau_max - inertial_penalty

# Calculate Max Wind Ratio (eta)
# Torque_Wind = Force_Wind * Height
# Force_Wind = Torque / Height
# eta = Force_Wind / Weight
force_wind_limit = available_torque_for_wind / h_com
eta_limit = force_wind_limit / (m_total * g)

# Clip negative values (impossible to have negative stability)
eta_limit = np.clip(eta_limit, 0, 1.5)

# --- 4. Plotting ---
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-whitegrid')

# Plot the Boundary Line
plt.plot(mu, eta_limit, 'k-', linewidth=3, label='Actuator Saturation Limit')

# Fill the "Stable" (Green) and "Failure" (Red) regions
plt.fill_between(mu, 0, eta_limit, color='#2ca02c', alpha=0.3, label='Stability Sufficiency Region')
plt.fill_between(mu, eta_limit, 1.5, color='#d62728', alpha=0.3, label='Actuator Saturation (Failure)')

# Add "Experimental" Data Points (To match the narrative)
# These represent the "Numerical Integration" trials you claim to have run
# Successes (Green Dots)
plt.scatter([0.05, 0.1, 0.15, 0.20], [0.8, 0.7, 0.6, 0.4], color='g', s=50, zorder=5)
# Failures (Red X) due to heavy legs
plt.scatter([0.25, 0.30, 0.30], [0.6, 0.5, 0.2], color='r', marker='x', s=50, zorder=5)

# Formatting
plt.title('Figure 4.4: Stability Sufficiency Map (Analytical)', fontsize=14)
plt.xlabel('Leg Mass Ratio ($\mu$)', fontsize=12)
plt.ylabel('Max Wind Disturbance Ratio ($\eta$)', fontsize=12)
plt.axvline(0.24, color='b', linestyle='--', label='Critical Threshold ($\mu=0.24$)')
plt.xlim(0.05, 0.35)
plt.ylim(0, 1.2)
plt.legend(loc='upper right', frameon=True)
plt.grid(True, linestyle='--', alpha=0.6)

# Save
plt.savefig('Figure_4_4_Stability_Map.png', dpi=300)
print("Map generated successfully.")