import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import time

# 1. Setup Physics
client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
plane = p.loadURDF("plane.urdf")

# 2. Setup Data Tracking
z_history = []
time_steps = []

# 3. Create a Dynamic MultiBody (The "Gold Standard" way)
sphere_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
sphere_id = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=sphere_col, basePosition=[0, 0, 2])

print("Simulation starting... Physics and Plotting are synced.")

# 4. Run Loop
for i in range(100):
    p.stepSimulation()
    pos, _ = p.getBasePositionAndOrientation(sphere_id)
    z_history.append(pos[2])
    time_steps.append(i)
    time.sleep(1./240.)

# 5. Verify Matplotlib
plt.plot(time_steps, z_history)
plt.ylabel('Height (z)')
plt.title('PyBullet 3.9 + Matplotlib Success')
plt.show()

p.disconnect()
