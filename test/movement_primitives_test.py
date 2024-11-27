"""
=======================
DMP with Final Velocity
=======================

Not all DMPs allow a final velocity > 0. In this case we analyze the effect
of changing final velocities in an appropriate variation of the DMP
formulation that allows to set the final velocity.
"""
print(__doc__)


import matplotlib.pyplot as plt
import numpy as np
from movement_primitives.dmp import DMPWithFinalVelocity


# radius = 1.5
# theta = np.linspace(np.pi, 0, 101)  # Angle from 0 to pi for a semicircle
# x_semi = radius * np.cos(theta) + radius  # x = r * cos(theta)
# y_semi = radius * np.sin(theta)  # y = r * sin(theta)
# z_semi = theta
# Y = np.column_stack((x_semi, y_semi, z_semi))

# traj = np.empty((0, 3))
traj = []
# Read the file and process each line
with open("semicyl_painter/temp/bmaric.csv", "r") as file:
    for line in file:
        # Remove any whitespace or newline characters
        line = line.strip()
        
        # Evaluate the line as a list of floats
        # If the line contains something like "[1.0, 2.0, 3.0]", eval will convert it into a Python list [1.0, 2.0, 3.0]
        point = list(map(float, line.split(',')))
        # point = tuple(float(line.split(',')))
        
        # Append the parsed point to the points list
        traj.append(point)

Y = np.array(traj)
dt = 0.01
execution_time = Y.shape[0] * 0.01
T = np.arange(0, execution_time, dt)
# Y = np.column_stack((np.cos(np.pi * T), -np.cos(np.pi * T), np.cos(np.pi * T)))

print(Y.shape)
dmp = DMPWithFinalVelocity(n_dims=2, execution_time=execution_time)
dmp.imitate(T, Y)

plt.figure(figsize=(14, 8))
ax1 = plt.subplot(231)
ax1.set_title("Dimension 1")
ax1.set_xlabel("Time")
ax1.set_ylabel("Position")
ax2 = plt.subplot(232)
ax2.set_title("Dimension 2")
ax2.set_xlabel("Time")
ax2.set_ylabel("Position")
ax3 = plt.subplot(233)
ax3.set_title("Dimension 3")
ax3.set_xlabel("Time")
ax3.set_ylabel("Position")

ax4 = plt.subplot(234)
ax4.set_xlabel("Time")
ax4.set_ylabel("Velocity")
ax5 = plt.subplot(235)
ax5.set_xlabel("Time")
ax5.set_ylabel("Velocity")
ax6 = plt.subplot(236)
ax6.set_xlabel("Time")
ax6.set_ylabel("Velocity")

ax1.plot(T, Y[:, 0], label="Demo")
ax2.plot(T, Y[:, 1], label="Demo")
# ax3.plot(T, Y[:, 2], label="Demo")
ax4.plot(T, np.gradient(Y[:, 0]) / dmp.dt_)
ax5.plot(T, np.gradient(Y[:, 1]) / dmp.dt_)
# ax6.plot(T, np.gradient(Y[:, 2]) / dmp.dt_)
ax4.scatter([T[-1]], (Y[-1, 0] - Y[-2, 0]) / dmp.dt_)
ax5.scatter([T[-1]], (Y[-1, 1] - Y[-2, 1]) / dmp.dt_)
# ax6.scatter([T[-1]], (Y[-1, 2] - Y[-2, 2]) / dmp.dt_)

dmp.configure(goal_yd=np.array([1.0, 0.0]))
# dmp.configure(goal_y=np.array([1, 0, 1]), goal_yd=np.array([goal_yd, goal_yd, goal_yd]))
T, Y = dmp.open_loop(run_t=execution_time)
ax1.plot(T, Y[:, 0], label="goal_yd = %g" % 1.0)
ax2.plot(T, Y[:, 1], label="goal_yd = %g" % 0.0)
# ax3.plot(T, Y[:, 2], label="goal_yd = %g" % goal_yd)
ax4.plot(T, np.gradient(Y[:, 0]) / dmp.dt_)
print(np.gradient(Y[:, 0]) / dmp.dt_)
ax5.plot(T, np.gradient(Y[:, 1]) / dmp.dt_)
# ax6.plot(T, np.gradient(Y[:, 2]) / dmp.dt_)
ax4.scatter([T[-1]], [1.0])
ax5.scatter([T[-1]], [0.0])
# ax6.scatter([T[-1]], [goal_yd])

ax1.legend()
plt.tight_layout()
plt.show()
