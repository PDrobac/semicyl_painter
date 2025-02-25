#!/usr/bin/env python3

import math
import matplotlib.pyplot as plt
import numpy as np
import dmp_node as dmp

def plot_old_2d(traj, welding_pattern, dmp):
    # Extract x and y coordinates
    x_traj = [point[0] for point in traj]
    y_traj = [point[1] for point in traj]
    x_dmp = [point[0] for point in dmp]
    y_dmp = [point[1] for point in dmp]
    x_wp = [point[0] for point in welding_pattern]
    y_wp = [point[1] for point in welding_pattern]

    # Create the plot
    plt.plot(x_traj, y_traj, color='blue', label='Original trajectory')
    plt.plot(x_dmp, y_dmp, color='red', label='DMP trajectory')
    plt.plot(x_wp, y_wp, color='green', label='Welding pattern')
    plt.title("Trajectory comparison")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.show()

def plot_new_2d(traj, pos):
    dt = 0.1
    execution_time = len(traj) * dt
    T = np.arange(0, execution_time + dt, dt)
    size_1 = len(traj)
    size_2 = T.shape[0]
    diff = size_2 - size_1
    if(diff > 0):
        T = T[:-diff]

    plt.figure(2, figsize=(10, 6))
    ax1 = plt.subplot(231)
    ax1.set_title("Dimension 1")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Position [m]")
    ax2 = plt.subplot(232)
    ax2.set_title("Dimension 2")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Position [m]")
    ax3 = plt.subplot(233)
    ax3.set_title("Dimension Sum")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Position [m]")

    ax4 = plt.subplot(234)
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Velocity [m/s]")
    ax5 = plt.subplot(235)
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Velocity [m/s]")
    ax6 = plt.subplot(236)
    ax6.set_xlabel("Time [s]")
    ax6.set_ylabel("Velocity [m/s]")

    Y = np.array(traj)

    Y_ = [math.sqrt(x*x + y*y) for x, y in zip(Y[:, 0], Y[:, 1])]
    pos_ = [math.sqrt(x*x + y*y) for x, y in zip(np.array(pos)[:, 0], np.array(pos)[:, 1])]

    ax1.plot(T, Y[:, 0], label="Demo")
    ax2.plot(T, Y[:, 1], label="Demo")
    ax3.plot(T, Y_, label="Demo")
    ax4.plot(T, np.gradient(Y[:, 0]) / dt)
    ax5.plot(T, np.gradient(Y[:, 1]) / dt)
    ax6.plot(T, np.gradient(Y_) / dt)
    #ax4.scatter([T[-1]], (Y[-1, 0] - Y[-2, 0]) / dmp.dt_)
    #ax5.scatter([T[-1]], (Y[-1, 1] - Y[-2, 1]) / dmp.dt_)
    # ax6.scatter([T[-1]], (Y[-1, 2] - Y[-2, 2]) / dmp.dt_)
    # dmp.configure(goal_y=np.array([1, 0, 1]), goal_yd=np.array([goal_yd, goal_yd, goal_yd]))
    dt = 0.1 * len(traj) / len(pos)
    execution_time = len(pos) * dt
    T = np.arange(0, execution_time, dt)
    size_1 = np.array(pos)[:, 0].shape[0]
    size_2 = T.shape[0]
    if(size_2 > size_1):
        T = T[:-1]

    ax1.plot(T, np.array(pos)[:, 0], label="DMP")
    ax2.plot(T, np.array(pos)[:, 1], label="DMP")
    ax3.plot(T, pos_, label="DMP")
    ax4.plot(T, np.gradient(np.array(pos)[:, 0]) / dt)
    ax5.plot(T, np.gradient(np.array(pos)[:, 1]) / dt)
    ax6.plot(T, np.gradient(pos_) / dt)
    #ax4.scatter([T[-1]], [1.0])
    #ax5.scatter([T[-1]], [0.0])
    # ax6.scatter([T[-1]], [goal_yd])

    ax1.legend()
    plt.tight_layout()

def main():
    radius = np.pi/2
    theta = np.linspace(-np.pi, 2 * np.pi, 300)  # Angle from 0 to pi for a semicircle
    x_semi = radius * np.cos(theta) + radius + 0.5 * (theta + np.pi) # x = r * cos(theta)
    y_semi = radius * np.sin(theta) # y = r * sin(theta)
    traj = [(x, y) for x, y in zip(x_semi, y_semi)]
    # traj.insert(0, traj[0])
    # traj.append(traj[-1])

    traj1 = traj[:len(traj) // 2]
    traj2 = traj[len(traj) // 2:]

    for _ in range(20):
        point = (traj1[-1][0] - 0.05, traj1[-1][1])
        traj1.append(point)  # Insert the middle value

    for tr in traj2:
        point = (tr[0] - 1, tr[1])
        traj1.append(point)  # Insert the middle value

    traj = traj1

    dist_to_goal = math.sqrt((traj[-1][0]-traj[0][0])**2 + (traj[-1][1]-traj[0][1])**2)

    path = dmp.calculate_dmp(traj)
    welding_pattern = [(dist_to_goal*i/len(path), 0.05*math.sin(200*i)) for i in range(len(path))]

    phi_list = [0.0]
    for i in range(1, len(path)-1):
        d_x = path[i+1][0] - path[i-1][0]
        d_y = path[i+1][1] - path[i-1][1]
        phi_list.append(math.atan2(d_y, d_x))
    phi_list.append(0.0)

    waypoints = []
    for i, phi in enumerate(phi_list):
        d = welding_pattern[i]
        d_x = -d[1] * math.sin(phi)
        d_y = d[1] * math.cos(phi)
        wp_x = path[i][0] + d_x
        wp_y = path[i][1] + d_y
        waypoints.append((wp_x, wp_y))


    # plot_old_2d(traj, welding_pattern, path)
    plot_old_2d(traj, welding_pattern, waypoints)

if __name__ == "__main__":
    main()