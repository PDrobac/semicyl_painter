#!/usr/bin/env python3
import math
import roslib
import copy
import csv
roslib.load_manifest('dmp')
import rospy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dmp.srv import *
from dmp.msg import *

# Learn a DMP from demonstration data
def makeLFDRequest(dims, traj, dt, K_gain, D_gain, num_bases):
    demotraj = DMPTraj()

    for i in range(len(traj)):
        pt = DMPPoint()
        pt.positions = traj[i]
        demotraj.points.append(pt)
        demotraj.times.append(dt * i)

    k_gains = [K_gain] * dims
    d_gains = [D_gain] * dims

    print("Starting LfD...")
    rospy.wait_for_service('learn_dmp_from_demo')
    try:
        lfd = rospy.ServiceProxy('learn_dmp_from_demo', LearnDMPFromDemo)
        resp = lfd(demotraj, k_gains, d_gains, num_bases)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
    print("LfD done")

    return resp

# Set a DMP as active for planning
def makeSetActiveRequest(dmp_list):
    try:
        sad = rospy.ServiceProxy('set_active_dmp', SetActiveDMP)
        sad(dmp_list)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

# Generate a plan from a DMP
def makePlanRequest(x_0, x_dot_0, t_0, goal, goal_thresh, seg_length, tau, dt, integrate_iter):
    print("Starting DMP planning...")
    rospy.wait_for_service('get_dmp_plan')
    try:
        gdp = rospy.ServiceProxy('get_dmp_plan', GetDMPPlan)
        resp = gdp(x_0, x_dot_0, t_0, goal, goal_thresh, seg_length, tau, dt, integrate_iter)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
    print("DMP planning done")

    return resp

def plot_dmp_2d(traj, dmp, goal):
    """
    Plot the trajectory, scaled trajectory and dmp.
    
    Parameters:
    traj (list): A list of (x, y) coordinates of the original trajectory.
    dmp (list): A list of (x, y) coordinates of the generated dmp trajectory.
    goal (list): Coordinates of the intended goal pose.
    """
    # Calculate scaling factors
    #scale_x = (goal[0] - traj[0][0]) / (traj[-1][0]- traj[0][0])
    #scale_y = (goal[1] - traj[0][1]) / (traj[-1][1]- traj[0][1])

    # Extract x and y coordinates
    x_traj = [point[0] for point in traj]
    y_traj = [point[1] for point in traj]
    #x_scaled = [(point[0] - traj[0][0]) * scale_x + traj[0][0] for point in traj]
    #y_scaled = [(point[1] - traj[0][1]) * scale_y + traj[0][1] for point in traj]
    x_dmp = [point[0] for point in dmp]
    y_dmp = [point[1] for point in dmp]

    # Create the plot
    plt.figure(1)
    plt.plot(x_traj, y_traj, color='blue', label='Original trajectory')
    #plt.plot(x_scaled, y_scaled, marker='o', linestyle='-', color='green', label='Scaled trajectory')
    plt.plot(x_dmp, y_dmp, color='red', label='DMP trajectory')
    plt.title("Trajectory comparison")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()

def plot_dmp_3d(traj, dmp, goal):
    """
    Plot the 3D trajectory, scaled trajectory, and DMP.
    
    Parameters:
    traj (list): A list of (x, y, z) coordinates of the original trajectory.
    dmp (list): A list of (x, y, z) coordinates of the generated DMP trajectory.
    goal (list): Coordinates of the intended goal pose.
    """
    # Extract x, y, and z coordinates for the original trajectory
    x_traj = [point[0] for point in traj]
    y_traj = [point[1] for point in traj]
    z_traj = [point[2] for point in traj]
    
    # Extract x, y, and z coordinates for the DMP trajectory
    x_dmp = [point[0] for point in dmp]
    y_dmp = [point[1] for point in dmp]
    z_dmp = [point[2] for point in dmp]
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original trajectory
    ax.plot(x_traj, y_traj, z_traj, marker='o', linestyle='-', color='blue', label='Original trajectory')
    
    # Plot DMP trajectory
    ax.plot(x_dmp, y_dmp, z_dmp, marker='o', linestyle='-', color='red', label='DMP trajectory')
    
    # Plot the goal point
    ax.scatter(goal[0], goal[1], goal[2], color='green', s=100, label='Goal', marker='*')
    
    # Add labels and legend
    ax.set_title("3D Trajectory Comparison")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.legend()
    
    # plt.show()

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
    plt.show()

def plot_new_3d(traj, pos):
    dt = 0.01
    execution_time = len(traj) * 0.01
    T = np.arange(0, execution_time, dt)

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

    Y = np.array(traj)

    ax1.plot(T, Y[:, 0], label="Demo")
    ax2.plot(T, Y[:, 1], label="Demo")
    ax3.plot(T, Y[:, 2], label="Demo")
    ax4.plot(T, np.gradient(Y[:, 0]) / dt)
    ax5.plot(T, np.gradient(Y[:, 1]) / dt)
    ax6.plot(T, np.gradient(Y[:, 2]) / dt)
    #ax4.scatter([T[-1]], (Y[-1, 0] - Y[-2, 0]) / dmp.dt_)
    #ax5.scatter([T[-1]], (Y[-1, 1] - Y[-2, 1]) / dmp.dt_)
    # ax6.scatter([T[-1]], (Y[-1, 2] - Y[-2, 2]) / dmp.dt_)
    # dmp.configure(goal_y=np.array([1, 0, 1]), goal_yd=np.array([goal_yd, goal_yd, goal_yd]))
    dt = 0.01 * len(traj) / len(pos)
    execution_time = len(pos) * dt
    T = np.arange(0, execution_time, dt)
    ax1.plot(T, np.array(pos)[:, 0])
    ax2.plot(T, np.array(pos)[:, 1])
    ax3.plot(T, np.array(pos)[:, 2])
    ax4.plot(T, np.gradient(np.array(pos)[:, 0]) / dt)
    ax5.plot(T, np.gradient(np.array(pos)[:, 1]) / dt)
    ax6.plot(T, np.gradient(np.array(pos)[:, 2]) / dt)
    #ax4.scatter([T[-1]], [1.0])
    #ax5.scatter([T[-1]], [0.0])
    # ax6.scatter([T[-1]], [goal_yd])

    ax1.legend()
    plt.tight_layout()
    plt.show()

def dmp_original():
    # Create a DMP from a 2-D trajectory
    dims = 2
    dt = 0.1
    K = 100
    D = 2.0 * np.sqrt(K)
    num_bases = 4
    traj = [[1.0, 1.0], [2.0, 2.0], [3.0, 4.0], [6.0, 8.0]]
    resp = makeLFDRequest(dims, traj, dt, K, D, num_bases)

    # Set it as the active DMP
    makeSetActiveRequest(resp.dmp_list)

    # Now, generate a plan
    x_0 = [0.0, 0.0]           # Plan starting at a different point than demo
    x_dot_0 = [0.0, 0.0]
    t_0 = 0
    goal = [8.0, 7.0]          # Plan to a different goal than demo
    goal_thresh = [0.05, 0.05]
    seg_length = -1            # Plan until convergence to goal
    tau = 2 * resp.tau         # Desired plan should take twice as long as demo
    dt = 1.0
    integrate_iter = 5         # dt is rather large, so this is > 1
    plan = makePlanRequest(x_0, x_dot_0, t_0, goal, goal_thresh, seg_length, tau, dt, integrate_iter)

    # print(plan)

    plan_positions = []
    plan_velocities = []
    for point in plan.plan.points:
        plan_positions.append(point.positions)
        plan_velocities.append(point.velocities)

    plot_dmp_2d(traj, plan_positions, goal)

def dmp_paralelogram():
    # Create a DMP from a 2-D trajectory
    dims = 2
    dt = 1.0
    K = 100
    D = 2.0 * np.sqrt(K)
    num_bases = 4
    traj = [[1.0, 1.0], [2.0, 2.0], [3.0, 2.0], [4.0, 1.0]]
    resp = makeLFDRequest(dims, traj, dt, K, D, num_bases)

    # Set it as the active DMP
    makeSetActiveRequest(resp.dmp_list)

    # Now, generate a plan
    x_0 = [0.0, 0.0]           # Plan starting at a different point than demo
    x_dot_0 = [0.0, 0.0]
    t_0 = 0
    goal = [4.0, 1.0]          # Plan to a different goal than demo
    goal_thresh = [0.05, 0.05]
    seg_length = -1            # Plan until convergence to goal
    tau = 2 * resp.tau         # Desired plan should take twice as long as demo
    dt = 1.0
    integrate_iter = 5         # dt is rather large, so this is > 1
    plan = makePlanRequest(x_0, x_dot_0, t_0, goal, goal_thresh, seg_length, tau, dt, integrate_iter)

    # print(plan)

    plan_positions = []
    plan_velocities = []
    for point in plan.plan.points:
        plan_positions.append(point.positions)
        plan_velocities.append(point.velocities)

    #plot_dmp_2d(traj, plan_positions, goal)

    abs_velocities = []
    for vels in plan_velocities:
        #abs_velocities.append(math.sqrt(vels[0]*vels[0] + vels[1]*vels[1]))
        abs_velocities.append(vels[0])

    t = np.linspace(0, len(abs_velocities), len(abs_velocities))
    #print(t)
    #print(abs_velocities)
    plt.plot(t, abs_velocities, marker='o', linestyle='-', color='blue', label='Velocities')
    plt.show()

def dmp_semicircle():
    # Create a DMP from a 2-D trajectory
    dims = 2
    dt = 1.0
    K = 100
    D = 2.0 * np.sqrt(K)
    num_bases = 4
    radius = 1.5
    theta = np.linspace(np.pi, 0, 100)  # Angle from 0 to pi for a semicircle
    x_semi = radius * np.cos(theta) + radius  # x = r * cos(theta)
    y_semi = radius * np.sin(theta)  # y = r * sin(theta)
    traj = [(x, y) for x, y in zip(x_semi, y_semi)]
    resp = makeLFDRequest(dims, traj, dt, K, D, num_bases)

    # Set it as the active DMP
    makeSetActiveRequest(resp.dmp_list)

    # Now, generate a plan
    x_0 = [0.0, 0.0]           # Plan starting at a different point than demo
    x_dot_0 = [0.0, 0.0]
    t_0 = 0
    goal = [4.0, 0.0]          # Plan to a different goal than demo
    goal_thresh = [0.05, 0.05]
    seg_length = -1            # Plan until convergence to goal
    tau = 2 * resp.tau         # Desired plan should take twice as long as demo
    dt = 1.0
    integrate_iter = 5         # dt is rather large, so this is > 1
    plan = makePlanRequest(x_0, x_dot_0, t_0, goal, goal_thresh, seg_length, tau, dt, integrate_iter)

    #print(plan)

    plan_positions = []
    plan_velocities = []
    for point in plan.plan.points:
        plan_positions.append(point.positions)
        plan_velocities.append(point.velocities)

    plot_new_2d(traj, plan_positions)
    plot_dmp_2d(traj, plan_positions, goal)

    # abs_velocities = []
    # for vels in plan_velocities:
    #     abs_velocities.append(math.sqrt(vels[0]*vels[0] + vels[1]*vels[1]))

    # t = np.linspace(0, len(abs_velocities), len(abs_velocities))
    # #print(t)
    # #print(abs_velocities)
    # plt.plot(t, abs_velocities, marker='o', linestyle='-', color='blue', label='Velocities')
    # plt.show()

def dmp_original_3d():
    # Create a DMP from a 2-D trajectory
    dims = 3
    dt = 1.0
    K = 100
    D = 2.0 * np.sqrt(K)
    num_bases = 4
    traj = [[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 4.0, 0.0], [6.0, 8.0, 0.0]]
    resp = makeLFDRequest(dims, traj, dt, K, D, num_bases)

    # Set it as the active DMP
    makeSetActiveRequest(resp.dmp_list)

    # Now, generate a plan
    x_0 = [0.0, 0.0, 0.0]           # Plan starting at a different point than demo
    x_dot_0 = [0.0, 0.0, 0.0]
    t_0 = 0
    goal = [8.0, 7.0, 0.0]          # Plan to a different goal than demo
    goal_thresh = [0.2, 0.2, 0.2]
    seg_length = -1            # Plan until convergence to goal
    tau = 2 * resp.tau         # Desired plan should take twice as long as demo
    dt = 1.0
    integrate_iter = 5         # dt is rather large, so this is > 1
    plan = makePlanRequest(x_0, x_dot_0, t_0, goal, goal_thresh, seg_length, tau, dt, integrate_iter)

    # print(plan)

    plan_positions = []
    plan_velocities = []
    for point in plan.plan.points:
        plan_positions.append(point.positions)
        plan_velocities.append(point.velocities)

    plot_dmp_3d(traj, plan_positions, goal)

def dmp_semicircle_3d():
    # Create a DMP from a 3-D trajectory
    dims = 3
    dt = 1.0
    K = 100
    D = 2.0 * np.sqrt(K)
    num_bases = 4
    radius = 1.5
    theta = np.linspace(np.pi, 0, 4)  # Angle from 0 to pi for a semicircle
    x_semi = radius * np.cos(theta) + radius  # x = r * cos(theta)
    y_semi = radius * np.sin(theta)  # y = r * sin(theta)
    z_semi = theta
    traj = [(x, y, z) for x, y, z in zip(x_semi, y_semi, z_semi)]
    resp = makeLFDRequest(dims, traj, dt, K, D, num_bases)

    # Set it as the active DMP
    makeSetActiveRequest(resp.dmp_list)

    # Now, generate a plan
    x_0 = [0.0, 0.0, 0.0]           # Plan starting at a different point than demo
    x_dot_0 = [0.0, 0.0, 0.0]
    t_0 = 0
    goal = [8.0, 1.0, 0.0]          # Plan to a different goal than demo
    goal_thresh = [0.05, 0.05, 0.05]
    seg_length = -1            # Plan until convergence to goal
    tau = 2 * resp.tau         # Desired plan should take twice as long as demo
    dt = 1.0
    integrate_iter = 5         # dt is rather large, so this is > 1
    plan = makePlanRequest(x_0, x_dot_0, t_0, goal, goal_thresh, seg_length, tau, dt, integrate_iter)

    #print(plan)

    plan_positions = []
    plan_velocities = []
    for point in plan.plan.points:
        plan_positions.append(point.positions)
        plan_velocities.append(point.velocities)

    plot_dmp_3d(traj, plan_positions, goal)

def dmp_data():
    # Create a DMP from a 2-D trajectory
    dims = 3
    dt = 1.0
    K = 100
    D = 2.0 * np.sqrt(K)
    num_bases = 4
    traj = []
    # Read the file and process each line
    with open("semicyl_painter/temp/points_data.txt", "r") as file:
        for line in file:
            # Remove any whitespace or newline characters
            line = line.strip()
            
            # Evaluate the line as a list of floats
            # If the line contains something like "[1.0, 2.0, 3.0]", eval will convert it into a Python list [1.0, 2.0, 3.0]
            point = eval(line)
            
            # Append the parsed point to the points list
            if(len(traj) == 0):
                traj.append(point)
                traj.append(point)
            traj.append(point)

    traj.append(point)
    traj.append(point)
    resp = makeLFDRequest(dims, traj, dt, K, D, num_bases)

    # Set it as the active DMP
    makeSetActiveRequest(resp.dmp_list)

    # Now, generate a plan
    #x_0 = [0.0, 0.0, 0.0]           # Plan starting at a different point than demo
    x_0 = traj[0]
    x_dot_0 = [0.0, 0.0, 0.0]
    t_0 = 0
    #goal = [0.2, 1.0, 1.0]          # Plan to a different goal than demo
    #goal = [f - l for f, l in zip(traj[-1], traj[0])]
    goal = copy.deepcopy(traj[-1])
    #goal[0] -= 0.1
    goal_thresh = [0.01, 0.01, 0.01]
    seg_length = -1            # Plan until convergence to goal
    tau = 2 * resp.tau         # Desired plan should take twice as long as demo
    dt = 1.0
    integrate_iter = 5         # dt is rather large, so this is > 1
    plan = makePlanRequest(x_0, x_dot_0, t_0, goal, goal_thresh, seg_length, tau, dt, integrate_iter)

    # print(plan)

    plan_positions = []
    plan_velocities = []
    for point in plan.plan.points:
        plan_positions.append(point.positions)
        plan_velocities.append(point.velocities)

    plot_new_3d(traj, plan_positions)

    # plot_dmp_3d(traj, plan_positions, goal)

    # abs_velocities = []
    # for vels in plan_velocities:
    #     abs_velocities.append(math.sqrt(vels[0]*vels[0] + vels[1]*vels[1] + vels[2]*vels[2]))

    # t = np.linspace(0, len(abs_velocities), len(abs_velocities))
    # #print(t)
    # #print(abs_velocities)
    # plt.plot(t, abs_velocities, marker='o', linestyle='-', color='blue', label='Velocities')
    # plt.show()

def dmp_bmaric():
    # Create a DMP from a 2-D trajectory
    dims = 2
    dt = 0.1
    K = 100
    D = 2.0 * np.sqrt(K)
    num_bases = 4
    traj = []
    # Read the file and process each line
    with open("data/bmaric.csv", "r") as file:
        for line in file:
            # Remove any whitespace or newline characters
            line = line.strip()
            
            # Evaluate the line as a list of floats
            # If the line contains something like "[1.0, 2.0, 3.0]", eval will convert it into a Python list [1.0, 2.0, 3.0]
            point = list(map(float, line.split(',')))
            # point = tuple(float(line.split(',')))
            
            # Append the parsed point to the points list
            if(len(traj) == 0):
                traj.append(point)
                traj.append(point)
                traj.append(point)
            traj.append(point)

    traj.append(point)
    traj.append(point)
    traj.append(point)
    resp = makeLFDRequest(dims, traj, dt, K, D, num_bases)

    # Set it as the active DMP
    makeSetActiveRequest(resp.dmp_list)

    # Now, generate a plan
    x_0 = [0.0, 0.0]           # Plan starting at a different point than demo
    x_dot_0 = [0.0, 0.0]
    t_0 = 0
    #goal = [0.2, 1.0]          # Plan to a different goal than demo
    goal = [2*(f - l) for f, l in zip(traj[-1], traj[0])]
    #goal[1] -= 0.2
    goal_thresh = [0.01, 0.01]
    seg_length = -1            # Plan until convergence to goal
    tau = resp.tau         # Desired plan should take twice as long as demo
    dt = 0.1
    integrate_iter = 10         # dt is rather large, so this is > 1
    plan = makePlanRequest(x_0, x_dot_0, t_0, goal, goal_thresh, seg_length, tau, dt, integrate_iter)

    # print(plan)

    plan_positions = []
    plan_velocities = []
    for point in plan.plan.points:
        plan_positions.append(point.positions)
        plan_velocities.append(point.velocities)

    plot_dmp_2d(traj, plan_positions, goal)
    plot_new_2d(traj, plan_positions)

    # abs_velocities = []
    # for vels in plan_velocities:
    #     abs_velocities.append(math.sqrt(vels[0]*vels[0] + vels[1]*vels[1]))

    # t = np.linspace(0, len(abs_velocities), len(abs_velocities))
    # #print(t)
    # #print(abs_velocities)
    # plt.plot(t, abs_velocities, marker='o', linestyle='-', color='blue', label='Velocities')
    # plt.show()

def dmp_mould():
    # Create a DMP from a 3-D trajectory
    dims = 3
    dt = 1.0
    K = 100
    D = 2.0 * np.sqrt(K)
    num_bases = 4
    traj = []
    # Read the file and process each line
    # Define the file path
    file_path = "data/mould_filtered_path.csv"

    # Read the CSV file and create a list of 3-member arrays
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            traj.append([float(value) for value in row])  # Convert each value to float

    resp = makeLFDRequest(dims, traj, dt, K, D, num_bases)

    # Set it as the active DMP
    makeSetActiveRequest(resp.dmp_list)

    # Now, generate a plan
    #x_0 = [0.0, 0.0, 0.0]           # Plan starting at a different point than demo
    x_0 = traj[0]
    x_dot_0 = [0.0, 0.0, 0.0]
    t_0 = 0
    #goal = [0.2, 1.0, 1.0]          # Plan to a different goal than demo
    #goal = [f - l for f, l in zip(traj[-1], traj[0])]
    goal = copy.deepcopy(traj[-1])
    #goal[0] -= 0.1
    goal_thresh = [0.01, 0.01, 0.01]
    seg_length = -1            # Plan until convergence to goal
    tau = 2 * resp.tau         # Desired plan should take twice as long as demo
    dt = 1.0
    integrate_iter = 5         # dt is rather large, so this is > 1
    plan = makePlanRequest(x_0, x_dot_0, t_0, goal, goal_thresh, seg_length, tau, dt, integrate_iter)

    # print(plan)

    plan_positions = []
    plan_velocities = []
    for point in plan.plan.points:
        plan_positions.append(point.positions)
        plan_velocities.append(point.velocities)

    plot_dmp_3d(traj, plan_positions, goal)
    plot_new_3d(traj, plan_positions)

if __name__ == '__main__':
    rospy.init_node('dmp_tutorial_node')
    # dmp_original()
    # dmp_paralelogram()
    # dmp_semicircle()
    # dmp_original_3d()
    # dmp_semicircle_3d()
    # dmp_data()
    # dmp_bmaric()
    dmp_mould()