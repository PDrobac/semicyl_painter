#!/usr/bin/env python3
import roslib
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
    plt.plot(x_traj, y_traj, marker='o', linestyle='-', color='blue', label='Original trajectory')
    #plt.plot(x_scaled, y_scaled, marker='o', linestyle='-', color='green', label='Scaled trajectory')
    plt.plot(x_dmp, y_dmp, marker='o', linestyle='-', color='red', label='DMP trajectory')
    plt.title("Trajectory comparison")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.show()

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
    
    plt.show()

def dmp_original():
    # Create a DMP from a 2-D trajectory
    dims = 2
    dt = 1.0
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

    plot_dmp_2d(traj, plan_positions, goal)

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
    goal = [8.0, 1.0]          # Plan to a different goal than demo
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

    plot_dmp_2d(traj, plan_positions, goal)

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
    with open("temp/points_data.txt", "r") as file:
        for line in file:
            # Remove any whitespace or newline characters
            line = line.strip()
            
            # Evaluate the line as a list of floats
            # If the line contains something like "[1.0, 2.0, 3.0]", eval will convert it into a Python list [1.0, 2.0, 3.0]
            point = eval(line)
            
            # Append the parsed point to the points list
            traj.append(point)

    resp = makeLFDRequest(dims, traj, dt, K, D, num_bases)

    # Set it as the active DMP
    makeSetActiveRequest(resp.dmp_list)

    # Now, generate a plan
    x_0 = [0.0, 0.0, 0.0]           # Plan starting at a different point than demo
    x_dot_0 = [0.0, 0.0, 0.0]
    t_0 = 0
    goal = [0.2, 1.0, 1.0]          # Plan to a different goal than demo
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

if __name__ == '__main__':
    rospy.init_node('dmp_tutorial_node')
    # dmp_original()
    # dmp_paralelogram()
    # dmp_semicircle()
    # dmp_original_3d()
    # dmp_semicircle_3d()
    dmp_data()