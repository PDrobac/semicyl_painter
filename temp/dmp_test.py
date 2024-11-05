#!/usr/bin/env python3
import roslib
roslib.load_manifest('dmp')
import rospy
import numpy as np
import matplotlib.pyplot as plt
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

if __name__ == '__main__':
    rospy.init_node('dmp_tutorial_node')
    dmp_original()
    dmp_paralelogram()
    dmp_semicircle()