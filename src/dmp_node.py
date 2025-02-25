#!/usr/bin/env python3

import csv
import math
import rospy
import rospkg
import numpy as np
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2
import pose_conversions as P
from dmp.srv import *
from dmp.msg import *
import multiprocessing

# Learn a DMP from demonstration data
def __makeLFDRequest(dims, traj, dt, K_gain, D_gain, num_bases):
    demotraj = DMPTraj()

    for i in range(len(traj)):
        pt = DMPPoint()
        pt.positions = traj[i]
        demotraj.points.append(pt)
        demotraj.times.append(dt * i)

    k_gains = [K_gain] * dims
    d_gains = [D_gain] * dims

    #print("Starting LfD...")
    rospy.wait_for_service('learn_dmp_from_demo')
    #print("...")
    try:
        lfd = rospy.ServiceProxy('learn_dmp_from_demo', LearnDMPFromDemo)
        resp = lfd(demotraj, k_gains, d_gains, num_bases)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
    # print("LfD done")

    return resp

# Set a DMP as active for planning
def __makeSetActiveRequest(dmp_list):
    try:
        sad = rospy.ServiceProxy('set_active_dmp', SetActiveDMP)
        sad(dmp_list)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

# Generate a plan from a DMP
def __makePlanRequest(x_0, x_dot_0, t_0, goal, goal_thresh, seg_length, tau, dt, integrate_iter):
    # print("Starting DMP planning...")
    rospy.wait_for_service('get_dmp_plan')
    try:
        gdp = rospy.ServiceProxy('get_dmp_plan', GetDMPPlan)
        resp = gdp(x_0, x_dot_0, t_0, goal, goal_thresh, seg_length, tau, dt, integrate_iter)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
    # print("DMP planning done")

    return resp

def get_filtered_mould_path(theta):
    traj = []
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('semicyl_painter')
    file_path = f"{package_path}/data/mould_filtered_path.csv"

    # Read the CSV file and create a list of 3-member arrays
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            mould_dims = [float(value) for value in row]
            mould_dims_fixed = [mould_dims[1], -mould_dims[0], mould_dims[2]]
            mould_dims_rotated = [mould_dims_fixed[0],
                                mould_dims_fixed[1] * math.cos(theta) + mould_dims_fixed[2] * math.sin(theta),
                                mould_dims_fixed[2] * math.cos(theta) - mould_dims_fixed[1] * math.sin(theta)]
            traj.append(mould_dims_rotated)
    
    return traj[::-1]

def get_filtered_stroke_path(T_mould, theta):
    traj = []
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('semicyl_painter')
    file_path = f"{package_path}/data/stroke_recorded.csv"

    # trace_publisher = rospy.Publisher('/stroke_example', PointCloud2, queue_size=10)
    # theta = 0

    # Read the CSV file and create a list of 3-member arrays
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        cnt = 0
        for row in reader:
            if(cnt < 0):
                cnt += 1
                continue
            cnt = 0

            mould_dims = [float(value) for value in row]
            # mould_dims_fixed = [mould_dims[2], mould_dims[0], mould_dims[1]]
            mould_dims_fixed = [mould_dims[2], 0, mould_dims[1]]

            mould_dims_rotated = [mould_dims_fixed[0],
                                mould_dims_fixed[1] * math.cos(theta) + mould_dims_fixed[2] * math.sin(theta),
                                mould_dims_fixed[2] * math.cos(theta) - mould_dims_fixed[1] * math.sin(theta)]
            
            homogeneous_point = np.array([mould_dims_rotated[0], mould_dims_rotated[1], mould_dims_rotated[2], 1])
            transformed_homogeneous_point = np.dot(T_mould, homogeneous_point)
            mould_dims_fixed = transformed_homogeneous_point[:3]
            
            traj.append(mould_dims_fixed)
    
    # pointcloud = P.create_pointcloud2(traj)
    # trace_publisher.publish(pointcloud)
    # rospy.sleep(1)

    # Smooth traj using a moving average
    smoothed_traj = []
    window_size = 3  # Define the window size for smoothing
    half_window = window_size // 2

    for i in range(len(traj)):
        x_sum, y_sum, z_sum = 0, 0, 0
        count = 0

        # Compute the average within the window
        for j in range(max(0, i - half_window), min(len(traj), i + half_window + 1)):
            x_sum += traj[j][0]
            y_sum += traj[j][1]
            z_sum += traj[j][2]
            count += 1

        smoothed_traj.append([
            x_sum / count,
            y_sum / count,
            z_sum / count
        ])

    # waypoints = []
    # for mould_dims_fixed in smoothed_traj:
    #     mould_dims_rotated = [mould_dims_fixed[0],
    #     mould_dims_fixed[1] * math.cos(theta) + mould_dims_fixed[2] * math.sin(theta),
    #     mould_dims_fixed[2] * math.cos(theta) - mould_dims_fixed[1] * math.sin(theta)]
    #     waypoints.append(mould_dims_rotated)
    
    return smoothed_traj

def calculate_dmp_painter(start_pose, goal_pose, theta, T_mould=[]):
    if len(T_mould) == 0:
        T_mould = P.get_tf_from_frames("base_link", "mould")

    # Create a DMP from a 3-D trajectory
    dims = 3
    dt = 1.0
    K = 100
    D = 2.0 * np.sqrt(K)
    num_bases = 4
    traj = get_filtered_stroke_path(T_mould, theta)

    resp = __makeLFDRequest(dims, traj, dt, K, D, num_bases)

    # Set it as the active DMP
    __makeSetActiveRequest(resp.dmp_list)

    # Now, generate a plan
    #x_0 = [0.0, 0.0, 0.0]           # Plan starting at a different point than demo
    x_dot_0 = [0.0, 0.0, 0.0]
    t_0 = 0
    goal_thresh = [0.01, 0.01, 0.01]
    seg_length = -1            # Plan until convergence to goal
    tau = 2 * resp.tau         # Desired plan should take twice as long as demo
    dt = 1.0
    integrate_iter = 5         # dt is rather large, so this is > 1

    max_vels = [0, 0, 0]
    velocities = compute_velocity_with_time(traj, 0.1)
    # print(velocities)
    for vel in velocities:
        for i in range(dims):
            if abs(vel[i]) > max_vels[i]:
                max_vels[i] = abs(vel[i])
    # print("Input max_vel: ")
    # print(max_vel)
    max_vel = math.sqrt(max_vels[0]*max_vels[0] + max_vels[1]*max_vels[1] + max_vels[2]*max_vels[2])

    start_position = [start_pose.position.x, start_pose.position.y, start_pose.position.z]
    goal_position = [goal_pose.position.x, goal_pose.position.y, goal_pose.position.z]
    plan = __makePlanRequest(start_position, x_dot_0, t_0, goal_position, goal_thresh, seg_length, tau, dt, integrate_iter)

    waypoints = []
    # plan_positions = []
    # vel_scale = 1.0
    for point in plan.plan.points:
        # for i in range(len(max_vel)):
        #     if vel_scale > max_vel[i] / abs(point.velocities[i]):
        #         vel_scale = max_vel[i] / abs(point.velocities[i])

        pose = Pose()
        pose.position.x = point.positions[0]
        pose.position.y = point.positions[1]
        pose.position.z = point.positions[2]
        pose.orientation = start_pose.orientation
        #print(traj[0])
        # plan_positions.append(point.positions)

        waypoints.append(pose)

    # proc = multiprocessing.Process(target=plot_new_3d, args=[traj, plan_positions])
    # proc.start()
    # proc.join()
    # print("Adjusted vel_scale: " + str(vel_scale))

    return [waypoints, max_vel]

def calculate_dmp(traj):

    # Create a DMP from a 3-D trajectory
    dims = 2
    dt = 1.0
    K = 100
    D = 2.0 * np.sqrt(K)
    num_bases = 4

    resp = __makeLFDRequest(dims, traj, dt, K, D, num_bases)

    # Set it as the active DMP
    __makeSetActiveRequest(resp.dmp_list)

    # Now, generate a plan
    x_dot_0 = [0.0 for _ in range(dims)]
    t_0 = 0
    goal_thresh = [0.05 for _ in range(dims)]
    seg_length = -1            # Plan until convergence to goal
    tau = 2 * resp.tau         # Desired plan should take twice as long as demo
    dt = 1.0
    integrate_iter = 5         # dt is rather large, so this is > 1

    start_position = traj[0]
    goal_position = traj[-1]
    plan = __makePlanRequest(start_position, x_dot_0, t_0, goal_position, goal_thresh, seg_length, tau, dt, integrate_iter)

    waypoints = []
    for point in plan.plan.points:
        waypoints.append(point.positions)

    return waypoints

def plot_new_3d(traj, pos):
    import matplotlib.pyplot as plt

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

def compute_velocity_with_time(points, delta_t):
    velocities = []
    velocities.append([0, 0, 0])
    for i in range(1, len(points)):
        # print(points[i])
        delta_d = np.array(points[i]) - np.array(points[i-1])
        # print(delta_d)
        velocity = delta_d / delta_t if delta_t != 0 else 0  # Avoid division by zero
        velocities.append(velocity)
    
    return velocities