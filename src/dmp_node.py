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

# TODO add T_mould, fix orientation
def calculate_dmp(start_pose, goal_pose, theta):
    # Create a DMP from a 3-D trajectory
    dims = 3
    dt = 1.0
    K = 100
    D = 2.0 * np.sqrt(K)
    num_bases = 4
    traj = []
    orig = []
    # Read the file and process each line
    # Open a file to save or read the poses
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('semicyl_painter')
    file_path = f"{package_path}/data/mould_filtered_path.csv"

    trace_publisher = rospy.Publisher('/dmp_trace', PointCloud2, queue_size=10)

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
            orig.append(mould_dims_fixed)

    pointcloud = P.create_pointcloud2(orig)
    trace_publisher.publish(pointcloud)
    input("...")
    pointcloud = P.create_pointcloud2([])
    trace_publisher.publish(pointcloud)

    resp = __makeLFDRequest(dims, traj[::-1], dt, K, D, num_bases)

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

    start_position = [start_pose.position.x, start_pose.position.y, start_pose.position.z]
    goal_position = [goal_pose.position.x, goal_pose.position.y, goal_pose.position.z]
    plan = __makePlanRequest(start_position, x_dot_0, t_0, goal_position, goal_thresh, seg_length, tau, dt, integrate_iter)

    waypoints = []
    for point in plan.plan.points:
        pose = Pose()
        pose.position.x = point.positions[0]
        pose.position.y = point.positions[1]
        pose.position.z = point.positions[2]
        #print(traj[0])

        waypoints.append(pose)

    return waypoints