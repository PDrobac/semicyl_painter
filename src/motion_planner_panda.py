#!/usr/bin/env python3

import sys
import csv
import math
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import numpy as np
import tf.transformations as tft
from geometry_msgs.msg import Pose
from std_msgs.msg import String
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

class MotionPlanner(object):
    """MotionPlanner"""
    def __init__(self):
        super(MotionPlanner, self).__init__()

        ## BEGIN_SUB_TUTORIAL setup
        ##
        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        #rospy.init_node("move_group_python_interface_tutorial", anonymous=True)

        ## Instantiate a `RobotCommander`_ object. Provides information such as the robot's
        ## kinematic model and the robot's current joint states
        robot = moveit_commander.RobotCommander()

        ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        ## for getting, setting, and updating the robot's internal understanding of the
        ## surrounding world:
        scene = moveit_commander.PlanningSceneInterface()

        ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        ## to a planning group (group of joints).  In this tutorial the group is the primary
        ## arm joints in the Panda robot, so we set the group's name to "panda_arm".
        ## If you are using a different robot, change this value to the name of your robot
        ## arm planning group.
        ## This interface can be used to plan and execute motions:
        group_name = "panda_arm"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
        ## trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )

        ## END_SUB_TUTORIAL

        ## BEGIN_SUB_TUTORIAL basic_info
        ##
        ## Getting Basic Information
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^
        # We can get the name of the reference frame for this robot:
        planning_frame = move_group.get_planning_frame()
        # print("============ Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        eef_link = move_group.get_end_effector_link()
        # print("============ End effector link: %s" % eef_link)

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        # print("============ Available Planning Groups:", robot.get_group_names())

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        # print("============ Printing robot state")
        # print(robot.get_current_state())
        # print("")
        ## END_SUB_TUTORIAL

        # Misc variables
        self.box_name = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names

        # Goal poses
        self.start_poses = []
        self.end_poses = []
        self.start_hover_poses = []
        self.end_hover_poses = []

    def go_to_pose_goal_cartesian(self, waypoints):
        ## BEGIN_SUB_TUTORIAL plan_cartesian_path
        ##
        ## Cartesian Paths
        ## ^^^^^^^^^^^^^^^
        ## You can plan and execute a Cartesian path directly by specifying a list of waypoints
        ## for the end-effector to go through.

        # We want the Cartesian path to be interpolated at a resolution of 1 cm
        # which is why we will specify 0.01 as the eef_step in Cartesian
        # translation.  We will disable the jump threshold by setting it to 0.0,
        # ignoring the check for infeasible jumps in joint space, which is sufficient
        # for this tutorial.
        (plan, fraction) = self.move_group.compute_cartesian_path(
            waypoints, 0.01  # waypoints to follow  # eef_step
        )

        self.move_group.execute(plan, wait=True)

        ## END_SUB_TUTORIAL

    def go_to_pose_goal_dmp(self, start_pose, goal_pose, theta):
        quaternion = (
            start_pose.orientation.x,
            start_pose.orientation.y,
            start_pose.orientation.z,
            start_pose.orientation.w
        )

        roll_0, pitch_0, yaw_0 = tft.euler_from_quaternion(quaternion)

        # Create a DMP from a 3-D trajectory
        dims = 3
        dt = 1.0
        K = 100
        D = 2.0 * np.sqrt(K)
        num_bases = 4
        traj = []
        # Read the file and process each line
        # Define the file path
        file_path = "semicyl_painter/data/mould_filtered_path.csv"

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
                # traj.append([float(value) for value in row])  # Convert each value to float

        resp = makeLFDRequest(dims, traj, dt, K, D, num_bases)

        # Set it as the active DMP
        makeSetActiveRequest(resp.dmp_list)

        # Now, generate a plan
        #x_0 = [0.0, 0.0, 0.0]           # Plan starting at a different point than demo
        x_dot_0 = [0.0, 0.0, 0.0]
        t_0 = 0
        goal_thresh = [0.01, 0.01, 0.01]
        seg_length = -1            # Plan until convergence to goal
        tau = 2 * resp.tau         # Desired plan should take twice as long as demo
        dt = 1.0
        integrate_iter = 5         # dt is rather large, so this is > 1
        plan = makePlanRequest(start_pose.position, x_dot_0, t_0, goal_pose.position, goal_thresh, seg_length, tau, dt, integrate_iter)

        plan_points = []
        for point in plan.plan.points:
            plan_points.append(point.positions)

        waypoints = copy.deepcopy(plan_points)
        self.go_to_pose_goal_cartesian(waypoints)

def start_pose_callback(start_pose, planner):
        planner.start_poses.append(start_pose)

def end_pose_callback(end_pose, planner):
        planner.end_poses.append(end_pose)

def start_hover_pose_callback(start_hover_pose, planner):
        planner.start_hover_poses.append(start_hover_pose)

def end_hover_pose_callback(end_hover_pose, planner):
        planner.end_hover_poses.append(end_hover_pose)

def ready_callback (defalut_pose, planner):
        input("============ Press `Enter` to initiate the motion planner")

        # Go to first start point
        waypoints = []
        waypoints.append(copy.deepcopy(planner.start_hover_poses[0]))
        waypoints.append(copy.deepcopy(planner.start_poses[0]))
        planner.go_to_pose_goal_cartesian(waypoints)

        pose_0 = planner.start_poses[0]
        quaternion_0 = (
            pose_0.orientation.x,
            pose_0.orientation.y,
            pose_0.orientation.z,
            pose_0.orientation.w
        )
        pose_1 = planner.start_poses[1]
        quaternion_1 = (
            pose_1.orientation.x,
            pose_1.orientation.y,
            pose_1.orientation.z,
            pose_1.orientation.w
        )

        roll_0, pitch_0, yaw_0 = tft.euler_from_quaternion(quaternion_0)
        roll_1, pitch_1, yaw_1 = tft.euler_from_quaternion(quaternion_1)
        start_angle = pitch_0
        angle_step = pitch_1 - pitch_0
        
        for i in range(len(planner.start_poses) - 1):
            # Execute the tool stroke
            theta = start_angle + i * angle_step
            planner.go_to_pose_goal_dmp(planner.start_poses[i], planner.end_poses[i], theta)

            # Go to next start point
            waypoints = []
            waypoints.append(copy.deepcopy(planner.end_hover_poses[i]))
            waypoints.append(copy.deepcopy(planner.start_hover_poses[i+1]))
            waypoints.append(copy.deepcopy(planner.start_poses[i+1]))
            planner.go_to_pose_goal_cartesian(waypoints)

        # Go to default pose
        waypoints = []
        waypoints.append(copy.deepcopy(defalut_pose))

def main():
    # Initialize the ROS node
    rospy.init_node('motion_planner_node', anonymous=True)

    planner = MotionPlanner()

    # Initialize subscribers
    start_sub = rospy.Subscriber('start_poses', Pose, start_pose_callback, planner)
    end_sub = rospy.Subscriber('end_poses', Pose, end_pose_callback, planner)
    start_hover_sub = rospy.Subscriber('start_hover_poses', Pose, start_hover_pose_callback, planner)
    end_hover_sub = rospy.Subscriber('end_hover_poses', Pose, end_hover_pose_callback, planner)
    default_sub = rospy.Subscriber('default_pose', Pose, ready_callback, planner)

    # ready_pub = rospy.Publisher('ready_pose', String, queue_size=10)
    # ready_pub.publish('msg')

    # print("Published ready flag!")

    # Keep the node running
    rospy.spin()

if __name__ == "__main__":
    main()