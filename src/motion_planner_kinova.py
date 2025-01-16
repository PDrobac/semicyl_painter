#!/usr/bin/env python3

import sys
import csv
import math
import copy
import time
import rospy
import moveit_commander
import moveit_msgs.msg
import numpy as np
import tf.transformations as tft
from geometry_msgs.msg import Pose, TransformStamped
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from std_msgs.msg import String
from dmp.srv import *
from dmp.msg import *
from tf2_ros import TransformBroadcaster

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

    # print("Starting LfD...")
    rospy.wait_for_service('learn_dmp_from_demo')
    try:
        lfd = rospy.ServiceProxy('learn_dmp_from_demo', LearnDMPFromDemo)
        resp = lfd(demotraj, k_gains, d_gains, num_bases)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
    # print("LfD done")

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
    # print("Starting DMP planning...")
    rospy.wait_for_service('get_dmp_plan')
    try:
        gdp = rospy.ServiceProxy('get_dmp_plan', GetDMPPlan)
        resp = gdp(x_0, x_dot_0, t_0, goal, goal_thresh, seg_length, tau, dt, integrate_iter)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
    # print("DMP planning done")

    return resp


class MotionPlanner(object):
    """MotionPlanner"""
    def __init__(self):

        # Initialize the node
        super(MotionPlanner, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)

        try:
            self.is_gripper_present = rospy.get_param(rospy.get_namespace() + "is_gripper_present", False)
            if self.is_gripper_present:
                gripper_joint_names = rospy.get_param(rospy.get_namespace() + "gripper_joint_names", [])
                self.gripper_joint_name = gripper_joint_names[0]
            else:
                self.gripper_joint_name = ""
                self.degrees_of_freedom = rospy.get_param(rospy.get_namespace() + "degrees_of_freedom", 7)

            # Create the MoveItInterface necessary objects
            arm_group_name = "arm"
            self.robot = moveit_commander.RobotCommander("robot_description")
            self.scene = moveit_commander.PlanningSceneInterface(ns=rospy.get_namespace())
            self.arm_group = moveit_commander.MoveGroupCommander(arm_group_name, ns=rospy.get_namespace())
            self.display_trajectory_publisher = rospy.Publisher(rospy.get_namespace() + 'move_group/display_planned_path',
                                                        moveit_msgs.msg.DisplayTrajectory,
                                                        queue_size=20)

            if self.is_gripper_present:
                gripper_group_name = "gripper"
                self.gripper_group = moveit_commander.MoveGroupCommander(gripper_group_name, ns=rospy.get_namespace())

            rospy.loginfo("Initializing node in namespace " + rospy.get_namespace())
        except Exception as e:
            print (e)
            self.is_init_success = False
        else:
            self.is_init_success = True

        # Goal poses
        self.start_poses = []
        self.end_poses = []
        self.start_hover_poses = []
        self.end_hover_poses = []
        self.dmps = []

        self.publisher = rospy.Publisher('/mould', PointCloud2, queue_size=10)

    def publish_pointcloud(self):
        """
        Publishes the PointCloud2 message.
        """
        pointcloud = create_pointcloud2(self.dmps)
        self.publisher.publish(pointcloud)
        #rospy.loginfo(f"Published PointCloud2 message with {len(self.dmps)} points.")

    def get_cartesian_pose(self):
        arm_group = self.arm_group

        # Get the current pose and display it
        pose = arm_group.get_current_pose()
        #rospy.loginfo("Actual cartesian pose is : ")
        #rospy.loginfo(pose.pose)

        return pose.pose


    def go_to_pose_goal_cartesian(self, waypoints):
        for waypoint in waypoints:
            original_q = waypoint.orientation
            axis = (0, 1, 0)  # Rotate around Y-axis
            angle = 30  # Degrees
            waypoint.orientation = rotate_quaternion(original_q, axis, angle)
            p = waypoint.position
            self.dmps.append([p.x, p.y, p.z])
        
        (plan, fraction) = self.arm_group.compute_cartesian_path(
            waypoints, 0.01  # waypoints to follow  # eef_step
        )

        self.publish_pointcloud()
        self.arm_group.execute(plan, wait=True)

    def go_to_pose_goal_dmp(self, start_pose, goal_pose, theta):
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

        start_position = [start_pose.position.x, start_pose.position.y, start_pose.position.z]
        goal_position = [goal_pose.position.x, goal_pose.position.y, goal_pose.position.z]
        plan = makePlanRequest(start_position, x_dot_0, t_0, goal_position, goal_thresh, seg_length, tau, dt, integrate_iter)

        waypoints = []
        for point in plan.plan.points:
            pose = Pose()
            pose.position.x = point.positions[0]
            pose.position.y = point.positions[1]
            pose.position.z = point.positions[2]
            #print(traj[0])
            #pose.orientation = start_pose.orientation

            original_q = start_pose.orientation
            axis = (0, 0, 1)  # Rotate around Z-axis
            angle = math.degrees(-theta)  # Degrees
            pose.orientation = rotate_quaternion(original_q, axis, angle)

            # Tool offset
            #original_q = start_pose.orientation
            #axis = (1, 0, 0)  # Rotate around Y-axis
            #angle = 90  # Degrees
            #pose.orientation = rotate_quaternion(original_q, axis, angle)

            waypoints.append(pose)
        
        self.go_to_pose_goal_cartesian(waypoints)

def create_pointcloud2(points, frame_id="world"):
    """
    Create a PointCloud2 message from a list of 3D points.
    
    Args:
        points (list): List of tuples containing (x, y, z) coordinates.
        frame_id (str): The reference frame of the point cloud.

    Returns:
        PointCloud2: The generated PointCloud2 message.
    """
    header = Header()
    header.stamp = rospy.Time.now()  # ROS1 timestamp
    header.frame_id = frame_id

    # Define the fields of the PointCloud2 message
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    point_step = 12  # 4 bytes each for x, y, z
    row_step = point_step * len(points)

    # Pack the points into binary format
    data = b"".join([struct.pack("fff", *point) for point in points])

    return PointCloud2(
        header=header,
        height=1,
        width=len(points),
        fields=fields,
        is_bigendian=False,
        point_step=point_step,
        row_step=row_step,
        data=data,
        is_dense=True,
    )

def start_pose_callback(start_pose, planner):
        planner.start_poses.append(start_pose)

def end_pose_callback(end_pose, planner):
        planner.end_poses.append(end_pose)

def start_hover_pose_callback(start_hover_pose, planner):
        planner.start_hover_poses.append(start_hover_pose)

def end_hover_pose_callback(end_hover_pose, planner):
        planner.end_hover_poses.append(end_hover_pose)

def ready_callback (default_pose, planner):
        input("============ Press `Enter` to initiate the motion planner")

        # print("-- Moving to End")
        # waypoints = []
        # waypoints.append(copy.deepcopy(planner.end_poses[0]))
        # print("Goal pose:")
        # print(str(planner.end_poses[0].orientation))
        # planner.go_to_pose_goal_cartesian(waypoints)
        # print("Real pose:")
        # print(str(planner.get_cartesian_pose().orientation))

        roll_0, pitch_0, yaw_0 = extract_orientation_from_pose(planner.start_poses[0])
        roll_1, pitch_1, yaw_1 = extract_orientation_from_pose(planner.start_poses[1])

        default_pose.orientation = planner.start_poses[0].orientation

        # Go to first start point
        waypoints = []
        waypoints.append(copy.deepcopy(default_pose))
        print("-- Moving to Start")
        # print("Goal pose:")
        # print(str(planner.start_poses[0].orientation))
        planner.go_to_pose_goal_cartesian(waypoints)
        # print("Real pose:")
        # print(str(planner.get_cartesian_pose().orientation))
        start_angle = pitch_0
        angle_step = pitch_1 - pitch_0
        theta_list = []
        
        for i in range(len(planner.start_poses)):
            waypoints = []
            theta = start_angle + i * angle_step

            # Go to next start point
            waypoints.append(copy.deepcopy(rotate_point_z(planner.start_hover_poses[i], theta)))
            waypoints.append(copy.deepcopy(rotate_point_z(planner.start_poses[i], theta)))
            print("-- Moving to Pose#" + str(i+2) + " ---------------")
            print("Goal pose:")
            print(str(planner.start_poses[i].orientation))
            planner.go_to_pose_goal_cartesian(waypoints)
            waypoints = []
            print("Real pose:")
            print(str(planner.get_cartesian_pose().orientation))

            # Execute the tool stroke
            print("-- Executing DMP#" + str(i+1) + " -------------------#")
            t_s = time.time()
            print("Goal pose:")
            print(str(planner.end_poses[i].orientation))
            planner.go_to_pose_goal_dmp(planner.start_poses[i], planner.end_poses[i], theta)
            waypoints = []
            t_e = time.time()
            print("Real pose:")
            print(str(planner.get_cartesian_pose().orientation))
            print("-- DMP cpmplete, time elapsed: " + str(t_e-t_s) + " seconds")
            theta_list.append(math.degrees(theta))

            waypoints.append(copy.deepcopy(rotate_point_z(planner.end_hover_poses[i], theta)))
            planner.go_to_pose_goal_cartesian(waypoints)

        # Go to default pose
        #waypoints = []
        #waypoints.append(copy.deepcopy(defalut_pose))
        #print("-- Moving to Default")
        #planner.go_to_pose_goal_cartesian(waypoints)
        waypoints = []
        waypoints.append(copy.deepcopy(default_pose))
        print("-- Moving to Start")
        planner.go_to_pose_goal_cartesian(waypoints)

        print("-- Completed!")
        print(theta_list)

def euler_from_orientation(pose):
    quaternion = (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w
    )
    return tft.euler_from_quaternion(quaternion)

def extract_orientation_from_pose(pose):
    quaternion = (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w
    )
    return tft.euler_from_quaternion(quaternion)

def create_pose_with_orientation(roll, pitch, yaw):
    # Initialize a Pose object
    pose = Pose()

    # Set position (optional, here set to zero)
    pose.position.x = 0.0
    pose.position.y = 0.0
    pose.position.z = 0.0

    # Convert roll, pitch, yaw to a quaternion
    quaternion = tft.quaternion_from_euler(roll, pitch, yaw)

    # Set orientation
    pose.orientation.x = quaternion[0]
    pose.orientation.y = quaternion[1]
    pose.orientation.z = quaternion[2]
    pose.orientation.w = quaternion[3]

    return pose

def rotate_quaternion(q, axis, angle_deg):
    angle_rad = math.radians(angle_deg)
    half_angle = angle_rad / 2.0
    pose = Pose()

    # Compute the rotation quaternion r
    r_w = math.cos(half_angle)
    r_x = math.sin(half_angle) * axis[0]
    r_y = math.sin(half_angle) * axis[1]
    r_z = math.sin(half_angle) * axis[2]

    # Quaternion multiplication: r * q
    pose.orientation.x = r_w * q.x + r_x * q.w + r_y * q.z - r_z * q.y
    pose.orientation.y = r_w * q.y - r_x * q.z + r_y * q.w + r_z * q.x
    pose.orientation.z = r_w * q.z + r_x * q.y - r_y * q.x + r_z * q.w
    pose.orientation.w = r_w * q.w - r_x * q.x - r_y * q.y - r_z * q.z

    return pose.orientation

def rotate_point_z(pose, theta):
    new_pose = Pose()
    original_q = pose.orientation
    axis = (0, 0, 1)  # Rotate around Z-axis
    angle = math.degrees(-theta)  # Degrees
    new_pose.orientation = rotate_quaternion(original_q, axis, angle)
    new_pose.position = pose.position
    return new_pose

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