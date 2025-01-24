#!/usr/bin/env python3

import rospy
import rospkg
import numpy as np
from scipy.optimize import fmin
from geometry_msgs.msg import Pose
import robot_controller_kinova as rc
import pose_conversions as P
import tf_conversions
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

def objectiveFunPosition(p, R, T_data):
    """
    Optimization for position
    """
    T0A = T_data.copy()
    # Initial supposed transform
    Tx = np.eye(4)
    Tx[:3, :3] = R
    Tx[:3, 3] = p

    num_of_mesurs = len(T0A)  # holds the number of measurements
    f = 0  # optimization function value initialized to 0
    for i in range(0, num_of_mesurs):
        T_OI = np.matmul(T0A[i], Tx) 
        for j in range(0, num_of_mesurs):
            T_OJ = np.matmul(T0A[j], Tx) 
            p_ = T_OI[:3, 3] - T_OJ[:3, 3]
            f += np.linalg.norm(p_)  # Euclidean distance
    return f

def f_optimizePointOrientation(T_init, callibration_data_poses):
    R1 = T_init[:3, :3]  # Initial rotation matrix
    P1 = T_init[:3, 3]   # Initial position vector

    # Optimize position using fmin
    P1_optimized = fmin(func=objectiveFunPosition, x0=P1, args=(R1, callibration_data_poses), xtol=1e-10, ftol=1e-10,  disp=True)

    # print("optimization finished")
    # print(P1_optimized)
    T = np.eye(4)
    T[:3, 3] = P1_optimized

    return T

def calibrateTransformation(callibration_data_poses):
    T_init = np.eye(4, 4)
    T_pose_list = []
    for pose in callibration_data_poses:
        T = P.pose_to_matrix(pose)
        T_pose_list.append(T)

    T = f_optimizePointOrientation(T_init, T_pose_list)
    return T

def read_poses_from_file(file_path):
    """
    Reads poses from a text file and returns a list of Pose objects.
    
    Args:
        file_path (str): Path to the file containing poses.

    Returns:
        list of geometry_msgs.msg.Pose: A list of Pose objects.
    """
    poses = []
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    current_pose = None
    for line in lines:
        line = line.strip()
        if line.startswith("Pose"):
            if current_pose is not None:
                poses.append(current_pose)
            current_pose = Pose()
        elif line.startswith("Position"):
            parts = line.split(", ")
            current_pose.position.x = float(parts[0].split("=")[1])
            current_pose.position.y = float(parts[1].split("=")[1])
            current_pose.position.z = float(parts[2].split("=")[1])
        elif line.startswith("Orientation"):
            parts = line.split(", ")
            current_pose.orientation.x = float(parts[0].split("=")[1])
            current_pose.orientation.y = float(parts[1].split("=")[1])
            current_pose.orientation.z = float(parts[2].split("=")[1])
            current_pose.orientation.w = float(parts[3].split("=")[1])
    if current_pose is not None:
        poses.append(current_pose)
    
    return poses

def get_poses_from_robot(file_path):
    planner = rc.MotionPlanner()
    collected_poses = []

    with open(file_path, "w") as file:
        for i in range(5):
            input("Press `Enter` to collect " + str(i) + "/5")
            
            # Get the pose
            pose = planner.get_cartesian_pose()

            # Append to the list
            collected_poses.append(pose)

            # Save to the file
            file.write(f"Pose {i + 1}:\n")
            file.write(f"  Position: x={pose.position.x}, y={pose.position.y}, z={pose.position.z}\n")
            file.write(f"  Orientation: x={pose.orientation.x}, y={pose.orientation.y}, z={pose.orientation.z}, w={pose.orientation.w}\n\n")

        return collected_poses

def main():
    rospy.init_node('tf2_tip_frame_calibrator')
    read_from_file = rospy.get_param("~read_from_file", True)

    collected_poses = []

    # Open a file to save or read the poses
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('semicyl_painter')
    file_path = f"{package_path}/data/collected_tooltip_poses.txt"
    save_path = f"{package_path}/config/robot_tip.txt"

    if(read_from_file):
        collected_poses = read_poses_from_file(file_path)
    else:
        collected_poses = get_poses_from_robot(file_path)

    # Call the calibration function
    T_to_tip = calibrateTransformation(collected_poses)

    # # Convert needle to brush
    # tip_pose.position.y -= 0.005
    # tip_pose.position.z += 0.01

    # Print the calibrated pose
    # print(T_to_tip)
    np.savetxt(save_path, T_to_tip)
    
    rospy.loginfo(f"TF matrix saved to {save_path}, publishing tooltip_needle until killed.")

    rate = rospy.Rate(10)
    broadcaster = TransformBroadcaster()

    while not rospy.is_shutdown():
        # Create TransformStamped message
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "end_effector_link"
        transform.child_frame_id = "tooltip_needle"
        transform.transform.translation.x = T_to_tip[0,3]
        transform.transform.translation.y = T_to_tip[1,3]
        transform.transform.translation.z = T_to_tip[2,3]
        q = tf_conversions.transformations.quaternion_from_matrix(T_to_tip)
        transform.transform.rotation.x = q[0]
        transform.transform.rotation.y = q[1]
        transform.transform.rotation.z = q[2]
        transform.transform.rotation.w = q[3]

        # Publish the transform
        broadcaster.sendTransform(transform)
        rate.sleep()

if __name__ == "__main__":
    main()

