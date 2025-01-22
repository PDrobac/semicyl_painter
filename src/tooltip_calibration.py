#!/usr/bin/env python3

import os
import sys
import csv
import rospy
import rospkg
import moveit_commander
import moveit_msgs.msg
import numpy as np
from scipy.optimize import fmin
import tf.transformations as tft
from geometry_msgs.msg import Pose

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

    def get_cartesian_pose(self):
        arm_group = self.arm_group

        # Get the current pose and display it
        pose = arm_group.get_current_pose()
        #rospy.loginfo("Actual cartesian pose is : ")
        #rospy.loginfo(pose.pose)

        return pose.pose

def compute_unit_vector(p1, p2):
    """Compute the unit vector from p1 to p2."""
    vec = np.array(p2) - np.array(p1)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def write_frame_to_csv(frame_data):
    """Write frame data to a CSV file."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, '../data/frame_pose.csv')
    with open(file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["frame_id", "x", "y", "z", "qx", "qy", "qz", "qw"])
        writer.writeheader()
        writer.writerow(frame_data)
        rospy.loginfo(f"Frame data written to {file_path}")

def translate_pose_local(pose, dx, dy, dz):
    """
    Translates a pose by (dx, dy, dz) in its own local frame.

    Args:
        pose (Pose): The input pose to translate.
        dx, dy, dz (float): Translation along the local x, y, and z axes.

    Returns:
        Pose: The translated pose.
    """
    # Extract position and orientation
    position = pose.position
    orientation = pose.orientation

    # Convert quaternion to rotation matrix
    quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
    rotation_matrix = tft.quaternion_matrix(quaternion)

    # Create a local translation vector
    local_translation = [dx, dy, dz, 1.0]  # Homogeneous coordinates

    # Transform the local translation to the global frame
    global_translation = rotation_matrix.dot(local_translation)

    # Update the pose's position
    translated_pose = Pose()
    translated_pose.orientation = pose.orientation  # Orientation remains the same
    translated_pose.position.x = position.x + global_translation[0]
    translated_pose.position.y = position.y + global_translation[1]
    translated_pose.position.z = position.z + global_translation[2]

    return translated_pose

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

    print("optimization finished")
    print(P1_optimized)
    T = np.eye(4)
    T[:3, 3] = P1_optimized

    return T

def calibrateTransformation(callibration_data_poses):
        T_init = np.eye(4, 4)
        T_pose_list = []
        for pose in callibration_data_poses:
            T = pose_to_transformation_matrix(pose)
            T_pose_list.append(T)

        T = f_optimizePointOrientation(T_init, T_pose_list)
        return transformation_matrix_to_pose(T)

def pose_to_transformation_matrix(pose):
    """
    Converts a Pose to a 4x4 transformation matrix.
    Args:
        pose (geometry_msgs/Pose): The pose to convert.

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    # Extract translation
    translation = (pose.position.x, pose.position.y, pose.position.z)
    
    # Extract quaternion
    quaternion = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
    
    # Create the transformation matrix
    matrix = tft.quaternion_matrix(quaternion)
    
    # Set the translation
    matrix[0:3, 3] = translation

    return matrix

def transformation_matrix_to_pose(matrix):
    """
    Converts a 4x4 transformation matrix to a Pose.
    Args:
        np.ndarray: A 4x4 transformation matrix.

    Returns:
        pose (geometry_msgs/Pose): The pose to convert.
    """
    # Extract translation (position)
    translation = matrix[:3, 3]

    # Extract rotation (quaternion)
    quaternion = tft.quaternion_from_matrix(matrix)

    # Create a Pose message
    pose = Pose()
    pose.position.x = translation[0]
    pose.position.y = translation[1]
    pose.position.z = translation[2]
    pose.orientation.x = quaternion[0]
    pose.orientation.y = quaternion[1]
    pose.orientation.z = quaternion[2]
    pose.orientation.w = quaternion[3]

    return pose

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
    planner = MotionPlanner()
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
    read_from_file = True

    collected_poses = []

    # Open a file to save or read the poses
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('semicyl_painter')
    file_path = f"{package_path}/data/collected_poses.txt"

    if(read_from_file):
        collected_poses = read_poses_from_file(file_path)
    else:
        collected_poses = get_poses_from_robot(file_path)

    # Call the calibration function
    tip_pose = calibrateTransformation(collected_poses)

    # Convert needle to brush
    tip_pose.position.y -= 0.005
    tip_pose.position.z += 0.01

    # Print the calibrated pose
    print(tip_pose)

if __name__ == "__main__":
    main()

