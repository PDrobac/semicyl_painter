#!/usr/bin/env python3

import os
import sys
import csv
import rospy
import moveit_commander
import moveit_msgs.msg
import numpy as np
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

def main():
    rospy.init_node('tf2_frame_broadcaster')

    planner = MotionPlanner()

    input("============ Set robot to sp_first (Top Left) of the mould, then press `Enter` to continue")
    sp_first = planner.get_cartesian_pose()
    sp_first = translate_pose_local(sp_first, 0, 0.1085, 0.041)
    print(sp_first.position)
    input("============ Set robot to sp_last (Top Right) of the mould, then press `Enter` to continue")
    sp_last = planner.get_cartesian_pose()
    sp_last = translate_pose_local(sp_last, 0, 0.1085, 0.041)
    print(sp_last.position)
    input("============ Set robot to ep_first (Bottom Left) of the mould, then press `Enter` to continue")
    ep_first = planner.get_cartesian_pose()
    ep_first = translate_pose_local(ep_first, 0, 0.1085, 0.041)
    print(ep_first.position)
    input("============ Set robot to ep_last (Bottom right) of the mould, then press `Enter` to continue")
    ep_last = planner.get_cartesian_pose()
    ep_last = translate_pose_local(ep_last, 0, 0.1085, 0.041)
    print(ep_last.position)

    # Define the points
    P1 = [sp_first.position.x, sp_first.position.y, sp_first.position.z]
    P2 = [ep_first.position.x, ep_first.position.y, ep_first.position.z]
    P3 = [ep_last.position.x, ep_last.position.y, ep_last.position.z]
    P4 = [sp_last.position.x, sp_last.position.y, sp_last.position.z]

    PC = []
    for i1,i2 in zip(P1, P4):
        PC.append((i1 + i2)/2)

    # Compute unit vectors for the axes
    x_axis_1 = compute_unit_vector(P2, P1)
    x_axis_2 = compute_unit_vector(P3, P4)
    x_axis = []
    for i1,i2 in zip(x_axis_1, x_axis_2):
        x_axis.append((i1 + i2)/2)

    y_axis_1 = compute_unit_vector(P4, P1)
    y_axis_2 = compute_unit_vector(P3, P2)
    y_axis = []
    for i1,i2 in zip(y_axis_1, y_axis_2):
        y_axis.append((i1 + i2)/2)

    z_axis = np.cross(x_axis, y_axis)  # Ensure right-handed coordinate system
    x_axis = np.cross(y_axis, z_axis)  # Recompute X to ensure orthogonality

    # Normalize all axes
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis /= np.linalg.norm(z_axis)

    # Create rotation matrix
    rotation_matrix = np.array([
        [x_axis[0], y_axis[0], z_axis[0], 0],
        [x_axis[1], y_axis[1], z_axis[1], 0],
        [x_axis[2], y_axis[2], z_axis[2], 0],
        [0, 0, 0, 1]
    ])

    # Convert rotation matrix to quaternion
    quaternion = tft.quaternion_from_matrix(rotation_matrix)

    # Define the frame data
    frame_data = {
        "frame_id": "mould",
        "x": PC[0],
        "y": PC[1],
        "z": PC[2],
        "qx": quaternion[0],
        "qy": quaternion[1],
        "qz": quaternion[2],
        "qw": quaternion[3]
    }

    # Write to CSV file
    write_frame_to_csv(frame_data)

if __name__ == "__main__":
    main()