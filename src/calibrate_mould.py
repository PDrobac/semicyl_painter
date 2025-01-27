#!/usr/bin/env python3

import rospy
import rospkg
from geometry_msgs.msg import Pose, PoseArray
import robot_controller_kinova as rc
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

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
        for i in range(4):
            input("Press `Enter` to collect pose #" + str(i))
            
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
    rospy.init_node('tf2_mould_frame_calibrator')
    pub = rospy.Publisher('mould_pose_array', PoseArray, queue_size=10)
    read_from_file = True

    collected_poses = []

    # Open a file to save or read the poses
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('semicyl_painter')
    file_path = f"{package_path}/data/collected_mould_poses.txt"

    if(read_from_file):
        collected_poses = read_poses_from_file(file_path)
    else:
        collected_poses = get_poses_from_robot(file_path)

    # Create a PoseArray message
    pose_array = PoseArray()
    pose_array.header.stamp = rospy.Time.now()
    pose_array.header.frame_id = "base_link"  # Set the appropriate reference frame
    pose_array.poses = collected_poses        # Add all poses to the PoseArray

    # Publish poses
    pub.publish(pose_array)

    rate = rospy.Rate(10)
    broadcaster = TransformBroadcaster()
    planner = rc.MotionPlanner()

    while not rospy.is_shutdown():
        # Create TransformStamped message
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "base_link"
        transform.child_frame_id = "mould"
        transform.transform.translation = planner.get_cartesian_pose().position
        transform.transform.rotation = planner.get_cartesian_pose().orientation

        # Publish the transform
        broadcaster.sendTransform(transform)
        rate.sleep()
    rospy.spin()

if __name__ == "__main__":
    main()

