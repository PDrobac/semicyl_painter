#!/usr/bin/env python3

import rospy
from moveit_commander import PlanningSceneInterface
from geometry_msgs.msg import PoseStamped

def add_obstacle():
    rospy.init_node("add_obstacle_node")

    # Create a PlanningSceneInterface object
    scene = PlanningSceneInterface()
    rospy.sleep(2)  # Allow some time for initialization

    # Define the obstacle (e.g., a box)
    box_name = "obstacle_box"
    box_pose = PoseStamped()
    box_pose.header.frame_id = "base_link"  # Use the appropriate frame (e.g., "world" or "base_link")
    box_pose.pose.position.x = 0.5     # X position of the box
    box_pose.pose.position.y = 0.0     # Y position of the box
    box_pose.pose.position.z = 0.0    # Z position of the box (height above the ground)
    box_pose.pose.orientation.x = 0.0  # Orientation (quaternion)
    box_pose.pose.orientation.y = 0.0  # Orientation (quaternion)
    box_pose.pose.orientation.z = 0.0  # Orientation (quaternion)
    box_pose.pose.orientation.w = 1.0  # Orientation (quaternion)

    box_size = (0.4, 0.5, 0.25)  # Size of the box (x, y, z)

    # Add the box to the planning scene
    rospy.loginfo("Adding obstacle to the planning scene...")
    scene.add_box(box_name, box_pose, box_size)

    # Wait to ensure the obstacle is added
    rospy.sleep(1)

if __name__ == "__main__":
    try:
        add_obstacle()
    except rospy.ROSInterruptException:
        pass
