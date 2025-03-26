#!/usr/bin/env python

import rospy
import csv
from geometry_msgs.msg import PoseStamped, PointStamped

# List to store pose data
pose_list = []
pitch = 0

# CSV file path
csv_filename = "/home/pero/Jelena/bruno1.csv"  # Change this path as needed

def sriod_callback(msg: PointStamped):
    global pitch

    pitch = msg.point.z

def pose_callback(msg):
    """Callback function to store received poses and write to CSV."""
    global pose_list
    global pitch

    # Extract position and orientation
    position = msg.pose.position
    orientation = msg.pose.orientation

    # Store in list
    pose_list.append([position.x, position.y, position.z,
                      orientation.x, orientation.y, orientation.z, orientation.w,
                      pitch])

    rospy.loginfo(f"Stored pose: {pose_list[-1]}")

    # Save to CSV
    with open(csv_filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(pose_list[-1])  # Write the latest pose

def pose_listener():
    """Initialize the ROS node and subscribe to the pose topic."""
    rospy.init_node('pose_collector', anonymous=True)

    # Create CSV file and write header
    with open(csv_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "qx", "qy", "qz", "qw", "sriod"])

    rospy.Subscriber("/vrpn_client_node/spahtla/pose", PoseStamped, pose_callback)
    rospy.Subscriber("/sriod_data_stamped", PointStamped, sriod_callback)
    rospy.spin()  # Keep the node running

if __name__ == "__main__":
    try:
        pose_listener()
    except rospy.ROSInterruptException:
        pass
