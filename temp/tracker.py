#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg

class PoseToPointCloud2Node:

    def __init__(self):
        # Initialize the node
        rospy.init_node("pose_to_pointcloud2_node", anonymous=True)

        # Create subscribers to the /kdno/TOOL and /kdno/INST_POINT topics
        rospy.Subscriber("kdno/TOOL", PoseStamped, self.pose_callback)
        rospy.Subscriber("kdno/INST_POINT", PoseStamped, self.pose_callback)

        # Create a publisher for the /tracker topic with PointCloud2
        self.pointcloud_pub = rospy.Publisher("tracker", PointCloud2, queue_size=10)

        # Initialize an empty list to store points
        self.points = []

        # Variable to track the last message time
        self.last_received_time = rospy.Time.now()
        self.timeout_flag = False

        # Start a timer to check if no messages are received within 1 second
        rospy.Timer(rospy.Duration(1), self.timer_callback)

    def point_exists(self, point):
        # Check if a point with the same coordinates exists in the list
        return any(p[0] == point[0] and p[1] == point[1] and p[2] == point[2] for p in self.points)

    def points_to_txt(self, output_file):
        # Open the file in write mode
        with open(output_file, 'w') as file:
            for point in self.points:
                # Write each point's coordinates to the file
                file.write(f"{point}\n")
        rospy.loginfo(f"Exported Points to {output_file}")

    def pose_callback(self, pose_msg):
        # Convert the Pose to a PointCloud2-compatible point
        point = [
            pose_msg.pose.position.x,
            pose_msg.pose.position.y,
            pose_msg.pose.position.z
        ]

        # Add this point to the list of points
        if not self.point_exists(point):
            self.points.append(point)

        # Update the last received time
        self.last_received_time = rospy.Time.now()

        # Define the fields for the PointCloud2 message (x, y, z)
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]

        # Create the PointCloud2 message
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "world"  # Adjust as per your frame of reference

        # Generate the PointCloud2 message
        pointcloud_msg = pc2.create_cloud(header, fields, self.points)

        # Publish the PointCloud2 message
        self.pointcloud_pub.publish(pointcloud_msg)

        self.timeout_flag = False

    def timer_callback(self, event):
        # Check if more than 5 seconds has passed since the last message
        if (rospy.Time.now() - self.last_received_time).to_sec() > 5.0 and not self.timeout_flag:
            rospy.loginfo("No message received in the last second, exporting and clearing point cloud.")
            # Export the point cloud
            output_file = "temp/points_data.txt"
            self.points_to_txt(output_file)

            # Publish an empty PointCloud2 message to clear the point cloud
            empty_cloud = PointCloud2()
            empty_cloud.header = std_msgs.msg.Header()
            empty_cloud.header.stamp = rospy.Time.now()
            empty_cloud.header.frame_id = "world"  # Adjust the frame as needed
            self.pointcloud_pub.publish(empty_cloud)
            # Clear the points list
            self.points = []

            self.timeout_flag = True

if __name__ == "__main__":
    try:
        # Initialize and run the node
        node = PoseToPointCloud2Node()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
