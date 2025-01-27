#!/usr/bin/env python3

import csv
import math
import rospy
import rospkg
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
import pose_conversions as P

class PoseCollector:
    def __init__(self):
        # Initialize list to store poses
        self.points = []
        self.recording = False
        self.publisher = rospy.Publisher('/stroke_recorded', PointCloud2, queue_size=10)

        rospack = rospkg.RosPack()
        package_path = rospack.get_path('semicyl_painter')
        self.output_csv = f"{package_path}/data/stroke_recorded.csv"
        
        # Subscribe to the pose topic
        self.pose_sub = rospy.Subscriber(
            "/vrpn_client_node/Kalipen/pose", 
            PoseStamped, 
            self.pose_callback
        )
    
    def pose_callback(self, msg):
        """
        Callback function for the pose subscriber.
        Adds the received pose to the pose_list.
        """
        pose = msg.pose
        T = np.eye(4, 4)
        T[1,3] = -0.1325
        tf_pose = P.apply_local_tf_to_pose(pose, T)

        if(self.recording):
            p = [tf_pose.position.x, tf_pose.position.y, tf_pose.position.z]

            if len(self.points) == 0:
                self.points.append(p)

            x1, y1, z1 = p
            x2, y2, z2 = self.points[-1]
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
            if(distance > 0.001):
                self.points.append(p)
                print(p)

                pointcloud = P.create_pointcloud2(self.points)
                self.publisher.publish(pointcloud)
            # rospy.sleep(0.1)

    def save_points(self):
        """
        Saves the poses into the output CSV file.
        """
        try:
            # Open the output CSV file and write the filtered points
            with open(self.output_csv, mode='w', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerows(self.points)
            print(f"Stroke saved to {self.output_csv}")
        
        except Exception as e:
            print(f"Error saving CSV file: {e}")

if __name__ == "__main__":
    # Initialize ROS node
    rospy.init_node("pose_collector", anonymous=True)
        
    # Instantiate the PoseCollector and run it
    collector = PoseCollector()
    input("Press `Enter` to start recording")
    collector.recording = True
    input("Press `Enter` to stop recording")
    collector.recording = False
    collector.save_points()
