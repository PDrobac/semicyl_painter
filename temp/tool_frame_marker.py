#!/usr/bin/env python

import rospy
from visualization_msgs.msg import Marker

def publish_marker():
    rospy.init_node('marker_publisher')
    marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)

    rate = rospy.Rate(100)  # 100 Hz
    while not rospy.is_shutdown():
        marker = Marker()
        marker.header.frame_id = "tool_frame"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "example_marker"
        marker.id = 0
        marker.type = Marker.ARROW  # Use ARROW instead of SPHERE
        marker.action = Marker.ADD

        # Position and orientation
        marker.pose.position.x = 0.0  # Base of the arrow
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = -0.7071
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 0.7071

        # Scale: x = shaft length, y = shaft diameter, z = head diameter
        marker.scale.x = 0.1  # Arrow shaft length
        marker.scale.y = 0.001  # Arrow shaft diameter
        marker.scale.z = 0.001  # Arrow head diameter

        # Color: red
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # Alpha (transparency)

        rospy.loginfo("Publishing arrow marker")
        marker_pub.publish(marker)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_marker()
    except rospy.ROSInterruptException:
        pass
