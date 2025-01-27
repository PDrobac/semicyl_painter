#!/usr/bin/env python3

import math
import copy
import time
import rospy
import numpy as np
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2
import pose_conversions as P
import robot_controller_kinova as rc
import dmp_node as dmp

class MouldPainter(object):
    def __init__(self):
        self.robot = rc.MotionPlanner()
        self.T_mould = np.eye(4, 4)

        self.start_poses = []
        self.end_poses = []
        self.start_hover_poses = []
        self.end_hover_poses = []
        self.default_pose = []

        self.tip_trace = []
        self.trace_publisher = rospy.Publisher('/tip_trace', PointCloud2, queue_size=10)

    def robot_mould_goto_pose(self, pose):
        pose = P.reset_pose_orientation(pose)
        tf_pose = P.apply_global_tf_to_pose(pose, self.T_mould)
        tf_pose = P.rotate_pose_about_axis(tf_pose, 90, 'y')
        tf_pose = P.rotate_pose_about_axis(tf_pose, -90, 'z')
        tf_pose = P.rotate_pose_about_axis(tf_pose, -60, 'x')

        tf = P.offset_transform(tf_pose, self.tf_default_pose, 0.01)
        tf_pose = P.apply_transform_to_pose_position(tf_pose, tf)

        p = [tf_pose.position.x, tf_pose.position.y, tf_pose.position.z]
        self.tip_trace.append(p)
        pointcloud = P.create_pointcloud2(self.tip_trace)
        self.trace_publisher.publish(pointcloud)

        self.robot.go_to_pose_goal_cartesian([tf_pose])

    def robot_mould_goto_dmp(self, start_pose, end_pose, theta):
        start_pose = P.reset_pose_orientation(start_pose)
        tf_start_pose = P.apply_global_tf_to_pose(start_pose, self.T_mould)
        tf_start_pose = P.rotate_pose_about_axis(tf_start_pose, 90, 'y')
        tf_start_pose = P.rotate_pose_about_axis(tf_start_pose, -90, 'z')
        tf_start_pose = P.rotate_pose_about_axis(tf_start_pose, -60, 'x')

        tf = P.offset_transform(tf_start_pose, self.tf_default_pose, 0.01)

        end_pose = P.reset_pose_orientation(end_pose)
        tf_end_pose = P.apply_global_tf_to_pose(end_pose, self.T_mould)
        tf_end_pose = P.rotate_pose_about_axis(tf_end_pose, 90, 'y')
        tf_end_pose = P.rotate_pose_about_axis(tf_end_pose, -90, 'z')
        tf_end_pose = P.rotate_pose_about_axis(tf_end_pose, -60, 'x')

        waypoints = dmp.calculate_dmp(tf_start_pose, tf_end_pose, theta)
        offset_waypoints = []

        for wp in waypoints:
            offset_wp = P.apply_transform_to_pose_position(wp, tf)
            offset_waypoints.append(offset_wp)

            p = [offset_wp.position.x, offset_wp.position.y, offset_wp.position.z]
            self.tip_trace.append(p)
            pointcloud = P.create_pointcloud2(self.tip_trace, "base_link")
            self.trace_publisher.publish(pointcloud)

        self.robot.go_to_pose_goal_cartesian(offset_waypoints, 0.03)
    
    def execute(self):
        input("\n============ Press `Enter` to initiate the mould painter\n")
        
        self.T_mould = P.get_tf_from_frames("base_link", "mould")

        self.back_pose = Pose()
        self.back_pose.position.x = (self.end_hover_poses[0].position.x + self.end_hover_poses[-1].position.x) / 2
        self.back_pose.position.y = (self.end_hover_poses[0].position.y + self.end_hover_poses[-1].position.y) / 2
        self.back_pose.position.z = (self.end_hover_poses[0].position.z + self.end_hover_poses[-1].position.z) / 2
        self.back_pose.orientation = self.default_pose.orientation

        self.tf_default_pose = P.reset_pose_orientation(self.default_pose)
        self.tf_default_pose = P.apply_global_tf_to_pose(self.tf_default_pose, self.T_mould)
        self.tf_default_pose = P.rotate_pose_about_axis(self.tf_default_pose, 90, 'y')
        self.tf_default_pose = P.rotate_pose_about_axis(self.tf_default_pose, -90, 'z')
        self.tf_default_pose = P.rotate_pose_about_axis(self.tf_default_pose, -60, 'x')

        roll_0, pitch_0, yaw_0 = P.get_euler_from_pose(self.start_poses[0])
        roll_1, pitch_1, yaw_1 = P.get_euler_from_pose(self.start_poses[1])

        # Go to first start point
        print("-- Moving to Start")
        self.robot_mould_goto_pose(self.default_pose)
        print("-- Start reached")
        
        # input("============ Press `Enter` to continue")

        start_angle = pitch_0
        angle_step = pitch_1 - pitch_0
        theta_list = []

        # print("####### angle list: ")
        # for sp in self.start_poses:
        #     roll, pitch, yaw = P.get_euler_from_pose(sp)
        #     print(str(math.degrees(pitch)))

        # print("#######")
             
        # print("####### second angle: " + str(math.degrees(pitch_1)))
        # print("####### angle step: " + str(math.degrees(angle_step)))

        print("-- # of poses: " + str(len(self.start_poses)))
        
        for i in range(len(self.start_poses)):
            theta = start_angle + i * angle_step

            # Go to next start point
            print("-- Moving to Pose#" + str(i+1) + " ---------------")
            self.robot_mould_goto_pose(self.default_pose)
            self.robot_mould_goto_pose(self.start_poses[i])

            # input("============ Press `Enter` to continue")

            print("-- Executing DMP#" + str(i+1) + " -------------------#")
            t_s = time.time()
            self.robot_mould_goto_dmp(self.start_poses[i], self.end_poses[i], theta)
            t_e = time.time()
            print("-- DMP complete, time elapsed: " + str(t_e-t_s) + " seconds")
            theta_list.append(math.degrees(theta))

            self.robot_mould_goto_pose(self.back_pose)

        print("-- Moving to Start")
        self.robot_mould_goto_pose(self.default_pose)

        print("-- Completed!")
        print(theta_list)

def start_pose_callback(start_pose, mould_painter):
        mould_painter.start_poses.append(start_pose)

def end_pose_callback(end_pose, mould_painter):
        mould_painter.end_poses.append(end_pose)

def start_hover_pose_callback(start_hover_pose, mould_painter):
        mould_painter.start_hover_poses.append(start_hover_pose)

def end_hover_pose_callback(end_hover_pose, mould_painter):
        mould_painter.end_hover_poses.append(end_hover_pose)

def ready_callback (default_pose, mould_painter):
        mould_painter.default_pose = default_pose
        mould_painter.execute()

def main():
    # Initialize the ROS node
    rospy.init_node('motion_planner_node', anonymous=True)

    mould_painter = MouldPainter()

    # Initialize subscribers
    start_sub = rospy.Subscriber('start_poses', Pose, start_pose_callback, mould_painter)
    end_sub = rospy.Subscriber('end_poses', Pose, end_pose_callback, mould_painter)
    start_hover_sub = rospy.Subscriber('start_hover_poses', Pose, start_hover_pose_callback, mould_painter)
    end_hover_sub = rospy.Subscriber('end_hover_poses', Pose, end_hover_pose_callback, mould_painter)
    default_sub = rospy.Subscriber('default_pose', Pose, ready_callback, mould_painter)

    rospy.spin()

if __name__ == "__main__":
    main()