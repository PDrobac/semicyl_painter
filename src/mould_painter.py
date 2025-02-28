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
import multiprocessing

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

    def robot_mould_goto_pose(self, pose, theta=0):
        pose = P.reset_pose_orientation(pose)
        tf_pose = P.apply_global_tf_to_pose(pose, self.T_mould)
        tf_pose = P.rotate_pose_about_axis(tf_pose, 90, 'y')
        tf_pose = P.rotate_pose_about_axis(tf_pose, -90, 'z')
        tf_pose = P.rotate_pose_about_axis(tf_pose, -45, 'x')
        
        tf_pose = P.pose_brush_padding(tf_pose, math.degrees(theta), 0.0075)

        p = [tf_pose.position.x, tf_pose.position.y, tf_pose.position.z]
        self.tip_trace.append(p)
        pointcloud = P.create_pointcloud2(self.tip_trace)
        self.trace_publisher.publish(pointcloud)

        self.robot.go_to_pose_goal_cartesian([tf_pose])

    def robot_mould_goto_dmp(self, start_pose, end_pose, theta):
        start_pose = P.reset_pose_orientation(start_pose)
        start_pose = P.apply_global_tf_to_pose(start_pose, self.T_mould)
        start_pose = P.rotate_pose_about_axis(start_pose, 90, 'y')
        start_pose = P.rotate_pose_about_axis(start_pose, -90, 'z')
        start_pose = P.rotate_pose_about_axis(start_pose, -45, 'x')
        tf_start_pose = P.pose_brush_padding(start_pose, math.degrees(theta), 0.0075)

        end_pose = P.reset_pose_orientation(end_pose)
        end_pose = P.apply_global_tf_to_pose(end_pose, self.T_mould)
        end_pose = P.rotate_pose_about_axis(end_pose, 90, 'y')
        end_pose = P.rotate_pose_about_axis(end_pose, -90, 'z')
        end_pose = P.rotate_pose_about_axis(end_pose, -45, 'x')
        tf_end_pose = P.pose_brush_padding(end_pose, math.degrees(theta), 0.0075)

        [waypoints, max_vel] = dmp.calculate_dmp_painter(tf_start_pose, tf_end_pose, theta)

        # proc = multiprocessing.Process(target=plot_waypoints, args=[waypoints])
        # proc.start()

        # wps = []
        for wp in waypoints:
            # wp = P.reset_pose_orientation(wp)     
            # wp = P.apply_global_tf_to_pose(wp, self.T_mould)
            # wp = P.rotate_pose_about_axis(wp, 90, 'y')
            # wp = P.rotate_pose_about_axis(wp, -90, 'z')
            # wp = P.rotate_pose_about_axis(wp, -45, 'x')
            # wp = P.pose_brush_padding(wp, math.degrees(theta), 0.01)    

            # wps.append(wp)

            p = [wp.position.x, wp.position.y, wp.position.z]
            self.tip_trace.append(p)
            pointcloud = P.create_pointcloud2(self.tip_trace, "base_link")
            self.trace_publisher.publish(pointcloud)

        self.robot.go_to_pose_goal_cartesian(waypoints)
        # proc.join()
    
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
        self.tf_default_pose = P.rotate_pose_about_axis(self.tf_default_pose, -45, 'x')

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
            self.robot_mould_goto_pose(self.start_poses[i], theta)

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

def plot_waypoints(waypoints):
    # print(waypoints[::9])

    import matplotlib.pyplot as plt 
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_traj = [pose.position.x for pose in waypoints]
    y_traj = [pose.position.y for pose in waypoints]
    z_traj = [pose.position.z for pose in waypoints]
    
    # Plot original trajectory
    ax.plot(x_traj, y_traj, z_traj, marker='o', linestyle='-', color='blue', label='Original trajectory')
    
    # Add labels and legend
    ax.set_title("3D Trajectory")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.legend()
    
    plt.show()

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