#!/usr/bin/env python3

import sys
import rospy
import rospkg
import moveit_commander
import moveit_msgs.msg
import numpy as np
import pose_conversions as P

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

        rospack = rospkg.RosPack()
        package_path = rospack.get_path('semicyl_painter')
        file_path = f"{package_path}/config/robot_tip.txt"
        self.T_to_tip = np.loadtxt(file_path)

    def get_cartesian_pose(self):
        arm_group = self.arm_group
        pose = arm_group.get_current_pose()

        transformed_pose = P.apply_local_tf_to_pose(pose.pose, self.T_to_tip)

        return transformed_pose
    
    def go_to_pose_goal_cartesian(self, waypoints):
        wps = []
        for waypoint in waypoints:
            T_to_tool = P.invert_tf(self.T_to_tip)
            wp = P.apply_local_tf_to_pose(waypoint, T_to_tool)
            wps.append(wp)
        
        (plan, fraction) = self.arm_group.compute_cartesian_path(
            wps, 0.01  # waypoints to follow  # eef_step
        )
        rospy.sleep(0.1)

        self.arm_group.execute(plan, wait=True)
