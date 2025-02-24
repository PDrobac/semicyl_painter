#!/usr/bin/env python3

import math
import multiprocessing
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
            
            self.arm_group.set_max_velocity_scaling_factor(0.75)  # 75% of max velocity
            self.arm_group.set_max_acceleration_scaling_factor(0.75)  # 75% of max acceleration
            # self.max_vel = 0.02

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
    
    def go_to_pose_goal_cartesian(self, waypoints, max_vel=-1):
        wps = []
        for waypoint in waypoints:
            T_to_tool = P.invert_tf(self.T_to_tip)
            wp = P.apply_local_tf_to_pose(waypoint, T_to_tool)
            # wp.position.z += 0.005
            wps.append(wp)
        
        (plan, fraction) = self.arm_group.compute_cartesian_path(
            wps, 0.01  # waypoints to follow  # eef_step
        )
        # offset = 0

        # if plan:
            # for point in plan.joint_trajectory.points:
            #     jacobian = self.arm_group.get_jacobian_matrix(list(point.positions))
            #     eef_speed = compute_ee_speed(point.velocities, jacobian)
            #     vel_sum = math.sqrt(eef_speed[0]*eef_speed[0] + eef_speed[1]*eef_speed[1] + eef_speed[2]*eef_speed[2])
            #     # print("vel_sum:")
            #     # print(vel_sum)
            #     velocity_scaling_factor = vel_sum / self.max_vel
            #     if offset == 0:
            #         offset = point.time_from_start * velocity_scaling_factor - point.time_from_start
            #     else:
            #         offset += point.time_from_start * velocity_scaling_factor - point.time_from_start

                # point.time_from_start += offset
            # print("Output vel_scale")
            # print(velocity_scaling_factor)
        # print(velocity_scaling_factor)
        # print(offset)

        # if(max_vel > 0):
        #     plan = scale_trajectory_speed(plan, max_vel)

        success = self.arm_group.execute(plan, wait=True)

        velocity_scaling_factor = 1.0

        # Execute the plan and wait for it to complete
        while not success:
            # rospy.sleep(1.0)
            # Ensure the arm stops completely to avoid residual movement
            self.arm_group.stop()

            # Clear any targets to avoid interference with future plans
            self.arm_group.clear_pose_targets()
            rospy.sleep(1.0)

            # (plan, fraction) = self.arm_group.compute_cartesian_path(
            #     wps, 0.01  # waypoints to follow  # eef_step
            # )

            # t = 0
            # for i, point in enumerate(plan.joint_trajectory.points):
            #     t_n = point.time_from_start.to_sec()
            #     if t > t_n:
            #         print("GOOBER ALERT")
            #         # print("point " + str(i-1) + "/" + str(len(plan.joint_trajectory.points)) + ":")
            #         # print(t)
            #         # print("point " + str(i) + "/" + str(len(plan.joint_trajectory.points)) + ":")
            #         # print(t_n)
            #         point.time_from_start = rospy.Duration(plan.joint_trajectory.points[i-1].time_from_start.to_sec() + point.time_from_start.to_sec())
            #         point.velocities = [0.0] * len(point.velocities)
            #         point.accelerations = [0.0] * len(point.accelerations)
            #     t = t_n

            if velocity_scaling_factor <= 0.5:
                rospy.logwarn(f"Execution failed. Reseting velocity...")
                velocity_scaling_factor = 1.0
                # self.arm_group.set_start_state_to_current_state()
                (plan, fraction) = self.arm_group.compute_cartesian_path(
                    wps, 0.01  # waypoints to follow  # eef_step
                )
                # p = self.get_cartesian_pose()
                # p.position.z += 0.001
                # (plan, fraction) = self.arm_group.compute_cartesian_path(
                #     [p], 0.01  # waypoints to follow  # eef_step
                # )
                rospy.sleep(1.0)
            else:
                # Reduce the velocity scaling factor
                velocity_scaling_factor -= 0.25
                rospy.logwarn(f"Execution failed. Reducing velocity scaling to {velocity_scaling_factor}.")

                # Modify the plan's trajectory using the scaling factor
                for point in plan.joint_trajectory.points:
                    point.time_from_start = point.time_from_start / velocity_scaling_factor

            # if(max_vel > 0):
            #     plan = scale_trajectory_speed(plan, max_vel)

            rospy.sleep(2.0)
            success = self.arm_group.execute(plan, wait=True)

def scale_trajectory_speed(plan, target_vel):
    """
    Scale the velocity of a MoveIt RobotTrajectory to match the target velocity.
    
    :param plan: The MoveIt! RobotTrajectory
    :param target_vel: Desired end-effector velocity (m/s or rad/s)
    :return: Scaled trajectory
    """
    if not plan.joint_trajectory.points:
        rospy.logwarn("Trajectory plan is empty!")
        return plan

    # Extract current max velocity
    # max_vel = max(max(p.velocities) for p in plan.joint_trajectory.points if p.velocities)  # Get max joint velocity
    # max_vel = math.sqrt(max(p.velocities[0]*p.velocities[0] + p.velocities[1]*p.velocities[1] + p.velocities[2]*p.velocities[2]) for p in plan.joint_trajectory.points if p.velocities)
    max_vel = 0
    for p in plan.joint_trajectory.points:
        if max_vel < p.velocities[0]*p.velocities[0] + p.velocities[1]*p.velocities[1] + p.velocities[2]*p.velocities[2]:
            max_vel = p.velocities[0]*p.velocities[0] + p.velocities[1]*p.velocities[1] + p.velocities[2]*p.velocities[2]

    if max_vel == 0:
        rospy.logwarn("Max velocity is zero, scaling is not possible!")
        return plan

    # Compute velocity scaling factor
    velocity_scaling = target_vel / max_vel
    rospy.loginfo(f"Scaling factor: {velocity_scaling:.3f}")

    # Scale velocity, acceleration, and time_from_start
    for point in plan.joint_trajectory.points:
        point.velocities = [v * velocity_scaling for v in point.velocities]
        point.accelerations = [a * velocity_scaling**2 for a in point.accelerations]
        point.time_from_start = rospy.Duration(point.time_from_start.to_sec() / velocity_scaling)

    return plan

def compute_ee_speed(joint_velocities, jacobian):
    """ Compute end-effector linear speed from joint velocities using the Jacobian """
    ee_velocity = np.dot(jacobian, joint_velocities)  # v = J * q̇
    return ee_velocity[:3]  # Take the norm of linear velocity part

def plot_new_3d(vels_1, vels_2, times_1, times_2):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 8))

    plt.plot(times_1, vels_1, label="Demo")
    plt.plot(times_2, vels_2, label="Demo")

    plt.tight_layout()
    plt.show()