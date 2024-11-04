#!/usr/bin/env python

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import Pose

class MotionPlanner(object):
    """MotionPlanner"""
    def __init__(self):
        super(MotionPlanner, self).__init__()

        ## BEGIN_SUB_TUTORIAL setup
        ##
        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        #rospy.init_node("move_group_python_interface_tutorial", anonymous=True)

        ## Instantiate a `RobotCommander`_ object. Provides information such as the robot's
        ## kinematic model and the robot's current joint states
        robot = moveit_commander.RobotCommander()

        ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        ## for getting, setting, and updating the robot's internal understanding of the
        ## surrounding world:
        scene = moveit_commander.PlanningSceneInterface()

        ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        ## to a planning group (group of joints).  In this tutorial the group is the primary
        ## arm joints in the Panda robot, so we set the group's name to "panda_arm".
        ## If you are using a different robot, change this value to the name of your robot
        ## arm planning group.
        ## This interface can be used to plan and execute motions:
        group_name = "panda_arm"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
        ## trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )

        ## END_SUB_TUTORIAL

        ## BEGIN_SUB_TUTORIAL basic_info
        ##
        ## Getting Basic Information
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^
        # We can get the name of the reference frame for this robot:
        planning_frame = move_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        eef_link = move_group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        print("============ Available Planning Groups:", robot.get_group_names())

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        print(robot.get_current_state())
        print("")
        ## END_SUB_TUTORIAL

        # Misc variables
        self.box_name = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names

        # Goal poses
        self.start_poses = []
        self.end_poses = []
        self.start_hover_poses = []
        self.end_hover_poses = []

    def go_to_pose_goal_cartesian(self, waypoints):
        ## BEGIN_SUB_TUTORIAL plan_cartesian_path
        ##
        ## Cartesian Paths
        ## ^^^^^^^^^^^^^^^
        ## You can plan and execute a Cartesian path directly by specifying a list of waypoints
        ## for the end-effector to go through.

        # We want the Cartesian path to be interpolated at a resolution of 1 cm
        # which is why we will specify 0.01 as the eef_step in Cartesian
        # translation.  We will disable the jump threshold by setting it to 0.0,
        # ignoring the check for infeasible jumps in joint space, which is sufficient
        # for this tutorial.
        (plan, fraction) = self.move_group.compute_cartesian_path(
            waypoints, 0.01  # waypoints to follow  # eef_step
        )

        self.move_group.execute(plan, wait=True)

        ## END_SUB_TUTORIAL

    def go_to_pose_goal_dmp(self, goal_point):
        # TODO: DMP

        # TEMP solution
        waypoints = []
        waypoints.append(copy.deepcopy(goal_point))
        self.go_to_pose_goal_cartesian(waypoints)

def start_pose_callback(start_pose, planner):
        planner.start_poses.append(start_pose)

def end_pose_callback(end_pose, planner):
        planner.end_poses.append(end_pose)

def start_hover_pose_callback(start_hover_pose, planner):
        planner.start_hover_poses.append(start_hover_pose)

def end_hover_pose_callback(end_hover_pose, planner):
        planner.end_hover_poses.append(end_hover_pose)

def ready_callback (defalut_pose, planner):
        input("============ Press `Enter` to initiate the motion planner")

        # Go to first start point
        waypoints = []
        waypoints.append(copy.deepcopy(planner.start_hover_poses[0]))
        waypoints.append(copy.deepcopy(planner.start_poses[0]))
        planner.go_to_pose_goal_cartesian(waypoints)
        
        for i in range(len(planner.start_poses) - 1):
            # Execute the tool stroke
            planner.go_to_pose_goal_dmp(planner.end_poses[i])

            # Go to next start point
            waypoints = []
            waypoints.append(copy.deepcopy(planner.end_hover_poses[i]))
            waypoints.append(copy.deepcopy(planner.start_hover_poses[i+1]))
            waypoints.append(copy.deepcopy(planner.start_poses[i+1]))
            planner.go_to_pose_goal_cartesian(waypoints)

        # Go to default pose
        waypoints = []
        waypoints.append(copy.deepcopy(defalut_pose))

def main():
    # Initialize the ROS node
    rospy.init_node('motion_planner_node', anonymous=True)

    planner = MotionPlanner()

    # Initialize subscribers
    start_sub = rospy.Subscriber('start_poses', Pose, start_pose_callback, planner)
    end_sub = rospy.Subscriber('end_poses', Pose, end_pose_callback, planner)
    start_hover_sub = rospy.Subscriber('start_hover_poses', Pose, start_hover_pose_callback, planner)
    end_hover_sub = rospy.Subscriber('end_hover_poses', Pose, end_hover_pose_callback, planner)
    default_sub = rospy.Subscriber('default_pose', Pose, ready_callback, planner)

    # Keep the node running
    rospy.spin()

if __name__ == "__main__":
    main()