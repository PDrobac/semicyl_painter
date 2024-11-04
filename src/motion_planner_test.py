#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from tf.transformations import quaternion_from_euler

class MotionPlanner(object):
    """MotionPlanner"""

    def __init__(self):
        super(MotionPlanner, self).__init__()

        ## BEGIN_SUB_TUTORIAL setup
        ##
        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("move_group_python_interface_tutorial", anonymous=True)

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

    def go_to_pose_goal(self, pose_goal):
        ## BEGIN_SUB_TUTORIAL plan_to_pose
        ##
        ## Planning to a Pose Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## We can plan a motion for this group to a desired pose for the
        ## end-effector:

        self.move_group.set_pose_target(pose_goal)

        ## Now, we call the planner to compute the plan and execute it.
        # `go()` returns a boolean indicating whether the planning and execution was successful.
        success = self.move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        self.move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets().
        self.move_group.clear_pose_targets()

        ## END_SUB_TUTORIAL

    def plan_cartesian_path(self, waypoints):
        ## BEGIN_SUB_TUTORIAL plan_cartesian_path
        ##
        ## Cartesian Paths
        ## ^^^^^^^^^^^^^^^
        ## You can plan a Cartesian path directly by specifying a list of waypoints
        ## for the end-effector to go through. If executing  interactively in a
        ## Python shell, set scale = 1.0.

        # We want the Cartesian path to be interpolated at a resolution of 1 cm
        # which is why we will specify 0.01 as the eef_step in Cartesian
        # translation.  We will disable the jump threshold by setting it to 0.0,
        # ignoring the check for infeasible jumps in joint space, which is sufficient
        # for this tutorial.
        (plan, fraction) = self.move_group.compute_cartesian_path(
            waypoints, 0.01  # waypoints to follow  # eef_step
        )

        # Note: We are just planning, not asking move_group to actually move the robot yet:
        return plan, fraction

        ## END_SUB_TUTORIAL

    def execute_plan(self, plan):
        ## BEGIN_SUB_TUTORIAL execute_plan
        ##
        ## Executing a Plan
        ## ^^^^^^^^^^^^^^^^
        ## Use execute if you would like the robot to follow
        ## the plan that has already been computed:
        self.move_group.execute(plan, wait=True)

        ## **Note:** The robot's current joint state must be within some tolerance of the
        ## first waypoint in the `RobotTrajectory`_ or ``execute()`` will fail
        ## END_SUB_TUTORIAL

if __name__ == "__main__":
    planner = MotionPlanner()

    input("============ Press `Enter` to execute a movement using a pose goal ...")
    pose_goal = geometry_msgs.msg.Pose()
    #pose_goal.orientation.w = 1.0
    pose_goal.position.x = 0.5
    pose_goal.position.y = 0
    pose_goal.position.z = 0
    initial_orientation_q = quaternion_from_euler(pi, 0, pi/4)
    pose_goal.orientation.x = initial_orientation_q[0]
    pose_goal.orientation.y = initial_orientation_q[1]
    pose_goal.orientation.z = initial_orientation_q[2]
    pose_goal.orientation.w = initial_orientation_q[3]
    planner.go_to_pose_goal(pose_goal)

    # input("============ Press `Enter` to plan and display a Cartesian path ...")
    # waypoints = []

    # wpose = planner.move_group.get_current_pose().pose
    # wpose.position.z -= 0.1  # First move up (z)
    # wpose.position.y += 0.2  # and sideways (y)
    # initial_orientation_q = quaternion_from_euler(0, 0, pi/4)
    # wpose.orientation.x = initial_orientation_q[0]
    # wpose.orientation.y = initial_orientation_q[1]
    # wpose.orientation.z = initial_orientation_q[2]
    # wpose.orientation.w = initial_orientation_q[3]
    # waypoints.append(copy.deepcopy(wpose))

    # wpose.position.x += 0.1  # Second move forward/backwards in (x)
    # waypoints.append(copy.deepcopy(wpose))

    # wpose.position.y -= 0.1  # Third move sideways (y)
    # waypoints.append(copy.deepcopy(wpose))
    # cartesian_plan, fraction = planner.plan_cartesian_path(waypoints)

    # input("============ Press `Enter` to execute a saved path ...")
    # planner.execute_plan(cartesian_plan)

    print("============ Python tutorial demo complete!")