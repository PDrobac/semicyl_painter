#!/usr/bin/env python

import rospy
import csv
import rospy
import numpy as np
from geometry_msgs.msg import Pose, PoseArray
import dmp_node as dmp
import tf.transformations as tft
import matplotlib.pyplot as plt
import pose_conversions as P

def plot_quats(traj, pos):
    plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(241)
    ax1.set_title("Dimension 1")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Position")
    ax2 = plt.subplot(242)
    ax2.set_title("Dimension 2")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Position")
    ax3 = plt.subplot(243)
    ax3.set_title("Dimension 3")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Position")
    ax4 = plt.subplot(244)
    ax4.set_title("Dimension 4")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Position")

    ax5 = plt.subplot(245)
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Velocity")
    ax6 = plt.subplot(246)
    ax6.set_xlabel("Time")
    ax6.set_ylabel("Velocity")
    ax7 = plt.subplot(247)
    ax7.set_xlabel("Time")
    ax7.set_ylabel("Velocity")
    ax8 = plt.subplot(248)
    ax8.set_xlabel("Time")
    ax8.set_ylabel("Velocity")

    Y = np.array(traj)

    dt = 0.01
    T = []
    # quats = np.empty(4)
    from scipy.spatial.transform import Rotation as R
    for i in range(len(Y)-1):
        T.append(dt * i)

    #     r = R.from_euler('xyz', [Y[i, 3], Y[i, 4], Y[i, 5]])
    #     qx, qy, qz, qw = r.as_quat()
    #     quats = np.vstack((quats, -np.array([qx, qy, qz, qw])))

    # quats = quats[1:]

    # List to store pose data
    quats = []

    # CSV file path
    csv_filename = "/home/pero/Jelena/bruno1.csv"  # Change this path as needed

    # Read CSV file
    with open(csv_filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row if it exists
        for row in reader:
            #pose_list.append([float(x) for x in row])  # Convert strings to floats
            # Convert strings to floats
            x, y, z, qx, qy, qz, qw, sriod = map(float, row)
            
            # Convert quaternion to Euler angles (roll, pitch, yaw)
            # roll, pitch, yaw = tft.euler_from_quaternion([qx, qy, qz, qw])
            # qx, qy, qz, qw = tft.quaternion_from_euler(roll, pitch, yaw)
            q = [qx, qy, qz, qw]

            # q_mat = tft.quaternion_matrix(q)
            # Rot = np.array([[0, -1, 0],
            #             [0, 0, 1],
            #             [-1, 0, 0]])
            # new_q_mat = np.eye(4)
            # new_q_mat[:3, :3] = Rot @ q_mat[:3, :3]
            # q = tft.quaternion_from_matrix(new_q_mat)
            
            # Store as [x, y, z, roll, pitch, yaw]
            quats.append(q)
            # pose_list.append([x, y, z, 0.0, 0.0, 0.0])

    quats = np.array(quats)
    ax1.plot(T, quats[:, 0], label="Demo")
    ax2.plot(T, quats[:, 1], label="Demo")
    ax3.plot(T, quats[:, 2], label="Demo")
    ax4.plot(T, quats[:, 3], label="Demo")
    ax5.plot(T, np.gradient(quats[:, 0]) / dt)
    ax6.plot(T, np.gradient(quats[:, 1]) / dt)
    ax7.plot(T, np.gradient(quats[:, 2]) / dt)
    ax8.plot(T, np.gradient(quats[:, 3]) / dt)
    #ax4.scatter([T[-1]], (Y[-1, 0] - Y[-2, 0]) / dmp.dt_)
    #ax5.scatter([T[-1]], (Y[-1, 1] - Y[-2, 1]) / dmp.dt_)
    # ax6.scatter([T[-1]], (Y[-1, 2] - Y[-2, 2]) / dmp.dt_)
    # dmp.configure(goal_y=np.array([1, 0, 1]), goal_yd=np.array([goal_yd, goal_yd, goal_yd]))
    dt = 0.01*0.7
    T = []
    quats = np.empty(4)
    for i in range(len(pos)):
        T.append(dt * i)

        Rot = np.array([[0, -1, 0],
                        [0, 0, 1],
                        [-1, 0, 0]])
        
        Rot = np.eye(3)
        
        rotation_from_euler = R.from_euler('xyz', np.array(pos)[i, 3:6]).as_matrix()  # Convert to rotation matrix
        
        rpy_mat = Rot @ rotation_from_euler
        rpy = R.from_matrix(rpy_mat).as_euler('xyz')

        r = R.from_euler('xyz', rpy)
        qx, qy, qz, qw = r.as_quat()
        quats = np.vstack((quats, -np.array([qx, qy, qz, qw])))

    quats = quats[1:]

    ax1.plot(T, quats[:, 0], label="Dmp")
    ax2.plot(T, quats[:, 1], label="Dmp")
    ax3.plot(T, quats[:, 2], label="Dmp")
    ax4.plot(T, quats[:, 3], label="Dmp")
    ax5.plot(T, np.gradient(quats[:, 0]) / dt)
    ax6.plot(T, np.gradient(quats[:, 1]) / dt)
    ax7.plot(T, np.gradient(quats[:, 2]) / dt)
    ax8.plot(T, np.gradient(quats[:, 3]) / dt)
    #ax4.scatter([T[-1]], [1.0])
    #ax5.scatter([T[-1]], [0.0])
    # ax6.scatter([T[-1]], [goal_yd])

    ax1.legend()
    plt.tight_layout()
    plt.show()

def plot_new_4d(traj, pos):
    plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(241)
    ax1.set_title("Dimension 1")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Position")
    ax2 = plt.subplot(242)
    ax2.set_title("Dimension 2")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Position")
    ax3 = plt.subplot(243)
    ax3.set_title("Dimension 3")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Position")
    ax4 = plt.subplot(244)
    ax4.set_title("Dimension 4")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Position")

    ax5 = plt.subplot(245)
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Velocity")
    ax6 = plt.subplot(246)
    ax6.set_xlabel("Time")
    ax6.set_ylabel("Velocity")
    ax7 = plt.subplot(247)
    ax7.set_xlabel("Time")
    ax7.set_ylabel("Velocity")
    ax8 = plt.subplot(248)
    ax8.set_xlabel("Time")
    ax8.set_ylabel("Velocity")

    Y = np.array(traj)

    dt = 0.01
    T = []
    for i in range(len(Y)):
        T.append(dt * i)

    ax1.plot(T, Y[:, 3], label="Demo")
    ax2.plot(T, Y[:, 4], label="Demo")
    ax3.plot(T, Y[:, 5], label="Demo")
    ax4.plot(T, Y[:, 6], label="Demo")
    ax5.plot(T, np.gradient(Y[:, 3]) / dt)
    ax6.plot(T, np.gradient(Y[:, 4]) / dt)
    ax7.plot(T, np.gradient(Y[:, 5]) / dt)
    ax8.plot(T, np.gradient(Y[:, 6]) / dt)
    #ax4.scatter([T[-1]], (Y[-1, 0] - Y[-2, 0]) / dmp.dt_)
    #ax5.scatter([T[-1]], (Y[-1, 1] - Y[-2, 1]) / dmp.dt_)
    # ax6.scatter([T[-1]], (Y[-1, 2] - Y[-2, 2]) / dmp.dt_)
    # dmp.configure(goal_y=np.array([1, 0, 1]), goal_yd=np.array([goal_yd, goal_yd, goal_yd]))
    dt = 0.01*0.7
    T = []
    for i in range(len(pos)):
        T.append(dt * i)
    ax1.plot(T, np.array(pos)[:, 3])
    ax2.plot(T, np.array(pos)[:, 4])
    ax3.plot(T, np.array(pos)[:, 5])
    ax4.plot(T, np.array(pos)[:, 6])
    ax5.plot(T, np.gradient(np.array(pos)[:, 3]) / dt)
    ax6.plot(T, np.gradient(np.array(pos)[:, 4]) / dt)
    ax7.plot(T, np.gradient(np.array(pos)[:, 5]) / dt)
    ax8.plot(T, np.gradient(np.array(pos)[:, 6]) / dt)
    #ax4.scatter([T[-1]], [1.0])
    #ax5.scatter([T[-1]], [0.0])
    # ax6.scatter([T[-1]], [goal_yd])

    ax1.legend()
    plt.tight_layout()
    plt.show()

def plot_2d(traj):
    # Extract x and y coordinates
    x_traj = [point[0] for point in traj]
    y_traj = [point[1] for point in traj]

    # Create the plot
    plt.plot(x_traj, y_traj, 'b-', label='DMP trajectory')
    # plt.plot(x_wp, y_wp, color='green', label='Welding pattern')
    # plt.plot(milestones[:, 0], milestones[:, 1], 'go', label="Milestones")
    plt.title("Trajectory comparison")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.show()

def rotation_matrix(A, B):
    A, B = np.array(A[:3]), np.array(B[:3])
    
    # Compute scale factor s
    norm_A, norm_B = np.linalg.norm(A), np.linalg.norm(B)
    # s = norm_A / norm_B if norm_B != 0 else 0  # Avoid division by zero
    
    # Normalize A and B
    A_unit, B_unit = A / norm_A, B / norm_B
    
    # Compute rotation matrix R using the outer product and cross product
    v = np.cross(B_unit, A_unit)
    c = np.dot(B_unit, A_unit)
    s = np.linalg.norm(v)
    Vx = np.array([[  0,   -v[2],  v[1]],
                   [ v[2],  0,   -v[0]],
                   [-v[1],  v[0],  0  ]])  # Skew-symmetric cross product matrix

    R = np.eye(3) + Vx + (1 - c) * np.dot(Vx, Vx) / (s ** 2) if c > -1 else -np.eye(3)  # Rodrigues' formula

    # Compute final transformation matrix
    return R

def generate_equivalent_euler_angles(roll, pitch, yaw):
    """
    Generate multiple equivalent Euler angle representations for the same orientation.

    Parameters:
        roll (float): Rotation around X-axis.
        pitch (float): Rotation around Y-axis.
        yaw (float): Rotation around Z-axis.

    Returns:
        list of tuples: Equivalent Euler angle triplets.
    """
    equivalents = set()  # Use a set to avoid duplicates

    # Base angles
    equivalents.add((roll, pitch, yaw))

    # Add periodicity (Euler angles are periodic with 2π)
    for k in [-1, 0, 1]:  # Adjust in multiples of 2π
        for m in [-1, 0, 1]:
            for n in [-1, 0, 1]:
                equivalents.add((roll + 2 * np.pi * k, pitch + 2 * np.pi * m, yaw + 2 * np.pi * n))

    # Gimbal Lock handling: If pitch is ±π/2
    if np.isclose(np.abs(pitch), np.pi / 2):
        equivalents.add((roll + yaw, pitch, 0))
        equivalents.add((-roll - yaw, pitch, 0))
        equivalents.add((0, pitch, roll + yaw))
        equivalents.add((0, pitch, - roll - yaw))

    others = [
        (roll, pitch, yaw),
        (roll + np.pi, np.pi - pitch, yaw + np.pi),
        (roll - np.pi, np.pi - pitch, yaw + np.pi),
        (roll + np.pi, np.pi - pitch, yaw - np.pi),
        (roll - np.pi, np.pi - pitch, yaw - np.pi),
    ]

    return np.vstack((list(equivalents), others))

def closest_euler_angles(new_angles, prev_angles):
    """
    Find the Euler representation closest to the previous one, considering angle wrapping.
    """
    candidates = generate_equivalent_euler_angles(*new_angles)

    # Choose the candidate that minimizes the wrapped difference
    ang = min(
        candidates,
        key=lambda angles: sum(abs(angles[i] - prev_angles[i]) for i in range(3))
    )

    return ang

def correct_pose(pose, prev_angles=[0, 0, 0]):
    """
    Convert a Pose message to a list [x, y, z, roll, pitch, yaw],
    ensuring the Euler angles are the closest representation to prev_angles with smooth transitions.
    """
    x, y, z, roll, pitch, yaw, sriod = pose

    # Find the closest Euler angle representation while handling wrapping
    roll, pitch, yaw = closest_euler_angles([roll, pitch, yaw], prev_angles)

    return [x, y, z, roll, pitch, yaw, sriod]

def correct_poses_array(pose_list):
    """
    Convert an array of Pose messages into an array of 6-element lists.
    """
    p_list = []
    prev = [0, 0, 0]
    for pose in pose_list:
        p_6 = correct_pose(pose, prev)
        p_list.append(p_6)
        prev = p_6[3:]

    return p_list

def generate_pose_array(pose_list):
    """Generate a PoseArray forming a straight line with a looping roll orientation"""
    pose_array = PoseArray()
    pose_array.header.frame_id = "map"  # Change this to match your TF frame
    pose_array.header.stamp = rospy.Time.now()
    printed = False

    for pose in pose_list:
        # Convert roll to quaternion (rotation around X-axis)
        qx, qy, qz, qw = tft.quaternion_from_euler(pose[3], pose[4], pose[5])

        # Define pose
        pose_ = Pose()
        pose_.position.x = pose[0]
        pose_.position.y = pose[1]
        pose_.position.z = pose[2]
        pose_.orientation.x = qx
        pose_.orientation.y = qy
        pose_.orientation.z = qz
        pose_.orientation.w = qw

        # Add pose to PoseArray
        pose_array.poses.append(pose_)

        # if not printed:
        #     print(pose_)
        #     printed = True

    return pose_array

def list_to_pose(pose_list):
    """
    Convert a 6-element list [x, y, z, roll, pitch, yaw] into a Pose message.
    """
    x, y, z, roll, pitch, yaw, sriod = pose_list
    quaternion = tft.quaternion_from_euler(roll, pitch, yaw)
    
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.x = quaternion[0]
    pose.orientation.y = quaternion[1]
    pose.orientation.z = quaternion[2]
    pose.orientation.w = quaternion[3]

    return pose

def convert_lists_to_poses(pose_lists):
    """
    Convert an array of 6-element lists into an array of Pose messages.
    """
    return [list_to_pose(pose_list) for pose_list in pose_lists]

def fetch_pose_list():
    # List to store pose data
    pose_list = []

    # CSV file path
    csv_filename = "/home/pero/Jelena/bruno1.csv"  # Change this path as needed

    # Read CSV file
    with open(csv_filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row if it exists
        for row in reader:
            #pose_list.append([float(x) for x in row])  # Convert strings to floats
            # Convert strings to floats
            x, y, z, qx, qy, qz, qw, sriod = map(float, row)
            
            # Convert quaternion to Euler angles (roll, pitch, yaw)
            roll, pitch, yaw = tft.euler_from_quaternion([qx, qy, qz, qw])
            
            # Store as [x, y, z, roll, pitch, yaw]
            pose_list.append([x, y, z, roll, pitch, yaw, sriod])
            # pose_list.append([x, y, z, 0.0, 0.0, 0.0])

    theta = np.radians(90) # Flip z-axis
    R = np.eye(3)
    R[1, 1] = np.cos(theta)
    R[1, 2] = -np.sin(theta)
    R[2, 1] = np.sin(theta)
    R[2, 2] = np.cos(theta)

    R = np.array([[0, -1, 0],
                  [0, 0, 1],
                  [-1, 0, 0]])
    
    pose_list = P.apply_rotation_to_pose_6d_array(pose_list, R.T)

    return correct_poses_array(pose_list)

def export_vel_list(waypoints):
    csv_filename = "/home/pero/Jelena/bruno1_vels.csv"
    dt = 1.0/100
    T = []
    for i in range(len(waypoints)):
        T.append(dt*i)

    n_dims = 7
    DT = np.gradient(T)
    Yd = np.empty_like(waypoints)
    for d in range(n_dims-1):
        Yd[:, d] = np.gradient(waypoints[:, d]) / DT
        # Yd[:, d] = waypoints[:, d]
    Yd[:, n_dims-1] = waypoints[:, n_dims-1]

    with open(csv_filename, 'a') as f:
        writer = csv.writer(f)
        for yd in Yd:
            writer.writerow(yd)  # Write the latest pose

def main():
    rospy.init_node("pose_array_publisher")
    pub = rospy.Publisher("/pose_array", PoseArray, queue_size=10)
    pub_dmp = rospy.Publisher("/pose_array_dmp", PoseArray, queue_size=10)
    rate = rospy.Rate(1000)  # 1 Hz

    pose_list = fetch_pose_list()
    pose_list.append(pose_list[-1])
    pose_array = generate_pose_array(pose_list)

    theta = np.radians(0)
    R = np.eye(3)
    R[0, 0] = np.cos(theta)
    R[0, 1] = -np.sin(theta)
    R[1, 0] = np.sin(theta)
    R[1, 1] = np.cos(theta)

    # print(R)

    tau = dmp.learn_dmp(pose_list)
    start = np.array(pose_list[0])
    end = np.array(pose_list[-1])

    start_p = [start[0], start[1], start[2], 0.0, 0.0, 0.0, 0.0]

    start_r = start - start_p
    new_start_r = P.apply_rotation_to_pose_6d(start_r, R)
    new_start = new_start_r + start_p

    relative_end = end - start_p
    new_relative_end = P.apply_rotation_to_pose_6d(relative_end, R)
    new_end = new_relative_end + start_p
    # new_end = [new_end[0], new_end[1], new_end[2], 0.0, 0.0, 0.0]
    # R = rotation_matrix(A=relative_end, B=new_relative_end)
    # print(R)

    #print(new_start)
    #print(new_end)
    # print(tau)

    [waypoints, v] = dmp.generate_dmp_7d(tau/0.7, new_start, new_end, R=R.T)
    # print(len(pose_list))
    # print(len(waypoints))
    # plot_new_4d(np.array(pose_list), np.array(waypoints))
    R = np.array([[0, -1, 0],
                  [0, 0, 1],
                  [-1, 0, 0]])
    
    waypoints = P.apply_rotation_to_pose_6d_array(waypoints, R)
    plot_quats(np.array(pose_list), np.array(waypoints))
    # plot_new_4d(np.array(pose_list), np.array(waypoints))
    # print("DMP generated")

    export_vel_list(np.array(waypoints))

    wp_array = convert_lists_to_poses(waypoints)

    # pose_list_R = [p - pose_list[0] for p in np.array(pose_list)]
    # pose_list_R = P.apply_rotation_to_pose_6d_array(pose_list_R, R.T)
    # pose_list_R = [p + pose_list[0] for p in np.array(pose_list_R)]

    while not rospy.is_shutdown():
        pose_array.poses = convert_lists_to_poses(pose_list)
        pub.publish(pose_array)
        pose_array.poses = wp_array
        pub_dmp.publish(pose_array)
        rate.sleep()

if __name__ == "__main__":
    try:
        # publish_pose_array()
        main()
        # dmp_6d()
    except rospy.ROSInterruptException:
        pass