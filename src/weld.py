#!/usr/bin/env python3

import csv
import rospkg
import time
import math
import rospy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tf.transformations as tft
import numpy as np
from geometry_msgs.msg import Pose
from scipy.interpolate import CubicSpline
from sensor_msgs.msg import PointCloud2
import pose_conversions as P
import robot_controller_kinova as rc
from scipy.signal import savgol_filter

import dmp_node as dmp
# from movement_primitives.dmp import DMPWithFinalVelocity

def plot_new_3d(traj, pos):
    dt = 1
    execution_time = len(traj) * dt
    T = np.arange(0, execution_time, dt)

    plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(231)
    ax1.set_title("Dimension 1")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Position")
    ax2 = plt.subplot(232)
    ax2.set_title("Dimension 2")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Position")
    ax3 = plt.subplot(233)
    ax3.set_title("Dimension 3")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Position")

    ax4 = plt.subplot(234)
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Velocity")
    ax5 = plt.subplot(235)
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Velocity")
    ax6 = plt.subplot(236)
    ax6.set_xlabel("Time")
    ax6.set_ylabel("Velocity")

    Y = np.array(traj)

    ax1.plot(T, Y[:, 0], label="Demo")
    ax2.plot(T, Y[:, 1], label="Demo")
    ax3.plot(T, Y[:, 2], label="Demo")
    ax4.plot(T, np.gradient(Y[:, 0]) / dt)
    ax5.plot(T, np.gradient(Y[:, 1]) / dt)
    ax6.plot(T, np.gradient(Y[:, 2]) / dt)
    #ax4.scatter([T[-1]], (Y[-1, 0] - Y[-2, 0]) / dmp.dt_)
    #ax5.scatter([T[-1]], (Y[-1, 1] - Y[-2, 1]) / dmp.dt_)
    # ax6.scatter([T[-1]], (Y[-1, 2] - Y[-2, 2]) / dmp.dt_)
    # dmp.configure(goal_y=np.array([1, 0, 1]), goal_yd=np.array([goal_yd, goal_yd, goal_yd]))
    dt *= len(traj) / len(pos)
    execution_time = len(pos) * dt
    T = np.arange(0, execution_time, dt)
    ax1.plot(T, np.array(pos)[:, 0])
    ax2.plot(T, np.array(pos)[:, 1])
    ax3.plot(T, np.array(pos)[:, 2])
    ax4.plot(T, np.gradient(np.array(pos)[:, 0]) / dt)
    ax5.plot(T, np.gradient(np.array(pos)[:, 1]) / dt)
    ax6.plot(T, np.gradient(np.array(pos)[:, 2]) / dt)
    #ax4.scatter([T[-1]], [1.0])
    #ax5.scatter([T[-1]], [0.0])
    # ax6.scatter([T[-1]], [goal_yd])

    ax1.legend()
    plt.tight_layout()
    plt.show()

def compute_pose(A, B):
    """
    Computes a PoseStamped message where:
    - The position is taken from point A.
    - The x-axis of the orientation points towards point B.
    - The z-axis is pointed downward.
    
    :param A: First point (x, y, z)
    :param B: Second point (x, y, z)
    :return: PoseStamped message
    """
    A = np.array(A)
    B = np.array(B)

    # Compute direction vector (B - A)
    direction = B - A
    direction /= np.linalg.norm(direction)  # Normalize to unit vector

    # Define the coordinate system where:
    x_axis = direction  # Our computed direction
    z_axis = np.array([0, 0, -1])  # Z-down assumption
    y_axis = np.cross(z_axis, x_axis)  # Compute Y-axis as perpendicular to X and Z
    y_axis /= np.linalg.norm(y_axis)  # Normalize
    z_axis = np.cross(x_axis, y_axis)  # Recompute Z-axis for orthogonality

    # Construct a 3x3 rotation matrix
    rotation_matrix = np.eye(4)  # 4x4 for homogeneous transformation
    rotation_matrix[:3, 0] = x_axis
    rotation_matrix[:3, 1] = y_axis
    rotation_matrix[:3, 2] = z_axis

    # Convert rotation matrix to quaternion
    roll, pitch, yaw = tft.euler_from_matrix(rotation_matrix)

    pose = np.array([A[0], A[1], A[2], roll, pitch, yaw])

    return pose

def interpolate_trajectory(poses, num_points=100):
    """
    Interpolates a trajectory to ensure a more constant velocity.

    :param poses: List of (x, y) or (x, y, theta) tuples
    :param num_points: Number of points in the resampled trajectory
    :return: Interpolated trajectory as a NumPy array
    """
    poses = np.array(poses)
    distances = np.cumsum(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1))
    distances = np.insert(distances, 0, 0)  # Insert 0 at the beginning

    # Create splines for x, y (and theta if available)
    spline_x = CubicSpline(distances, poses[:, 0])
    spline_y = CubicSpline(distances, poses[:, 1])
    
    if poses.shape[1] > 2:  # If theta is present
        spline_theta = CubicSpline(distances, poses[:, 2])
    
    # Resample at uniform intervals
    new_distances = np.linspace(0, distances[-1], num_points)
    new_x = spline_x(new_distances)
    new_y = spline_y(new_distances)
    
    if poses.shape[1] > 2:
        new_theta = spline_theta(new_distances)
        return np.column_stack((new_x, new_y, new_theta))
    
    return np.column_stack((new_x, new_y))

def plot_2d(traj, dmp, milestones=[]):
    # Extract x and y coordinates
    x_traj = [point[0] for point in traj]
    y_traj = [point[1] for point in traj]
    x_dmp = [point[0] for point in dmp]
    y_dmp = [point[1] for point in dmp]

    # Create the plot
    plt.plot(x_traj, y_traj, 'b-', label='Original trajectory')
    plt.plot(x_dmp, y_dmp, 'r-', label='DMP trajectory')
    # plt.plot(x_wp, y_wp, color='green', label='Welding pattern')
    plt.plot(milestones[:, 0], milestones[:, 1], 'go', label="Milestones")
    plt.title("Trajectory comparison")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.show()

def plot_vels_2d(traj, pos):
    dt = 0.1
    execution_time = len(traj) * dt
    T = np.arange(0, execution_time + dt, dt)
    size_1 = len(traj)
    size_2 = T.shape[0]
    diff = size_2 - size_1
    if(diff > 0):
        T = T[:-diff]

    plt.figure(2, figsize=(10, 6))
    ax1 = plt.subplot(231)
    ax1.set_title("Dimension 1")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Position [m]")
    ax2 = plt.subplot(232)
    ax2.set_title("Dimension 2")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Position [m]")
    ax3 = plt.subplot(233)
    ax3.set_title("Dimension Sum")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Position [m]")

    ax4 = plt.subplot(234)
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Velocity [m/s]")
    ax5 = plt.subplot(235)
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Velocity [m/s]")
    ax6 = plt.subplot(236)
    ax6.set_xlabel("Time [s]")
    ax6.set_ylabel("Velocity [m/s]")

    Y = np.array(traj)

    Y_ = [math.sqrt(x*x + y*y) for x, y in zip(Y[:, 0], Y[:, 1])]
    pos_ = [math.sqrt(x*x + y*y) for x, y in zip(np.array(pos)[:, 0], np.array(pos)[:, 1])]

    ax1.plot(T, Y[:, 0], label="Demo")
    ax2.plot(T, Y[:, 1], label="Demo")
    ax3.plot(T, Y_, label="Demo")
    ax4.plot(T, np.gradient(Y[:, 0]) / dt)
    ax5.plot(T, np.gradient(Y[:, 1]) / dt)
    ax6.plot(T, np.gradient(Y_) / dt)
    #ax4.scatter([T[-1]], (Y[-1, 0] - Y[-2, 0]) / dmp.dt_)
    #ax5.scatter([T[-1]], (Y[-1, 1] - Y[-2, 1]) / dmp.dt_)
    # ax6.scatter([T[-1]], (Y[-1, 2] - Y[-2, 2]) / dmp.dt_)
    # dmp.configure(goal_y=np.array([1, 0, 1]), goal_yd=np.array([goal_yd, goal_yd, goal_yd]))
    dt = 0.1 * len(traj) / len(pos)
    execution_time = len(pos) * dt
    T = np.arange(0, execution_time, dt)
    size_1 = np.array(pos)[:, 0].shape[0]
    size_2 = T.shape[0]
    if(size_2 > size_1):
        T = T[:-1]

    ax1.plot(T, np.array(pos)[:, 0], label="DMP")
    ax2.plot(T, np.array(pos)[:, 1], label="DMP")
    ax3.plot(T, pos_, label="DMP")
    ax4.plot(T, np.gradient(np.array(pos)[:, 0]) / dt)
    ax5.plot(T, np.gradient(np.array(pos)[:, 1]) / dt)
    ax6.plot(T, np.gradient(pos_) / dt)
    #ax4.scatter([T[-1]], [1.0])
    #ax5.scatter([T[-1]], [0.0])
    # ax6.scatter([T[-1]], [goal_yd])

    ax1.legend()
    plt.tight_layout()

def funny_loop():
    radius = np.pi/2
    theta = np.linspace(-np.pi, 2 * np.pi, 500)  # Angle from 0 to pi for a semicircle
    x_semi = radius * np.cos(theta) + radius + 0.5 * (theta + np.pi) # x = r * cos(theta)
    y_semi = radius * np.sin(theta) # y = r * sin(theta)
    traj = [[x, y, 0.0] for x, y in zip(x_semi, y_semi)]
    # traj.insert(0, traj[0])
    # traj.append(traj[-1])

    traj1 = traj[:len(traj) // 2]
    traj2 = traj[len(traj) // 2:]

    # for _ in range(50):
    #     point = [traj1[-1][0] - 0.02, traj1[-1][1], 0.0]
    #     traj1.append(point)  # Insert the middle value

    for tr in traj2:
        point = [tr[0] - 1, tr[1], 0.0]
        traj1.append(point)  # Insert the middle value

    return traj1

def serious_loop():
    radius = np.pi/2
    theta = np.linspace(-np.pi, np.pi, 500)  # Angle from 0 to pi for a semicircle
    x_semi = radius * np.cos(theta) + radius + 0.5 * (theta + np.pi) # x = r * cos(theta)
    y_semi = radius * np.sin(theta) # y = r * sin(theta)
    traj = [[x, y, 0.0] for x, y in zip(x_semi, y_semi)]

    return traj

def crescent():
    return np.array([[-np.cos(2*np.pi*i/100)+i/100, np.sin(2*np.pi*i/200), 0.0] for i in range(200)])

def eight():
    n = 200
    return np.array([[np.sin(2*np.pi*i/(n/2))+i/(n/2), -np.sin(2*np.pi*i/n), 0.0] for i in range(n)])

def eight_pose():
    n = 200
    p_list = []
    current = [0.0, 0.0, 0.0]
    for i in range(1, n+1):
        next = [np.sin(2*np.pi*i/(n/2))+i/(n/2), -np.sin(2*np.pi*i/n), 0.0]
        p = compute_pose(current, next)
        current = next
        p_list.append(p)
    return np.array(p_list)

def tangent_pose(p_list):
    pose_list = []
    current = p_list[0]
    for i in range(1, len(p_list)):
        next = p_list[i]
        pose_list.append(compute_pose(current, next))
        current = next
    final_pose = compute_pose(p_list[-2], p_list[-1])
    final_pose[:3] = p_list[-1][:3]
    pose_list.append(final_pose)
    return pose_list

def resample_curve(points, d, g):
    points = np.array(points)
    points3 = points[:, :3]
    distances = np.sqrt(np.sum(np.diff(points3, axis=0) ** 2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)  # Cumulative distance along the curve
    
    new_points = [points[0]]  # Start with the first point
    current_distance = g

    while current_distance < cumulative_distances[-1]:  # Stay within the original curve length
        new_x = np.interp(current_distance, cumulative_distances, points[:, 0])
        new_y = np.interp(current_distance, cumulative_distances, points[:, 1])
        new_z = 0.0
        new_roll = np.interp(current_distance, cumulative_distances, points[:, 3])
        new_pitch = np.interp(current_distance, cumulative_distances, points[:, 4])
        new_yaw = np.interp(current_distance, cumulative_distances, points[:, 5])
        new_points.append([new_x, new_y, new_z, new_roll, new_pitch, new_yaw])
        current_distance += d

    current_distance += g - d
    new_x = np.interp(current_distance, cumulative_distances, points[:, 0])
    new_y = np.interp(current_distance, cumulative_distances, points[:, 1])
    new_z = 0.0
    new_roll = np.interp(current_distance, cumulative_distances, points[:, 3])
    new_pitch = np.interp(current_distance, cumulative_distances, points[:, 4])
    new_yaw = np.interp(current_distance, cumulative_distances, points[:, 5])
    new_points.append([new_x, new_y, new_z, new_roll, new_pitch, new_yaw])

    return np.array(new_points)

def find_tangents(points):
    phi_list = []
    d_x = points[1][0] - points[0][0]
    d_y = points[1][1] - points[0][1]
    phi_list.append(math.atan2(d_y, d_x))
    for i in range(1, len(points)-1):
        d_x = points[i+1][0] - points[i-1][0]
        d_y = points[i+1][1] - points[i-1][1]
        phi_list.append(math.atan2(d_y, d_x))
    d_x = points[-1][0] - points[-2][0]
    d_y = points[-1][1] - points[-2][1]
    phi_list.append(math.atan2(d_y, d_x))

    return phi_list

def warp_curve_arc(points, phi_start, phi_end):
    """
    Warps the given curve into an arc that smoothly transitions from phi_start to phi_end,
    while handling arbitrary start-end orientations.
    """
    p0, p1 = points[0], points[-1]  # Start and end points
    
    # Compute unit vector along chord direction
    chord_vec = p1 - p0
    chord_length = np.linalg.norm(chord_vec)
    chord_dir = chord_vec / chord_length  # Normalize
    chord_ang = math.atan2(chord_vec[1], chord_vec[0])
    
    # Compute a perpendicular vector to chord
    perp_dir = np.array([-chord_dir[1], chord_dir[0], chord_dir[2]])  # 90-degree rotation

    # Compute radius using intersection formula
    theta = (phi_end - phi_start) / 2  # Half the angle difference
    if theta == 0: 
        return points
    radius = chord_length / (2 * np.sin(theta)) if np.sin(theta) != 0 else np.inf
    
    # Compute center of the circular arc
    midpoint = (p0 + p1) / 2
    center = midpoint + theta * perp_dir * np.sqrt(radius**2 - (chord_length / 2) ** 2) / abs(theta)

    # Compute angles for interpolation in the local frame
    start_angle = np.arctan2(p0[1] - center[1], p0[0] - center[0])
    end_angle = np.arctan2(p1[1] - center[1], p1[0] - center[0])

    if start_angle - end_angle > np.pi:
        end_angle -= 2 * np.pi * end_angle / abs(end_angle)

    # Generate new warped curve along the arc
    # angles = np.linspace(start_angle, end_angle, len(points)) # Ovo je krivo, nadi angles da pripadaju x vrijednostima
    warped_points = np.empty((0, 3))
    for i, point in enumerate(points):
        len_progress = ((point[0] - points[0][0]) * np.cos(chord_ang) + (point[1] - points[0][1]) * np.sin(chord_ang)) / chord_length
        radius_offset = - (point[0] - points[0][0]) * np.sin(chord_ang) + (point[1] - points[0][1]) * np.cos(chord_ang)
        angle = start_angle + len_progress * (end_angle - start_angle)
        # print(radius_offset)
        wp = [center[0] + theta * (radius - radius_offset) * np.cos(angle) / abs(theta),
              center[1] + theta * (radius - radius_offset) * np.sin(angle) / abs(theta),
              0.0]
        # print(warped_points)
        # print(wp)
        warped_points = np.vstack((warped_points, wp))

    return warped_points

def pose_callback(poses):
    # File to save poses
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('semicyl_painter')
    csv_filename = f"{package_path}/data/poses.csv"
    """Callback function to save pose data to CSV"""
    with open(csv_filename, "a", newline="") as file:
        writer = csv.writer(file)
        for pose in poses:
            writer.writerow([
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
            ])
    rospy.loginfo("Poses saved to CSV")

def main():
    t0 = time.time()
    # demo_path_x = np.linspace(0.3, 0.7, 20)
    # demo_path = np.empty((0, 3))
    # for x in demo_path_x:
    #     demo_path = np.vstack((demo_path, [x, x*x*x, 0.0]))

    demo_path = np.array(funny_loop()) * 0.1

    demo_pattern = eight_pose() * 0.05

    seg_len = math.sqrt((demo_pattern[-1][0] - demo_pattern[0][0])**2 + (demo_pattern[-1][1] - demo_pattern[0][1])**2)
    path_len = 0.0
    for i in range(1, len(demo_path)):
        path_len += math.sqrt((demo_path[i][0] - demo_path[i-1][0])**2 + (demo_path[i][1] - demo_path[i-1][1])**2)

    print(path_len)
    print(seg_len)

    num_segments = math.trunc((path_len - seg_len) / seg_len)
    end_seg_len = (path_len - num_segments * seg_len) / 2  # Leftover time for start and end segment

    demo_poses = tangent_pose(demo_path)
    milestones = resample_curve(demo_poses, seg_len, end_seg_len)
    # phi_list = find_tangents(milestones)

    planned_path = np.empty((0, 6))  # Ensure planned_pattern is a 2D array
    v = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # v = [1, -1, 0.0]
    print(v)
    tau = dmp.learn_dmp(traj=demo_pattern)
    orig = demo_pattern[-1] - demo_pattern[0]

    for i in range(len(milestones) - 1):
        curr = milestones[i]
        next = milestones[i + 1]
        d = next - curr
        # theta = math.atan2(d[1], d[0])
        R = P.rotation_matrix_from_positions(orig, d)
        # next = curr + orig
        # R = np.eye(3)

        # goal_dot = [0.0, 0.0, 0.0]
        goal_dot = demo_pattern[-1] - demo_pattern[-2]
        dx = demo_pattern[-1] - demo_pattern[0]
        dx_new = next - curr
        mult =  np.linalg.norm(dx_new)/np.linalg.norm(dx)
        # print(mult)
        goal_dot *= mult
        # v = goal_dot

        [pattern_increment, v] = dmp.generate_dmp(tau=tau, x_0=curr, x_goal=next, x_dot_0=v, goal_dot=goal_dot, R=R.T)

        # waypoints = warp_curve_arc(pattern_increment, phi_list[i], phi_list[i + 1])
        waypoints = pattern_increment

        planned_path = np.vstack((planned_path, waypoints))

        plot_2d(demo_path, planned_path, milestones)
        #plot_new_3d(demo_path, waypoints)

    # planned_path = np.vstack((planned_path, demo_path[-1]))

    # planned_path = interpolate_trajectory(planned_path, 500)

    plot_new_3d(demo_path, planned_path)
    # plot_2d(demo_path, planned_path)

    t1 = time.time()
    print("Execution time: " + str(t1 - t0) + "s")

    rospy.init_node('weld_node', anonymous=True)
    trace_publisher = rospy.Publisher('/tip_trace', PointCloud2, queue_size=10)
    pointcloud = P.create_pointcloud2(planned_path, "base_link")
    trace_publisher.publish(pointcloud)

if __name__ == "__main__":
    main()