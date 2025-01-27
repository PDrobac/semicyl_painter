#!/usr/bin/env python3

import struct
import rospy
import tf2_ros
import math
import numpy as np
import tf.transformations as tft
from geometry_msgs.msg import Pose, Transform
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory

def pose_to_matrix(pose: Pose):
    """
    Convert a Pose message to a 4x4 transformation matrix.

    Args:
        pose (Pose): The Pose message to convert.

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    # Create the translation matrix
    translation = tft.translation_matrix([pose.position.x, pose.position.y, pose.position.z])

    # Create the rotation matrix from quaternion
    quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    rotation = tft.quaternion_matrix(quaternion)

    # Combine translation and rotation into a single transformation matrix
    return np.dot(translation, rotation)

def matrix_to_pose(matrix: np.ndarray):
    """
    Convert a 4x4 transformation matrix to a Pose message.

    Args:
        matrix (np.ndarray): The 4x4 transformation matrix.

    Returns:
        Pose: The converted Pose message.
    """
    pose = Pose()

    # Extract translation
    translation = tft.translation_from_matrix(matrix)
    pose.position.x = translation[0]
    pose.position.y = translation[1]
    pose.position.z = translation[2]

    # Extract rotation as quaternion
    quaternion = tft.quaternion_from_matrix(matrix)
    pose.orientation.x = quaternion[0]
    pose.orientation.y = quaternion[1]
    pose.orientation.z = quaternion[2]
    pose.orientation.w = quaternion[3]

    return pose

def apply_global_tf_to_pose(pose: Pose, tf_matrix: np.ndarray):
    """
    Apply a transformation matrix to a Pose.

    Args:
        pose (Pose): The Pose to transform.
        tf_matrix (np.ndarray): A 4x4 transformation matrix to apply.

    Returns:
        Pose: The transformed Pose.
    """
    # Convert the Pose to a transformation matrix
    pose_matrix = pose_to_matrix(pose)

    # Apply the transformation by matrix multiplication
    transformed_matrix = np.dot(tf_matrix, pose_matrix)

    # Convert the resulting matrix back to a Pose
    return matrix_to_pose(transformed_matrix)

def apply_local_tf_to_pose(pose: Pose, tf_matrix: np.ndarray):
    """
    Apply a transformation matrix to a Pose.

    Args:
        pose (Pose): The Pose to transform.
        tf_matrix (np.ndarray): A 4x4 transformation matrix to apply.

    Returns:
        Pose: The transformed Pose.
    """
    # Convert the Pose to a transformation matrix
    pose_matrix = pose_to_matrix(pose)

    # Apply the transformation by matrix multiplication
    transformed_matrix = np.dot(pose_matrix, tf_matrix)

    # Convert the resulting matrix back to a Pose
    return matrix_to_pose(transformed_matrix)

def translate_pose_local(pose: Pose, dx: float, dy: float, dz: float):
    """
    Translates a pose by (dx, dy, dz) in its own local frame.

    Args:
        pose (Pose): The input pose to translate.
        dx, dy, dz (float): Translation along the local x, y, and z axes.

    Returns:
        Pose: The translated pose.
    """
    # Extract position and orientation
    position = pose.position
    orientation = pose.orientation

    # Convert quaternion to rotation matrix
    quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
    rotation_matrix = tft.quaternion_matrix(quaternion)

    # Create a local translation vector
    local_translation = [dx, dy, dz, 1.0]  # Homogeneous coordinates

    # Transform the local translation to the global frame
    global_translation = rotation_matrix.dot(local_translation)

    # Update the pose's position
    translated_pose = Pose()
    translated_pose.orientation = pose.orientation  # Orientation remains the same
    translated_pose.position.x = position.x + global_translation[0]
    translated_pose.position.y = position.y + global_translation[1]
    translated_pose.position.z = position.z + global_translation[2]

    return translated_pose

def rotate_pose_about_axis(pose: Pose, angle_deg: float, axis: str):
    # Convert angle from degrees to radians
    angle_rad = angle_deg * (3.141592653589793 / 180.0)
    
    # Current orientation quaternion of the pose
    current_q = [
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ]

    # Create rotation quaternion for the desired axis
    if axis == 'x':
        rotation_q = tft.quaternion_about_axis(angle_rad, (1, 0, 0))
    elif axis == 'y':
        rotation_q = tft.quaternion_about_axis(angle_rad, (0, 1, 0))
    elif axis == 'z':
        rotation_q = tft.quaternion_about_axis(angle_rad, (0, 0, 1))
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    # Multiply the rotation quaternion with the current quaternion
    new_q = tft.quaternion_multiply(current_q, rotation_q)

    # Update the pose's orientation
    pose.orientation.x = new_q[0]
    pose.orientation.y = new_q[1]
    pose.orientation.z = new_q[2]
    pose.orientation.w = new_q[3]

    return pose

def get_euler_from_pose(pose: Pose):
    q = (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w
    )
    return tft.euler_from_quaternion(q)

def reset_pose_orientation(pose: Pose):
    new_pose = Pose()
    new_pose.position = pose.position

    return new_pose

def offset_transform(orig_pose: Pose, default_pose: Pose, distance: float):
    """
    Interpolates between orig_pose and default_pose at a distance d away from orig_pose
    while retaining the orientation of orig_pose.
    
    Args:
        orig_pose (Pose): The input pose to interpolate from.
        default_pose (Pose): The default pose to interpolate to.
        distance(float): Distance from the original pose to interpolated pose.

    Returns:
        Transform: Interpolated transform.
    """
    # Extract positions
    x1, y1, z1 = orig_pose.position.x, orig_pose.position.y, orig_pose.position.z
    x2, y2, z2 = default_pose.position.x, default_pose.position.y, default_pose.position.z
    
    # Compute the direction vector from orig_pose to default_pose
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    total_distance = math.sqrt(dx**2 + dy**2 + dz**2)
    
    # Handle the case where orig_pose and default_pose are the same
    if total_distance == 0:
        rospy.logwarn("orig_pose and default_pose are identical. Returning identity transform.")
        transform = Transform()
        transform.translation.x = 0.0
        transform.translation.y = 0.0
        transform.translation.z = 0.0
        transform.rotation = orig_pose.orientation
        return transform
    
    # Normalize the direction vector
    unit_dx = dx / total_distance
    unit_dy = dy / total_distance
    unit_dz = dz / total_distance
    
    # Calculate the translation for the transform
    transform = Transform()
    transform.translation.x = unit_dx * distance
    transform.translation.y = unit_dy * distance
    transform.translation.z = unit_dz * distance
    
    # Retain the orientation of orig_pose
    transform.rotation = orig_pose.orientation
    
    return transform

def apply_transform_to_pose_position(pose: Pose, transform: Transform):
    """
    Apply a Transform to a Pose while keeping the original orientation.
    
    Args:
        pose (Pose): Pose to be transformed.
        transform (Transform): Transform to be applied.

    Returns:
        transformed_pose (Pose): Transformed pose.
    """
    # Extract position from the pose
    px, py, pz = pose.position.x, pose.position.y, pose.position.z
    
    # Extract translation from the transform
    tx, ty, tz = transform.translation.x, transform.translation.y, transform.translation.z
    
    # Apply translation to the position
    transformed_position_x = px + tx
    transformed_position_y = py + ty
    transformed_position_z = pz + tz
    
    # Construct the transformed pose
    transformed_pose = Pose()
    transformed_pose.position.x = transformed_position_x
    transformed_pose.position.y = transformed_position_y
    transformed_pose.position.z = transformed_position_z
    transformed_pose.orientation = pose.orientation
    
    return transformed_pose


def invert_tf(matrix: np.ndarray):
    """
    Inverts a 4x4 transformation matrix.

    Args:
        matrix (numpy.ndarray): A 4x4 transformation matrix.

    Returns:
        numpy.ndarray: The inverted 4x4 transformation matrix.
    """
    # Ensure the matrix is 4x4
    if matrix.shape != (4, 4):
        raise ValueError("Input must be a 4x4 transformation matrix.")

    # Separate the rotation and translation components
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]

    # Invert the rotation (transpose for orthogonal matrices)
    rotation_inv = rotation.T

    # Invert the translation
    translation_inv = -np.dot(rotation_inv, translation)

    # Construct the inverted matrix
    inverted_matrix = np.eye(4)
    inverted_matrix[:3, :3] = rotation_inv
    inverted_matrix[:3, 3] = translation_inv

    return inverted_matrix

def create_pointcloud2(points: list, frame_id="world"):
    """
    Create a PointCloud2 message from a list of 3D points.
    
    Args:
        points (list): List of tuples containing (x, y, z) coordinates.
        frame_id (str): The reference frame of the point cloud.

    Returns:
        PointCloud2: The generated PointCloud2 message.
    """
    header = Header()
    header.stamp = rospy.Time.now()  # ROS1 timestamp
    header.frame_id = frame_id

    # Define the fields of the PointCloud2 message
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    point_step = 12  # 4 bytes each for x, y, z
    row_step = point_step * len(points)

    # Pack the points into binary format
    data = b"".join([struct.pack("fff", *point) for point in points])

    return PointCloud2(
        header=header,
        height=1,
        width=len(points),
        fields=fields,
        is_bigendian=False,
        point_step=point_step,
        row_step=row_step,
        data=data,
        is_dense=True,
    )

def get_tf_from_frames(target_frame, source_frame):
    try:
        buffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(buffer)

        # Wait for the transform to become available
        transform = buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(4.0))
        
        # Extract translation and rotation
        trans = transform.transform.translation
        rot = transform.transform.rotation
        
        # Convert to a 4x4 transformation matrix
        translation = np.array([trans.x, trans.y, trans.z])
        rotation = np.array([rot.x, rot.y, rot.z, rot.w])
        
        # Construct the matrix
        transform_matrix = np.dot(tft.translation_matrix(translation), tft.quaternion_matrix(rotation))
        return transform_matrix
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"Could not find transformation: {e}")
        return None

def fix_trajectory_timestamps(robot_trajectory: JointTrajectory, start_time=0.0, time_step=0.05):
    """
    Fixes the `time_from_start` field for a given RobotTrajectory message.
    :param robot_trajectory: RobotTrajectory message from MoveIt
    :param start_time: Initial time offset (default: 0.0 seconds)
    :param time_step: Time increment between waypoints (default: 0.1 seconds)
    :return: Fixed RobotTrajectory message
    """
    cumulative_time = start_time
    # prev = rospy.Duration(start_time)

    for point in robot_trajectory.joint_trajectory.points:
        # Update time_from_start for each point
        # curr = point.time_from_start
        # if prev > curr:
        #     curr += 2*(prev - curr)
        # prev = curr
        point.time_from_start = rospy.Duration(cumulative_time)
        cumulative_time += time_step

    return robot_trajectory