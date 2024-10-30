import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import tf.transformations as tft
from geometry_msgs.msg import Pose, Point
from mpl_toolkits.mplot3d import Axes3D

def string_to_point(point_string):
    # Split the string
    coordinates = point_string.split()
    
    # Convert the split strings to float and unpack them into x, y, z
    x, y, z = map(float, coordinates)

    # Create and return a Point object
    return Point(x=x, y=y, z=z)

def vector_between_points(p1, p2):
    # Create numpy arrays from the Point objects
    vector = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])
    return vector

def translate_poses(poses, vector):
    """
    Sum each pose's position in a list of geometry_msgs/Pose with a numpy array vector,
    preserving each pose's orientation, and return a new list of poses.

    Parameters:
    poses (list of Pose): The original list of Pose objects.
    vector (np.array): A 3D vector (array of shape (3,)) to add to each Pose's position.

    Returns:
    list of Pose: A new list of Pose objects with updated positions.
    """
    # Create a new list to hold the resulting poses
    result_poses = []
    
    # Sum each pose's position with the vector
    for pose in poses:
        new_pose = Pose()
        
        # Update the position by adding the vector to the pose's position
        pose_position = np.array([pose.position.x, pose.position.y, pose.position.z])
        new_position = pose_position + vector
        
        new_pose.position.x = new_position[0]
        new_pose.position.y = new_position[1]
        new_pose.position.z = new_position[2]
        
        # Keep the original orientation of the pose
        new_pose.orientation = pose.orientation
        
        # Append the new pose to the result list
        result_poses.append(new_pose)
    
    return result_poses

def convert_poses_to_points(poses):
    """
    Convert a list of geometry_msgs/Pose objects into a list of geometry_msgs/Point objects.

    Parameters:
    poses (list of Pose): The original list of Pose objects.

    Returns:
    list of Point: A new list of Point objects representing the positions of the Poses.
    """
    # Create a new list to hold the resulting points
    points = []
    
    # Convert each Pose to a Point
    for pose in poses:
        point = Point()
        point.x = pose.position.x
        point.y = pose.position.y
        point.z = pose.position.z
        points.append(point)
    
    return points

def angle_to_quaternion(theta):
    """
    Convert a 2D angle in radians to a quaternion representing a rotation around the Z-axis.
    
    Parameters:
    theta (float): Angle in radians.
    
    Returns:
    tuple: Quaternion (x, y, z, w)
    """
    # For a rotation around the Z-axis
    qx = 0
    qy = 0
    qz = math.sin(theta / 2)
    qw = math.cos(theta / 2)
    return (qx, qy, qz, qw)

def calculate_tool_strokes(radius, tool_width):
    """
    Calculate the number of tool strokes needed to paint the inner surface of a hollow half-cylinder,
    ensuring the first and last strokes are half the tool width away from the edges.
    
    Parameters:
    radius (float): The radius of the half-cylinder.
    tool_width (float): The width of the tooltip.
    
    Returns:
    int: The number of tool strokes required (rounded up to ensure full coverage).
    """
    # Effective length of the semicircle to be painted (excluding half-tool widths on both sides)
    effective_length = np.pi * radius - tool_width
    
    # Calculate the number of strokes (rounding up to ensure full coverage)
    num_strokes = math.ceil(effective_length / tool_width)
    return num_strokes

# REDUNDANT
# def find_start_points_on_semicircle(radius, num_strokes, tool_width):
#     """
#     Find the start points of each tool stroke on the inner surface of the half-cylinder (semicircle),
#     ensuring the first and last strokes are half the tool width away from the edges.
    
#     Parameters:
#     radius (float): The radius of the semicircle.
#     num_strokes (int): The number of equally spaced points (strokes).
#     tool_width (float): The width of the tooltip.
    
#     Returns:
#     list: A list of (x, y) coordinates for the start points on the semicircle.
#     """
#     points = []
    
#     # Angular range to cover (excluding half-tool widths on both sides)
#     start_angle = tool_width / (2 * radius)
#     end_angle = np.pi - tool_width / (2 * radius)
    
#     # Calculate the angular step (divide the semicircle into num_strokes equal parts)
#     angle_step = (end_angle - start_angle) / (num_strokes - 1)
    
#     # Calculate the (x, y) coordinates for each start point
#     for i in range(num_strokes):
#         theta = start_angle + i * angle_step  # Angle for each point
#         x = radius * np.cos(theta) + radius
#         y = radius * np.sin(theta)
#         points.append((x, y))
    
#     return points

def find_start_poses_on_semicircle(radius, num_strokes, tool_width):
    """
    Find the start poses of each tool stroke on the inner surface of the half-cylinder (semicircle),
    ensuring the first and last strokes are half the tool width away from the edges. Each pose includes
    both position (x, y, z) and orientation (quaternion) pointing outward from the center of the semicircle.
    
    Parameters:
    radius (float): The radius of the semicircle.
    num_strokes (int): The number of equally spaced points (strokes).
    tool_width (float): The width of the tool.
    
    Returns:
    list: A list of geometry_msgs/Pose objects representing each start pose.
    """
    poses = []
    
    # Angular range to cover (excluding half-tool widths on both sides)
    start_angle = tool_width / (2 * radius)
    end_angle = math.pi - tool_width / (2 * radius)
    
    # Calculate the angular step (divide the semicircle into num_strokes equal parts)
    angle_step = (end_angle - start_angle) / (num_strokes - 1)
    
    # Calculate the position and orientation for each start pose
    for i in range(num_strokes):
        theta = start_angle + i * angle_step  # Angle for each point
        x = radius * np.cos(theta) + radius
        y = radius * np.sin(theta)
        
        # Orientation is outward, so we add pi/2 to theta
        orientation_angle = theta + np.pi / 2
        qx, qy, qz, qw = angle_to_quaternion(orientation_angle)
        
        # Create Pose object
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0  # 2D plane, so z is 0
        
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        
        poses.append(pose)
    
    return poses

def transform_points(points, transformation_matrix):
    """
    Transform a list of 3D np.array points using a transformation matrix
    
    Parameters:
    points (list): A list of 2D or 3D np.array points.
    transformation_matrix (np.array): An np.array 4x4 transformation matrix.
    
    Returns:
    list: A list of transformed 3D np.array points.
    """
    # Transform the points
    transformed_points = []
    for point in points:
        # Convert to 3D if necessary
        if np.size(point) == 2:
            point = np.append(point, 0)

        # Convert to homogeneous coordinates by appending a 1
        point_homogeneous = np.append(point, 1)

        # Apply the transformation matrix
        transformed_point_homogeneous = np.dot(transformation_matrix, point_homogeneous)
        
        # Truncate the last value (the homogeneous coordinate 1)
        transformed_point = transformed_point_homogeneous[:3]
        
        # Append the transformed point to the list
        transformed_points.append(transformed_point)

    return transformed_points

def transform_poses(poses, transformation_matrix):
    """
    Transform a list of geometry_msgs/Pose objects using a transformation matrix,
    including both position and orientation.

    Parameters:
    poses (list): A list of geometry_msgs/Pose objects.
    transformation_matrix (np.array): A 4x4 np.array transformation matrix.

    Returns:
    list: A list of transformed geometry_msgs/Pose objects.
    """
    transformed_poses = []

    for pose in poses:
        # Extract position and orientation from Pose
        position = np.array([pose.position.x, pose.position.y, pose.position.z, 1])
        orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        
        # Convert quaternion to rotation matrix
        rotation_matrix = tft.quaternion_matrix(orientation)[:3, :3]
        
        # Transform position using the transformation matrix
        transformed_position = transformation_matrix @ position
        
        # Apply the transformation to the rotation
        transformed_rotation_matrix = np.eye(4)
        transformed_rotation_matrix[:3, :3] = transformation_matrix[:3, :3] @ rotation_matrix
        
        # Convert the transformed rotation matrix back to quaternion
        transformed_orientation = tft.quaternion_from_matrix(transformed_rotation_matrix)

        # Create new Pose object with transformed position and orientation
        new_pose = Pose()
        new_pose.position.x = transformed_position[0]
        new_pose.position.y = transformed_position[1]
        new_pose.position.z = transformed_position[2]
        new_pose.orientation.x = transformed_orientation[0]
        new_pose.orientation.y = transformed_orientation[1]
        new_pose.orientation.z = transformed_orientation[2]
        new_pose.orientation.w = transformed_orientation[3]

        transformed_poses.append(new_pose)

    return transformed_poses

# REDUNDANT
# def plot_semicircle(radius, start_points):
#     """
#     Plot the start points of the tool strokes on a semicircle.
    
#     Parameters:
#     radius (float): The radius of the semicircle.
#     start_points (list): A list of (x, y) coordinates for the start points on the semicircle.
#     """
#     x_vals = [p[0] for p in start_points]
#     y_vals = [p[1] for p in start_points]

#     theta = np.linspace(0, np.pi, 100)  # Angle from 0 to pi for a semicircle
#     x_semi = radius * np.cos(theta)  # x = r * cos(theta)
#     y_semi = radius * np.sin(theta)  # y = r * sin(theta)

#     plt.plot(x_semi, y_semi, color='black') # Plot semicircle
#     plt.plot(x_vals, y_vals, 'bo')  # Plot start points as blue circles
#     plt.plot(x_vals, y_vals, 'r--')  # Connect points with a red dashed line
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.title(f"Start points of {num_strokes} tool strokes on the semicircle")
#     plt.xlabel("X-axis")
#     plt.ylabel("Y-axis")
#     plt.show()

def plot_strokes_3d(sp_first, stroke_vector, diameter_vector, start_points, end_points, transformation_matrix, radius):
    """
    Plot the start points of the tool strokes on a semicircle in 3D.
    
    Parameters:
    sp_first (np.array): Start point of the first stroke.
    stroke_vector (np.array): The stroke vector.
    diameter_vector (np.array): The diameter vector.
    start_points_tf (list): List of 3D np.array stroke start points.
    end_points_tf (list): List of 3D np.array stroke end points.
    radius (float): The radius of the semicylinder.
    """
    # Prepare data
    sp_x_vals = [p.x for p in start_points]
    sp_y_vals = [p.y for p in start_points]
    sp_z_vals = [p.z for p in start_points]

    # 3D Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot stroke vector
    ax.quiver(sp_first.x, sp_first.y, sp_first.z, stroke_vector[0], stroke_vector[1], stroke_vector[2], color='blue', label='Stroke vector')

    # Plot diameter vector
    ax.quiver(sp_first.x, sp_first.y, sp_first.z, diameter_vector[0], diameter_vector[1], diameter_vector[2], color='red', label='Diameter vector')

    # Calculate semicircle points
    theta = np.linspace(0, np.pi, 100)  # Angle from 0 to pi for a semicircle
    x_semi = radius * np.cos(theta) + radius  # x = r * cos(theta)
    y_semi = radius * np.sin(theta)  # y = r * sin(theta)
    z_semi = np.zeros_like(theta)  # z = 0
    semicircle = [np.array([x, y, z]) for x, y, z in zip(x_semi, y_semi, z_semi)]

    # Calculate 2 transformed semicircles
    semicircle_tf_1 = transform_points(semicircle, transformation_matrix)
    semicircle_tf_2 = np.array(semicircle_tf_1) + vector_between_points(start_points[0], end_points[0])

    x_semi_1 = [p[0] for p in semicircle_tf_1]
    y_semi_1 = [p[1] for p in semicircle_tf_1]
    z_semi_1 = [p[2] for p in semicircle_tf_1]
    x_semi_2 = [p[0] for p in semicircle_tf_2]
    y_semi_2 = [p[1] for p in semicircle_tf_2]
    z_semi_2 = [p[2] for p in semicircle_tf_2]

    sp_1 = semicircle_tf_1[0]
    ep_1 = semicircle_tf_2[0]
    sp_2 = semicircle_tf_1[-1]
    ep_2 = semicircle_tf_2[-1]

    # Plot semicylinder outline
    ax.plot(x_semi_1, y_semi_1, z_semi_1, color='black')
    ax.plot(x_semi_2, y_semi_2, z_semi_2, color='black')
    ax.plot([sp_1[0], ep_1[0]], [sp_1[1], ep_1[1]], [sp_1[2], ep_1[2]], color='black')
    ax.plot([sp_2[0], ep_2[0]], [sp_2[1], ep_2[1]], [sp_2[2], ep_2[2]], color='black')

    # Plot strokes
    for start, end in zip(start_points, end_points):
        ax.plot([start.x, end.x], [start.y, end.y], [start.z, end.z], color='green')

    # Plot start and end points
    ax.plot(sp_x_vals, sp_y_vals, sp_z_vals, 'bo')  # Plot start points as blue dots
    ax.plot(sp_x_vals, sp_y_vals, sp_z_vals, 'r--')  # Connect points with a red dashed line

    # Set plot limits and labels
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()

# Input from the user
tool_width = float(input("Enter the width of the tooltip: "))
overlap = float(input("Enter the minimum allowed overlap between strokes: "))

sp_first_str = input("Enter the coordinates of the first start point separated by spaces: ")
ep_first_str = input("Enter the coordinates of the first end point separated by spaces: ")
sp_last_str = input("Enter the coordinates of the last start point separated by spaces: ")
#ep_last_str = input("Enter the coordinates of the last end point separated by spaces: ")

# Truncate tool width to include overlap
tool_width -= overlap
if tool_width <= 0.0:
    raise ValueError("The tool width must be greater than the overlap.")

# Convert the point input strings into individual values
sp_first = string_to_point(sp_first_str)
ep_first = string_to_point(ep_first_str)
sp_last = string_to_point(sp_last_str)

# Calculate the diameter vector from sp_first to sp_last
diameter_vector = vector_between_points(sp_first, sp_last) # This is our x-axis
diameter = np.linalg.norm(diameter_vector)
if diameter == 0:
    raise ValueError("Diameter cannot be zero.")
diameter_vector /= diameter  # This is our x-axis

# Calculate the depth vector
depth_vector = np.cross(vector_between_points(sp_first, ep_first), diameter_vector)
depth = np.linalg.norm(depth_vector)
if depth == 0:
    raise ValueError("Depth cannot be zero.")
depth_vector /= depth  # This is our y-axis

# Calculate the stroke direction vector
stroke_vector = np.cross(diameter_vector, depth_vector)# This is our z-axis
tf_vector = np.linalg.norm(vector_between_points(sp_first, ep_first)) * stroke_vector

# Calculate cylinder radius
radius = diameter / 2

# Calculate the number of tool strokes
num_strokes = calculate_tool_strokes(radius, tool_width)
if num_strokes < 2:
    raise ValueError("Number of strokes must be at least 2 (tool width too big).")

# Find the start points of each stroke on the semicircle
# start_points = find_start_points_on_semicircle(radius, num_strokes, tool_width)
start_poses = find_start_poses_on_semicircle(radius, num_strokes, tool_width)

# Calculate the transformed start and end points
transformation_matrix = np.eye(4)
transformation_matrix[:3, :3] = np.column_stack((diameter_vector, depth_vector, stroke_vector))
transformation_matrix[:3, 3] = np.array([sp_first.x, sp_first.y, sp_first.z])

start_poses_tf = transform_poses(start_poses, transformation_matrix)
end_poses_tf = translate_poses(start_poses_tf, tf_vector)
# Plot the start and end points on the cylinder
plot_strokes_3d(sp_first, stroke_vector, diameter_vector, convert_poses_to_points(start_poses_tf), convert_poses_to_points(end_poses_tf), transformation_matrix, radius)
