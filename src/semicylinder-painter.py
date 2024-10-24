import sys
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    num_strokes = ceil(effective_length / tool_width)
    return num_strokes

def find_start_points_on_semicircle(radius, num_strokes, tool_width):
    """
    Find the start points of each tool stroke on the inner surface of the half-cylinder (semicircle),
    ensuring the first and last strokes are half the tool width away from the edges.
    
    Parameters:
    radius (float): The radius of the semicircle.
    num_strokes (int): The number of equally spaced points (strokes).
    tool_width (float): The width of the tooltip.
    
    Returns:
    list: A list of (x, y) coordinates for the start points on the semicircle.
    """
    points = []
    
    # Angular range to cover (excluding half-tool widths on both sides)
    start_angle = tool_width / (2 * radius)
    end_angle = np.pi - tool_width / (2 * radius)
    
    # Calculate the angular step (divide the semicircle into num_strokes equal parts)
    angle_step = (end_angle - start_angle) / (num_strokes - 1)
    
    # Calculate the (x, y) coordinates for each start point
    for i in range(num_strokes):
        theta = start_angle + i * angle_step  # Angle for each point
        x = radius * np.cos(theta) + radius
        y = radius * np.sin(theta)
        points.append((x, y))
    
    return points

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

def plot_semicircle(radius, start_points):
    """
    Plot the start points of the tool strokes on a semicircle.
    
    Parameters:
    radius (float): The radius of the semicircle.
    start_points (list): A list of (x, y) coordinates for the start points on the semicircle.
    """
    x_vals = [p[0] for p in start_points]
    y_vals = [p[1] for p in start_points]

    theta = np.linspace(0, np.pi, 100)  # Angle from 0 to pi for a semicircle
    x_semi = radius * np.cos(theta)  # x = r * cos(theta)
    y_semi = radius * np.sin(theta)  # y = r * sin(theta)

    plt.plot(x_semi, y_semi, color='black') # Plot semicircle
    plt.plot(x_vals, y_vals, 'bo')  # Plot start points as blue circles
    plt.plot(x_vals, y_vals, 'r--')  # Connect points with a red dashed line
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Start points of {num_strokes} tool strokes on the semicircle")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

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
    sp_x_vals = [p[0] for p in start_points]
    sp_y_vals = [p[1] for p in start_points]
    sp_z_vals = [p[2] for p in start_points]

    # 3D Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot stroke vector
    ax.quiver(sp_first[0], sp_first[1], sp_first[2], stroke_vector[0], stroke_vector[1], stroke_vector[2], color='blue', label='Stroke vector')

    # Plot diameter vector
    ax.quiver(sp_first[0], sp_first[1], sp_first[2], diameter_vector[0], diameter_vector[1], diameter_vector[2], color='red', label='Diameter vector')

    # Calculate semicircle points
    theta = np.linspace(0, np.pi, 100)  # Angle from 0 to pi for a semicircle
    x_semi = radius * np.cos(theta) + radius  # x = r * cos(theta)
    y_semi = radius * np.sin(theta)  # y = r * sin(theta)
    z_semi = np.zeros_like(theta)  # z = 0
    semicircle = [np.array([x, y, z]) for x, y, z in zip(x_semi, y_semi, z_semi)]

    # Calculate 2 transformed semicircles
    semicircle_tf_1 = transform_points(semicircle, transformation_matrix)
    semicircle_tf_2 = np.array(semicircle_tf_1) + (end_points[0] - start_points[0])

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
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='green')

    # Plot start and end points
    ax.plot(sp_x_vals, sp_y_vals, sp_z_vals, 'bo')  # Plot start points as blue circles
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

# Input from the user (old)
# radius = float(input("Enter the radius of the half-cylinder: "))
# tool_width = float(input("Enter the width of the tooltip: "))
# overlap = float(input("Enter the minimum allowed overlap between strokes: "))

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
sp_first = np.array(list(map(float, sp_first_str.split())))
ep_first = np.array(list(map(float, ep_first_str.split())))
sp_last = np.array(list(map(float, sp_last_str.split())))
#ep_last = np.array(list(map(float, ep_last_str.split())))

# Calculate the diameter vector from sp_first to sp_last
diameter_vector = sp_last - sp_first # This is our x-axis
diameter = np.linalg.norm(diameter_vector)
if diameter == 0:
    raise ValueError("Diameter cannot be zero.")
diameter_vector /= diameter  # This is our x-axis

# Calculate the depth vector
depth_vector = np.cross(ep_first - sp_first, diameter_vector)
depth = np.linalg.norm(depth_vector)
if depth == 0:
    raise ValueError("Depth cannot be zero.")
depth_vector /= depth  # This is our y-axis

# Calculate the stroke direction vector
stroke_vector = np.cross(diameter_vector, depth_vector)# This is our z-axis
tf_vector = np.linalg.norm(ep_first - sp_first) * stroke_vector

# Calculate cylinder radius
radius = diameter / 2

# Calculate the number of tool strokes
num_strokes = calculate_tool_strokes(radius, tool_width)
if num_strokes < 2:
    raise ValueError("Number of strokes must be at least 2 (tool width too big).")

# Find the start points of each stroke on the semicircle
start_points = find_start_points_on_semicircle(radius, num_strokes, tool_width)

# Calculate the transformed start and end points
transformation_matrix = np.eye(4)
transformation_matrix[:3, :3] = np.column_stack((diameter_vector, depth_vector, stroke_vector))
transformation_matrix[:3, 3] = sp_first

start_points_tf = transform_points(start_points, transformation_matrix)
end_points_tf = np.array(start_points_tf) + tf_vector

# Plot the start and end points on the cylinder
plot_strokes_3d(sp_first, stroke_vector, diameter_vector, start_points_tf, end_points_tf, transformation_matrix, radius)
