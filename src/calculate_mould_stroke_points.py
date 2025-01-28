#!/usr/bin/env python3
 
import math
import roslib
import csv
import numpy as np
import matplotlib.pyplot as plt
import tf.transformations as tft
import rospy
import rospkg
import tf2_ros
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import String
roslib.load_manifest('dmp')
from dmp.srv import *
from dmp.msg import *

# Learn a DMP from demonstration data
def makeLFDRequest(dims, traj, dt, K_gain, D_gain, num_bases):
    demotraj = DMPTraj()

    for i in range(len(traj)):
        pt = DMPPoint()
        pt.positions = traj[i]
        demotraj.points.append(pt)
        demotraj.times.append(dt * i)

    k_gains = [K_gain] * dims
    d_gains = [D_gain] * dims

    print("Starting LfD...")
    rospy.wait_for_service('learn_dmp_from_demo')
    try:
        lfd = rospy.ServiceProxy('learn_dmp_from_demo', LearnDMPFromDemo)
        resp = lfd(demotraj, k_gains, d_gains, num_bases)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
    print("LfD done")

    return resp

# Set a DMP as active for planning
def makeSetActiveRequest(dmp_list):
    try:
        sad = rospy.ServiceProxy('set_active_dmp', SetActiveDMP)
        sad(dmp_list)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

# Generate a plan from a DMP
def makePlanRequest(x_0, x_dot_0, t_0, goal, goal_thresh, seg_length, tau, dt, integrate_iter):
    print("Starting DMP planning...")
    rospy.wait_for_service('get_dmp_plan')
    try:
        gdp = rospy.ServiceProxy('get_dmp_plan', GetDMPPlan)
        resp = gdp(x_0, x_dot_0, t_0, goal, goal_thresh, seg_length, tau, dt, integrate_iter)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
    print("DMP planning done")

    return resp

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

def find_default_pose(point1, point2):
    # Calculate the midpoint position
    midpoint = Point(
        x=(point1.x + point2.x) / 2,
        y=(point1.y + point2.y) / 2,
        z=(point1.z + point2.z) / 2
    )

    # Set default orientation (identity quaternion)
    default_orientation = Quaternion(x=0, y=0.7071, z=0, w=0.7071)

    # Create and return the Pose
    pose = Pose()
    pose.position = midpoint
    pose.orientation = default_orientation

    return pose

def find_poses_on_semicircle(radius, num_strokes, tool_width, hover):
    """
    Find the start and hover poses of each tool stroke on the inner surface of the half-cylinder (semicircle),
    ensuring the first and last strokes are half the tool width away from the edges. Each pose includes
    both position (x, y, z) and orientation (quaternion) pointing outward from the center of the semicircle.
    
    Parameters:
    radius (float): The radius of the semicircle.
    num_strokes (int): The number of equally spaced points (strokes).
    tool_width (float): The width of the tool.
    hover (float): The hover distance.
    
    Returns:
    tuple: A tuple containing two lists:
        - list: A list of geometry_msgs/Pose objects representing each start pose.
        - list: A list of geometry_msgs/Pose objects representing each hover pose.
    """
    end_poses = []
    hover_poses = []
    
    # Angular range to cover (excluding half-tool widths on both sides)
    start_angle = tool_width / (2 * radius)
    end_angle = math.pi - tool_width / (2 * radius)
    
    # Calculate the angular step (divide the semicircle into num_strokes equal parts)
    angle_step = (end_angle - start_angle) / (num_strokes - 1)
    
    # Calculate the position and orientation for each pose
    for i in range(num_strokes):
        theta = start_angle + i * angle_step  # Angle for each point
        x_start = radius * np.cos(theta) + radius
        y_start = radius * np.sin(theta)

        x_hover = (radius - hover) * np.cos(theta) + radius
        y_hover = (radius - hover) * np.sin(theta)
        
        # Orientation is outward, so we add pi/2 to theta
        orientation_angle = theta + np.pi / 2
        qx, qy, qz, qw = angle_to_quaternion(orientation_angle)
        
        # Create start Pose
        end_pose = Pose()
        end_pose.position.x = x_start
        end_pose.position.y = y_start
        end_pose.position.z = 0  # 2D plane, so z is 0
        
        end_pose.orientation.x = qx
        end_pose.orientation.y = qy
        end_pose.orientation.z = qz
        end_pose.orientation.w = qw
        
        end_poses.append(end_pose)

        # Create hover Pose
        hover_pose = Pose()
        hover_pose.position.x = x_hover
        hover_pose.position.y = y_hover
        hover_pose.position.z = 0  # 2D plane, so z is 0
        
        hover_pose.orientation.x = qx
        hover_pose.orientation.y = qy
        hover_pose.orientation.z = qz
        hover_pose.orientation.w = qw
        
        hover_poses.append(hover_pose)
    
    return end_poses, hover_poses

def vector_from_projection_to_point(point1, direction_vector, point2):
    """
    Calculate the vector from the projection of point2 on the line defined by point1 and direction_vector to point2.
    
    Parameters:
    point1 (Point): A ROS Point object on the line.
    direction_vector (tuple): The direction vector of the line as (a, b, c).
    point2 (Point): A ROS Point object for the point we want to project.
    
    Returns:
    Point: A ROS Point object representing the vector from the projection of point2 to point2.
    """
    # Convert point1 and point2 to numpy arrays for easier calculation
    p1 = np.array([point1.x, point1.y, point1.z])
    p2 = np.array([point2.x, point2.y, point2.z])
    v = np.array(direction_vector)
    
    # Calculate the vector from p1 to p2
    w = p2 - p1
    
    # Project w onto v
    proj_w_on_v = (np.dot(w, v) / np.dot(v, v)) * v
    
    # Calculate the projection point on the line
    p_proj = p1 + proj_w_on_v
    
    # Calculate the vector from the projection point to point2
    vector_projection_to_point = p2 - p_proj
    
    return vector_projection_to_point

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
        transformed_position = np.dot(transformation_matrix, position)
        
        # Apply the transformation to the rotation
        transformed_rotation_matrix = np.eye(4)
        transformed_rotation_matrix[:3, :3] = np.dot(transformation_matrix[:3, :3], rotation_matrix)
        
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

def swap_every_second_element(list1, list2):
    """
    Swap every second element between two lists.
    
    Parameters:
    list1 (list): The first list of poses.
    list2 (list): The second list of poses.
    
    Returns:
    tuple: The modified list1 and list2 after swapping every second element.
    """
    # Ensure both lists are of equal length
    if len(list1) != len(list2):
        raise ValueError("Both lists must be of the same length.")
    
    # Swap every second element (start from index 1 to get every second element)
    for i in range(1, len(list1), 2):
        list1[i], list2[i] = list2[i], list1[i]
    
    return list1, list2

def plot_strokes_3d(ep_first, forward_vector, diameter_vector, end_points, start_points, end_points_hover, start_points_hover, tf_vector, transformation_matrix, start_radius, end_radius, start_angle, angle_step):
    """
    Plot the start points of the tool strokes on a semicylinder in 3D.
    
    Parameters:
    ep_first (np.array): Start point of the first stroke.
    forward_vector (np.array): The stroke vector.
    diameter_vector (np.array): The diameter vector.
    end_points (list): List of 3D np.array stroke start points.
    start_points (list): List of 3D np.array stroke end points.
    end_points_hover (list): List of 3D np.array stroke start hover points.
    start_points_hover (list): List of 3D np.array stroke end hover points.
    tf_vector (np.array): Height vector of the semicylinder.
    start_radius (float): The start radius of the semicylinder.
    end_radius (float): The end radius of the semicylinder.
    """
    # Prepare data
    sp_x_vals = [p.x for p in end_points]
    sp_y_vals = [p.y for p in end_points]
    sp_z_vals = [p.z for p in end_points]

    # 3D Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot stroke vector
    # ax.quiver(ep_first.x, ep_first.y, ep_first.z, forward_vector[0], forward_vector[1], forward_vector[2], color='blue', label='Stroke vector')

    # Plot diameter vector
    # ax.quiver(ep_first.x, ep_first.y, ep_first.z, diameter_vector[0], diameter_vector[1], diameter_vector[2], color='red', label='Diameter vector')

    # Calculate start semicircle points
    theta = np.linspace(0, np.pi, 100)  # Angle from 0 to pi for a semicircle
    x_semi = start_radius * np.cos(theta) + start_radius  # x = r * cos(theta)
    y_semi = start_radius * np.sin(theta)  # y = r * sin(theta)
    z_semi = np.zeros_like(theta)  # z = 0
    start_semicircle = [np.array([x, y, z]) for x, y, z in zip(x_semi, y_semi, z_semi)]

    # Calculate end semicircle points
    x_semi = end_radius * np.cos(theta) + end_radius  # x = r * cos(theta)
    y_semi = end_radius * np.sin(theta)  # y = r * sin(theta)
    z_semi = np.zeros_like(theta)  # z = 0
    end_semicircle = [np.array([x, y, z]) for x, y, z in zip(x_semi, y_semi, z_semi)]

    # Calculate 2 transformed semicircles
    semicircle_tf_1 = transform_points(start_semicircle, transformation_matrix)
    semicircle_tf = transform_points(end_semicircle, transformation_matrix)
    semicircle_tf_2 = np.array(semicircle_tf) + tf_vector

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
    # ax.plot([sp_1[0], ep_1[0]], [sp_1[1], ep_1[1]], [sp_1[2], ep_1[2]], color='black')
    # ax.plot([sp_2[0], ep_2[0]], [sp_2[1], ep_2[1]], [sp_2[2], ep_2[2]], color='black')

    cnt = 0
    # Plot strokes
    for start, end in zip(end_points, start_points):
        start_np = np.array([start.x, start.y, start.z])
        end_np = np.array([end.x, end.y, end.z])
        theta = start_angle + cnt * angle_step  # Angle for each point
        cnt += 1

        stroke = dmp_mould(start_np, end_np, theta)
        x, y, z = zip(*stroke)

        ax.plot(x, y, z, color='blue')
        #ax.plot([start.x, end.x], [start.y, end.y], [start.z, end.z], color='blue')

    # Plot start points
    # ax.plot(sp_x_vals, sp_y_vals, sp_z_vals, 'bo')  # Plot start points as blue dots
    # ax.plot(sp_x_vals, sp_y_vals, sp_z_vals, 'r--')  # Connect points with a red dashed line

    # Plot hover strokes
    # for p1, p2 in zip(start_points[1:], start_points_hover[1:]):
    #     ax.plot([p1.x, p2.x], [p1.y, p2.y], [p1.z, p2.z], color='green')

    # for p1, p2 in zip(end_points[:-1], end_points_hover[:-1]):
    #     ax.plot([p1.x, p2.x], [p1.y, p2.y], [p1.z, p2.z], color='green')

    # swap_every_second_element(end_points_hover, start_points_hover)
    
    # for p1, p2 in zip(start_points_hover[1:], end_points_hover[:-1]):
    #     ax.plot([p1.x, p2.x], [p1.y, p2.y], [p1.z, p2.z], color='green')

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

def dmp_mould(start, goal, theta):
    # Create a DMP from a 3-D trajectory
    dims = 3
    dt = 1.0
    K = 100
    D = 2.0 * np.sqrt(K)
    num_bases = 4
    traj = []
    # Read the file and process each line
    # Define the file path
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('semicyl_painter')
    file_path = f"{package_path}/data/mould_filtered_path.csv"

    # Read the CSV file and create a list of 3-member arrays
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            mould_dims = [float(value) for value in row]
            mould_dims_fixed = [mould_dims[1], -mould_dims[0], mould_dims[2]]
            mould_dims_rotated = [mould_dims_fixed[0],
                                  mould_dims_fixed[1] * math.cos(theta) + mould_dims_fixed[2] * math.sin(theta),
                                  mould_dims_fixed[2] * math.cos(theta) - mould_dims_fixed[1] * math.sin(theta)]
            traj.append(mould_dims_rotated)
            # traj.append([float(value) for value in row])  # Convert each value to float

    resp = makeLFDRequest(dims, traj, dt, K, D, num_bases)

    # Set it as the active DMP
    makeSetActiveRequest(resp.dmp_list)

    # Now, generate a plan
    #x_0 = [0.0, 0.0, 0.0]           # Plan starting at a different point than demo
    x_dot_0 = [0.0, 0.0, 0.0]
    t_0 = 0
    goal_thresh = [0.01, 0.01, 0.01]
    seg_length = -1            # Plan until convergence to goal
    tau = 2 * resp.tau         # Desired plan should take twice as long as demo
    dt = 1.0
    integrate_iter = 5         # dt is rather large, so this is > 1
    plan = makePlanRequest(start, x_dot_0, t_0, goal, goal_thresh, seg_length, tau, dt, integrate_iter)

    plan_positions = []
    for point in plan.plan.points:
        plan_positions.append(point.positions)

    return plan_positions

def publish_poses(end_poses, start_poses, end_hover_poses, start_hover_poses):
    """
    Publish the poses of the tool strokes.
    
    Parameters:
    end_poses (list): List of Pose stroke start poses.
    start_poses (list): List of Pose stroke end poses.
    end_hover_poses (list): List of pose stroke start hover poses.
    start_hover_poses (list): List of pose stroke end hover poses.
    """
    # Initialize the ROS node
    # rospy.init_node('pose_publisher_node_goober', anonymous=True)

    # Create publishers for Pose messages
    end_pub = rospy.Publisher('end_poses', Pose, queue_size=100)
    start_pub = rospy.Publisher('start_poses', Pose, queue_size=100)
    end_hover_pub = rospy.Publisher('end_hover_poses', Pose, queue_size=100)
    start_hover_pub = rospy.Publisher('start_hover_poses', Pose, queue_size=100)
    default_pub = rospy.Publisher('default_pose', Pose, queue_size=10)

    rospy.sleep(1)

    #swap_every_second_element(end_poses, start_poses)
    #swap_every_second_element(end_hover_poses, start_hover_poses)

    # Publish start poses
    for pose in end_poses:
        end_pub.publish(pose)
        #rospy.loginfo(f"Published End Pose: {apply_transformation_to_pose(tf_matrix, pose)}")

    # Publish end poses
    for pose in start_poses:
        start_pub.publish(pose)
        #rospy.loginfo(f"Published End Pose: {pose}")

    # Publish start hover poses
    for pose in end_hover_poses:
        end_hover_pub.publish(pose)
        #rospy.loginfo(f"Published Start Hover Pose: {pose}")

    # Publish end hover poses
    for pose in start_hover_poses:
        start_hover_pub.publish(pose)
        #rospy.loginfo(f"Published End Hover Pose: {pose}")

    default_pose = find_default_pose(start_hover_poses[0].position, start_hover_poses[-1].position)

    rospy.sleep(1)

    default_pub.publish(default_pose)
    #rospy.loginfo(f"All poses published!")

def get_transformation_matrix(buffer, target_frame, source_frame):
    try:
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
    
def apply_transformation_to_pose(matrix, pose):
    """
    Applies a 4x4 transformation matrix to a geometry_msgs.msg.Pose.

    Args:
        matrix (np.ndarray): A 4x4 transformation matrix.
        pose (Pose): A ROS Pose object with position and orientation.

    Returns:
        Pose: A new Pose object with the transformed position and orientation.
    """
    # Convert Pose position to a homogeneous vector
    position = np.array([pose.position.x, pose.position.y, pose.position.z, 1.0])
    
    # Apply the transformation matrix to the position
    transformed_position = np.dot(matrix, position)

    # Extract the quaternion from the pose
    quaternion = [
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w
    ]

    # Apply the rotation part of the transformation matrix to the quaternion
    full_matrix = tft.quaternion_matrix(quaternion)  # Convert quaternion to 4x4 matrix
    transformed_orientation_matrix = np.dot(matrix, full_matrix)  # Apply transformation
    transformed_quaternion = tft.quaternion_from_matrix(transformed_orientation_matrix)

    # Create a new Pose object with transformed position and orientation
    transformed_pose = Pose()
    transformed_pose.position.x = transformed_position[0]
    transformed_pose.position.y = transformed_position[1]
    transformed_pose.position.z = transformed_position[2]
    transformed_pose.orientation.x = transformed_quaternion[0]
    transformed_pose.orientation.y = transformed_quaternion[1]
    transformed_pose.orientation.z = transformed_quaternion[2]
    transformed_pose.orientation.w = transformed_quaternion[3]

    return transformed_pose

def main():
    # Initialize the ROS node
    rospy.init_node('pose_publisher_node', anonymous=True)

    rospy.sleep(2)

    # Input from the user
    # tool_width = float(input("Enter the width of the tooltip: "))
    # overlap = float(input("Enter the minimum allowed overlap between strokes: "))

    # ep_first_str = input("Enter the coordinates of the first start point separated by spaces: ")
    # sp_first_str = input("Enter the coordinates of the first end point separated by spaces: ")
    # ep_last_str = input("Enter the coordinates of the last start point separated by spaces: ")

    # Input from ros launch file
    tool_width = rospy.get_param("~tool_width", 0.01)  # Default value is 0.05
    overlap = rospy.get_param("~tool_overlap", 0.0)  # Default value is 0.0

    #ep_first_str = rospy.get_param("~ep_first", "0.3 -0.2 0.5")  # Default is "0.2 0.0 0.0"
    #sp_first_str = rospy.get_param("~sp_first", "0.4 -0.205 0.5")  # Default is "0.7 0.0 0.0"
    #ep_last_str = rospy.get_param("~ep_last", "0.3 -0.3 0.5")  # Default is "0.2 -0.5 0.0"
    #ep_first_str = rospy.get_param("~ep_first", "0.6 -0.2 0.5")  # Default is "0.2 0.0 0.0"
    #sp_first_str = rospy.get_param("~sp_first", "0.7 -0.205 0.5")  # Default is "0.7 0.0 0.0"
    #ep_last_str = rospy.get_param("~ep_last", "0.6 -0.3 0.5")  # Default is "0.2 -0.5 0.0"
    # ep_first_str = rospy.get_param("~ep_first", "0.3 0.0 0.5")  # Default is "0.2 0.0 0.0"
    # sp_first_str = rospy.get_param("~sp_first", "0.57 -0.01875 0.5")  # Default is "0.7 0.0 0.0"
    # ep_last_str = rospy.get_param("~ep_last", "0.3 -0.063 0.5")  # Default is "0.2 -0.5 0.0"

    ep_first_str = rospy.get_param("~ep_first", "-0.165 0.032 0.0")  # Default is "-0.16 0.0315 0.0"
    sp_first_str = rospy.get_param("~sp_first", "0.0 0.01275 0.0")  # Default is "0.0 0.01275 0.0"
    ep_last_str = rospy.get_param("~ep_last", "-0.165 -0.032 0.0")  # Default is "-0.16 -0.0315 0.0"
    padding = 0.0

    # Truncate tool width to include overlap
    tool_width -= overlap
    if tool_width <= 0.0:
        raise ValueError("The tool width must be greater than the overlap.")

    # Convert the point input strings into individual values
    ep_first = string_to_point(ep_first_str)
    sp_first = string_to_point(sp_first_str)
    ep_last = string_to_point(ep_last_str)

    ep_first.y -= padding
    sp_first.y -= padding
    ep_first.y += padding

    # Calculate the diameter vector from ep_first to ep_last
    diameter_vector = vector_between_points(ep_first, ep_last)
    diameter = np.linalg.norm(diameter_vector)
    if diameter == 0:
        raise ValueError("Diameter cannot be zero.")
    diameter_vector /= diameter  # This is our x-axis

    # Calculate the depth vector
    depth_vector = np.cross(vector_between_points(ep_first, sp_first), diameter_vector)
    depth = np.linalg.norm(depth_vector)
    if depth == 0:
        raise ValueError("Depth cannot be zero.")
    depth_vector /= depth  # This is our y-axis

    # Calculate the stroke direction vector
    forward_vector = np.cross(diameter_vector, depth_vector)# This is our z-axis

    # Calculate cylinder radius, side offset, second radius and tf_vector
    start_radius = diameter / 2
    offset_vector = vector_from_projection_to_point(ep_first, forward_vector, sp_first)
    end_radius = start_radius - np.linalg.norm(offset_vector)
    tf_vector = vector_from_projection_to_point(ep_first, diameter_vector, sp_first) + offset_vector

    # Calculate the number of tool strokes
    num_strokes = calculate_tool_strokes(start_radius, tool_width)
    if num_strokes < 2:
        raise ValueError("Number of strokes must be at least 2 (tool width too big).")

    # Find the start points of each stroke on the semicircle
    # end_points = find_end_points_on_semicircle(radius, num_strokes, tool_width)
    end_poses, end_hover_poses = find_poses_on_semicircle(start_radius, num_strokes, tool_width, tool_width*2)

    # Find the end points of each stroke on the semicircle
    start_poses, start_hover_poses = find_poses_on_semicircle(end_radius, num_strokes, tool_width, tool_width)

    for i in range(len(start_hover_poses)):
        end_hover_poses[i].orientation = start_hover_poses[i].orientation

    # Calculate the transformed start and end points
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = np.column_stack((diameter_vector, depth_vector, forward_vector))
    transformation_matrix[:3, 3] = np.array([ep_first.x, ep_first.y, ep_first.z])

    end_poses_tf = transform_poses(end_poses, transformation_matrix)
    end_poses_hover_tf = transform_poses(end_hover_poses, transformation_matrix)

    start_poses_temp = transform_poses(start_poses, transformation_matrix)
    start_poses_hover_temp = transform_poses(start_hover_poses, transformation_matrix)

    start_poses_tf = translate_poses(start_poses_temp, tf_vector)
    start_poses_hover_tf = translate_poses(start_poses_hover_temp, tf_vector)

    # pose_0 = end_poses_tf[0]
    # quaternion_0 = (
    #     pose_0.orientation.x,
    #     pose_0.orientation.y,
    #     pose_0.orientation.z,
    #     pose_0.orientation.w
    # )
    # pose_1 = end_poses_tf[1]
    # quaternion_1 = (
    #     pose_1.orientation.x,
    #     pose_1.orientation.y,
    #     pose_1.orientation.z,
    #     pose_1.orientation.w
    # )

    # roll_0, pitch_0, yaw_0 = tft.euler_from_quaternion(quaternion_0)
    # roll_1, pitch_1, yaw_1 = tft.euler_from_quaternion(quaternion_1)
    

    # # Plot the start and end points on the cylinder
    # plot_strokes_3d(
    #     ep_first, forward_vector, diameter_vector,
    #     convert_poses_to_points(end_poses_tf), convert_poses_to_points(start_poses_tf),
    #     convert_poses_to_points(end_poses_hover_tf), convert_poses_to_points(start_poses_hover_tf),
    #     tf_vector, transformation_matrix, start_radius, end_radius, pitch_0, pitch_1 - pitch_0)

    publish_poses(end_poses_tf, start_poses_tf, end_poses_hover_tf, start_poses_hover_tf)
    print("Published!")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass