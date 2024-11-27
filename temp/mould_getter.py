import rospy
import rosbag
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import csv

def get_last_pointcloud_from_bag(bag_file_path, topic_name):
    last_message = None
    last_time = None
    
    # Open the bag file and read through the specified topic
    with rosbag.Bag(bag_file_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            last_message = msg
            last_time = t
    
    return last_message, last_time

def save_pointcloud_to_csv(pointcloud_msg, csv_file_path):
    """
    Converts a PointCloud2 message to CSV format and saves it.

    Args:
        pointcloud_msg (PointCloud2): The PointCloud2 message.
        csv_file_path (str): The path to the CSV file.
    """
    if pointcloud_msg is None:
        rospy.logwarn("No PointCloud2 message found to save.")
        return
    
    # Parse the point cloud data
    points = pc2.read_points(pointcloud_msg, skip_nans=True, field_names=("x", "y", "z"))
    
    # Write points to CSV
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # writer.writerow(["x", "y", "z"])  # Write header
        
        rospy.loginfo(f"Saving PointCloud2 data to '{csv_file_path}'...")
        for point in points:
            writer.writerow(point[:3])  # Only save x, y, z coordinates
    
    rospy.loginfo(f"PointCloud2 data saved to '{csv_file_path}'.")

if __name__ == "__main__":
    # Define bag file path and topic names
    bag_file_path = '/home/pero/Pero - diplomski/OMCO_0705_brusenje/obradjeni_bag/umjeravanje_2.bag'
    input_topic = '/localize_mould/transformed_cloud_pcl'
    csv_file_path = 'data/mould.csv'  # Path to save the CSV
    
    # Retrieve the last message
    last_msg, last_msg_time = get_last_pointcloud_from_bag(bag_file_path, input_topic)
    
    # Save the last message to CSV
    if last_msg:
        save_pointcloud_to_csv(last_msg, csv_file_path)
    else:
        rospy.logwarn("No message found on the specified topic in the bag file.")
