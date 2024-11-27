import rospy
import rosbag
from sensor_msgs.msg import PointCloud2

def get_last_pointcloud_from_bag(bag_file_path, topic_name):
    last_message = None
    last_time = None
    
    # Open the bag file and read through the specified topic
    with rosbag.Bag(bag_file_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            last_message = msg
            last_time = t
    
    return last_message, last_time

def publish_pointcloud(last_msg, output_topic):
    # Initialize ROS node
    rospy.init_node('mould_publisher', anonymous=True)
    
    # Create a publisher for the PointCloud2 message
    pub = rospy.Publisher(output_topic, PointCloud2, queue_size=10)
    
    # Check if last_msg exists
    if last_msg is not None:
        rospy.loginfo(f"Publishing last PointCloud2 message from topic '{output_topic}'")
        
        rospy.sleep(1)
        # Publish the message
        pub.publish(last_msg)
        
        # Allow some time to publish
        rospy.spin()
    else:
        rospy.logwarn("No PointCloud2 message found to publish")

if __name__ == "__main__":
    # Define bag file path and topic names
    bag_file_path = '/home/pero/Pero - diplomski/OMCO_0705_brusenje/obradjeni_bag/umjeravanje_2.bag'
    input_topic = '/localize_mould/transformed_cloud_pcl'
    output_topic = '/mould'
    
    # Retrieve the last message
    last_msg, last_msg_time = get_last_pointcloud_from_bag(bag_file_path, input_topic)
    
    # Publish the last message
    if last_msg:
        publish_pointcloud(last_msg, output_topic)
    else:
        rospy.logwarn("No message found on the specified topic in the bag file.")
