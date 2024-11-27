import rospy
import struct
import csv
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


def create_pointcloud2(points, frame_id="world"):
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


class PointCloudPublisher:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.publisher = rospy.Publisher('/mould', PointCloud2, queue_size=10)

    def read_csv(self):
        """
        Reads the points from a CSV file.

        Returns:
            list: A list of (x, y, z) tuples.
        """
        points = []
        try:
            with open(self.csv_file, "r") as file:
                reader = csv.reader(file)
                for row in reader:
                    points.append((float(row[0]), float(row[1]), float(row[2])))
        except Exception as e:
            rospy.logerr(f"Error reading CSV file: {e}")
        return points

    def publish_pointcloud(self):
        """
        Publishes the PointCloud2 message.
        """
        points = self.read_csv()
        if not points:
            rospy.logerr("No points found in the CSV file.")
            return
        pointcloud = create_pointcloud2(points)
        self.publisher.publish(pointcloud)
        rospy.loginfo(f"Published PointCloud2 message with {len(points)} points.")


def main():
    rospy.init_node('pointcloud_publisher', anonymous=True)
    csv_file = "data/mould.csv"
    pointcloud_publisher = PointCloudPublisher(csv_file)

    rate = rospy.Rate(1)  # Publish at 1 Hz
    while not rospy.is_shutdown():
        pointcloud_publisher.publish_pointcloud()
        rate.sleep()


if __name__ == "__main__":
    main()
