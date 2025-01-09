import csv
import math

class PointCloudSaver:
    def __init__(self, input_csv, output_csv):
        self.input_csv = input_csv
        self.output_csv = output_csv

    def read_and_filter_csv(self):
        """
        Reads the points from the input CSV file, removes points where x < -0.35 and z < 0.348,
        and rotates the points by 90 degrees around the x-axis.
        
        Returns:
            list: A list of (x', y', z') tuples after rotation.
        """
        filtered_points = []
        try:
            with open(self.input_csv, "r") as infile:
                reader = csv.reader(infile)
                # header = next(reader)  # Read the header
                # filtered_points.append(header)  # Include header in output CSV

                points = []  # To store the rotated points
                # Loop through each row in the CSV and filter points
                for row in reader:
                    x, y, z = float(row[0]), float(row[1]), float(row[2])

                    # Only include points where x >= -0.35 and z >= 0.348
                    if x >= -0.35 and z >= 0.348:
                        # Apply rotation: rotate by 90 degrees around the y-axis
                        x_rot = z
                        y_rot = y
                        z_rot = -x
                        points.append([x_rot, y_rot, z_rot])

                if len(points) > 1:
                    # Sort points by ascending y value
                    points.sort(key=lambda point: point[1])

                    # Calculate the angle for Z-axis rotation
                    x1, y1, z1 = points[0]
                    x2, y2, z2 = points[-1]
                    theta = math.atan2(y2 - y1, x2 - x1)

                    # Rotate points around the Z-axis
                    for i in range(len(points)):
                        x, y, z = points[i]
                        x_new = x * math.cos(theta) - y * math.sin(theta)
                        y_new = x * math.sin(theta) + y * math.cos(theta)
                        # points[i] = [x_new, y_new, z]
                        points[i] = [x1, y, z]
                    
                    # Smooth points using a moving average
                    smoothed_points = []
                    window_size = 3  # Define the window size for smoothing
                    half_window = window_size // 2

                    for i in range(len(points)):
                        x_sum, y_sum, z_sum = 0, 0, 0
                        count = 0

                        # Compute the average within the window
                        for j in range(max(0, i - half_window), min(len(points), i + half_window + 1)):
                            x_sum += points[j][0]
                            y_sum += points[j][1]
                            z_sum += points[j][2]
                            count += 1

                        smoothed_points.append([
                            x_sum / count,
                            y_sum / count,
                            z_sum / count
                        ])

                    filtered_points.extend(smoothed_points)
                else:
                    # If there's only one point, just append it without modification
                    filtered_points.extend(points)

        except Exception as e:
            print(f"Error reading CSV file: {e}")
        return filtered_points

    def save_filtered_points(self):
        """
        Saves the filtered and rotated points into the output CSV file.
        """
        filtered_points = self.read_and_filter_csv()
        if not filtered_points:
            print("No filtered points to save.")
            return
        
        try:
            # Open the output CSV file and write the filtered points
            with open(self.output_csv, mode='w', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerows(filtered_points)
            print(f"Filtered and rotated points saved to {self.output_csv}")
        
        except Exception as e:
            print(f"Error saving CSV file: {e}")


def main():
    input_csv = "data/mould.csv"
    output_csv = "data/mould_filtered_path.csv"
    pointcloud_saver = PointCloudSaver(input_csv, output_csv)

    # Save the filtered and rotated points into the output CSV file
    pointcloud_saver.save_filtered_points()


if __name__ == "__main__":
    main()
