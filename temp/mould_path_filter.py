import csv

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
                header = next(reader)  # Read the header
                filtered_points.append(header)  # Include header in output CSV

                # Loop through each row in the CSV and filter points
                for row in reader:
                    x, y, z = float(row[0]), float(row[1]), float(row[2])

                    # Only include points where x >= -0.35 and z >= 0.348
                    if x >= -0.35 and z >= 0.348:
                        # Apply rotation: rotate by 90 degrees around the x-axis
                        x_rot = z
                        y_rot = y
                        z_rot = -x
                        filtered_points.append([x_rot, y_rot, z_rot])

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
