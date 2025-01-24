#!/usr/bin/env python3

import rospy
import rospkg
import numpy as np
import pose_conversions as P

if __name__ == "__main__":
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('semicyl_painter')
    file_path = f"{package_path}/../rf-loc/config/S00523_M_transform.txt"
    T_mould = np.loadtxt(file_path)
    print(P.matrix_to_pose(T_mould))

