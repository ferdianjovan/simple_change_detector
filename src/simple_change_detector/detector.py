#!/usr/bin/env python

import math
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import read_points


class ChangeDetector(object):

    def __init__(self, depth_topic="/head_xtion/depth/points"):
        rospy.loginfo("Initialising change detector...")
        self._depth = {"WayPoint3": list()}
        self._is_learning = True
        self._data = list()
        rospy.Subscriber(depth_topic, PointCloud2, self._depth_cb, None, 10)
        rospy.sleep(0.1)

    def _depth_cb(self, msg):
        # print msg.height, msg.width
        # 480 640
        # print msg.is_bigendian, msg.point_step, msg.row_step, msg.is_dense
        # False, 16, 10240, False
        # print len(msg.data)
        # 4915200
        data = list()
        for point in read_points(msg, field_names=("x", "y", "z")):
            data.append(point)
        if self._is_learning:
            if len(self._depth["WayPoint3"]) < 20:
                self._depth["WayPoint3"].append(data)
                print len(self._depth["WayPoint3"])
                rospy.sleep(1)
            if len(self._depth["WayPoint3"]) >= 20:
                for base in self._depth["WayPoint3"]:
                    if self._data == list():
                        self._data = np.array(base)
                    else:
                        self._data = self._data + np.array(base)
                self._data = self._data / float(len(self._depth["WayPoint3"]))
                self._is_learning = False
        else:
            # smallest_rmse = [float("inf"), float("inf"), float("inf")]
            # for base in self._depth["WayPoint3"]:
            mse = [0.0, 0.0, 0.0]
            for ind, point in enumerate(data):
                if self._is_nan(self._data[ind]) or self._is_nan(point):
                    continue
                mse[0] += (self._data[ind][0] - point[0])**2
                mse[1] += (self._data[ind][0] - point[0])**2
                mse[2] += (self._data[ind][0] - point[0])**2
            rmse = [math.sqrt(i/float(len(data))) for i in mse]
            print rmse
            #     if sum(rmse) < sum(smallest_rmse):
            #         smallest_rmse = rmse
            # print smallest_rmse
            rospy.sleep(0.1)

    def _is_nan(self, point):
        return math.isnan(point[0]) or math.isnan(point[1]) or math.isnan(
            point[2]
        )

if __name__ == '__main__':
    rospy.init_node("change_detector")
    ChangeDetector()
    rospy.spin()
