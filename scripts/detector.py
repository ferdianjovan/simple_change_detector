#!/usr/bin/env python

import rospy
from simple_change_detector.stationary_shifting_detection import StationaryShiftingDetection


class DetectorManager(object):

    def __init__(self):
        img_topic = rospy.get_param("~img_topic", "/head_xtion/rgb/image_raw")
        centroid_history_size = rospy.get_param("~centroid_history_size", 20)
        wait_time = rospy.get_param("~wait_time", 5)
        self.ssd = StationaryShiftingDetection(
            topic_img=img_topic, max_centroid_history_size=centroid_history_size,
            wait_time=wait_time
        )
        self.ssd.publish_shifting_message()
        rospy.spin()


if __name__ == '__main__':
    img_topic = rospy.get_param("~img_topic", "/head_xtion/rgb/image_raw")
    tmp = img_topic.split("/")
    rospy.init_node("%s_change_detection" % tmp[1])
    dm = DetectorManager()
