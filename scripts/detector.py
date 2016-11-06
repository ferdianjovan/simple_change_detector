#!/usr/bin/env python

import rospy
from simple_change_detector.stationary_shifting_detection import StationaryShiftingDetection


class DetectorManager(object):

    def __init__(self):
        img_topic = rospy.get_param("~img_topic", "/head_xtion/rgb/image_raw")
        sample_size = rospy.get_param("~sample_size", 20)
        wait_time = rospy.get_param("~wait_time", 5)
        soma_config = rospy.get_param("~soma_config", "")
        self.ssd = StationaryShiftingDetection(
            topic_img=img_topic, sample_size=sample_size, wait_time=wait_time,
            soma_config=soma_config
        )
        self.ssd.publish_shifting_message()


if __name__ == '__main__':
    img_topic = rospy.get_param("~img_topic", "/head_xtion/rgb/image_raw")
    tmp = img_topic.split("/")
    rospy.init_node("%s_change_detection" % tmp[1])
    dm = DetectorManager()
    rospy.spin()
