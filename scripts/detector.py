#!/usr/bin/env python

import rospy
from simple_change_detector.stationary_shifting_detection import StationaryShiftingDetection


class DetectorManager(object):

    def __init__(self):
        img_topic = rospy.get_param("~img_topic", "/head_xtion/rgb/image_raw")
        sample_size = rospy.get_param("~sample_size", 20)
        wait_time = rospy.get_param("~wait_time", 5)
        publish_image = rospy.get_param("~publish_image", True)
        save_mode = rospy.get_param("~save_mode", False)
        self.ssd = StationaryShiftingDetection(
            topic_img=img_topic, sample_size=sample_size, wait_time=wait_time,
            publish_image=publish_image, save_mode=save_mode
        )
        self.ssd.publish_shifting_message()


if __name__ == '__main__':
    img_topic = rospy.get_param("~img_topic", "/head_xtion/rgb/image_raw")
    tmp = img_topic.split("/")
    rospy.init_node("%s_change_detection" % tmp[1])
    dm = DetectorManager()
    rospy.spin()
