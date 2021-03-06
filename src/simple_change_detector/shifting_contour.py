#!/usr/bin/env python

import cv2
import rospy
import argparse
import numpy as np
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2
from simple_change_detector.baseline import BaselineImage


class ShiftingContour(object):

    def __init__(
        self, topic_img="/head_xtion/rgb/image_raw", min_dist=5, max_dist=100,
        min_contour_area=900, publish_contour=False, sample_size=20
    ):
        self._min_dist = min_dist
        self._max_dist = max_dist
        self._sample_size = sample_size
        self._min_contour_area = min_contour_area
        self.reset()
        self._pause = False
        self._bridge = CvBridge()
        if publish_contour:
            tmp = topic_img.split("/")
            pub_topic = "/" + tmp[1] + "/" + tmp[2] + "/image_contour"
            self._pub = rospy.Publisher(pub_topic, Image, queue_size=10)
            rospy.Timer(rospy.Duration(0, 50000000), self._print_contours)
        tmp = topic_img.split("/")
        subs = [
            message_filters.Subscriber(topic_img, Image),
            message_filters.Subscriber(tmp[1]+"/depth/points", PointCloud2)
        ]
        ts = message_filters.ApproximateTimeSynchronizer(
            subs, queue_size=1, slop=0.15
        )
        ts.registerCallback(self._img_cb)

    def reset(self):
        # CONTOUR STUFF
        rospy.loginfo("Resetting all values...")
        self._counter = 0
        self._img = None
        self._img_color = None
        self._depth = None
        self.contours = list()
        self._contours = list()
        self._previous_contours = list()
        self._fgbg = cv2.BackgroundSubtractorMOG2()
        # baseline image stuff
        self._base = BaselineImage(sample_size=self._sample_size)

    def _img_cb(self, img, depth):
        if not self._pause:
            try:
                self._img_color = self._bridge.imgmsg_to_cv2(img)
                self._depth = depth
                if self._base.baseline is not None:
                    self._img = self._fgbg.apply(self._img_color)
                    self._counter = (self._counter + 1) % self._sample_size
                    if self._counter == 0:
                        self._fgbg = cv2.BackgroundSubtractorMOG2()
                    self.contours = self._find_contours(self._img)
                else:
                    self._base.get_baseline(self._img_color)
            except CvBridgeError as e:
                rospy.logerr(e)
        rospy.sleep(0.1)

    def _find_contours(self, img):
        contours, _ = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        tmp = list()
        for cnt in contours:
            m = cv2.moments(cnt)
            # area greater than 30 x 30 pixels
            if m['m00'] > self._min_contour_area:
                # storing contour, and its centroid on depth
                centroid = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
                tmp.append((cnt, centroid))
        contours = tmp
        try:
            contours = [
                cnt for cnt in contours if self._base.is_contour_deviated(
                    self._img_color, cnt[0]
                )
            ]
        except TypeError:
            rospy.logerr("Baseline has been reset, no value can be accessed!")
        return contours

    def _print_contours(self, event):
        if not self._pause:
            if self._img is None:
                return
            # create black image
            _, img = cv2.threshold(self._img, 255, 255, 0)
            img = np.array(img, dtype=np.float32)
            cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # adding contours
            if len(self.contours) > 0:
                contours = zip(*self.contours)[0]
                cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
            img = cv2.transform(img, np.array([[1, 1, 1]]))
            self._pub.publish(self._bridge.cv2_to_imgmsg(img))


if __name__ == '__main__':
    parser_arg = argparse.ArgumentParser(prog=rospy.get_name())
    parser_arg.add_argument(
        "-t", dest="img_topic", default="/head_xtion/rgb/image_raw",
        help="Image topic to subscribe to (default=/head_xtion/rgb/image_raw)"
    )
    parser_arg.add_argument(
        "-s", dest="sample_size", default="20",
        help="The number of sampling (default=20)"
    )
    args = parser_arg.parse_args()
    tmp = args.img_topic.split("/")
    name = tmp[1] + "_" + tmp[2] + "_image_contour"
    rospy.init_node("shifting_contour_%s" % name)
    ShiftingContour(
        topic_img=args.img_topic, publish_contour=True,
        sample_size=int(args.sample_size)
    )
    rospy.spin()
