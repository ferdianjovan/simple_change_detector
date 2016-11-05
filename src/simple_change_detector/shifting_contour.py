#!/usr/bin/env python

import cv2
import rospy
import argparse
import numpy as np
from sensor_msgs.msg import Image
from scipy.spatial.distance import euclidean
from cv_bridge import CvBridge, CvBridgeError
from upper_body_detector.msg import UpperBodyDetector
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
        self._bridge = CvBridge()
        if publish_contour:
            tmp = topic_img.split("/")
            pub_topic = "/" + tmp[1] + "/" + tmp[2] + "/image_contour"
            self._pub = rospy.Publisher(pub_topic, Image, queue_size=10)
            rospy.Timer(rospy.Duration(0, 100000000), self._print_contours)
        rospy.Subscriber(topic_img, Image, self._img_cb, None, 2)

    def reset(self):
        # CONTOUR STUFF
        rospy.loginfo("Resetting all values...")
        self._counter = 0
        self._img = None
        self._img_color = None
        self.contours = list()
        self._contours = list()
        self._previous_contours = list()
        self._fgbg = cv2.BackgroundSubtractorMOG2()
        # baseline image stuff
        self._base = BaselineImage(sample_size=self._sample_size)

    def _img_cb(self, msg):
        try:
            self._img_color = self._bridge.imgmsg_to_cv2(msg)
            if self._base.baseline is not None:
                self._img = self._img_substractor(self._img_color)
                # self._previous_contours = self._contours
                # self._contours = self._find_contours(self._img)
                # self.contours = self._find_shifting_contours(
                #     self._contours, self._previous_contours
                # )
                self.contours = self._find_contours(self._img)
            else:
                self._base.get_baseline(self._img_color)
        except CvBridgeError as e:
            rospy.logerr(e)
        rospy.sleep(0.05)

    def _img_substractor(self, img):
        if "depth" in self._pub.name.split("/"):
            img = np.array(img, dtype=np.uint8)
            # img = cv2.fastNlMeansDenoising(img, None, 10, 10, 7, 21)
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            _, fgimg = cv2.threshold(
                blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU
            )
        fgimg = self._fgbg.apply(img)
        self._counter = (self._counter + 1) % self._sample_size
        if self._counter == 0:
            self._fgbg = cv2.BackgroundSubtractorMOG2()
        return fgimg

    def _find_contours(self, img):
        contours, _ = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        tmp = list()
        for cnt in contours:
            m = cv2.moments(cnt)
            if m['m00'] > self._min_contour_area:  # area greater than 30 x 30 pixels
                # storing contour, its area, and its centroid
                centroid = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
                tmp.append((cnt, m['m00'], centroid))
        contours = sorted(tmp, key=lambda i: i[1], reverse=True)
        try:
            contours = [
                cnt for cnt in contours if self._base.is_contour_deviated(self._img_color, cnt[0])
            ]
        except TypeError as e:
            rospy.logerr("Baseline has been reset, no value can be accessed!")
        return contours

    def _find_shifting_contours(self, contours, previous_contours):
        shifting_contours = list()
        if len(previous_contours) > 0 and len(contours) > 0:
            nearest = self._get_nearest_contours(contours, previous_contours)
            if len(nearest) > 0:
                shifting_contours = zip(*nearest)[0]
        return shifting_contours

    def _get_nearest_contours(self, contours1, contours2):
        # comparing two sets of contours if there is a movement
        # returning a pair of closest contours (contours1, contours2)
        nearest_contours = list()
        for cnt1 in contours1:
            closest = None
            for cnt2 in contours2:
                dist = euclidean(cnt1[2], cnt2[2])
                if self._min_dist <= dist and dist <= self._max_dist:
                    if closest is None or dist < euclidean(cnt1[2], closest[2]):
                        closest = cnt2
            if closest is not None:
                nearest_contours.append((cnt1, closest))
        nearest_contours = sorted(nearest_contours, key=lambda i: i[0][1], reverse=True)
        return nearest_contours

    def _print_contours(self, event):
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
