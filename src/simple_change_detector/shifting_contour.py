#!/usr/bin/env python

import cv2
import rospy
import argparse
import numpy as np
from sensor_msgs.msg import Image
from scipy.spatial.distance import euclidean
from cv_bridge import CvBridge, CvBridgeError


class ShiftingContour(object):

    def __init__(
        self, topic_img="/head_xtion/rgb/image_raw", min_dist=5, max_dist=100,
        min_contour_area=1600, publish_contour=False, max_centroid_history_size=20
    ):
        self._min_dist = min_dist
        self._max_dist = max_dist
        self._min_contour_area = min_contour_area
        self._max_centroid_history_size = max_centroid_history_size
        self.reset()
        self._bridge = CvBridge()
        rospy.sleep(0.1)
        if publish_contour:
            tmp = topic_img.split("/")
            pub_topic = "/" + tmp[1] + "/" + tmp[2] + "/image_contour"
            self._pub = rospy.Publisher(pub_topic, Image, queue_size=10)
            rospy.Timer(rospy.Duration(0, 100000000), self._print_contours)
        rospy.Subscriber(topic_img, Image, self._img_cb, None, 10)

    def reset(self):
        # CONTOUR STUFF
        rospy.loginfo("Resetting all values...")
        self._img = None
        self.contours = list()
        self._contours = list()
        self._previous_contours = list()
        self._fgbg = cv2.BackgroundSubtractorMOG2()
        # centroid stuff
        self._counter = 0
        self._centroid_history = dict()

    def _img_cb(self, msg):
        try:
            img = self._bridge.imgmsg_to_cv2(msg)
            self._img = self._img_substractor(img)
            self._previous_contours = self._contours
            self._contours = self._find_contours(self._img)
            self.contours = self._find_shifting_contours(
                self._contours, self._previous_contours
            )
        except CvBridgeError as e:
            rospy.logerr(e)
        rospy.sleep(0.05)

    def _img_substractor(self, img):
        # should only be applied to depth/image_raw
        # if "depth" in self._pub.name.split("/"):
        #     fgimg = np.array(img * 255, dtype=np.uint8)
        #     fgimg = cv2.adaptiveThreshold(
        #         fgimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #         cv2.THRESH_BINARY, 11, 2
        #     )
        fgimg = self._fgbg.apply(img)
        # otsu binarization + gaussian filter threshold
        blur = cv2.GaussianBlur(fgimg, (5, 5), 0)
        _, fgimg = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU
        )
        # _, fgimg = cv2.threshold(fgimg, 127, 255, 0)  # std threshold
        # fgimg = cv2.cvtColor(fgimg, cv2.COLOR_GRAY2BGR)  # color transformation
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
        contours = sorted(tmp, key=lambda i: i[1], reverse=True)[:5]
        # adding new centroids to centroid history
        tmp = list()
        for cnt in contours:
            too_close = False
            for cntr in self._centroid_history.values():
                if euclidean(cnt[2], cntr[1]) < self._min_dist:
                    too_close = True
                    tmp.append(cnt)
                    break
            if not too_close and self._max_centroid_history_size > 0:
                self._centroid_history.update({self._counter: (cnt[1], cnt[2])})
                self._counter = (self._counter + 1) % self._max_centroid_history_size
        if len(tmp) > 0:
            contours = [cnt for cnt in contours if cnt[2] not in zip(*tmp)[2]]
        return contours

    def _find_shifting_contours(self, contours, previous_contours):
        # if self._previous_contours is not None and self._previous_contours != list():
        shifting_contours = list()
        if len(previous_contours) > 0 and len(contours) > 0:
            nearest = self._get_nearest_contours(contours, previous_contours)
            if len(nearest) > 0:
                # print nearest[0][0][1], nearest[0][0][2], nearest[0][1][1], nearest[0][1][2]
                # print euclidean(nearest[0][0][2], nearest[0][1][2])
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
            # creating rectangle around the objects
            # for cnt in zip(*self._previous_contours)[0]:
            #     x, y, w, h = cv2.boundingRect(cnt)
            #     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)
        img = cv2.transform(img, np.array([[1, 1, 1]]))
        self._pub.publish(self._bridge.cv2_to_imgmsg(img))


if __name__ == '__main__':
    parser_arg = argparse.ArgumentParser(prog=rospy.get_name())
    parser_arg.add_argument(
        "-t", dest="img_topic", default="/head_xtion/rgb/image_raw",
        help="Image topic to subscribe to (default=/head_xtion/rgb/image_raw)"
    )
    parser_arg.add_argument(
        "-c", dest="centroid_history_size", default="20",
        help="Size of centroid history (default=20)"
    )
    args = parser_arg.parse_args()
    tmp = args.img_topic.split("/")
    name = tmp[1] + "_" + tmp[2] + "_image_contour"
    rospy.init_node("shifting_contour_%s" % name)
    ShiftingContour(
        topic_img=args.img_topic, publish_contour=True,
        max_centroid_history_size=int(args.centroid_history_size)
    )
    rospy.spin()
