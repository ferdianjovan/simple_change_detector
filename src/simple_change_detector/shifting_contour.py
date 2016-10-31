#!/usr/bin/env python

import cv2
import math
import rospy
import argparse
import numpy as np
from sensor_msgs.msg import Image
from scipy.spatial.distance import euclidean
from cv_bridge import CvBridge, CvBridgeError
from upper_body_detector.msg import UpperBodyDetector


class ShiftingContour(object):

    def __init__(
        self, topic_img="/head_xtion/rgb/image_raw", min_dist=5, max_dist=100,
        min_contour_area=1600, publish_contour=False, max_centroid_history_size=20
    ):
        self._h_factor = 5
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
        rospy.Subscriber(
            "/upper_body_detector/detections", UpperBodyDetector, self._ubd_cb, None, 10
        )

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
        # ubd stuff for excepting people from being ignored by contour
        self._ubd_pos = list()

    def _ubd_cb(self, ubd):
        if len(ubd.pos_x) > 0:
            self._ubd_pos = zip(ubd.pos_x, ubd.pos_y, ubd.width, ubd.height)

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
        # remove noise with local means denoising (slow)
        # if "depth" in self._pub.name.split("/"):
        #     img = cv2.fastNlMeansDenoising(img, None, 10, 10, 7, 21)
        # else:
        #     img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        fgimg = self._fgbg.apply(img)
        # remove coloured noise
        # fgimg = cv2.adaptiveThreshold(
        #     fgimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #     cv2.THRESH_BINARY, 11, 2
        # )
        # otsu binarization + gaussian filter threshold for BW img
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
        # contour exception for those inside ubd rect
        exceptional_contours = list()
        for cnt in contours:
            # for pos in self._ubd_pos.values():
            for pos in self._ubd_pos:
                is_inside = pos[0] <= cnt[2][0] and cnt[2][0] <= pos[0]+pos[2]
                is_inside = is_inside and pos[1] <= cnt[2][1]
                is_inside = is_inside and cnt[2][1] <= pos[1]+(self._h_factor*pos[3])
                if is_inside:
                    exceptional_contours.append(cnt)
        # adding new centroids to centroid history
        tmp = list()
        for cnt in contours:
            if cnt in exceptional_contours:
                continue
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
