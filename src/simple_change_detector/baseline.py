#!/usr/bin/env python

import cv2
import rospy
import numpy as np


class BaselineImage(object):

    def __init__(self, sample_size=20, deviate_value=0.01):
        self._counter = 0
        self.std_dev = None
        self.baseline = None
        self.sample_size = sample_size
        self._deviate_value = deviate_value
        self._imgs = [None for i in range(sample_size)]

    def get_baseline(self, img):
        rospy.loginfo("Observing scene, %d observations..." % self._counter)
        self._imgs[self._counter] = img
        self._counter = (self._counter + 1) % self.sample_size
        if None not in self._imgs:
            rospy.loginfo("Calculating baseline of the scene...")
            self.baseline = sum(self._imgs) / float(self.sample_size)
            self.std_dev = np.sqrt(sum((self._imgs - self.baseline)**2))

    def is_contour_deviated(self, img, contour):
        # creating rectangle around the contour
        x, y, w, h = cv2.boundingRect(contour)
        y1, x1 = max(y, 0), max(x, 0)
        y2, x2 = y+h, x+w
        img_cnt = img[y1:y2, x1:x2]
        base_cnt = self.baseline[y1:y2, x1:x2]
        diff = np.abs(img_cnt - base_cnt) > (self.std_dev[y1:y2, x1:x2])
        diff = np.array(diff, dtype=int)
        return (np.sum(diff) / float(diff.size)) >= self._deviate_value
