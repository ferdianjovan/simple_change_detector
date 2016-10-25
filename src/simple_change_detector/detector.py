#!/usr/bin/env python

import cv2
import copy
import rospy
import roslib
import argparse
import threading
import actionlib
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from strands_navigation_msgs.msg import TopologicalMap
from simple_change_detector.msg import ChangeDetectionMsg
from mongodb_store.message_store import MessageStoreProxy
from simple_change_detector.msg import ChangeDetectionAction
from simple_change_detector.msg import ChangeDetectionResult


class ChangeDetector(object):

    def __init__(
        self, depth_topic="/head_xtion/depth/image_raw", num_obs=50, threshold=5
    ):
        rospy.loginfo(
            "Number observation %d and threshold %.2f..." % (
                num_obs, threshold
            )
        )
        self._is_active = False
        self.threshold = threshold
        self._counter = 0
        self.num_of_obs = num_obs
        self._baseline = np.array(list())
        self._std_dev = np.array(list())
        self._lock = threading.Lock()

        rospy.loginfo("Subscribing to %s" % depth_topic)
        self._depth = None
        self._bridge = CvBridge()
        rospy.Subscriber(depth_topic, Image, self._depth_cb, None, 1)

        self.topo_map = None
        self._get_topo_info()
        self.topo_map = self.topo_map.map

        rospy.loginfo("Publishing topic %s/detections..." % rospy.get_name())
        self._pub = rospy.Publisher(
            rospy.get_name()+"/detections", ChangeDetectionMsg, queue_size=10
        )
        collection = rospy.get_name()[1:]+"_detections"
        self._db_detect = MessageStoreProxy(collection=collection)
        self._load_baseline()
        rospy.loginfo(
            "Creating an action server %s/action..." % rospy.get_name()
        )
        self.server = actionlib.SimpleActionServer(
            rospy.get_name()+"/action", ChangeDetectionAction,
            self.execute, False
        )
        self.server.start()
        rospy.sleep(0.1)

    def _topo_cb(self, msg):
        self.topo_map = msg

    def _get_topo_info(self):
        topo_sub = rospy.Subscriber(
            "/topological_map", TopologicalMap, self._topo_cb, None, 10
        )
        rospy.loginfo("Getting information from /topological_map...")
        while self.topo_map is None:
            rospy.sleep(0.1)
        topo_sub.unregister()

    def _depth_cb(self, msg):
        if self._is_active:
            self._lock.acquire()
            try:
                cv_image = self._bridge.imgmsg_to_cv2(msg, "16UC1")
                self._depth = np.array(cv_image, dtype=np.float32)
                cv2.normalize(self._depth, self._depth, 0, 1, cv2.NORM_MINMAX)
                self._depth = self._depth.reshape(640, 480)
            except CvBridgeError as e:
                rospy.logerr(e)
            self._lock.release()
        rospy.sleep(0.1)

    def _load_baseline(self):
        rospy.loginfo(
            "Load baseline from db with map name: %s..." % self.topo_map
        )
        fname = self.topo_map
        try:
            baseline = np.fromfile(
                roslib.packages.get_pkg_dir(
                    "simple_change_detector"
                ) + ("/config/%s_baseline.yaml" % fname),
                dtype=np.float32
            )
            self._baseline = baseline.reshape(640, 480)
            std_dev = np.fromfile(
                roslib.packages.get_pkg_dir(
                    "simple_change_detector"
                ) + ("/config/%s_std_dev.yaml" % fname),
                dtype=np.float32
            )
            self._std_dev = std_dev.reshape(640, 480)
            rospy.loginfo("Baseline and Std dev for map %s are obtained..." % fname)
        except:
            rospy.sleep(0.1)

    def _learning(self, goal):
        baseline = list()
        # calculate average in each x, y, z
        bases = list()
        for i in range(0, self.num_of_obs):
            self._lock.acquire()
            base = copy.deepcopy(self._depth)
            self._lock.release()
            bases.append(base)
            if self.server.is_preempt_requested() or rospy.is_shutdown():
                return True
            if baseline == list():
                baseline = np.array(base)
            else:
                baseline = baseline + np.array(base)
            rospy.loginfo("Learning %d observation(s)..." % (i+1))
            rospy.sleep(0.1)
        baseline = baseline / float(self.num_of_obs)
        self._baseline = baseline
        self._is_active = False
        # calculate standard deviation in each x, y, z
        std_dev = list()
        for base in bases:
            if std_dev == list():
                std_dev = (np.array(base) - baseline)**2
            else:
                std_dev += (np.array(base) - baseline)**2
        std_dev = np.sqrt(std_dev / float(self.num_of_obs))
        self._std_dev = std_dev
        # storing to file
        fname = self.topo_map
        rospy.loginfo("Storing baseline for map %s..." % fname)
        baseline.tofile(
            roslib.packages.get_pkg_dir(
                "simple_change_detector"
            ) + ("/config/%s_baseline.yaml" % fname)
        )
        rospy.loginfo("Storing std_dev for map %s..." % fname)
        std_dev.tofile(
            roslib.packages.get_pkg_dir(
                "simple_change_detector"
            ) + ("/config/%s_std_dev.yaml" % fname)
        )
        rospy.loginfo("Finish learning...")
        return False

    def execute(self, goal):
        start = rospy.Time.now()
        self._is_active = True
        preempted = False
        while self._depth is None:
            rospy.sleep(0.1)
        if goal.is_learning:
            self._baseline = np.array(list())
            self._std_dev = np.array(list())
            preempted = self._learning(goal)
        elif self._baseline != np.array(list()):
            preempted = self._predicting(start, goal.duration)
        else:
            rospy.logwarn("There is no baseline for this map")
        self._is_active = False
        if preempted:
            rospy.loginfo("The action has been preempted...")
            self.server.set_preempted()
        else:
            rospy.loginfo("The action succeeded...")
            self.server.set_succeeded(ChangeDetectionResult())

    def _predicting(self, start, duration):
        counter = [0.0 for i in range(0, 5)]
        previous_data = None
        while (rospy.Time.now() - start) < duration:
            if self.server.is_preempt_requested() or rospy.is_shutdown():
                return True
            self._lock.acquire()
            data = copy.deepcopy(self._depth)
            self._lock.release()
            outlier_percentage = self._is_changing(
                previous_data, data
            )
            counter[self._counter % len(counter)] = outlier_percentage
            previous_data = data
            self._counter += 1
            is_changing = (0.0 not in counter)
            rospy.loginfo("Is changing right now: %s" % is_changing)
            msg = ChangeDetectionMsg(
                Header(self._counter, rospy.Time.now(), ''),
                is_changing, counter
            )
            self._pub.publish(msg)
            self._db_detect.insert(
                msg, {
                    "threshold": self.threshold,
                    "map": self.topo_map,
                }
            )
            rospy.sleep(0.1)
        return False

    def _is_changing(self, previous, current):
        if previous is None:
            return 0.0
        outliers = np.abs(previous - current) > self._std_dev*2
        outliers = np.array(outliers, dtype=int)
        out_percentage = (outliers.sum() / float(outliers.size)) * 100.0
        if out_percentage < self.threshold:
            out_percentage = 0.0
        # row, column = np.where(outliers==1)
        blob_size = 11
        blob = np.ones([blob_size, blob_size], dtype=int)
        row_idx = 0
        col_idx = 0
        blob_idxs = list()
        while row_idx+blob_size <= outliers.shape[0]:
            while col_idx+blob_size <= outliers.shape[1]:
                is_skipped = False
                for blob_idx in blob_idxs:
                    cond = row_idx >= blob_idx[0][0]
                    cond = cond and row_idx <= blob_idx[1][0]
                    cond = cond and col_idx >= blob_idx[0][1]
                    cond = cond and col_idx <= blob_idx[1][1]
                    if cond:
                        is_skipped = True
                        break
                if is_skipped:
                    col_idx += 1
                    continue
                submatrix = outliers[
                    np.ix_(
                        [i for i in range(row_idx, row_idx+blob_size)],
                        [i for i in range(col_idx, col_idx+blob_size)]
                    )
                ]
                intersect = (blob | submatrix).sum() / float(blob.size)
                if intersect >= 0.75:
                    blob_idxs.append([
                        (row_idx, col_idx),
                        (row_idx+(blob_size-1), col_idx+(blob_size-1))
                    ])
                    col_idx += blob_size
                    continue
                col_idx += 1
            row_idx += 1
        print str(len(blob_idxs)), str(blob_idxs[-4:])
        return out_percentage


if __name__ == '__main__':
    rospy.init_node("change_detector")
    parser_arg = argparse.ArgumentParser(prog=rospy.get_name())
    parser_arg.add_argument(
        "-d", dest="depth_topic", default="/head_xtion/depth",
        help="Depth topic to subscribe to (default=/head_xtion/depth)"
    )
    parser_arg.add_argument(
        "-o", dest="num_obs", default="100",
        help="The number of observations for baseline (default=100)"
    )
    parser_arg.add_argument(
        "-t", dest="threshold", default="5",
        help="Threshold percentage for saying something has changed (default=5)"
    )
    args = parser_arg.parse_args()
    ChangeDetector(
        args.depth_topic+"/image_raw", int(args.num_obs), float(args.threshold)
    )
    rospy.spin()
