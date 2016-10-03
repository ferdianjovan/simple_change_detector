#!/usr/bin/env python

import copy
import math
import rospy
import roslib
import argparse
import threading
import actionlib
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.point_cloud2 import read_points
from sensor_msgs.msg import PointCloud2, JointState
from scitos_ptu.msg import PtuGotoAction, PtuGotoGoal
from strands_navigation_msgs.msg import TopologicalMap
from simple_change_detector.msg import ChangeDetectionMsg
from mongodb_store.message_store import MessageStoreProxy
from simple_change_detector.msg import ChangeDetectionAction
from simple_change_detector.msg import ChangeDetectionResult


class ChangeDetector(object):

    def __init__(
        self, depth_topic="/head_xtion/depth/points", num_obs=50, threshold=0.05
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
        self._baseline = dict()
        self._std_dev = dict()
        self._upbound = dict()
        self._lowbound = dict()
        self._lock = threading.Lock()

        rospy.loginfo("Subscribing to %s" % depth_topic)
        self._depth = None
        rospy.Subscriber(depth_topic, PointCloud2, self._depth_cb, None, 1)

        self.topo_map = None
        self._topo_info = list()
        self._get_topo_info()

        self.ptu = None
        rospy.loginfo("Connecting with /SetPTUState action server...")
        self.ptu_action = actionlib.SimpleActionClient(
            "SetPTUState", PtuGotoAction
        )
        self.ptu_action.wait_for_server()
        self._ptu_info = dict()

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

    def _ptu_cb(self, msg):
        if self._is_active:
            self.ptu = msg
        rospy.sleep(0.1)

    def _topo_cb(self, msg):
        self.topo_map = msg

    def _depth_cb(self, msg):
        if self._is_active:
            self._lock.acquire()
            self._depth = [
                point for point in read_points(msg, field_names=("x", "y", "z"))
            ]
            self._lock.release()
        rospy.sleep(0.1)

    def _get_topo_info(self):
        topo_sub = rospy.Subscriber(
            "/topological_map", TopologicalMap, self._topo_cb, None, 10
        )
        rospy.loginfo("Getting information from /topological_map...")
        while self.topo_map is None:
            rospy.sleep(0.1)
        topo_sub.unregister()
        for wp in self.topo_map.nodes:
            self._topo_info.append(wp.name)
        self.topo_map = self.topo_map.map

    def _load_baseline(self):
        rospy.loginfo(
            "Load baseline from db with map name: %s..." % self.topo_map
        )
        for wp in self._topo_info:
            fname = self.topo_map + "_" + wp
            try:
                baseline = np.fromfile(
                    roslib.packages.get_pkg_dir(
                        "simple_change_detector"
                    ) + ("/config/%s_baseline.yaml" % fname),
                    dtype=np.float64
                )
                self._baseline[wp] = baseline.reshape(baseline.size/3, 3)
                std_dev = np.fromfile(
                    roslib.packages.get_pkg_dir(
                        "simple_change_detector"
                    ) + ("/config/%s_std_dev.yaml" % fname),
                    dtype=np.float64
                )
                self._std_dev[wp] = std_dev.reshape(std_dev.size/3, 3)
                self._ptu_info[wp] = np.fromfile(
                    roslib.packages.get_pkg_dir(
                        "simple_change_detector"
                    ) + ("/config/%s_ptu.yaml" % fname),
                    dtype=np.float64
                )
                rospy.loginfo("Data for node %s are obtained..." % wp)
                rospy.loginfo("Calculate lower and upper bound at each node...")
                self._calculate_low_up_bound(wp)
            except:
                rospy.sleep(0.1)

    def _calculate_low_up_bound(self, topo_node):
        baseline = self._baseline[topo_node]
        std_dev = self._std_dev[topo_node]
        self._upbound[topo_node] = baseline + 2*std_dev
        self._lowbound[topo_node] = baseline - 2*std_dev

    def _learning(self, goal):
        baseline = list()
        rospy.loginfo("Subscribing to /ptu/state")
        subs = rospy.Subscriber(
            "/ptu/state", JointState, self._ptu_cb, None, 10
        )
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
        self._baseline[goal.topological_node] = baseline
        self._is_active = False
        self._ptu_info[goal.topological_node] = np.array(self.ptu.position)
        subs.unregister()
        # calculate standard deviation in each x, y, z
        std_dev = list()
        for base in bases:
            if std_dev == list():
                std_dev = (np.array(base) - baseline)**2
            else:
                std_dev += (np.array(base) - baseline)**2
        std_dev = np.sqrt(std_dev / float(self.num_of_obs))
        self._std_dev[goal.topological_node] = std_dev
        # storing to file
        rospy.loginfo("Storing ptu for %s..." % goal.topological_node)
        fname = self.topo_map + "_" + goal.topological_node
        np.array(self.ptu.position).tofile(
            roslib.packages.get_pkg_dir(
                "simple_change_detector"
            ) + ("/config/%s_ptu.yaml" % fname)
        )
        rospy.loginfo("Storing baseline for %s..." % goal.topological_node)
        baseline.tofile(
            roslib.packages.get_pkg_dir(
                "simple_change_detector"
            ) + ("/config/%s_baseline.yaml" % fname)
        )
        rospy.loginfo("Storing std_dev for %s..." % goal.topological_node)
        std_dev.tofile(
            roslib.packages.get_pkg_dir(
                "simple_change_detector"
            ) + ("/config/%s_std_dev.yaml" % fname)
        )
        # calculate upper and lower bound confidence
        rospy.loginfo("Calculate lower and upper bound at each node...")
        self._calculate_low_up_bound(goal.topological_node)
        rospy.loginfo("Finish learning...")
        return False

    def execute(self, goal):
        start = rospy.Time.now()
        self._is_active = True
        preempted = False
        while self._depth is None:
            rospy.sleep(0.1)
        if goal.is_learning:
            self._baseline[goal.topological_node] = list()
            self._std_dev[goal.topological_node] = list()
            self._ptu_info[goal.topological_node] = None
            preempted = self._learning(goal)
        elif goal.topological_node in self._baseline:
            preempted = self._predicting(goal, start)
        else:
            rospy.logwarn(
                "There is no baseline for change detection in %s" % (
                    goal.topological_node
                )
            )
        self._is_active = False
        if preempted:
            rospy.loginfo("The action has been preempted...")
            self.server.set_preempted()
        else:
            rospy.loginfo("The action succeeded...")
            self.server.set_succeeded(ChangeDetectionResult())

    def _predicting(self, goal, start):
        wp = goal.topological_node
        counter = [False for i in range(0, 1)]
        self._moving_ptu(
            self._ptu_info[wp],
            start, goal.duration
        )
        while (rospy.Time.now() - start) < goal.duration:
            if self.server.is_preempt_requested() or rospy.is_shutdown():
                return True
            self._lock.acquire()
            data = copy.deepcopy(self._depth)
            self._lock.release()
            outliers = list()
            higher = np.array(data).flatten() > self._upbound[wp].flatten()
            lower = np.array(data).flatten() < self._lowbound[wp].flatten()
            outliers = np.array(higher, dtype=int) + np.array(lower, dtype=int)
            outliers = np.array(np.array(outliers, dtype=bool), dtype=int)
            out_percentage = sum(outliers) / float(len(outliers))
            rospy.loginfo("Outlier percentage: %.2f" % out_percentage)
            counter[self._counter % len(counter)] = (
                out_percentage >= self.threshold
            )
            self._counter += 1
            is_changing = False
            if False not in counter:
                is_changing = True
            msg = ChangeDetectionMsg(
                Header(self._counter, rospy.Time.now(), ''),
                wp, is_changing
            )
            self._pub.publish(msg)
            self._db_detect.insert(
                msg, {
                    "threshold": self.threshold,
                    "map": self.topo_map,
                    "ptu_pan": self._ptu_info[wp][0],
                    "ptu_tilt": self._ptu_info[wp][1]
                }
            )
            rospy.sleep(0.1)
        return False

    def _moving_ptu(self, ptu_pos, start, duration):
        pan = float(ptu_pos[0]*180.0/math.pi)
        tilt = float(ptu_pos[1]*180/math.pi)
        rospy.loginfo(
            "Moving ptu %.2f pan and %.2f tilt..." % (pan, tilt)
        )
        self.ptu_action.send_goal(PtuGotoGoal(pan, tilt, 30, 30))
        self.ptu_action.wait_for_result()
        result = self.ptu_action.get_result()
        while len(result.state.position) == 0 and (rospy.Time.now() - start) < duration:
            if self.server.is_preempt_requested() or rospy.is_shutdown():
                return
            rospy.logwarn("PTU does not seem to move, try again...")
            self.ptu_action.send_goal(PtuGotoGoal(pan, tilt, 30, 30))
            self.ptu_action.wait_for_result()
            result = self.ptu_action.get_result()
            rospy.sleep(0.1)
        rospy.sleep(1)

    def _is_nan(self, point):
        return math.isnan(point[0]) or math.isnan(point[1]) or math.isnan(
            point[2]
        )


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
        "-t", dest="threshold", default="0.05",
        help="Threshold for saying something has changed (default=0.05)"
    )
    args = parser_arg.parse_args()
    ChangeDetector(
        args.depth_topic+"/points", int(args.num_obs), float(args.threshold)
    )
    rospy.spin()
