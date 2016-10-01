#!/usr/bin/env python

import copy
import math
import rospy
import argparse
import actionlib
import numpy as np
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from sensor_msgs.point_cloud2 import read_points
from sensor_msgs.msg import PointCloud2, JointState
from strands_navigation_msgs.msg import TopologicalMap
from mongodb_store.message_store import MessageStoreProxy
from simple_change_detector.msg import BaselineDetectionMsg, ChangeDetectionMsg
from simple_change_detector.msg import ChangeDetectionAction, ChangeDetectionResult


class ChangeDetector(object):

    def __init__(self, depth_topic="/head_xtion/depth/points", num_obs=50):
        rospy.loginfo("Initialising change detector...")
        self._counter = 0
        self.num_of_obs = num_obs
        self._baseline = dict()

        rospy.loginfo("Subscribing to %s" % depth_topic)
        self._depth = None
        self._depths = list()
        self._is_learning = False
        self._has_learnt = False
        rospy.Subscriber(depth_topic, PointCloud2, self._depth_cb, None, 10)

        self.topo_map = None
        self._topo_info = dict()
        self._get_topo_info()

        rospy.loginfo("Subscribing to /ptu/state")
        self.ptu = None
        self._ptu_info = dict()
        rospy.Subscriber("/ptu/state", JointState, self._ptu_cb, None, 10)

        rospy.loginfo("Publishing topic %s/detections..." % rospy.get_name())
        self._pub = rospy.Publisher(
            rospy.get_name()+"/detections", ChangeDetectionMsg, queue_size=10
        )
        collection = rospy.get_name()[1:]+"_baseline"
        self._db = MessageStoreProxy(collection=collection)
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
        self.ptu = msg

    def _topo_cb(self, msg):
        self.topo_map = msg

    def _depth_cb(self, msg):
        self._depth = [
            point for point in read_points(
                msg, field_names=("x", "y", "z")
            )
        ]
        if self._is_learning and not self._has_learnt:
            self._depths.append(self._depth)
            rospy.loginfo("Learning %d observation(s)..." % len(self._depths))
        rospy.sleep(0.1)

    def _get_topo_info(self):
        topo_sub = rospy.Subscriber(
            "/topological_map", TopologicalMap, self._topo_cb, None, 10
        )
        rospy.loginfo("Getting information from /topological_map...")
        while self.topo_map is None:
            rospy.sleep(0.1)
        topo_sub.unregister()
        self._topo_info = dict()
        for wp in self.topo_map.nodes:
            self._topo_info[wp.name] = wp

    def _array_to_point(self, array):
        points = [
            Point(i[0], i[1], i[2]) for i in array
        ]
        return points

    def _point_to_array(self, points):
        array = [
            [i.x, i.y, i.z] for i in points
        ]
        return array

    def _load_baseline(self):
        wp = self._topo_info.values()[0]
        rospy.loginfo("Load baseline from db with map name: %s..." % wp.map)
        query = {"topological_node.map": wp.map}
        logs = self._db.query(BaselineDetectionMsg._type, query, {})
        if len(logs) > 0:
            self._has_learnt = True
            for log in logs:
                self._baseline[
                    log[0].topological_node.name
                ] = self._point_to_array(log[0].baseline)
                self._ptu_info[
                    log[0].topological_node.name
                ] = log[0].ptu_state
                self._topo_info[
                    log[0].topological_node.name
                ] = log[0].topological_node

    def _learning(self, goal, start):
        assert len(self._depths) == 0, "len:%d (len should be 0)" % (
            len(self._depths)
        )
        self._is_learning = True
        while len(self._depths) < self.num_of_obs:
            if self.server.is_preempt_requested():
                self._is_learning = False
                self._depths = list()
                return
            rospy.sleep(0.1)
        self._is_learning = False
        baseline = list()
        for base in self._depths:
            if self.server.is_preempt_requested():
                self._depths = list()
                return
            if baseline == list():
                baseline = np.array(base)
                continue
            baseline = baseline + np.array(base)
        baseline = baseline / float(len(self._depth))
        self._baseline[goal.topological_node] = baseline
        self._ptu_info[goal.topological_node] = self.ptu
        baseline = self._array_to_point(baseline)
        self._db.insert(
            BaselineDetectionMsg(
                baseline,
                self._ptu_info[goal.topological_node],
                self._topo_info[goal.topological_node]
            ), {}
        )
        self._depths = list()

    def execute(self, goal):
        # print msg.height, msg.width
        # 480 640
        # print msg.is_bigendian, msg.point_step, msg.row_step, msg.is_dense
        # False, 16, 10240, False
        # print len(msg.data)
        # 4915200
        start = rospy.Time.now()
        if goal.is_learning and not self._has_learnt:
            self._learning(goal, start)
        elif not goal.is_learning:
            self._predicting(goal, start)
        return ChangeDetectionResult()

    def _predicting(self, goal, start):
        counter = [False for i in range(0, 5)]
        while (rospy.Time.now() - start) < goal.duration:
            if self.server.is_preempt_requested():
                return
            data = copy.deepcopy(self._depth)
            mse = [0.0, 0.0, 0.0]
            for ind, point in enumerate(data):
                if self._is_nan(
                    self._baseline[goal.topological_node][ind]
                ) or self._is_nan(point):
                    continue
                mse[0] += (self._baseline[goal.topological_node][0]-point[0])**2
                mse[1] += (self._baseline[goal.topological_node][1]-point[1])**2
                mse[2] += (self._baseline[goal.topological_node][2]-point[2])**2
            rmse = [math.sqrt(i/float(len(data))) for i in mse]
            counter[self._counter % len(counter)] = (sum(rmse)/len(rmse) >= 0.1)
            self._counter += 1
            is_changing = False
            if False not in counter:
                is_changing = True
            self._pub.publish(
                ChangeDetectionMsg(
                    Header(self._counter, rospy.Time.now(), ''),
                    goal.topological_node, is_changing
                )
            )
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
    args = parser_arg.parse_args()
    ChangeDetector(args.depth_topic+"/points", int(args.num_obs))
    rospy.spin()
