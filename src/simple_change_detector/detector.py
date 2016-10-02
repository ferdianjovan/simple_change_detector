#!/usr/bin/env python

import copy
import math
import rospy
import argparse
import threading
import actionlib
import numpy as np
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from sensor_msgs.point_cloud2 import read_points
from sensor_msgs.msg import PointCloud2, JointState
from scitos_ptu.msg import PtuGotoAction, PtuGotoGoal
from strands_navigation_msgs.msg import TopologicalMap
from mongodb_store.message_store import MessageStoreProxy
from simple_change_detector.msg import ChangeDetectionAction
from simple_change_detector.msg import ChangeDetectionResult
from simple_change_detector.msg import BaselineDetectionMsg, ChangeDetectionMsg


class ChangeDetector(object):

    def __init__(
        self, depth_topic="/head_xtion/depth/points", num_obs=50, threshold=0.05
    ):
        rospy.loginfo(
            "Initialising change detector with num observation %d and threshold %.2f..." % (
                num_obs, threshold
            )
        )
        self._is_active = False
        self.threshold = threshold
        self._counter = 0
        self.num_of_obs = num_obs
        self._baseline = dict()
        self._lock = threading.Lock()

        rospy.loginfo("Subscribing to %s" % depth_topic)
        self._depth = None
        rospy.Subscriber(depth_topic, PointCloud2, self._depth_cb, None, 10)

        self.topo_map = None
        self._topo_info = dict()
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
        rospy.sleep(1)

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
            rospy.loginfo("%d entries are being obtained..." % len(logs))
            for log in logs:
                if log[0].topological_node.name in self._ptu_info:
                    if log[0].ptu_state.header.stamp < self._ptu_info[
                        log[0].topological_node.name
                    ].header.stamp:
                        continue
                    else:
                        rospy.loginfo(
                            "A newer baseline for %s is found, updating..." % log[0].topological_node.name
                        )
                self._baseline[
                    log[0].topological_node.name
                ] = self._point_to_array(log[0].baseline)
                self._ptu_info[
                    log[0].topological_node.name
                ] = log[0].ptu_state
                self._topo_info[
                    log[0].topological_node.name
                ] = log[0].topological_node

    def _learning(self, goal):
        baseline = list()
        rospy.loginfo("Subscribing to /ptu/state")
        subs = rospy.Subscriber(
            "/ptu/state", JointState, self._ptu_cb, None, 10
        )
        for i in range(0, self.num_of_obs):
            self._lock.acquire()
            base = copy.deepcopy(self._depth)
            self._lock.release()
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
        self._ptu_info[goal.topological_node] = self.ptu
        self._db.insert(
            BaselineDetectionMsg(
                self._array_to_point(baseline),
                self._ptu_info[goal.topological_node],
                self._topo_info[goal.topological_node]
            ), {}
        )
        subs.unregister()
        return False

    def execute(self, goal):
        start = rospy.Time.now()
        self._is_active = True
        preempted = False
        while self._depth is None:
            rospy.sleep(0.1)
        if goal.is_learning:
            self._baseline[goal.topological_node] = list()
            self._ptu_info[goal.topological_node] = None
            preempted = self._learning(goal)
        elif goal.topological_node in self._baseline:
            preempted = self._predicting(goal, start)
        else:
            rospy.logwarn(
                "There is no baseline for change detection in %s" % goal.topological_node
            )
        self._is_active = False
        if preempted:
            rospy.loginfo("The action has been preempted...")
            self.server.set_preempted()
        else:
            rospy.loginfo("The action succeeded...")
            self.server.set_succeeded(ChangeDetectionResult())

    def _predicting(self, goal, start):
        counter = [False for i in range(0, 2)]
        self._moving_ptu(
            self._ptu_info[goal.topological_node].position,
            start, goal.duration
        )
        while (rospy.Time.now() - start) < goal.duration:
            if self.server.is_preempt_requested() or rospy.is_shutdown():
                return True
            self._lock.acquire()
            data = copy.deepcopy(self._depth)
            self._lock.release()
            mse = [0.0, 0.0, 0.0]
            for ind, point in enumerate(data):
                if self._is_nan(
                    self._baseline[goal.topological_node][ind]
                ) or self._is_nan(point):
                    continue
                mse[0] += (
                    self._baseline[goal.topological_node][ind][0]-point[0]
                )**2
                mse[1] += (
                    self._baseline[goal.topological_node][ind][1]-point[1]
                )**2
                mse[2] += (
                    self._baseline[goal.topological_node][ind][2]-point[2]
                )**2
            rmse = [math.sqrt(i/float(len(data))) for i in mse]
            rospy.loginfo(rmse)
            counter[self._counter % len(counter)] = (
                sum(rmse)/len(rmse) >= self.threshold
            )
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
        return False

    def _moving_ptu(self, ptu_pos, start, duration):
        pan = int(ptu_pos[0]*180.0/math.pi)
        tilt = int(ptu_pos[1]*180/math.pi)
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
