#!/usr/bin/env python

import copy
import rospy
import argparse
import actionlib
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point
from scipy.spatial.distance import euclidean
from scitos_ptu.msg import PtuGotoAction, PtuGotoGoal
from mongodb_store.message_store import MessageStoreProxy
from simple_change_detector.msg import ChangeDetectionMsg
from simple_change_detector.shifting_contour import ShiftingContour


class StationaryShiftingDetection(object):

    def __init__(
        self, topic_img="/head_xtion/rgb/image_raw",
        sample_size=20, wait_time=5
    ):
        # local vars
        self._counter = 0
        self._max_dist = 0.01
        self._wait_time = wait_time
        self.wait_time = rospy.Duration(wait_time)
        rospy.loginfo("Subcribe to /ptu/state...")
        self._ptu = JointState()
        self._ptu.position = [0, 0]
        self._ptu_counter = 0
        self._is_ptu_changing = [True for i in range(wait_time)]
        rospy.Subscriber("/ptu/state", JointState, self._ptu_cb, None, 1)
        self._ptu_client = actionlib.SimpleActionClient(
            'SetPTUState', PtuGotoAction
        )
        rospy.loginfo("Wait for PTU action server")
        self._ptu_client.wait_for_server(rospy.Duration(60))
        rospy.loginfo("Subcribe to /robot_pose...")
        self._robot_pose = Pose()
        self._robot_pose_counter = 0
        self._is_robot_moving = [True for i in range(wait_time)]
        rospy.Subscriber("/robot_pose", Pose, self._robot_cb, None, 1)
        self._img_contour = ShiftingContour(
            topic_img=topic_img, publish_contour=True, sample_size=sample_size
        )
        # publishing stuff
        collection = rospy.get_name()[1:]
        self._db = MessageStoreProxy(collection=collection)
        self._pub = rospy.Publisher(
            rospy.get_name()+"/detections", ChangeDetectionMsg, queue_size=10
        )
        self._is_publishing = False

    def _ptu_cb(self, ptu):
        dist = euclidean(ptu.position, self._ptu.position)
        self._is_ptu_changing[self._ptu_counter] = dist >= self._max_dist
        # print "is ptu moving: %s" % str(self._is_ptu_changing)
        self._ptu_counter = (self._ptu_counter+1) % self._wait_time
        self._ptu = ptu
        rospy.sleep(1)

    def _robot_cb(self, pose):
        dist = euclidean(
            [
                pose.position.x, pose.position.y,
                pose.orientation.z, pose.orientation.w
            ],
            [
                self._robot_pose.position.x, self._robot_pose.position.y,
                self._robot_pose.orientation.z, self._robot_pose.orientation.w
            ]
        )
        self._is_robot_moving[self._robot_pose_counter] = dist >= self._max_dist
        self._robot_pose_counter = (self._robot_pose_counter+1) % self._wait_time
        self._robot_pose = pose
        rospy.sleep(1)

    def publish_shifting_message(self):
        while not rospy.is_shutdown():
            if True not in self._is_robot_moving and True not in self._is_ptu_changing:
                if not self._is_publishing:
                    rospy.loginfo(
                        "Robot has not been moving for a while, start detection in %d seconds" % self._wait_time
                    )
                    if self._ptu.position[0] == 0.0 and self._ptu.position[1] == 0.0:
                        self._ptu_client.send_goal(PtuGotoGoal(0, 20, 30, 30))
                        self._ptu_client.wait_for_result(rospy.Duration(5, 0))
                    self._is_publishing = True
                    self._img_contour.reset()
                    rospy.sleep(self.wait_time)
                else:
                    contours = copy.deepcopy(self._img_contour.contours)
                    rospy.loginfo(
                        "%d object(s) are detected moving" % len(contours)
                    )
                    if len(contours) > 0:
                        centroids = [
                            Point(i[2][0], i[2][1], 0) for i in contours
                        ]
                        areas = zip(*contours)[1]
                    else:
                        areas = list()
                        centroids = list()
                    self._counter += 1
                    msg = ChangeDetectionMsg(
                        Header(self._counter, rospy.Time.now(), ''),
                        self._robot_pose, self._ptu, centroids, areas
                    )
                    self._pub.publish(msg)
                    if len(contours) > 0:
                        self._db.insert(msg)
                    rospy.sleep(0.9)
            else:
                self._is_publishing = False
            rospy.sleep(0.1)


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
    parser_arg.add_argument(
        "-w", dest="wait_time", default="5",
        help="Waiting time before publishing detection (default=5 (seconds))"
    )
    args = parser_arg.parse_args()
    tmp = args.img_topic.split("/")
    name = tmp[1] + "_" + tmp[2] + "_image_contour"
    rospy.init_node("shifting_detection_%s" % name)
    ssd = StationaryShiftingDetection(
        topic_img=args.img_topic, sample_size=int(args.sample_size),
        wait_time=int(args.wait_time)
    )
    ssd.publish_shifting_message()
    rospy.spin()
