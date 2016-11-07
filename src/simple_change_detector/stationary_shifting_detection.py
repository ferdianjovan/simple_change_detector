#!/usr/bin/env python

import copy
import rospy
import argparse
import actionlib
import numpy as np
from scipy.spatial.distance import euclidean

from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point
from tf.transformations import euler_from_quaternion

from scitos_ptu.msg import PtuGotoAction, PtuGotoGoal
from mongodb_store.message_store import MessageStoreProxy

from simple_change_detector.msg import ChangeDetectionMsg
from simple_change_detector.shifting_contour import ShiftingContour


class StationaryShiftingDetection(object):

    def __init__(
        self, topic_img="/head_xtion/rgb/image_raw", sample_size=20, wait_time=5
    ):
        # local vars
        self._counter = 0
        self._max_dist = 0.1
        self._wait_time = wait_time
        # ptu
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
        # robot pose
        rospy.loginfo("Subcribe to /robot_pose...")
        self._robot_pose = Pose()
        self._robot_pose_counter = 0
        self._is_robot_moving = [True for i in range(wait_time)]
        rospy.Subscriber("/robot_pose", Pose, self._robot_cb, None, 1)
        # img
        self._img_contour = ShiftingContour(
            topic_img=topic_img, publish_contour=True, sample_size=sample_size
        )
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
                        self._ptu_client.send_goal(PtuGotoGoal(0, 15, 30, 30))
                        self._ptu_client.wait_for_result(rospy.Duration(5, 0))
                    self._is_publishing = True
                    self._img_contour.reset()
                    while self._img_contour._base.baseline is None:
                        rospy.sleep(0.1)
                else:
                    contours = copy.deepcopy(self._img_contour.contours)
                    rospy.loginfo(
                        "%d object(s) are detected moving" % len(contours)
                    )
                    centroids = list()
                    for i in contours:
                        centroid = self.convert_to_world_frame(
                            Point(i[1][0], i[1][1], i[1][2]),
                            self._robot_pose, self._ptu
                        )
                        centroids.append(centroid)
                    self._counter += 1
                    msg = ChangeDetectionMsg(
                        Header(self._counter, rospy.Time.now(), ''),
                        self._robot_pose, self._ptu, centroids
                    )
                    self._pub.publish(msg)
                    if len(contours) > 0:
                        self._db.insert(msg)
                    rospy.sleep(0.1)
            else:
                self._is_publishing = False
            rospy.sleep(0.1)

    # @PDuckworth's code
    def convert_to_world_frame(self, point, robot_pose, ptu):
        """Convert a single camera frame coordinate into a map frame coordinate"""
        y,z,x = point.x, point.y, point.z

        xr = robot_pose.position.x
        yr = robot_pose.position.y
        zr = robot_pose.position.z
        ax = robot_pose.orientation.x
        ay = robot_pose.orientation.y
        az = robot_pose.orientation.z
        aw = robot_pose.orientation.w
        roll, pr, yawr = euler_from_quaternion([ax, ay, az, aw])

        yawr += ptu.position[ptu.name.index('pan')]
        pr += ptu.position[ptu.name.index('tilt')]

        # transformation from camera to map
        rot_y = np.matrix([
            [np.cos(pr), 0, np.sin(pr)],
            [0, 1, 0],
            [-np.sin(pr), 0, np.cos(pr)]
        ])
        rot_z = np.matrix([
            [np.cos(yawr), -np.sin(yawr), 0],
            [np.sin(yawr), np.cos(yawr), 0],
            [0, 0, 1]
        ])
        rot = rot_z*rot_y

        pos_r = np.matrix([[xr], [yr], [zr+1.66]]) # robot's position in map frame
        pos_p = np.matrix([[x], [-y], [-z]]) # person's position in camera frame

        map_pos = rot*pos_p+pos_r # person's position in map frame
        x_mf = map_pos[0,0]
        y_mf = map_pos[1,0]
        z_mf = map_pos[2,0]

        print "_________"
        print point
        print ">>" , x_mf, y_mf, z_mf
        return Point(x_mf, y_mf, z_mf)


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
