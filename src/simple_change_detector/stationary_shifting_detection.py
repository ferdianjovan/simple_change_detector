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
from sensor_msgs.point_cloud2 import read_points
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray

from strands_executive_msgs.srv import GetActiveTasks
from scitos_ptu.msg import PtuGotoAction, PtuGotoGoal
from mongodb_store.message_store import MessageStoreProxy

from simple_change_detector.msg import ChangeDetectionMsg
from simple_change_detector.shifting_contour import ShiftingContour


class StationaryShiftingDetection(object):

    def __init__(
        self, topic_img="/head_xtion/rgb/image_raw", sample_size=20,
        wait_time=5, publish_image=True, save_mode=False,
        save_duration=rospy.Duration(10),
        non_interrupt_tasks=list()
    ):
        # subscribe to active tasks if necessary
        if non_interrupt_tasks == list() or non_interrupt_tasks[0] == "":
            self._non_interrupt_tasks = list()
            rospy.loginfo("Have control on PTU all the time...")
        else:
            self._active_tasks = rospy.ServiceProxy(
                "/task_executor/get_active_tasks", GetActiveTasks
            )
            self._active_tasks.wait_for_service()
            self._non_interrupt_tasks = non_interrupt_tasks
        # save mode vars
        self._save_mode = save_mode
        self._save_dur = save_duration
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
            topic_img=topic_img, publish_contour=publish_image,
            sample_size=sample_size
        )
        collection = rospy.get_name()[1:]
        self._db = MessageStoreProxy(collection=collection)
        self._pub = rospy.Publisher(
            rospy.get_name()+"/detections", ChangeDetectionMsg, queue_size=10
        )
        self._pub_marker = rospy.Publisher(
            rospy.get_name()+"/marker", MarkerArray, queue_size=10
        )
        self._is_publishing = False
        self.detection = ChangeDetectionMsg()
        rospy.Timer(rospy.Duration(1), self.publish_detections)

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
        self._robot_pose_counter = (
            self._robot_pose_counter+1
        ) % self._wait_time
        self._robot_pose = pose
        rospy.sleep(1)

    def publish_shifting_message(self):
        start = rospy.Time.now()
        end = rospy.Time.now()
        while not rospy.is_shutdown():
            if self._save_mode:
                end = rospy.Time.now()
            if True not in (self._is_robot_moving+self._is_ptu_changing):
                if not self._is_publishing:
                    self.tilting_ptu()
                    self._img_contour._pause = False
                    rospy.loginfo(
                        "Robot has not been moving for a while..."
                    )
                    rospy.loginfo(
                        "Start detection in %d seconds" % self._wait_time
                    )
                    self._is_publishing = True
                    self._img_contour.reset()
                    while self._img_contour._base.baseline is None:
                        rospy.sleep(0.1)
                    start = rospy.Time.now()
                elif (end-start) > self._save_dur:
                    self._img_contour._pause = not self._img_contour._pause
                    start = rospy.Time.now()
                else:
                    if self._img_contour._pause:
                        centroids = []
                    else:
                        contours = copy.deepcopy(self._img_contour.contours)
                        centroids = self._get_centroid_on_map_frame(contours)
                    self._counter += 1
                    self.detection = ChangeDetectionMsg(
                        Header(self._counter, rospy.Time.now(), ''),
                        self._robot_pose, self._ptu, centroids
                    )
                    self._draw_detections(centroids)
            else:
                self.tilting_ptu(False)
                self._img_contour._pause = True
                self._is_publishing = False
            rospy.sleep(0.1)

    def tilting_ptu(self, is_robot_stationary=True):
        non_interruption = False
        if self._non_interrupt_tasks != list():
            tasks = self._active_tasks()
            tasks = [i.action for i in tasks.task]
            for i in self._non_interrupt_tasks:
                if i in tasks:
                    non_interruption = True
                    break
        ptu_cond = self._ptu.position[0] == 0.0 and self._ptu.position[1] == 0.0
        if not non_interruption and ptu_cond and is_robot_stationary:
            self._ptu_client.send_goal(PtuGotoGoal(0, 15, 30, 30))
            self._ptu_client.wait_for_result(rospy.Duration(5))
        elif not non_interruption and not ptu_cond and not is_robot_stationary:
            self._ptu_client.send_goal(PtuGotoGoal(0, 0, 30, 30))
            self._ptu_client.wait_for_result(rospy.Duration(5))

    def publish_detections(self, event):
        if self._is_publishing:
            self._pub.publish(self.detection)
            if len(self.detection.object_centroids) > 0:
                self._db.insert(self.detection)

    def _draw_detections(self, centroids):
        markers = MarkerArray()
        for ind, centroid in enumerate(centroids):
            marker = Marker()
            marker.header.frame_id = "/map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "change_detection_markers"
            marker.action = Marker.ADD
            marker.pose.position = centroid
            marker.pose.orientation.w = 1.0
            marker.id = ind
            marker.type = Marker.SPHERE
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.b = 1.0
            markers.markers.append(marker)
        if len(centroids) == 0:
            for ind in range(10):
                marker = Marker()
                marker.header.frame_id = "/map"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "change_detection_markers"
                marker.action = Marker.DELETE
                marker.id = ind
                marker.type = Marker.SPHERE
                marker.color.a = 0.0
                markers.markers.append(marker)
        self._pub_marker.publish(markers)

    def _get_centroid_on_map_frame(self, contours):
        centroids = list()
        depth = [
            i for i in read_points(
                self._img_contour._depth,
                field_names=("x", "y", "z")
            )
        ]
        depth = np.array(depth, dtype="float").reshape(
            self._img_contour._depth.height,
            self._img_contour._depth.width, 3
        )
        for i in contours:
            centroid = depth[i[1][1], i[1][0]]
            if True in np.isnan(centroid):
                rospy.loginfo("Reflective object, ignore...")
                continue
            centroid = self.convert_to_world_frame(
                Point(centroid[0], centroid[1], centroid[2]),
                self._robot_pose, self._ptu
            )
            centroids.append(centroid)
        return centroids

    # @PDuckworth's code
    def convert_to_world_frame(self, point, robot_pose, ptu):
        """
            Convert a single camera frame coordinate into a map frame coordinate
        """
        y, z, x = point.x, point.y, point.z

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

        # robot's position in map frame
        pos_r = np.matrix([[xr], [yr], [zr+1.66]])
        # person's position in camera frame
        pos_p = np.matrix([[x], [-y], [-z]])
        # person's position in map frame
        map_pos = rot * pos_p + pos_r
        x_mf = map_pos[0, 0]
        y_mf = map_pos[1, 0]
        z_mf = map_pos[2, 0]

        # print "_________"
        # print point
        # print ">>", x_mf, y_mf, z_mf
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
