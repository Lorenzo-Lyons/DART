#!/usr/bin/env python

import rospy
import tf
import tf2_ros
import geometry_msgs.msg
import yaml
import numpy as np
from collections import deque
import sys
import os

class MocapObjectCalibration:
    def __init__(self, object_name, calibration_board_name=None):
        self.object_name = object_name  # Store object_name in the class
        self.object_topic = "/mocap/{}/pose".format(object_name)
        self.calibration_board_topic = None
        self.calibration_board_defined = calibration_board_name is not None
        
        if self.calibration_board_defined:
            self.calibration_board_topic = "/mocap/{}/pose".format(calibration_board_name)
        
        self.calibration_board_poses = deque(maxlen=500)
        self.object_poses = deque(maxlen=500)

        self.calibration_board_pose_avg = None
        self.object_pose_avg = None

        if self.calibration_board_defined:
            self.calibration_board_sub = rospy.Subscriber(
                self.calibration_board_topic, geometry_msgs.msg.PoseStamped, self.calibration_board_callback)
        self.object_sub = rospy.Subscriber(
            self.object_topic, geometry_msgs.msg.PoseStamped, self.object_callback)

        rospy.on_shutdown(self.shutdown_hook)

    def calibration_board_callback(self, msg):
        self.calibration_board_poses.append(msg.pose)

    def object_callback(self, msg):
        self.object_poses.append(msg.pose)

    def calculate_average_pose(self, poses):
        positions = np.array([[p.position.x, p.position.y, p.position.z] for p in list(poses)])
        orientations = np.array([[p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w] for p in list(poses)])

        avg_position = np.mean(positions, axis=0)
        avg_orientation = np.mean(orientations, axis=0)
        avg_orientation /= np.linalg.norm(avg_orientation)  # Normalize the quaternion

        avg_pose = geometry_msgs.msg.Pose()
        avg_pose.position.x = avg_position[0]
        avg_pose.position.y = avg_position[1]
        avg_pose.position.z = avg_position[2]
        avg_pose.orientation.x = avg_orientation[0]
        avg_pose.orientation.y = avg_orientation[1]
        avg_pose.orientation.z = avg_orientation[2]
        avg_pose.orientation.w = avg_orientation[3]

        return avg_pose

    def calculate_relative_pose(self):
        translation = [
             self.calibration_board_pose_avg.position.x - self.object_pose_avg.position.x,
             self.calibration_board_pose_avg.position.y - self.object_pose_avg.position.y,
             self.calibration_board_pose_avg.position.z - self.object_pose_avg.position.z
        ]

        quat_calibration = [
            self.calibration_board_pose_avg.orientation.x,
            self.calibration_board_pose_avg.orientation.y,
            self.calibration_board_pose_avg.orientation.z,
            self.calibration_board_pose_avg.orientation.w
        ]

        quat_object = [
            self.object_pose_avg.orientation.x,
            self.object_pose_avg.orientation.y,
            self.object_pose_avg.orientation.z,
            self.object_pose_avg.orientation.w
        ]

        relative_quat = tf.transformations.quaternion_multiply(
            quat_calibration,
            tf.transformations.quaternion_inverse(quat_object)
        )

        return translation, relative_quat

    def save_relative_pose(self, translation, orientation):
        def convert_to_python_type(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, np.generic):
                return value.item()
            return value

        relative_pose = {
            'position': {
                'x': convert_to_python_type(translation[0]),
                'y': convert_to_python_type(translation[1]),
                'z': convert_to_python_type(translation[2])
            },
            'orientation': {
                'x': convert_to_python_type(orientation[0]),
                'y': convert_to_python_type(orientation[1]),
                'z': convert_to_python_type(orientation[2]),
                'w': convert_to_python_type(orientation[3])
            }
        }

        # Create the calibrations directory if it doesn't exist
        calibrations_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'calibrations')
        if not os.path.exists(calibrations_dir):
            os.makedirs(calibrations_dir)

        # Save the YAML file with the same name as the object_name
        file_path = os.path.join(calibrations_dir, '{}.yaml'.format(self.object_name))
        with open(file_path, 'w') as yaml_file:
            yaml.dump(relative_pose, yaml_file, default_flow_style=False)

        rospy.loginfo("Relative pose saved to {}".format(file_path))


    def run(self):
        rate = rospy.Rate(50)
        start_time = rospy.Time.now()

        while not rospy.is_shutdown() and (rospy.Time.now() - start_time).to_sec() < 2.0:
            rospy.loginfo("Collecting data: {} calibration poses, {} object poses".format(len(self.calibration_board_poses), len(self.object_poses)))
            rate.sleep()

        if self.calibration_board_defined and len(self.calibration_board_poses) == 0:
            rospy.logerr("No calibration board poses received.")
            return

        if len(self.object_poses) > 0:
            self.object_pose_avg = self.calculate_average_pose(self.object_poses)

            if self.calibration_board_defined:
                self.calibration_board_pose_avg = self.calculate_average_pose(self.calibration_board_poses)
            else:
                self.calibration_board_pose_avg = geometry_msgs.msg.Pose()
                self.calibration_board_pose_avg.position.x = 0
                self.calibration_board_pose_avg.position.y = 0
                self.calibration_board_pose_avg.position.z = 0
                self.calibration_board_pose_avg.orientation.x = 0
                self.calibration_board_pose_avg.orientation.y = 0
                self.calibration_board_pose_avg.orientation.z = 0
                self.calibration_board_pose_avg.orientation.w = 1

            translation, orientation = self.calculate_relative_pose()
            self.save_relative_pose(translation, orientation)
        else:
            rospy.logwarn("Insufficient data to calculate average poses.")

    def shutdown_hook(self):
        rospy.loginfo("Shutting down mocap_object_calibration node.")

if __name__ == '__main__':
    rospy.init_node('mocap_object_calibration', anonymous=True)
    if len(sys.argv) < 2:
        rospy.logerr("Object name argument is required.")
        sys.exit(1)

    object_name = sys.argv[1]
    calibration_board_name = sys.argv[2] if len(sys.argv) > 2 else None
    mocap_calibration = MocapObjectCalibration(object_name, calibration_board_name)
    mocap_calibration.run()
