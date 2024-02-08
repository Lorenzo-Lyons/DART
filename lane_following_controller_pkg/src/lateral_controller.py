#!/usr/bin/env python


import numpy as np
import os
import sys
import rospy
from std_msgs.msg import Float32, Float32MultiArray, Bool
from datetime import datetime
import csv
import rospkg
from custom_msgs_optitrack.msg import custom_opti_pose_stamped_msg
from geometry_msgs.msg import PointStamped
import random
from tf.transformations import euler_from_quaternion
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from functions_for_controllers import find_s_of_closest_point_on_global_path, produce_track,produce_marker_array_rviz,produce_marker_rviz, steer_angle_2_command
from visualization_msgs.msg import MarkerArray, Marker


class steering_controller_class:
	def __init__(self, car_number):
		#self.controller_type = 'linear'
		self.controller_type = 'pursuit'

		#set up variables
		self.car_number = car_number

		# initialize state variables
		# [x y theta]
		self.state = [0, 0, 0]
		self.t_prev = 0.0
		self.previous_path_index = 0 # initial index for closest point in global path
		self.sensors = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		self.w = 0.0
		self.v = 0.0


		# initiate steering variables
		self.steering_command_prev = 0

		# set up publisher
		self.steering_publisher = rospy.Publisher('steering_' + str(car_number), Float32, queue_size=1)
		#self.throttle_publisher = rospy.Publisher('throttle_' + str(car_number), Float32, queue_size=1)
		self.rviz_global_path_publisher = rospy.Publisher('rviz_global_path_' + str(self.car_number), MarkerArray, queue_size=10)
		self.rviz_closest_point_on_path = rospy.Publisher('rviz_closest_point_on_path_' + str(self.car_number), Marker, queue_size=10)

		#subscribers
		self.global_pose_subscriber = rospy.Subscriber('odom_' + str(car_number), Odometry, self.odometry_callback)
		self.v_encoder_subscriber = rospy.Subscriber('sensors_and_input_' + str(car_number), Float32MultiArray, self.sensors_callback)
		self.tf_listener = tf.TransformListener()

	def odometry_callback(self,odometry_msg):
		quaternion = (
		    odometry_msg.pose.pose.orientation.x,
		    odometry_msg.pose.pose.orientation.y,
		    odometry_msg.pose.pose.orientation.z,
		    odometry_msg.pose.pose.orientation.w)
		euler = euler_from_quaternion(quaternion)
		roll = euler[0]
		pitch = euler[1]
		yaw = euler[2]
		self.state = [odometry_msg.pose.pose.position.x, odometry_msg.pose.pose.position.y, yaw]

	def sensors_callback(self, msg):
		self.sensors = np.array(msg.data)
		self.w = self.sensors[5]
		self.v = self.sensors[6]

	def generate_track(self, track_choice):
		# choice = 'savoiardo'
		# choice = 'straight_line'

		# number of checkpoints to be used to define each spline of the track
		n_checkpoints = 1000

		# cartesian coordinates (x-y) of the points defining the track and the gates
		Checkpoints_x, Checkpoints_y = produce_track(track_choice, n_checkpoints)

		# associate these points to the global path x-y-z variables
		self.x_vals_global_path = Checkpoints_x
		self.y_vals_global_path = Checkpoints_y

		# from the x-y-z points obtain the sum of the arc lenghts between each point, i.e. the path seen as a function of s
		spline_discretization = len(Checkpoints_x)
		self.s_vals_global_path = np.zeros(spline_discretization)
		for i in range(1,spline_discretization):
		    self.s_vals_global_path[i] = self.s_vals_global_path[i-1] + np.sqrt((self.x_vals_global_path[i]-self.x_vals_global_path[i-1])**2+
				                                                        (self.y_vals_global_path[i]-self.y_vals_global_path[i-1])**2)

		# s_vals_global_path = sum of the arc lenght along the original path, which are going to be used to re-parametrize the path

		# generate splines x(s) and y(s) where s is now the arc length value (starting from 0)
		#self.x_of_s = interpolate.CubicSpline(self.s_vals_global_path, self.x_vals_global_path)
		#self.y_of_s = interpolate.CubicSpline(self.s_vals_global_path, self.y_vals_global_path)

		# produce and send out global path message to rviz, which contains information about the track (i.e. the global path)
		rgba = [219.0, 0.0, 204.0, 0.6]
		marker_type = 4
		self.global_path_message = produce_marker_array_rviz(self.x_vals_global_path, self.y_vals_global_path, rgba, marker_type)
		self.rviz_global_path_publisher.publish(self.global_path_message)




	def compute_steering_control_action(self):

		# --- get point on path closest to the robot ---

		#get latest transform data for robot pose
		try:
			self.tf_listener.waitForTransform("/map", "/base_link_" + str(self.car_number), rospy.Time(), rospy.Duration(1.0))
			(robot_position,robot_quaternion) = self.tf_listener.lookupTransform("/map",  "/base_link_" + str(self.car_number), rospy.Time(0))
			# transform from quaternion to euler angles
			robot_euler = euler_from_quaternion(robot_quaternion)
			robot_theta = robot_euler[2]
		except:
			print('Failed to evaluate transform, trying again')
			robot_theta = 0.0
			robot_position = [0.0,0.0]


		#evaluate this for longitudinal controller coordination
		# adding delay compensation by projecting the position of the robot into the future
		delay =  0.175 # [s]
		robot_position[0] = robot_position[0] + np.cos(robot_theta) * self.v * delay
		robot_position[1] = robot_position[1] + np.sin(robot_theta) * self.v * delay
		robot_theta = robot_theta + self.w * delay
		# measure the closest point on the global path, returning the respective s parameter and its index
		estimated_ds = 1 # just a first guess for how far the robot has travelled along the path
		s, self.current_path_index = find_s_of_closest_point_on_global_path(np.array([robot_position[0], robot_position[1]]),self.s_vals_global_path,self.x_vals_global_path, self.y_vals_global_path,self.previous_path_index, estimated_ds)

		# update index along the path to know where to search in next iteration
		self.previous_path_index = self.current_path_index
		x_closest_point = self.x_vals_global_path[self.current_path_index]
		y_closest_point = self.y_vals_global_path[self.current_path_index]

		# plot closest point on the reference path
		rgba = [255.0, 0.0, 0.0, 0.6]
		marker_type = 2
		scale = 0.05
		closest_point_message = produce_marker_rviz(x_closest_point, y_closest_point, rgba, marker_type, scale)
		self.rviz_closest_point_on_path.publish(closest_point_message)





		# ----------------------------------------

		#evaluate control action
		if self.controller_type == 'linear':

			# determine tangent to path in closest point
			# using a 10 point index jump just to avoid numerical issues with very small numbers
			x_next_closest_point = self.x_vals_global_path[self.current_path_index+10]
			y_next_closest_point = self.y_vals_global_path[self.current_path_index+10]
			norm =  np.sqrt((x_next_closest_point-x_closest_point)**2+(y_next_closest_point-y_closest_point)**2)
			tangent = [x_next_closest_point-x_closest_point, y_next_closest_point-x_closest_point] / norm
			normal_right = [tangent[1],-tangent[0]]
			# evaluate lateral distance
			lateral_distance = normal_right[0] * (robot_position[0]-x_closest_point) + normal_right[1] * (robot_position[1]-y_closest_point)
			#evaluate relative orientation to path
			path_theta = np.arctan2(-y_closest_point+y_next_closest_point, -x_closest_point+x_next_closest_point)
			rel_theta = robot_theta - path_theta
			#print('lateral distance =', lateral_distance[0], 'relative theta =', rel_theta[0], 'path theta', path_theta[0])

			#evaluate control action
			kp =0.25 # 0.15
			kp_err_lat_dot = 0.25/ (np.pi/4) 
			# evaluating lateral error derivative
			err_lat_dot = np.sin(rel_theta) * (self.v + 0.5)
			steering = - kp * lateral_distance + kp_err_lat_dot * err_lat_dot 
			# check for saturation
			steering = np.min([1,steering])
			steering = np.max([-1,steering])

			# publish command
			# steering offset
			if car_number == '1':
				steering_offset = 0.03
			elif car_number == '2':
				steering_offset = -0.075

			self.steering_publisher.publish(steering+steering_offset) ## super temporary fix because the map is flipped!

		elif self.controller_type == 'pursuit':
			L = 0.175 # length of vehicle [m]
			look_ahead_dist = 1.0 #1.0 # look ahead distance on path [m]
			Px = np.interp(s+look_ahead_dist, self.s_vals_global_path, self.x_vals_global_path)
			Py = np.interp(s+look_ahead_dist, self.s_vals_global_path, self.y_vals_global_path)
			#Px = self.x_of_s(s+look_ahead_dist)
			#Py = self.y_of_s(s+look_ahead_dist)

			ld = np.sqrt((Py-robot_position[1])**2+(Px-robot_position[0])**2+0.001)

			alpha = np.arctan2((Py-robot_position[1]),(Px-robot_position[0])) - robot_theta # putting -theta corrects for the robot orinetation
			#print('Px =', Px, '   Py=',Py ,'x_robot=',robot_position[0],'y_robot=',robot_position[1],'robot theta',robot_theta)
			delta =np.arctan2(2*L*np.sin(alpha),ld)
			#print('delta=',delta)
			# convert from steering angle to steering command
			steering = steer_angle_2_command(delta,self.car_number)

			# saturate steering
			steering = np.min([steering,1])
			steering = np.max([steering, -1])

			self.steering_publisher.publish(steering)







if __name__ == '__main__':
	try:
		car_number = os.environ["car_number"]
		rospy.init_node('steering_control_node_' + str(car_number), anonymous=False)
		rate = rospy.Rate(10) #Hz
		global_path_message_rate = 50 # publish 1 every 50 control loops

		#set up steering controller
		vehicle_controller = steering_controller_class(car_number)
		# straight_line
		# savoiardo
		# straight_line_my_house
		#'straight_line_downstairs'
		#savoiardo_long
		vehicle_controller.generate_track('savoiardo_long')
		
		counter = 0
		
		while not rospy.is_shutdown():
			#run steering control loop
			
			vehicle_controller.compute_steering_control_action()


			# this is just to republish global path message every now and then
			if counter > global_path_message_rate:
				vehicle_controller.rviz_global_path_publisher.publish(vehicle_controller.global_path_message)
				counter = 0 # reset counter

			#update counter
			counter = counter + 1
			rate.sleep()



	except rospy.ROSInterruptException:
		pass
