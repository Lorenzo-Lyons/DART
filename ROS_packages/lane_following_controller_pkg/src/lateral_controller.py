#!/usr/bin/env python


import numpy as np
import os
import sys
import rospy
from std_msgs.msg import Float32, Float32MultiArray, Bool
from datetime import datetime
import csv
import rospkg
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

		self.overtaking = False


		# initiate steering variables
		self.steering_command_prev = 0

		# set up publisher
		self.steering_angle_publisher = rospy.Publisher('steering_angle_' + str(car_number), Float32, queue_size=1)
		#self.throttle_publisher = rospy.Publisher('throttle_' + str(car_number), Float32, queue_size=1)
		self.rviz_global_path_publisher = rospy.Publisher('rviz_global_path_' + str(self.car_number), MarkerArray, queue_size=10)
		self.rviz_global_path_publisher_overtake = rospy.Publisher('rviz_global_path__overtake' + str(self.car_number), MarkerArray, queue_size=10)
		
		self.rviz_closest_point_on_path = rospy.Publisher('rviz_closest_point_on_path_' + str(self.car_number), Marker, queue_size=10)

		#subscribers
		self.global_pose_subscriber = rospy.Subscriber('odom_' + str(car_number), Odometry, self.odometry_callback)
		self.v_encoder_subscriber = rospy.Subscriber('sensors_and_input_' + str(car_number), Float32MultiArray, self.sensors_callback)

		# only for platooning
		self.overtake_shift_y = 0.0
		rospy.Subscriber('overtaking_' + str(car_number), Bool, self.overtaking_callback)

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

	def overtaking_callback(self, msg):
		self.overtaking = msg.data


	def generate_track(self, track_choice,n_checkpoints,rotation_angle,x_shift,y_shift):
		# choice = 'savoiardo'
		# choice = 'straight_line'

		# cartesian coordinates (x-y) of the points defining the track and the gates
		Checkpoints_x, Checkpoints_y = produce_track(track_choice, n_checkpoints,rotation_angle,x_shift,y_shift)

		# associate these points to the global path x-y-z variables
		x_vals_global_path = Checkpoints_x
		y_vals_global_path = Checkpoints_y

		# from the x-y points obtain the sum of the arc lenghts between each point, i.e. the path seen as a function of s
		spline_discretization = len(Checkpoints_x)
		s_vals_global_path = np.zeros(spline_discretization)
		for i in range(1,spline_discretization):
		    s_vals_global_path[i] = s_vals_global_path[i-1] + np.sqrt((x_vals_global_path[i]-x_vals_global_path[i-1])**2+
				                                                        (y_vals_global_path[i]-y_vals_global_path[i-1])**2)


		# produce and send out global path message to rviz, which contains information about the track (i.e. the global path)
		rgba = [219.0, 0.0, 204.0, 0.6]
		marker_type = 4
		global_path_message = produce_marker_array_rviz(x_vals_global_path, y_vals_global_path, rgba, marker_type)
		

		return s_vals_global_path, x_vals_global_path, y_vals_global_path, global_path_message




	def compute_steering_control_action(self):
		if self.overtaking:
			x_vals_global_path = self.x_vals_global_path_overtake
			y_vals_global_path = self.y_vals_global_path_overtake
			s_vals_global_path = self.s_vals_global_path_overtake
		else:
			x_vals_global_path = self.x_vals_global_path
			y_vals_global_path = self.y_vals_global_path
			s_vals_global_path = self.s_vals_global_path			

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
		delay =  0.165 # [s]
		robot_position[0] = robot_position[0] + np.cos(robot_theta) * self.v * delay
		robot_position[1] = robot_position[1] + np.sin(robot_theta) * self.v * delay
		robot_theta = robot_theta + self.w * delay
		# measure the closest point on the global path, returning the respective s parameter and its index
		estimated_ds = 0.5 # just a first guess for how far the robot has travelled along the path
		s, self.current_path_index = find_s_of_closest_point_on_global_path(np.array([robot_position[0], robot_position[1]]),
									s_vals_global_path,x_vals_global_path,
									 y_vals_global_path ,self.previous_path_index, estimated_ds)

		# update index along the path to know where to search in next iteration
		self.previous_path_index = self.current_path_index
		x_closest_point = x_vals_global_path[self.current_path_index] 
		y_closest_point = y_vals_global_path[self.current_path_index] 

		# plot closest point on the reference path
		rgba = [255.0, 0.0, 0.0, 0.6]
		marker_type = 2
		scale = 0.05
		closest_point_message = produce_marker_rviz(x_closest_point, y_closest_point, rgba, marker_type, scale)
		self.rviz_closest_point_on_path.publish(closest_point_message)





		# ----------------------------------------
		L = 0.175 # length of vehicle [m]
		# if self.v > 1.0:
		# 	look_ahead_dist = 1 + (self.v-1)*2 
		# else:
		look_ahead_dist = 0.6 #look ahead distance on path [m]

		# account for path running out
		if s + look_ahead_dist > s_vals_global_path[-1]: # look ahead is beyond the path length
			look_ahead_s = s + look_ahead_dist - s_vals_global_path[-1]
		else:
			look_ahead_s = s + look_ahead_dist


		Px = np.interp(look_ahead_s, s_vals_global_path, x_vals_global_path)
		Py = np.interp(look_ahead_s, s_vals_global_path, y_vals_global_path )


		ld = np.sqrt((Py-robot_position[1])**2+(Px-robot_position[0])**2+0.001)

		alpha = np.arctan2((Py-robot_position[1]),(Px-robot_position[0])) - robot_theta # putting -theta corrects for the robot orinetation
		#print('Px =', Px, '   Py=',Py ,'x_robot=',robot_position[0],'y_robot=',robot_position[1],'robot theta',robot_theta)
		delta = np.arctan2(2*L*np.sin(alpha),ld)

		# publish steering angle 
		self.steering_angle_publisher.publish(delta)






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
		#square_vicon

		# DEMO ARENA TRACK
		main_track_choice = 'savoiardo_demo_arena'
		overtake_track_choice = 'savoiardo_demo_arena_internal'

		# DEMO ARENA TRACK 8x14
		main_track_choice = 'savoiardo_demo_arena_8x14'
		overtake_track_choice = 'savoiardo_demo_arena_internal_8x14'

		n_checkpoints = 100
		rotation_angle = -0.01

		# main lane offset
		# DEMO ARENA
		# x_shift_main = -0.1
		# y_shift_main = -0.02
		# #overtake lane offsets
		# x_shift_overtake = x_shift_main + 0.25
		# y_shift_overtake = -0.02
		# rotation_angle_overtake = -0.01

		# DEMO ARENA 8x14
		x_shift_main = -0.1
		y_shift_main = 0.025
		#overtake lane offsets
		x_shift_overtake = x_shift_main + 0.35
		y_shift_overtake = 0
		rotation_angle_overtake = 0


		# # VIKON LAB TRACK
		# main_track_choice = 'square_vicon'
		# overtake_track_choice = 'square_vicon_internal'

		# n_checkpoints = 100
		# rotation_angle = -0.0
		# # main lane offset
		# x_shift_main = -0.2 +1
		# y_shift_main = -0.15
		# #overtake lane offsets
		# x_shift_overtake = +0.15 + 1
		# y_shift_overtake = +0.2



		#main lane
		s_vals_global_path, x_vals_global_path, y_vals_global_path,global_path_message = vehicle_controller.generate_track(main_track_choice,n_checkpoints,rotation_angle,x_shift_main,y_shift_main)
		
		vehicle_controller.s_vals_global_path = s_vals_global_path
		vehicle_controller.x_vals_global_path = x_vals_global_path
		vehicle_controller.y_vals_global_path = y_vals_global_path
		vehicle_controller.global_path_message = global_path_message

		#overtake lane
		s_vals_global_path_overtake, x_vals_global_path_overtake, y_vals_global_path_overtake,global_path_message_overtake = vehicle_controller.generate_track(overtake_track_choice,n_checkpoints,rotation_angle_overtake,x_shift_overtake,y_shift_overtake)
		
		vehicle_controller.s_vals_global_path_overtake = s_vals_global_path_overtake
		vehicle_controller.x_vals_global_path_overtake = x_vals_global_path_overtake
		vehicle_controller.y_vals_global_path_overtake = y_vals_global_path_overtake
		vehicle_controller.global_path_message_overtake = global_path_message_overtake



		counter = 0
		
		while not rospy.is_shutdown():
			#run steering control loop
			
			vehicle_controller.compute_steering_control_action()


			# this is just to republish global path message every now and then
			if counter > global_path_message_rate:
				vehicle_controller.rviz_global_path_publisher.publish(vehicle_controller.global_path_message)
				vehicle_controller.rviz_global_path_publisher_overtake.publish(vehicle_controller.global_path_message_overtake)
				counter = 0 # reset counter

			#update counter
			counter = counter + 1
			rate.sleep()



	except rospy.ROSInterruptException:
		pass
1
