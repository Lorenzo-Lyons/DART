#!/usr/bin/env python


import numpy as np
import os
import rospy
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from std_msgs.msg import Float32,Float32MultiArray



# got reference code from here:
# https://gist.github.com/atotto/f2754f75bedb6ea56e3e0264ec405dcf

	



class odom_pub:
	def __init__(self, car_number):

		self.car_number = car_number

		# set up topics and nodes
		rospy.init_node('odometry_publisher_' + str(car_number), anonymous=True)
		self.odom_pub = rospy.Publisher("odom_"+ str(car_number), Odometry, queue_size=50)
		self.odom_broadcaster = tf.TransformBroadcaster()
		rospy.Subscriber('sensors_and_input_' + str(car_number), Float32MultiArray, self.callback_sensors_and_input)


		# initialize absolute coordinates
		self.x = 0.0
		self.y = 0.0
		self.theta = 0.0

		# initialize quantities for odometry
		self.w_IMU = 0.0
		self.vx = 0.0

		# initialize time counter
		self.current_time = rospy.Time.now()
		self.last_time = rospy.Time.now()


	#velocity callback function
	def callback_sensors_and_input(self, sensors_and_input_data):
		#[elapsed_time, current, voltage, acc_x, acc_y,omega, vel]
		self.w_IMU = sensors_and_input_data.data[5] # omega is in radians		
		self.vx = sensors_and_input_data.data[6]



	def publish_odometry(self,rate):

		r = rospy.Rate(rate)
		while not rospy.is_shutdown():
			self.current_time = rospy.Time.now()

			# compute odometry in a typical way given the velocities of the robot
			dt = (self.current_time - self.last_time).to_sec()

			# --- evaluate estimated movement in the absolute frame according to the kinematic bicycle model ---
			#simple kinematic bicycle model
			vx_map = self.vx * np.cos(self.theta) 
			vy_map = self.vx * np.sin(self.theta)
			w = self.w_IMU

			self.x = self.x + vx_map * dt
			self.y = self.y + vy_map * dt
			self.theta = self.theta + w * dt
			

			# since all odometry is 6DOF we'll need a quaternion created from yaw
			odom_quat = tf.transformations.quaternion_from_euler(0, 0, self.theta)

			# first, we'll publish the transform over tf - the transformation from the odometry frame to the base link frame
			self.odom_broadcaster.sendTransform(
			(self.x, self.y, 0.),
			odom_quat,
			self.current_time,
			"base_link_"+str(car_number),
			"odom_"+ str(car_number)
			)

			# next, we'll publish the odometry message over ROS
			odom = Odometry()
			odom.header.stamp = self.current_time
			odom.header.frame_id = "odom_"+ str(car_number)

			# set the position
			odom.pose.pose = Pose(Point(self.x, self.y, 0.), Quaternion(*odom_quat))

			# set the velocity NOTE that the velocity is expressed in the base link frame! (So the robot frame)
			odom.child_frame_id = "base_link_"+str(self.car_number)
			odom.twist.twist = Twist(Vector3(self.vx, 0, 0), Vector3(0, 0, w))

			# publish the message
			self.odom_pub.publish(odom)

			self.last_time = self.current_time
			r.sleep()

if __name__ == '__main__':

	car_number = os.environ["car_number"]
	print("publishing odometry car " + str(car_number))
	# instantiate odometry publishing class
	odom_pub_obj = odom_pub(car_number)
	#publish odometry
	rate = 10
	odom_pub_obj.publish_odometry(rate)

	rospy.spin()






