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
		rate_integration = 50 # set odom integration rate, pupblishing will be done every time there is a new arduino message

		self.car_number = car_number

		# set up topics and nodes
		rospy.init_node('odometry_publisher_' + str(car_number), anonymous=True)
		self.odom_pub = rospy.Publisher("odom_"+ str(car_number), Odometry, queue_size=50)
		self.odom_broadcaster = tf.TransformBroadcaster()
		#rospy.Subscriber('sensors_and_input_' + str(car_number), Float32MultiArray, self.callback_sensors_and_input)
		rospy.Subscriber('arduino_data_' + str(car_number), Float32MultiArray, self.callback_arduino)


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


		#rospy.spin()
		r = rospy.Rate(rate_integration)
		while not rospy.is_shutdown():
			#self.publish_odometry()
			self.integrate_odometry()
			r.sleep()


	def callback_arduino(self, msg):
		#[acc_x, acc_y,omega, vel]
		self.vx = msg.data[3]
		self.w_IMU = msg.data[2] # omega is in radians
		#publish the odometry message
		self.publish_odometry(self.x,self.y,self.theta,self.vx,self.w_IMU,self.current_time)


	def integrate_odometry(self):
		# r = rospy.Rate(rate)
		# while not rospy.is_shutdown():
		self.current_time = rospy.Time.now()

		# compute odometry in a typical way given the velocities of the robot
		dt = (self.current_time - self.last_time).to_sec()

		# --- evaluate estimated movement in the absolute frame according to the kinematic bicycle model ---
		#simple kinematic bicycle model
		vx_map = self.vx * np.cos(self.theta) 
		vy_map = self.vx * np.sin(self.theta)
		w = self.w_IMU

		# since there is some noise on W imu, if it the car is not moving set it to 0 to avoid drift
		# but if the car is picked up and turned do some w integrating
		if  self.vx < 0.01 and w < 0.01:
			w = 0.0

		self.x = self.x + vx_map * dt
		self.y = self.y + vy_map * dt
		self.theta = self.theta + w * dt
		#print('theta=',self.theta)

		# update time
		self.last_time = self.current_time
			#r.sleep()


	def publish_odometry(self,x,y,theta,vx,w,current_time):
		# since all odometry is 6DOF we'll need a quaternion created from yaw
		odom_quat = tf.transformations.quaternion_from_euler(0, 0, theta)
		# first, we'll publish the transform over tf - the transformation from the odometry frame to the base link frame
		self.odom_broadcaster.sendTransform(
		(x, y, 0.),
		odom_quat,
		current_time,
		"base_link_"+str(car_number),
		"odom_"+ str(car_number)
		)

		# next, we'll publish the odometry message over ROS
		odom = Odometry()
		odom.header.stamp = current_time
		odom.header.frame_id = "odom_"+ str(car_number)

		# set the position
		odom.pose.pose = Pose(Point(x, y, 0.), Quaternion(*odom_quat))

		# set the velocity NOTE that the velocity is expressed in the base link frame! (So the robot frame)
		odom.child_frame_id = "base_link_"+str(self.car_number)
		odom.twist.twist = Twist(Vector3(vx, 0, 0), Vector3(0, 0, w))

		# assign odom to odom_msg
		odom_msg = odom
		# publish the message
		self.odom_pub.publish(odom)


if __name__ == '__main__':

	car_number = os.environ["car_number"]
	print("publishing odometry car " + str(car_number))
	# instantiate odometry publishing class
	odom_pub_obj = odom_pub(car_number)


	rospy.spin()