#!/usr/bin/env python3


import numpy as np
import os
import sys
import rospy
from std_msgs.msg import Float32
import rospkg



class steering_angle_2_command:
	def __init__(self, car_number):

		# initialize stering_command-steering_angle lookup table
		self.steering_vec = np.linspace(-1,1,100)

		if car_number == str(1):
			# define steering command to steering angle static mapping
			a =  1.6379064321517944
			b =  0.3301370143890381
			c =  0.019644200801849365
			d =  0.37879398465156555
			e =  1.6578725576400757

		w = 0.5 * (np.tanh(30*(self.steering_vec+c))+1)
		steering_angle1 = b * np.tanh(a * (self.steering_vec + c)) 
		steering_angle2 = d * np.tanh(e * (self.steering_vec + c))
		self.steering_angle_vec = (w)*steering_angle1+(1-w)*steering_angle2

		#set up variables
		self.car_number = car_number

		#subscriber to steering angle
		self.steer_angle_subscriber = rospy.Subscriber('steering_angle_' + str(car_number), Float32, self.steer_angle_callback)

		# set up publisher
		self.steer_publisher = rospy.Publisher('steering_' + str(car_number), Float32, queue_size=1)

		
	def steer_angle_callback(self, msg):
		# evaluate steering command and publish it
		steering = np.interp(msg.data, self.steering_angle_vec, self.steering_vec)
		self.steer_publisher.publish(steering)







if __name__ == '__main__':
	try:
		car_number = os.environ["car_number"]
		rospy.init_node('steer_angle_2_steer_control_node_' + str(car_number), anonymous=False)

		#set up steer angle to steer command converter
		steering_angle_2_command_obj = steering_angle_2_command(car_number)

		rospy.spin()



	except rospy.ROSInterruptException:
		pass
