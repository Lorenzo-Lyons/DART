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

		elif car_number == str(2):

			# a =  0.7133021354675293
			# b =  0.5663600564002991
			# c =  0.02374953031539917
			# d =  0.42923927307128906
			# e =  1.1735163927078247
			a =  1.352936863899231
			b =  0.4281103014945984
			c =  -0.03932629153132439
			d =  0.6000000238418579
			e =  0.9139584302902222

		elif car_number == str(3):

			a =  1.5443377494812012
			b =  0.36627525091171265
			c =  -0.0053876787424087524
			d =  0.34682372212409973
			e =  1.8532332181930542
		
		elif car_number == str(4):

			# a =  1.5197126865386963
			# b =  0.47897201776504517
			# c =  0.003458067774772644
			# d =  0.44620275497436523
			# e =  1.6679414510726929

			a =  1.5300253629684448
			b =  0.322294145822525
			c =  0.018957361578941345
			d =  0.45186465978622437
			e =  1.095400094985962


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
