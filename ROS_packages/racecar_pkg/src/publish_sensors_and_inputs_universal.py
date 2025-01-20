#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32, Float32MultiArray
from std_msgs.msg import Bool
import os
import time
import numpy as np


class Pubsensors_and_input:
	
	def __init__(self, car_number):
		#Setup node and topics subscription
		print("Sensor publisher setup ros topics and node")

		#setup this node
		rospy.init_node('publish_sensors_and_input_'+str(car_number), anonymous=True)
		#setup publisher handles
		self.pub_sens_input = rospy.Publisher("sensors_and_input_" + str(car_number), Float32MultiArray, queue_size=1)
		# subscribe to inputs
		self.throttle = 0.0 
		self.steering = 0.0
		self.safety_value = 0.0
		rospy.Subscriber('safety_value', Float32, self.callback_safety)
		rospy.Subscriber('throttle_' + str(car_number), Float32, self.callback_throttle)
		rospy.Subscriber('steering_' + str(car_number), Float32, self.callback_steering)
		rospy.Subscriber('arduino_data_' + str(car_number), Float32MultiArray, self.callback_arduino_data)


		
		print("starting sensor data publishing")
		# Run the while loop for the controller
		rate = rospy.Rate(10)	#frequency !! note that the serial gives new measurements at 10 Hz so keep it like this, faster rates seem to bee too much for the battery sensor and the values stop being updated
		self.start_clock_time = rospy.get_rostime()
		rospy.spin()



		
		
	#Safety callback function
	def callback_safety(self, safety_val_incoming):
		self.safety_value = safety_val_incoming.data

	#Throttle callback function
	def callback_throttle(self, throttle_data):
		self.throttle = throttle_data.data

	#Steering callback function
	def callback_steering(self, steering_data):
		self.steering = steering_data.data
			
	#Arduino callback function
	def callback_arduino_data(self, arduino_msg):
		stop_clock_time = rospy.get_rostime()
		elapsed_time = (stop_clock_time - self.start_clock_time).to_sec()

		#arduino_msg = [	   acc_x, acc_y, omega_rad, vel]

		#define messages to send
		sensor_msg = Float32MultiArray()
		#  arduino_msg = [	   acc_x, acc_y, omega_rad, vel]
		#                  elapsed_time, current,voltage,IMU[0](acc x),IMU[1] (acc y),IMU[2] (omega rads),velocity, safety          ,throttle     ,steering
		sensor_msg.data = [elapsed_time, 0.0    ,0.0    , *arduino_msg.data                                       ,self.safety_value,self.throttle,self.steering]

		#publish messages
		self.pub_sens_input.publish(sensor_msg)	

if __name__ == '__main__':
	print("Starting pid-controller for velocity")
	try:	
		car_number = os.environ["car_number"]
		vel_publisher = Pubsensors_and_input(car_number)
	except rospy.ROSInterruptException:
		print('failed to lauch velocity publisher')
