#!/usr/bin/env python3

import rospy
import math
import os
from jetracer.nvidia_racecar import NvidiaRacecar
from std_msgs.msg import Float32,Float64MultiArray


class racecar:
	def __init__(self, car_number, steering_gain, steering_offset, throttle_gain):
		print("Starting racecar" + str(car_number))
		#Base features from cytron - initialize car variables and tune settings
		self.car = NvidiaRacecar()
		self.car.steering_gain = steering_gain	#do not change this value
		self.car.steering_offset = steering_offset
		self.car.throttle_gain = throttle_gain


		# set inputs to 0 when starting up
		self.car.steering = 0.0
		self.car.throttle = 0.0

		# additional features by Lyons
		self.car_number = float(car_number)
		self.safety_value = 0

		#set up ros nodes for this vehicle
		
		rospy.init_node('racecar_' + str(car_number), anonymous=True)
		rospy.Subscriber('steering_' + str(car_number), Float32, self.callback_steering)
		rospy.Subscriber('throttle_' + str(car_number), Float32, self.callback_throttle)
		rospy.Subscriber('safety_value', Float32, self.callback_safety)
		self.commands_timestamped = rospy.Publisher("commands_timestamped_car_" + str(car_number), Float64MultiArray, queue_size=1)
		print("finished setting up racecar " + str(car_number))	
		print('------------------------------')	

		self.last_safety_value_received = rospy.Time.now()

		# publish sensors to the vehicle every 30 hz
		self.throttle = 0.0
		rate = rospy.Rate(30)
		while not rospy.is_shutdown():
			
			# evaluate how long ago the last safety value was received 
			d = rospy.Time.now() - self.last_safety_value_received

			if d.to_sec() > 0.3:
				print('too long since last safety value!! setting it to 0')
				safety_value = 0
			else:
				safety_value = self.safety_value

			#publish throttle input
			self.car.throttle = self.throttle * safety_value
			rate.sleep()



	#Safety callback function
	def callback_safety(self, safety_val_incoming):
		self.safety_value = safety_val_incoming.data
		# reset timer on last safety value received
		self.last_safety_value_received = rospy.Time.now()
		
	
	#Steering control callback function
	def callback_steering(self, steer):
		self.car.steering = steer.data # execute the steering command

	#Throttle callback function
	def callback_throttle(self, throttle):
		self.throttle = throttle.data # update global throttle variable

	
if __name__ == '__main__':
	car_number = os.environ["car_number"]
	print('car_number = ', os.environ["car_number"])
	if float(car_number) == 1:
		#steering_gain = -0.5
		#steering_offset = + 0.08   # a negative value means go more to the right
		# changing steering gain sign!
		steering_gain = +0.5
		steering_offset = + 0.08   # a negative value means go more to the left

		print('setting steer gain and offset for car number 1')
		print('steering gain = ',steering_gain)
		print('steering offset = ',steering_offset)

		

	elif float(car_number) == 2:
		steering_gain = 0.5 #-0.5
		steering_offset = -0.18 * 0.5 # -0.18 * 0.5
		print('setting steer gain and offset for car number 2')

	elif float(car_number) == 3:
		steering_gain = 0.5
		steering_offset = +0.26 * 0.5
		print('setting steer gain and offset for car number 3')
	
	elif float(car_number) == 4:
		#steering_gain = -0.5
		#steering_offset = + 0.08   # a negative value means go more to the right
		# changing steering gain sign!
		steering_gain = +0.5
		steering_offset = -0.091   # a negative value means go more to the right

		print('setting steer gain and offset for car number 4')
		print('steering gain = ',steering_gain)
		print('steering offset = ',steering_offset)

	else:
		steering_gain = -0.5
		steering_offset = 0.0


	throttle_gain = 0.6
	print('throttle gain = ',throttle_gain)
	racecar(car_number, steering_gain, steering_offset, throttle_gain)





