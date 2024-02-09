#!/usr/bin/env python3

import rospy
import pygame
import time
from std_msgs.msg import Float32
import os

#Initialize pygame and gamepad
pygame.init()
j = pygame.joystick.Joystick(0)
j.init()
print ('Initialized Joystick : %s' % j.get_name())
print('remove safety by pressing R1 button')

def teleop_gamepad(car_number):

	#Setup topics publishing and nodes
	pub_throttle = rospy.Publisher('throttle_' + str(car_number), Float32, queue_size=8)
	pub_steering = rospy.Publisher('steering_' + str(car_number), Float32, queue_size=8)
	# also publishing safety value
	pub_safety_value = rospy.Publisher('safety_value', Float32, queue_size=8)

	rospy.init_node('teleop_gamepad' + str(car_number), anonymous=True)
	rate = rospy.Rate(10) # 10hz

	throttle = 0.0
	step = 0.005

	while not rospy.is_shutdown():
		pygame.event.pump()

		# --- sttering ---
		steering = -j.get_axis(2) * 0.1 #Right thumbstick X
		pub_steering.publish(steering)

		# --- throttle ---
		left_analog_stick = -j.get_axis(1) #Right thumbstick Y
		if j.get_button(4) == 1:
			throttle += step
			print('throttle = ', throttle)
		elif j.get_button(0) == 1:
			throttle -= step
			print('throttle = ', throttle)
		pub_throttle.publish(throttle * left_analog_stick)

		# --- safety value ---
		if j.get_button(7) == 1:
			pub_safety_value.publish(1)
		else:
			pub_safety_value.publish(0)

		rate.sleep()

if __name__ == '__main__':
	try:
		car_number = os.environ['car_number']
		teleop_gamepad(car_number)
	except rospy.ROSInterruptException:
		pass
