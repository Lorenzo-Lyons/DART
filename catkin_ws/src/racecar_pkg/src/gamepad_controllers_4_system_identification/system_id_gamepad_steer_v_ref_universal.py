#!/usr/bin/env python3

import rospy
import pygame
import time
from std_msgs.msg import Float32
import os
import numpy as np


# this gamepad directly controls the steering and publishes a reference velocity for the longitudinal motion.
# Use longitudinal_controller.py in the lane_following_controller_pkg to actually follow the reference.

#Initialize pygame and gamepad
pygame.init()
j = pygame.joystick.Joystick(0)
j.init()
print ('Initialized Joystick : %s' % j.get_name())
print('remove safety by pressing R1 button')

def teleop_gamepad(car_number):

	#Setup topics publishing and nodes
	pub_steering = rospy.Publisher('steering_' + str(car_number), Float32, queue_size=8)
	pub_v_ref = rospy.Publisher('v_ref_' + str(car_number), Float32, queue_size=8)
	pub_safety_value = rospy.Publisher('safety_value', Float32, queue_size=8)

	rospy.init_node('teleop_gamepad' + str(car_number), anonymous=True)
	rate = rospy.Rate(10) # 10hz

	# initialize v_ref
	v_ref = 0.0 # [m/s]
	incr = 0.1 # increase by this amount


	while not rospy.is_shutdown():
		pygame.event.pump()

		# Collect and publish steering
		steering = - j.get_axis(2) #Right thumbstick X  # NOTE that minus sign is needed to correct the direction of the analog stich (so that right means turn right)
		pub_steering.publish(steering)



		# Collect and publish safety value
		if j.get_button(7) == 1:
			print('safety off')
			pub_safety_value.publish(1)
		else:
			pub_safety_value.publish(0)

		# increase reference velocity by pressing the "Y" and "A" buttons
		for event in pygame.event.get(): # User did something.
			if event.type == pygame.JOYBUTTONDOWN: # button Y
				if j.get_button(4) == 1:
					v_ref = v_ref + incr
					print('v_ref = ', v_ref)

				if j.get_button(0) == 1: # button A
					v_ref = v_ref - incr
					print('v_ref = ', v_ref)

		# publish v_ref
		pub_v_ref.publish(v_ref)

		rate.sleep()

if __name__ == '__main__':
	try:
		car_number = os.environ['car_number']
		teleop_gamepad(car_number)
	except rospy.ROSInterruptException:
		pass
