#!/usr/bin/env python3

import rospy
import pygame
import time
from std_msgs.msg import Float32
import os
import numpy as np

#Initialize pygame and gamepad
pygame.init()
j = pygame.joystick.Joystick(0)
j.init()
print ('Initialized Joystick : %s' % j.get_name())
print('remove safety by pressing R1 button')

def teleop_gamepad(car_number):
	# this only controls the steering, the throttle can be controlled for example at a constant speed by running
	# leader controller in platooning_pkg_iros_2023

	#Setup topics publishing and nodes
	pub_steering = rospy.Publisher('steering_' + str(car_number), Float32, queue_size=8)
	pub_v_ref = rospy.Publisher('v_ref_' + str(car_number), Float32, queue_size=8)
	pub_safety_value = rospy.Publisher('safety_value', Float32, queue_size=8)

	rospy.init_node('teleop_gamepad' + str(car_number), anonymous=True)
	rate = rospy.Rate(5) # 10hz

	# initialize v_ref
	v_ref = 1.0 # [m/s]
	incr = 0.1 # increase by this amount

	#publish sinusoidal steering
	steer_step = 0.05
	steering = 0.0
	start_clock_time = rospy.get_rostime()


	while not rospy.is_shutdown():
		pygame.event.pump()






		# --- safety value --- 
		if j.get_button(7) == 1:
			#print('safety off')
			safety_value = 1
		else:
			safety_value = 0
		
		pub_safety_value.publish(safety_value)


		# --- steering ---
		if j.get_button(1) == 1:
			steering += steer_step
			print('steering = ', steering)
		elif j.get_button(3) == 1:
			steering -= steer_step
			print('steering = ', steering)


		#saturate steering
		steering = np.max([-1,steering])
		steering = np.min([1,steering])
		pub_steering.publish(steering)

		# --- reference velocity ---
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
