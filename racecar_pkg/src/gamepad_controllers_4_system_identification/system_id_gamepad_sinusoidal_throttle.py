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
	pub_throttle = rospy.Publisher('throttle_' + str(car_number), Float32, queue_size=8)
	pub_safety_value = rospy.Publisher('safety_value', Float32, queue_size=8)
	pub_freq = rospy.Publisher('freq_' + str(car_number), Float32, queue_size=8)

	rospy.init_node('teleop_gamepad' + str(car_number), anonymous=True)
	rate = rospy.Rate(30) # 10hz

	# initialize v_ref

	#publish sinusoidal steering
	freq = 0.1 #Hz
	amp = 0.1
	mean = 0.3
	
	start_clock_time = rospy.get_rostime()


	while not rospy.is_shutdown():
		pygame.event.pump()

		#Obtain gamepad values


		if j.get_button(1) == 1:
			freq += 0.001
			print('freq = ', freq)
		elif j.get_button(3) == 1:
			freq -= 0.001
			print('freq = ', freq)
		pub_freq.publish(freq)





		# --- safety value --- 
		if j.get_button(7) == 1:
			#print('safety off')
			safety_value = 1
		else:
			safety_value = 0
		
		pub_safety_value.publish(safety_value)


		# --- steering ---
		steering = -j.get_axis(2) * 0.1 #Right thumbstick X
		# saturate steering
		steering = np.max([-1,steering])
		steering = np.min([1,steering])
		pub_steering.publish(steering)
		pub_steering.publish(steering)

		# --- throttle ---
		# evaluate sinusoidal input
		stop_clock_time = rospy.get_rostime()
		elapsed_time = stop_clock_time.secs - start_clock_time.secs + (stop_clock_time.nsecs - start_clock_time.nsecs)/1000000000
		time_now = stop_clock_time.secs + (stop_clock_time.nsecs)/1000000000
		throttle =  mean + amp*np.sin(time_now * freq * 2 * np.pi)

		# ovverride to get reverse
		if j.get_button(4) == 1:
			throttle = -0.8
		pub_throttle.publish(throttle)






		rate.sleep()

if __name__ == '__main__':
	try:
		car_number = os.environ['car_number']
		teleop_gamepad(car_number)
	except rospy.ROSInterruptException:
		pass
