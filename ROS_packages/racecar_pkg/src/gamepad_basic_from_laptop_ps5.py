#!/usr/bin/env python3

import rospy
import pygame
import time
from std_msgs.msg import Float32
import os
#this allows to run the gamepad without a video display plugged in!
os.environ["SDL_VIDEODRIVER"] = "dummy"

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

	while not rospy.is_shutdown():
		pygame.event.pump()

		# Obtain gamepad values
		throttle = -j.get_axis(1) #Left thumbstick Y
		
		steering = -j.get_axis(3) #Right thumbstick X



		# Publish gamepad values
		pub_throttle.publish(throttle)
		pub_steering.publish(steering)

		#safety value publishing
		if j.get_button(5) == 1:
			#print('safety off')
			pub_safety_value.publish(1)
		else:
			pub_safety_value.publish(0)

		rate.sleep()

if __name__ == '__main__':
	try:
		try:
			car_number = os.environ['car_number']
		except:
			car_number = 1 # set to 1 if environment variable is not set
		teleop_gamepad(car_number)
	except rospy.ROSInterruptException:
		pass
