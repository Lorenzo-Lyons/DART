#!/usr/bin/env python3


import numpy as np
import os
import sys
import rospy
from std_msgs.msg import Float32, Float32MultiArray, Bool
import rospkg



class leader_longitudinal_controller_class:
	def __init__(self, car_number):
		# initialize tau-v_inf lookup table
		self.tau_v_table = data =np.array([[0.20,0.19],
						[0.21,0.38],
						[0.22,0.61],
						[0.23,0.85],
						[0.24,1.00],
						[0.25,1.19],
						[0.26,1.32],
						[0.27,1.64],
						[0.28,1.75],
						[0.29,1.91],
						[0.30,1.99],
						[0.31,2.11],
						[0.32,2.19],
						[0.35,2.60],
						[0.40,3.15],
							])
		

		#set up variables
		self.car_number = car_number

		# set controller gains
		self.v_ref = 0.0 #[m/s]
		self.kp = 0.2
		self.k_int = 0.01
		self.tau_int = 0.0

		# initialize state variables
		self.sensors_and_input = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0]
		self.v = 0.0
		self.safety_value = 0.0

		# initiate steering variables
		self.steering_command_prev = 0

		# set up publisher
		self.throttle_publisher = rospy.Publisher('throttle_' + str(car_number), Float32, queue_size=1)

		#subscribers
		self.v_encoder_subscriber = rospy.Subscriber('sensors_and_input_' + str(car_number), Float32MultiArray, self.sensors_and_input_callback)
		self.v_ref_subscriber = rospy.Subscriber('v_ref_' + str(car_number), Float32, self.v_ref_callback)
		rospy.Subscriber('safety_value', Float32, self.safety_callback)

		# set up feed forward action
		self.evaluate_reference_throttle(self.v_ref)


	def safety_callback(self,msg):
		self.safety_value = msg.data


	def sensors_and_input_callback(self, msg):
		self.sensors_and_input = np.array(msg.data)
		# [elapsed_time, current, voltage, acc_x, acc_y, omega_rad, vel, safety_value, throttle, steering]
		self.v = self.sensors_and_input[6]

	def v_ref_callback(self, msg):
		# re-evaluate ff action
		if self.v_ref != msg.data:
			self.evaluate_reference_throttle(self.v_ref)
			print('------------')
			self.v_ref = msg.data


	def evaluate_reference_throttle(self,v_ref):
		#func = lambda th : evaluate_Fx_2(v_ref, th)
		#th_first_guess = 0.14
		#self.tau_ff = optimize.fsolve (func, th_first_guess)[0]

		if v_ref < np.min(self.tau_v_table[:,1]):
			print('the requested velocity is less than minimum velocity in the provided data, seetting to 0')
		elif v_ref > np.max(self.tau_v_table[:,1]):
			print('the requested velocity is more than maximum velocity in the provided data, seetting to maximum')

		self.tau_ff = np.interp(v_ref, self.tau_v_table[:,1], self.tau_v_table[:,0],left=0) 
		#print ('feed forward action =', self.tau_ff)

	def compute_longitudinal_control_action(self):
		# evaluate control action as a feed forward part plus a feed back part
		if self.safety_value == 0.0:
			self.tau_int = 0.0
		else:
			self.tau_int = self.tau_int - self.k_int * (self.v-self.v_ref)
			self.tau_int = np.min([0.1,self.tau_int])
			self.tau_int = np.max([-0.1,self.tau_int])
		#print(self.tau_int)

		tau_fb = - self.kp * (self.v-self.v_ref)

		# apply saturation to feedbacck part
		tau_fb = np.min([0.1,tau_fb])
		tau_fb = np.max([-0.1,tau_fb])

		# sum the two contributions
		tau = self.tau_ff + tau_fb + self.tau_int

		# apply saturation to overall control action
		tau = np.min([1,tau])
		tau = np.max([-1,tau])
		#publish command
		self.throttle_publisher.publish(tau)






if __name__ == '__main__':
	try:
		try:
			car_number = os.environ['car_number']
		except:
			car_number = 1 # set to 1 if environment variable is not set

		rospy.init_node('longitudinal_control_node_' + str(car_number), anonymous=False)
		rate = rospy.Rate(10) #Hz

		#set up longitudinal controller
		vehicle_controller = leader_longitudinal_controller_class(car_number)

		while not rospy.is_shutdown():
			#run longitudinal controller loop
			vehicle_controller.compute_longitudinal_control_action()
			rate.sleep()



	except rospy.ROSInterruptException:
		pass
