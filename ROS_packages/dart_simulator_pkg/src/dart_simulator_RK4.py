#!/usr/bin/env python3

# this script simulates the output from the optitrack, so you can use it for tests
import sys

from Helper_functions.Analytical_jetracer_models import kinematic_bicycle_2, dynamic_bicycle_2, evaluate_vy_w_kin_bike

import rospy
from std_msgs.msg import String, Float32, Float32MultiArray
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy import integrate
from custom_msgs_optitrack.msg import custom_opti_pose_stamped_msg
from tf.transformations import quaternion_from_euler
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dynamic_reconfigure.server import Server
from dynamic_reconfigure_pkg.cfg import dart_simulator_guiConfig



class Forward_intergrate_GUI_manager:
    def __init__(self, vehicles_list):
        #fory dynamic parameter change using rqt_reconfigure GUI
        self.vehicles_list = vehicles_list
        srv = Server(dart_simulator_guiConfig, self.reconfig_callback_forwards_integrate)
        


    def reconfig_callback_forwards_integrate(self, config, level):
            print('reconfiguring parameters from dynamic_reconfig')
            # self.dt_int = config['dt_int'] # set the inegrating step
            # self.rate = rospy.Rate(1 / self.dt_int) # accordingly reset the output rate of a new measurement

            vehicle_model_choice = config['dynamic_model_choice']

            for i in range(len(self.vehicles_list)):
                if vehicle_model_choice == 1:
                    self.vehicles_list[i].vehicle_model = kinematic_bicycle_2

                elif vehicle_model_choice == 2:
                    self.vehicles_list[i].vehicle_model = dynamic_bicycle_2


            reset_state_x = config['reset_state_x']
            reset_state_y = config['reset_state_y']
            reset_state_theta = config['reset_state_theta']
            reset_state = config['reset_state']

            if reset_state:
                print('Resetting state')
                reset_vehicle_number = config['reset_vehicle_number']

                for i in range(len(self.vehicles_list)):
                    if self.vehicles_list[i].car_number == reset_vehicle_number:
                        self.vehicles_list[i].state = [reset_state_x, reset_state_y, reset_state_theta, 0, 0, 0]

            return config



class Forward_intergrate_vehicle:
    def __init__(self, car_number, vehicle_model, initial_state, dt_int):
        print("Starting vehicle integrator " + str(car_number))
        # set up ros nodes for this vehicle
        print("setting ros topics and node")

        # set up variables
        self.safety_value = 0
        self.steering = 0
        self.throttle = 0
        self.state = initial_state
        self.dt_int = dt_int
        self.vehicle_model = vehicle_model
        self.reset_state = False
        self.car_number = car_number


        
        rospy.Subscriber('steering_' + str(car_number), Float32, self.callback_steering)
        rospy.Subscriber('throttle_' + str(car_number), Float32, self.callback_throttle)
        rospy.Subscriber('safety_value', Float32, self.callback_safety)
        self.state_publisher = rospy.Publisher('Optitrack_data_topic_' + str(car_number), custom_opti_pose_stamped_msg, queue_size=10)
        self.vx_publisher = rospy.Publisher('vx_' + str(car_number), Float32, queue_size=10)
        self.vy_publisher = rospy.Publisher('vy_' + str(car_number), Float32, queue_size=10)
        self.omega_publisher = rospy.Publisher('omega_' + str(car_number), Float32, queue_size=10)
        # for rviz
        self.rviz_state_publisher = rospy.Publisher('rviz_data_' + str(car_number), PoseStamped, queue_size=10)




    # Safety callback function
    def callback_safety(self, safety_val_incoming):
        self.safety_value = safety_val_incoming.data

    # Steering control callback function
    def callback_steering(self, steer):
        self.steering = steer.data

    # Throttle callback function
    def callback_throttle(self, throttle):
        if self.safety_value == 1:
            self.throttle = throttle.data
        else:
            self.throttle = 0.0

    def integrating_function(self, t, z):  # RK4 wants a function that takes as input time and state
        x_dot = self.vehicle_model(z)
        # adding zeros to the derivatives of u since they are constant (in the timestep)
        zdot = np.array([0, 0, *x_dot])  #  we need to give the derivative of the full state (including the control inputs)
        return zdot

    def forward_integrate_1_timestep(self):

        # perform forwards integration
        t0 = 0
        t_bound = self.dt_int

        # only activates if safety is off
        y0 = np.array([self.throttle, self.steering] + self.state)


        # forwards integrate using RK4
        RK45_output = integrate.RK45(self.integrating_function, t0, y0, t_bound)
        RK45_output.step()

        # z = throttle delta x y theta vx vy w
        z_next = RK45_output.y
        self.state = z_next[2:].tolist()
        # non - elegant fix: if using kinematic bicycle that does not have Vy w_dot, assign it from kinematic realtions
        if self.vehicle_model == kinematic_bicycle_2:
            # evaluate vy w from kinematic relations (so this can't be in the integrating function)
            steer_input = z_next[1]
            vx = z_next[5]
            L = 0.175

            vy, w = evaluate_vy_w_kin_bike(steer_input,vx,L)
            self.state[4] = vy
            self.state[5] = w

        # publish results of the integration
        opti_state_message = custom_opti_pose_stamped_msg()
        opti_state_message.header.stamp = rospy.Time.now()
        opti_state_message.x = self.state[0]
        opti_state_message.y = self.state[1]
        opti_state_message.rotation = self.state[2]
        self.state_publisher.publish(opti_state_message)
        self.vx_publisher.publish(self.state[3])
        self.vy_publisher.publish(self.state[4])
        self.omega_publisher.publish(self.state[5])

        # publish rviz pose stamped
        rviz_message = PoseStamped()
        #to plot centre of arrow as centre of vehicle
        rviz_message.pose.position.x = self.state[0] - L/2 * np.cos(self.state[2])
        rviz_message.pose.position.y = self.state[1] - L/2 * np.sin(self.state[2])
        quat = quaternion_from_euler(0, 0, self.state[2])

        rviz_message.pose.orientation.x = quat[0]
        rviz_message.pose.orientation.y = quat[1]
        rviz_message.pose.orientation.z = quat[2]
        rviz_message.pose.orientation.w = quat[3]

            # frame data is necessary for rviz
        rviz_message.header.frame_id = 'map'
        self.rviz_state_publisher.publish(rviz_message)






    








if __name__ == '__main__':
    try:
        rospy.init_node('Vehicles_Integrator_node' , anonymous=True)

        dt_int = 0.01
        vehicle_model = kinematic_bicycle_2
        

        #vehicle 3         #x y theta vx vy w
        initial_state_3 = [0, -0.8, 0, 0.0, 0, 0]
        car_number_3 = 3
        vehicle_3_integrator = Forward_intergrate_vehicle(car_number_3, vehicle_model, initial_state_3, dt_int)


        #vehicle 1         #x y theta vx vy w
        initial_state_1 = [0, 0, 0, 0.0, 0, 0]
        car_number_1 = 1
        vehicle_1_integrator = Forward_intergrate_vehicle(car_number_1, vehicle_model, initial_state_1, dt_int)




        vehicles_list = [vehicle_1_integrator, vehicle_3_integrator]


        #set up GUI manager
        Forward_intergrate_GUI_manager_obj = Forward_intergrate_GUI_manager(vehicles_list)


        # forwards integrate
        rate = rospy.Rate(1 / dt_int)
        while not rospy.is_shutdown():
            # forwards integrate all vehicles
            for i in range(len(vehicles_list)):
                vehicles_list[i].forward_integrate_1_timestep()
            # wait one loop time
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
