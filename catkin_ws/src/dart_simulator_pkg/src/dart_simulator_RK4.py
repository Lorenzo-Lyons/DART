#!/usr/bin/env python3

# this script simulates the output from the optitrack, so you can use it for tests
import sys
import rospy
from std_msgs.msg import String, Float32, Float32MultiArray
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy import integrate
from custom_msgs_optitrack.msg import custom_opti_pose_stamped_msg
from tf.transformations import quaternion_from_euler
from dynamic_reconfigure.server import Server
from dynamic_reconfigure_pkg.cfg import dart_simulator_guiConfig


# define dynamic models
# hard coded vehicle parameters
L = 0.175
l_r = L/2
m = 1.67


def evaluate_steer_angle(steering_command):
    a =  1.6379064321517944
    b =  0.3301370143890381
    c =  0.019644200801849365
    d =  0.37879398465156555
    e =  1.6578725576400757

    w = 0.5 * (np.tanh(30*(steering_command+c))+1)
    steering_angle1 = b * np.tanh(a * (steering_command + c)) 
    steering_angle2 = d * np.tanh(e * (steering_command + c))
    steering_angle = (w)*steering_angle1+(1-w)*steering_angle2 

    return steering_angle

def evaluate_motor_force(th,v):
    a =  28.887779235839844
    b =  5.986172199249268
    c =  -0.15045104920864105
    w = 0.5 * (np.tanh(100*(th+c))+1)
    Fm =  (a - v * b) * w * (th+c)
    return Fm

def evaluate_friction(v):
    a =  1.7194761037826538
    b =  13.312559127807617
    c =  0.289848655462265
    Ff = - a * np.tanh(b  * v) - v * c
    return Ff


def kinematic_bicycle(t,z):  # RK4 wants a function that takes as input time and state
    #z = throttle delta x y theta vx vy w
    u = z[0:2]
    x = z[2:]
    th = u[0]
    vx = x[3]
    vy = x[4]
    w = x[5]


    #evaluate steering angle 
    steering_angle = evaluate_steer_angle(u[1])

    #evaluate forward force
    Fx = evaluate_motor_force(th,vx) + evaluate_friction(vx)

    acc_x =  Fx / m # acceleration in the longitudinal direction

    #simple bycicle nominal model - using centre of vehicle as reference point
    w = vx * np.tan(steering_angle) / L
    vy = l_r * w

    xdot1 = vx * np.cos(x[2]) - vy * np.sin(x[2])
    xdot2 = vx * np.sin(x[2]) + vy * np.cos(x[2])
    xdot3 = w
    xdot4 =  acc_x  
    xdot5 = 0  # vy dot is not used
    xdot6 = 0  # w dot is not used

    # assemble derivatives [th, stter, x y theta vx vy omega], NOTE: # for RK4 you need to supply also the derivatives of the inputs (that are set to zero)
    zdot = np.array([0,0, xdot1, xdot2, xdot3, xdot4, xdot5, xdot6]) 
    return zdot



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
                    self.vehicles_list[i].vehicle_model = kinematic_bicycle

                elif vehicle_model_choice == 2:
                    self.vehicles_list[i].vehicle_model = dynamic_bicycle


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


    def forward_integrate_1_timestep(self):

        # perform forwards integration
        t0 = 0
        t_bound = self.dt_int

        # only activates if safety is off
        y0 = np.array([self.throttle, self.steering] + self.state)

        # forwards integrate using RK4
        RK45_output = integrate.RK45(kinematic_bicycle, t0, y0, t_bound)
        while RK45_output.status != 'finished':
            RK45_output.step()

        # z = throttle delta x y theta vx vy w
        z_next = RK45_output.y
        self.state = z_next[2:].tolist()
        # non - elegant fix: if using kinematic bicycle that does not have Vy w_dot, assign it from kinematic realtions
        if self.vehicle_model == kinematic_bicycle:
            # evaluate vy w from kinematic relations (so this can't be in the integrating function)
            steering_angle = evaluate_steer_angle(z_next[1])
            w = z_next[5] * np.tan(steering_angle) / L
            vy = l_r * z_next[5]
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
        vehicle_model = kinematic_bicycle
        

        #vehicle 1         #x y theta vx vy w
        initial_state_1 = [0, 0, 0, 0.0, 0, 0]
        car_number_1 = 1
        vehicle_1_integrator = Forward_intergrate_vehicle(car_number_1, vehicle_model, initial_state_1, dt_int)


        vehicles_list = [vehicle_1_integrator]


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
