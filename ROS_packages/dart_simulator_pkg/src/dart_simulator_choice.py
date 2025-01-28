#!/usr/bin/env python3

# this script simulates the output from the optitrack, so you can use it for tests
import sys
import rospy
from std_msgs.msg import String, Float32, Float32MultiArray
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import numpy as np
from scipy import integrate
import tf_conversions
from dynamic_reconfigure.server import Server
from dart_simulator_pkg.cfg import dart_simulator_guiConfig
from tf.transformations import quaternion_from_euler
from scipy.stats import truncnorm

## Selected noise bounds

n_up = 0.5
n_low = - n_up


# define dynamic models
# hard coded vehicle parameters
l = 0.175
l_r = 0.54*l # the reference point taken by the data is not exaclty in the center of the vehicle
#lr = 0.06 # reference position from rear axel
l_f = l-l_r

m = 1.67
Jz = 0.006513 # uniform rectangle of shape 0.18 x 0.12


### FUNCTIONS WITHOUT NOISE ###
def steer_angle(steering_command):
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

def motor_force(th,v):
    a =  28.887779235839844
    b =  5.986172199249268
    c =  -0.15045104920864105
    w = 0.5 * (np.tanh(100*(th+c))+1)
    Fm =  (a - v * b) * w * (th+c)
    return Fm

def friction(v):
    a =  1.7194761037826538
    b =  13.312559127807617
    c =  0.289848655462265
    Ff = - a * np.tanh(b  * v) - v * c
    return Ff
def slip_angles(vx,vy,w,steering_angle):
    # evaluate slip angles
    Vy_wheel_r = vy - l_r * w # lateral velocity of the rear wheel
    Vx_wheel_r = vx 
    Vx_correction_term_r = 0.1*np.exp(-100*Vx_wheel_r**2) # this avoids the vx term being 0 and having undefined angles for small velocities
    # note that it will be negligible outside the vx = [-0.2,0.2] m/s interval.
    Vx_wheel_r = Vx_wheel_r + Vx_correction_term_r
    alpha_r = - np.arctan(Vy_wheel_r/ Vx_wheel_r) / np.pi * 180  #converting alpha into degrees
                
    # front slip angle
    Vy_wheel_f = (vy + w * l_f) #* np.cos(steering_angle) - vx * np.sin(steering_angle)
    Vx_wheel_f =  vx
    Vx_correction_term_f = 0.1*np.exp(-100*Vx_wheel_f**2)
    Vx_wheel_f = Vx_wheel_f + Vx_correction_term_f
    alpha_f = -( -steering_angle + np.arctan2(Vy_wheel_f, Vx_wheel_f)) / np.pi * 180  #converting alpha into degrees
    return alpha_f, alpha_r

def lateral_tire_forces(alpha_f,alpha_r):
    #front tire Pacejka tire model
    d =  2.9751534461975098
    c =  0.6866822242736816
    b =  0.29280123114585876
    e =  -3.0720443725585938
    #rear tire linear model
    c_r = 0.38921865820884705

    F_y_f = d * np.sin(c * np.arctan(b * alpha_f - e * (b * alpha_f -np.arctan(b * alpha_f))))
    F_y_r = c_r * alpha_r
    return F_y_f, F_y_r

def kinematic_bicycle(t,z):  # RK4 wants a function that takes as input time and state
    #z = throttle delta x y theta vx vy w
    u = z[0:2]
    x = z[2:]
    th = u[0]
    vx = x[3]
    vy = x[4]
    w = x[5]

    #evaluate steering angle 
    steering_angle = steer_angle(u[1])

    #evaluate forward force
    Fx = motor_force(th,vx) + friction(vx)

    acc_x =  Fx / m # acceleration in the longitudinal direction

    #simple bycicle nominal model - using centre of vehicle as reference point
    w = vx * np.tan(steering_angle) / l
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

def dynamic_bicycle(t,z):  # RK4 wants a function that takes as input time and state
    #z = throttle delta x y theta vx vy w
    u = z[0:2]
    x = z[2:]
    th = u[0]
    vx = x[3]
    vy = x[4]
    w = x[5]

    # evaluate steering angle 
    steering_angle = steer_angle(u[1])

    # evaluare slip angles
    alpha_f,alpha_r =slip_angles(vx,vy,w,steering_angle)

    #evaluate forward force
    Fx_wheels = motor_force(th,vx) + friction(vx)
    # assuming equally shared force among wheels
    Fx_f = Fx_wheels/2
    Fx_r = Fx_wheels/2

    # evaluate lateral tire forces
    F_y_f, F_y_r = lateral_tire_forces(alpha_f,alpha_r)

    # solve equations of motion for the rigid body
    A = np.array([[+np.cos(steering_angle),1,-np.sin(steering_angle),0],
                  [+np.sin(steering_angle),0, np.cos(steering_angle),1],
                  [l_f*np.sin(steering_angle),0             ,l_f     ,-l_r]])

    b = np.array([Fx_f,
                  Fx_r,
                  F_y_f,
                  F_y_r])

    [Fx, Fy, M] = A @ b

    acc_x =  Fx / m  + w * vy# acceleration in the longitudinal direction
    acc_y =  Fy / m  - w * vx# acceleration in the latera direction
    acc_w =  M / Jz # acceleration yaw


    xdot1 = vx * np.cos(x[2]) - vy * np.sin(x[2])
    xdot2 = vx * np.sin(x[2]) + vy * np.cos(x[2])
    xdot3 = w
    xdot4 = acc_x  
    xdot5 = acc_y  
    xdot6 = acc_w  

    # assemble derivatives [th, stter, x y theta vx vy omega], NOTE: # for RK4 you need to supply also the derivatives of the inputs (that are set to zero)
    zdot = np.array([0,0, xdot1, xdot2, xdot3, xdot4, xdot5, xdot6]) 
    return zdot

### FUNCTION FOR ADDING NOISE


def lateral_tire_forces_noise(alpha_f,alpha_r):
    #front tire Pacejka tire model
    d =  2.9751534461975098
    c =  0.6866822242736816
    b =  0.29280123114585876
    e =  -3.0720443725585938
    #rear tire linear model
    c_r = 0.38921865820884705

    F_y_f = d * np.sin(c * np.arctan(b * alpha_f - e * (b * alpha_f -np.arctan(b * alpha_f)))) + truncnorm(n_low, n_up, loc=0, scale=0.1).rvs()
    F_y_r = c_r * alpha_r + truncnorm(n_low, n_up, loc=0, scale=0.1).rvs()
    return F_y_f, F_y_r

def kinematic_bicycle_noise(t,z):  # RK4 wants a function that takes as input time and state
    #z = throttle delta x y theta vx vy w
    u = z[0:2]
    x = z[2:]
    th = u[0]
    vx = x[3]
    vy = x[4]
    w = x[5]

    #evaluate steering angle 
    steering_angle = steer_angle(u[1])

    #evaluate forward force
    Fx = motor_force(th,vx) + friction(vx) + truncnorm(n_low, n_up, loc=0, scale=0.2).rvs()

    acc_x =  Fx / m # acceleration in the longitudinal direction

    #simple bycicle nominal model - using centre of vehicle as reference point
    w = vx * np.tan(steering_angle) / l
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

def dynamic_bicycle_noise(t,z):  # RK4 wants a function that takes as input time and state
    #z = throttle delta x y theta vx vy w
    u = z[0:2]
    x = z[2:]
    th = u[0]
    vx = x[3]
    vy = x[4]
    w = x[5]

    # evaluate steering angle 
    steering_angle = steer_angle(u[1])

    # evaluare slip angles
    alpha_f,alpha_r =slip_angles(vx,vy,w,steering_angle)

    #evaluate forward force
    Fx_wheels = motor_force(th,vx) + friction(vx) + truncnorm(n_low, n_up, loc=0, scale=0.1).rvs()
    # assuming equally shared force among wheels
    Fx_f = Fx_wheels/2
    Fx_r = Fx_wheels/2

    # evaluate lateral tire forces
    F_y_f, F_y_r = lateral_tire_forces_noise(alpha_f,alpha_r)

    # solve equations of motion for the rigid body
    A = np.array([[+np.cos(steering_angle),1,-np.sin(steering_angle),0],
                  [+np.sin(steering_angle),0, np.cos(steering_angle),1],
                  [l_f*np.sin(steering_angle),0             ,l_f     ,-l_r]])

    b = np.array([Fx_f,
                  Fx_r,
                  F_y_f,
                  F_y_r])

    [Fx, Fy, M] = A @ b

    acc_x =  Fx / m  + w * vy# acceleration in the longitudinal direction
    acc_y =  Fy / m  - w * vx# acceleration in the latera direction
    acc_w =  M / Jz # acceleration yaw


    xdot1 = vx * np.cos(x[2]) - vy * np.sin(x[2])
    xdot2 = vx * np.sin(x[2]) + vy * np.cos(x[2])
    xdot3 = w
    xdot4 = acc_x  
    xdot5 = acc_y  
    xdot6 = acc_w  

    # assemble derivatives [th, stter, x y theta vx vy omega], NOTE: # for RK4 you need to supply also the derivatives of the inputs (that are set to zero)
    zdot = np.array([0,0, xdot1, xdot2, xdot3, xdot4, xdot5, xdot6]) 
    return zdot


class Forward_intergrate_GUI_manager:
    def __init__(self, vehicles_list):
        #fory dynamic parameter change using rqt_reconfigure GUI
        self.vehicles_list = vehicles_list
        srv = Server(dart_simulator_guiConfig, self.reconfig_callback_forwards_integrate)
        


    def reconfig_callback_forwards_integrate(self, config, level):
            print('----------------------------------------------')
            print('reconfiguring parameters from dynamic_reconfig')
            # self.dt_int = config['dt_int'] # set the inegrating step
            # self.rate = rospy.Rate(1 / self.dt_int) # accordingly reset the output rate of a new measurement

            vehicle_model_choice = config['dynamic_model_choice']

            for i in range(len(self.vehicles_list)):
                if vehicle_model_choice == 1:
                    print('vehicle model set to kinematic bicycle')
                    self.vehicles_list[i].vehicle_model = kinematic_bicycle

                elif vehicle_model_choice == 2:
                    print('vehicle model set to dynamic bicycle')
                    self.vehicles_list[i].vehicle_model = dynamic_bicycle


            reset_state_x = config['reset_state_x']
            reset_state_y = config['reset_state_y']
            reset_state_theta = config['reset_state_theta']
            reset_state = config['reset_state']

            if reset_state:
                print('resetting state')
                for i in range(len(self.vehicles_list)):
                    self.vehicles_list[i].state = [reset_state_x, reset_state_y, reset_state_theta, 0, 0, 0]

            return config
    

class Forward_intergrate_GUI_manager_noise:
    def __init__(self, vehicles_list):
        #fory dynamic parameter change using rqt_reconfigure GUI
        self.vehicles_list = vehicles_list
        srv = Server(dart_simulator_guiConfig, self.reconfig_callback_forwards_integrate_noise)
        


    def reconfig_callback_forwards_integrate_noise(self, config, level):
        print('----------------------------------------------')
        print('reconfiguring parameters from dynamic_reconfig')
        # self.dt_int = config['dt_int'] # set the inegrating step
        # self.rate = rospy.Rate(1 / self.dt_int) # accordingly reset the output rate of a new measurement

        vehicle_model_choice = config['dynamic_model_choice']

        for i in range(len(self.vehicles_list)):
            if vehicle_model_choice == 1:
                print('vehicle model set to kinematic bicycle with noise')
                self.vehicles_list[i].vehicle_model = kinematic_bicycle_noise

            elif vehicle_model_choice == 2:
                print('vehicle model set to dynamic bicycle with noise')
                self.vehicles_list[i].vehicle_model = dynamic_bicycle_noise


        reset_state_x = config['reset_state_x']
        reset_state_y = config['reset_state_y']
        reset_state_theta = config['reset_state_theta']
        reset_state = config['reset_state']

        if reset_state:
            print('resetting state')
            for i in range(len(self.vehicles_list)):
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
        self.initial_time = rospy.Time.now()


        
        rospy.Subscriber('steering_' + str(car_number), Float32, self.callback_steering)
        rospy.Subscriber('throttle_' + str(car_number), Float32, self.callback_throttle)
        rospy.Subscriber('safety_value', Float32, self.callback_safety)
        self.pub_motion_capture_state = rospy.Publisher('vicon/jetracer' + str(car_number), PoseWithCovarianceStamped, queue_size=10)
        self.pub_sens_input = rospy.Publisher("sensors_and_input_" + str(car_number), Float32MultiArray, queue_size=1)
        # for rviz
        self.pub_rviz_vehicle_visualization = rospy.Publisher('rviz_data_' + str(car_number), PoseStamped, queue_size=10)
        self.vx_publisher = rospy.Publisher('vx_' + str(car_number), Float32, queue_size=10)
        self.vy_publisher = rospy.Publisher('vy_' + str(car_number), Float32, queue_size=10)
        self.omega_publisher = rospy.Publisher('omega_' + str(car_number), Float32, queue_size=10)



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
        RK45_output = integrate.RK45(self.vehicle_model, t0, y0, t_bound)
        while RK45_output.status != 'finished':
            RK45_output.step()

        # z = throttle delta x y theta vx vy w
        z_next = RK45_output.y
        self.state = z_next[2:].tolist()
        # non - elegant fix: if using kinematic bicycle that does not have Vy w_dot, assign it from kinematic realtions
        if self.vehicle_model == kinematic_bicycle:
            # evaluate vy w from kinematic relations (so this can't be in the integrating function)
            steering_angle = steer_angle(z_next[1])
            w = z_next[5] * np.tan(steering_angle) / l
            vy = l_r * w
            self.state[4] = vy
            self.state[5] = w



        # simulate vicon motion capture system output
        vicon_msg = PoseWithCovarianceStamped()
        vicon_msg.header.stamp = rospy.Time.now()
        vicon_msg.header.frame_id = 'map'
        vicon_msg.pose.pose.position.x = self.state[0] 
        vicon_msg.pose.pose.position.y = self.state[1] 
        vicon_msg.pose.pose.position.z = 0.0
        quaternion = tf_conversions.transformations.quaternion_from_euler(0.0, 0.0, self.state[2])
        #type(pose) = geometry_msgs.msg.Pose
        vicon_msg.pose.pose.orientation.x = quaternion[0]
        vicon_msg.pose.pose.orientation.y = quaternion[1]
        vicon_msg.pose.pose.orientation.z = quaternion[2]
        vicon_msg.pose.pose.orientation.w = quaternion[3]
        self.pub_motion_capture_state.publish(vicon_msg)

        # simulate on-board sensor data
        sensor_msg = Float32MultiArray()
        #                  current,voltage,IMU[0](acc x),IMU[1] (acc y),IMU[2] (omega rads),velocity, safety, throttle, steering
        elapsed_time = rospy.Time.now() - self.initial_time
        sensor_msg.data = [elapsed_time.to_sec(), 0.0, 0.0, 0.0, 0.0, z_next[7], z_next[5], self.safety_value,self.throttle,self.steering]
        self.pub_sens_input.publish(sensor_msg)


        #publish messages for rviz visualization purposes
        self.vx_publisher.publish(self.state[3])
        self.vy_publisher.publish(self.state[4])
        self.omega_publisher.publish(self.state[5])


        #publish rviz vehicle visualization
        rviz_message = PoseStamped()
        # to plot centre of arrow as centre of vehicle shift the centre back by l_r
        rviz_message.pose.position.x = vicon_msg.pose.pose.position.x - l_r/2 * np.cos(self.state[2])
        rviz_message.pose.position.y = vicon_msg.pose.pose.position.y - l_r/2 * np.sin(self.state[2])
        rviz_message.pose.orientation = vicon_msg.pose.pose.orientation

        # frame data is necessary for rviz
        rviz_message.header.frame_id = 'map'
        self.pub_rviz_vehicle_visualization.publish(rviz_message)


class Forward_intergrate_vehicle_noise:
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
        self.initial_time = rospy.Time.now()


        
        rospy.Subscriber('steering_' + str(car_number), Float32, self.callback_steering)
        rospy.Subscriber('throttle_' + str(car_number), Float32, self.callback_throttle)
        rospy.Subscriber('safety_value', Float32, self.callback_safety)
        self.pub_motion_capture_state = rospy.Publisher('vicon/jetracer' + str(car_number), PoseWithCovarianceStamped, queue_size=10)
        self.pub_sens_input = rospy.Publisher("sensors_and_input_" + str(car_number), Float32MultiArray, queue_size=1)
        # for rviz
        self.pub_rviz_vehicle_visualization = rospy.Publisher('rviz_data_' + str(car_number), PoseStamped, queue_size=10)
        self.vx_publisher = rospy.Publisher('vx_' + str(car_number), Float32, queue_size=10)
        self.vy_publisher = rospy.Publisher('vy_' + str(car_number), Float32, queue_size=10)
        self.omega_publisher = rospy.Publisher('omega_' + str(car_number), Float32, queue_size=10)



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



    def forward_integrate_1_timestep_noise(self):

            # perform forwards integration
            t0 = 0
            t_bound = self.dt_int

            # only activates if safety is off
            y0 = np.array([self.throttle, self.steering] + self.state)

            # forwards integrate using RK4
            RK45_output = integrate.RK45(self.vehicle_model, t0, y0, t_bound)
            while RK45_output.status != 'finished':
                RK45_output.step()

            # z = throttle delta x y theta vx vy w
            z_next = RK45_output.y
            self.state = z_next[2:].tolist()
            # non - elegant fix: if using kinematic bicycle that does not have Vy w_dot, assign it from kinematic realtions
            if self.vehicle_model == kinematic_bicycle_noise:
                # evaluate vy w from kinematic relations (so this can't be in the integrating function)
                steering_angle = steer_angle(z_next[1])
                w = z_next[5] * np.tan(steering_angle) / l
                vy = l_r * w
                self.state[4] = vy
                self.state[5] = w



            # simulate vicon motion capture system output
            vicon_msg = PoseWithCovarianceStamped()
            vicon_msg.header.stamp = rospy.Time.now()
            vicon_msg.header.frame_id = 'map'
            vicon_msg.pose.pose.position.x = self.state[0] 
            vicon_msg.pose.pose.position.y = self.state[1] 
            vicon_msg.pose.pose.position.z = 0.0
            quaternion = tf_conversions.transformations.quaternion_from_euler(0.0, 0.0, self.state[2])
            #type(pose) = geometry_msgs.msg.Pose
            vicon_msg.pose.pose.orientation.x = quaternion[0]
            vicon_msg.pose.pose.orientation.y = quaternion[1]
            vicon_msg.pose.pose.orientation.z = quaternion[2]
            vicon_msg.pose.pose.orientation.w = quaternion[3]
            self.pub_motion_capture_state.publish(vicon_msg)

            # simulate on-board sensor data
            sensor_msg = Float32MultiArray()
            #                  current,voltage,IMU[0](acc x),IMU[1] (acc y),IMU[2] (omega rads),velocity, safety, throttle, steering
            elapsed_time = rospy.Time.now() - self.initial_time
            sensor_msg.data = [elapsed_time.to_sec(), 0.0, 0.0, 0.0, 0.0, z_next[7], z_next[5], self.safety_value,self.throttle,self.steering]
            self.pub_sens_input.publish(sensor_msg)


            #publish messages for rviz visualization purposes
            self.vx_publisher.publish(self.state[3])
            self.vy_publisher.publish(self.state[4])
            self.omega_publisher.publish(self.state[5])


            #publish rviz vehicle visualization
            rviz_message = PoseStamped()
            # to plot centre of arrow as centre of vehicle shift the centre back by l_r
            rviz_message.pose.position.x = vicon_msg.pose.pose.position.x - l_r/2 * np.cos(self.state[2])
            rviz_message.pose.position.y = vicon_msg.pose.pose.position.y - l_r/2 * np.sin(self.state[2])
            rviz_message.pose.orientation = vicon_msg.pose.pose.orientation

            # frame data is necessary for rviz
            rviz_message.header.frame_id = 'map'
            self.pub_rviz_vehicle_visualization.publish(rviz_message)












if __name__ == '__main__':
    try:
        rospy.init_node('Vehicles_Integrator_node' , anonymous=True)
        dt_int = 0.01
        # Ask user for choosing the mode
        user_choice = ''
        while user_choice not in ['1', '2']:
            print("Choose the simulation mode:")
            print("1: Simulator without noise.")
            print("2: Simulator with truncated gaussian noise.")
            user_choice = input("Enter your choice (1 or 2): ")

        # Configura il metodo di integrazione in base alla scelta dell'utente
        if user_choice == '1':
            print("You selected: Simulator without noise.")
            vehicle_model = dynamic_bicycle
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


        elif user_choice == '2':
            print("You selected: Simulator with truncated gaussian noise.")
            vehicle_model = dynamic_bicycle_noise
                    #vehicle 1         #x y theta vx vy w
            initial_state_1 = [0, 0, 0, 0.0, 0, 0]
            car_number_1 = 1
            vehicle_1_integrator = Forward_intergrate_vehicle_noise(car_number_1, vehicle_model, initial_state_1, dt_int)


            vehicles_list = [vehicle_1_integrator]


            #set up GUI manager
            Forward_intergrate_GUI_manager_obj = Forward_intergrate_GUI_manager_noise(vehicles_list)

            # forwards integrate
            rate = rospy.Rate(1 / dt_int)
            while not rospy.is_shutdown():
                # forwards integrate all vehicles
                for i in range(len(vehicles_list)):
                    vehicles_list[i].forward_integrate_1_timestep_noise()
                # wait one loop time
                rate.sleep()

        
    except rospy.ROSInterruptException:
        pass
