#!/usr/bin/env python3

# this script simulates the output from the optitrack, so you can use it for tests
import rospy
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import numpy as np
import tf_conversions
from dynamic_reconfigure.server import Server
from dart_simulator_pkg.cfg import dart_simulator_guiConfig
from scipy.stats import truncnorm


# import the model functions from previously installed python package
from DART_dynamic_models.dart_dynamic_models import model_functions,load_SVGPModel_actuator_dynamics_analytic



# instantiate the model functions
mf = model_functions()


def unpack_state(z):
    #z = throttle delta x y theta vx vy w
    th = z[0]
    st = z[1]
    x = z[2]
    y = z[3]
    yaw = z[4]
    vx = z[5]
    vy = z[6]
    w = z[7]

    return th, st, x, y, yaw, vx, vy, w

def produce_xdot(yaw,vx,vy,w,acc_x,acc_y,acc_w):
    xdot1 = vx * np.cos(yaw) - vy * np.sin(yaw)
    xdot2 = vx * np.sin(yaw) + vy * np.cos(yaw)
    xdot3 = w
    xdot4 = acc_x  
    xdot5 = acc_y  
    xdot6 = acc_w
    return np.array([xdot1,xdot2,xdot3,xdot4,xdot5,xdot6])


def activation_coeff(vx):
    c = 0.5
    sharpness = 40
    out = np.tanh(sharpness * (vx-c))*0.5+0.5
    # print with 2 decimal points
    #print('activation coefficient: ' + "{:.2f}".format(out))
    return out


def kinematic_bicycle(t,z,disturbance):  # RK4 wants a function that takes as input time and state
    
    th, st, x, y, yaw, vx, vy, w = unpack_state(z)

    #evaluate steering angle
    steering_angle = mf.steering_2_steering_angle(st,mf.a_s_self,mf.b_s_self,mf.c_s_self,mf.d_s_self,mf.e_s_self)

    # evaluate longitudinal forces
    Fx = + mf.motor_force(th,vx,mf.a_m_self,mf.b_m_self,mf.c_m_self)\
            + mf.rolling_friction(vx,mf.a_f_self,mf.b_f_self,mf.c_f_self,mf.d_f_self)

    acc_x =  Fx / mf.m_self # acceleration in the longitudinal direction

    #simple bycicle nominal model - using centre of mass as reference point
    w = vx * np.tan(steering_angle) / (mf.lr_self + mf.lf_self) # angular velocity
    vy = mf.l_COM_self * w

    if disturbance:
        # add disturbance if needed
        stdev_x = model_vx.max_stdev
        
        # draw a random sample with the appropriate covariance
        act_coeff = activation_coeff(vx)
        disturbance_x = act_coeff * np.random.normal(0, stdev_x) # np.random wants the standard deviation
        # add to xdot
        acc_x += disturbance_x

    xdot= produce_xdot(yaw,vx,vy,w,acc_x,0,0)

    # assemble derivatives [th, stter, x y theta vx vy omega], NOTE: # for RK4 you need to supply also the derivatives of the inputs (that are set to zero)
    #zdot = np.array([0,0, xdot1, xdot2, xdot3, xdot4, xdot5, xdot6]) 
    return xdot

def dynamic_bicycle(t,z,disturbance):  # RK4 wants a function that takes as input time and state
    # extract states
    th, st, x, y, yaw, vx, vy, w = unpack_state(z)

    #evaluate steering angle 
    steering_angle = mf.steering_2_steering_angle(st,mf.a_s_self,mf.b_s_self,mf.c_s_self,mf.d_s_self,mf.e_s_self)

    # # evaluate longitudinal forces
    Fx_wheels = + mf.motor_force(th,vx,mf.a_m_self,mf.b_m_self,mf.c_m_self)\
                + mf.rolling_friction(vx,mf.a_f_self,mf.b_f_self,mf.c_f_self,mf.d_f_self)\
                + mf.F_friction_due_to_steering(steering_angle,vx,mf.a_stfr_self,mf.b_stfr_self,mf.d_stfr_self,mf.e_stfr_self)

    c_front = (mf.m_front_wheel_self)/mf.m_self
    c_rear = (mf.m_rear_wheel_self)/mf.m_self

    # redistribute Fx to front and rear wheels according to normal load
    Fx_front = Fx_wheels * c_front
    Fx_rear = Fx_wheels * c_rear

    #evaluate slip angles
    alpha_f,alpha_r = mf.evaluate_slip_angles(vx,vy,w,mf.lf_self,mf.lr_self,steering_angle)

    #lateral forces
    Fy_wheel_f = mf.lateral_tire_force(alpha_f,mf.d_t_f_self,mf.c_t_f_self,mf.b_t_f_self,mf.m_front_wheel_self)
    Fy_wheel_r = mf.lateral_tire_force(alpha_r,mf.d_t_r_self,mf.c_t_r_self,mf.b_t_r_self,mf.m_rear_wheel_self)

    acc_x,acc_y,acc_w = mf.solve_rigid_body_dynamics(vx,vy,w,steering_angle,Fx_front,Fx_rear,Fy_wheel_f,Fy_wheel_r,mf.lf_self,mf.lr_self,mf.m_self,mf.Jz_self)

    if disturbance:
        stdev_x = model_vx.max_stdev
        stdev_y = model_vy.max_stdev
        stdev_w = model_w.max_stdev
        
        # draw a random sample with the appropriate covariance
        act_coeff = activation_coeff(vx)
        disturbance_x = act_coeff * np.random.normal(0, stdev_x) # np.random wants the standard deviation
        disturbance_y = act_coeff * np.random.normal(0, stdev_y)
        disturbance_w = act_coeff * np.random.normal(0, stdev_w)
        
        # add to xdot
        acc_x += disturbance_x
        acc_y += disturbance_y
        acc_w += disturbance_w

    xdot = produce_xdot(yaw,vx,vy,w,acc_x,acc_y,acc_w)

    return xdot




#get current folder path
#import os
import importlib.resources
#folder_path = os.path.dirname(os.path.realpath(__file__))


with importlib.resources.path('DART_dynamic_models', 'SVGP_saved_parameters') as data_path:
    folder_path = str(data_path)

with importlib.resources.path('DART_dynamic_models', 'SVGP_saved_parameters_slippery_floor') as data_path:
    folder_path_slippery_floor = str(data_path)


evalate_covariance_tag = False
model_vx,model_vy,model_w = load_SVGPModel_actuator_dynamics_analytic(folder_path,evalate_covariance_tag)

def SVGP(t,z,disturbance):  # RK4 wants a function that takes as input time and state
    th, st, x, y, yaw, vx, vy, w = unpack_state(z)

    #['vx body', 'vy body', 'w', 'throttle filtered' ,'steering filtered','throttle','steering']
    state_action_base_model = np.array([vx,vy,w,th,st])

    acc_x, cov_x = model_vx.forward(state_action_base_model)
    acc_y, cov_y = model_vy.forward(state_action_base_model)
    acc_w, cov_w = model_w.forward(state_action_base_model)
    

    # to avoid strange jittering close to 0 velocity, blend
    act_coeff = activation_coeff(vx)

    # add nominal model
    xdot_nom = dynamic_bicycle(t,z,False)
    acc_x = act_coeff * acc_x + xdot_nom[3]
    acc_y = act_coeff * acc_y + xdot_nom[4]
    acc_w = act_coeff * acc_w + xdot_nom[5]

    if disturbance:
        # add disturbance if needed
        stdev_x = model_vx.max_stdev
        stdev_y = model_vy.max_stdev
        stdev_w = model_w.max_stdev

        disturbance_x = act_coeff * np.random.normal(0, stdev_x) # np.random wants the standard deviation
        disturbance_y = act_coeff * np.random.normal(0, stdev_y)
        disturbance_w = act_coeff * np.random.normal(0, stdev_w)

        # add to xdot
        acc_x += disturbance_x
        acc_y += disturbance_y
        acc_w += disturbance_w
        # print disturbance values with 3 decimal points
        #print('disturbance x: ' + "{:.3f}".format(disturbance_x) + ' disturbance y: ' + "{:.3f}".format(disturbance_y) + ' disturbance w: ' + "{:.3f}".format(disturbance_w))

    xdot = produce_xdot(yaw,vx,vy,w,acc_x,acc_y,acc_w) 

    # assemble derivatives [th, stter, x y theta vx vy omega], NOTE: # for RK4 you need to supply also the derivatives of the inputs (that are set to zero)
    #xdot = np.array([0,0, xdot1, xdot2, xdot3, xdot4, xdot5, xdot6]) 
    return xdot


# make this cleaner
model_vx_slippery,model_vy_slippery,model_w_slippery = load_SVGPModel_actuator_dynamics_analytic(folder_path_slippery_floor,evalate_covariance_tag)


def SVGP_slippery(t,z,disturbance):  # RK4 wants a function that takes as input time and state
    th, st, x, y, yaw, vx, vy, w = unpack_state(z)

    #['vx body', 'vy body', 'w', 'throttle filtered' ,'steering filtered','throttle','steering']
    state_action_base_model = np.array([vx,vy,w,th,st])

    acc_x, cov_x = model_vx_slippery.forward(state_action_base_model)
    acc_y, cov_y = model_vy_slippery.forward(state_action_base_model)
    acc_w, cov_w = model_w_slippery.forward(state_action_base_model)
    

    # to avoid strange jittering close to 0 velocity, blend
    act_coeff = activation_coeff(vx)

    # add nominal model
    xdot_nom = dynamic_bicycle(t,z,False)
    acc_x = act_coeff * acc_x + xdot_nom[3]
    acc_y = act_coeff * acc_y + xdot_nom[4]
    acc_w = act_coeff * acc_w + xdot_nom[5]

    if disturbance:
        # add disturbance if needed
        stdev_x = model_vx_slippery.max_stdev
        stdev_y = model_vy_slippery.max_stdev
        stdev_w = model_w_slippery.max_stdev

        disturbance_x = act_coeff * np.random.normal(0, stdev_x) # np.random wants the standard deviation
        disturbance_y = act_coeff * np.random.normal(0, stdev_y)
        disturbance_w = act_coeff * np.random.normal(0, stdev_w)

        # add to xdot
        acc_x += disturbance_x
        acc_y += disturbance_y
        acc_w += disturbance_w
        # print disturbance values with 3 decimal points
        #print('disturbance x: ' + "{:.3f}".format(disturbance_x) + ' disturbance y: ' + "{:.3f}".format(disturbance_y) + ' disturbance w: ' + "{:.3f}".format(disturbance_w))

    xdot = produce_xdot(yaw,vx,vy,w,acc_x,acc_y,acc_w) 

    # assemble derivatives [th, stter, x y theta vx vy omega], NOTE: # for RK4 you need to supply also the derivatives of the inputs (that are set to zero)
    #xdot = np.array([0,0, xdot1, xdot2, xdot3, xdot4, xdot5, xdot6]) 
    return xdot






class Forward_intergrate_GUI_manager:
    def __init__(self, vehicles_list):
        #fory dynamic parameter change using rqt_reconfigure GUI
        self.vehicles_list = vehicles_list
        srv = Server(dart_simulator_guiConfig, self.reconfig_callback_forwards_integrate)
        #self.disturbance_type_options = ["None", "Truncated_Gaussian", "Flat"]
        


    def reconfig_callback_forwards_integrate(self, config, level):
            print('----------------------------------------------')
            print('reconfiguring parameters from dynamic_reconfig')

            vehicle_model_choice = config['dynamic_model_choice']

            for i in range(len(self.vehicles_list)):
                self.vehicles_list[i].actuator_dynamics = config['actuator_dynamics']
                self.vehicles_list[i].disturbance = config['disturbance']

                if vehicle_model_choice == 1:
                    print('vehicle model set to kinematic bicycle')
                    self.vehicles_list[i].vehicle_model = kinematic_bicycle

                elif vehicle_model_choice == 2:
                    print('vehicle model set to dynamic bicycle')
                    self.vehicles_list[i].vehicle_model = dynamic_bicycle

                elif vehicle_model_choice == 3:
                    print('vehicle model set to SVGP + dynamic bicycle as nominal model')
                    self.vehicles_list[i].vehicle_model = SVGP

                elif vehicle_model_choice == 4:
                    print('vehicle model set to SVGP + SVGP_dynamic_bicycle_slippery_floor')
                    self.vehicles_list[i].vehicle_model = SVGP_slippery



            reset_state_x = config['reset_state_x']
            reset_state_y = config['reset_state_y']
            reset_state_theta = config['reset_state_theta']
            reset_state = config['reset_state']

            if reset_state:
                print('resetting state')
                for i in range(len(self.vehicles_list)):
                    self.vehicles_list[i].state = [reset_state_x, reset_state_y, reset_state_theta, 0, 0, 0]

            return config



class Forward_intergrate_vehicle(model_functions):
    def __init__(self, car_number, vehicle_model, initial_state, dt_int,actuator_dynamics,disturbance):
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
        self.actuator_dynamics = actuator_dynamics
        self.disturbance = disturbance

        # internal states for actuator dynamics
        self.throttle_state = 0.0
        self.steering_state = 0.0

        
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


    def forward_integrate_1_timestep(self,rostime_begin_loop,dt_int):

        # perform forwards integration
        t0 = rostime_begin_loop.to_sec()
        t_bound = self.dt_int

        # only activates if safety is off
        if self.actuator_dynamics:
            y0 = np.array([self.throttle_state, self.steering_state, *self.state]) # use integrated throttle and steering
        else:
            y0 = np.array([self.throttle, self.steering, *self.state] )

        # using forward euler
        xdot = self.vehicle_model(t0,y0,self.disturbance)

        # evaluate elapsed time
        rostime_stop = rospy.get_rostime()
        #evalaute time needed to do the loop and print
        elapsed_dt = rostime_stop - rostime_begin_loop
        elapsed_dt = elapsed_dt.to_sec()

        if elapsed_dt > dt_int:
            # print elapesed time with 4 decimal points
            print('elapsed time exceeded simulator loop time, elapsed time: ' + "{:.4f}".format(elapsed_dt))
            
            dt_step = elapsed_dt
        else:
            dt_step = dt_int
        # step state
        #z_next = y0 + zdot * dt_step
        self.state += xdot * dt_step #z_next[2:].tolist()

        # now step internal states for actuators if needed
        if self.actuator_dynamics:
            throttle_dot = self.continuous_time_1st_order_dynamics(self.throttle_state,self.throttle,self.d_m_self)
            steering_dot = self.continuous_time_1st_order_dynamics(self.steering_state,self.steering,self.k_stdn_self)
            self.throttle_state += throttle_dot * dt_step
            self.steering_state += steering_dot * dt_step


        # non - elegant fix: if using kinematic bicycle that does not have Vy w_dot, assign it from kinematic realtions
        if self.vehicle_model == kinematic_bicycle:
            # z_next[1]
            steering_angle = mf.steering_2_steering_angle(self.steering,self.a_s_self,self.b_s_self,self.c_s_self,self.d_s_self,self.e_s_self)
            # evaluate vy w from kinematic relations (so this can't be in the integrating function)
            #steering_angle = steer_angle(z_next[1])
            vx = self.state[3]
            w = vx * np.tan(steering_angle) / (self.lf_self+self.lr_self)
            vy = self.l_COM_self * w
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
        sensor_msg.data = [elapsed_time.to_sec(), 0.0, 0.0, 0.0, 0.0, self.state[5], self.state[3], self.safety_value,self.throttle,self.steering]
        self.pub_sens_input.publish(sensor_msg)


        #publish messages for rviz visualization purposes
        self.vx_publisher.publish(self.state[3])
        self.vy_publisher.publish(self.state[4])
        self.omega_publisher.publish(self.state[5])


        #publish rviz vehicle visualization
        rviz_message = PoseStamped()
        # to plot centre of arrow as centre of vehicle shift the centre back by l_COM
        rviz_message.pose.position.x = vicon_msg.pose.pose.position.x - self.l_COM_self * np.cos(self.state[2])
        rviz_message.pose.position.y = vicon_msg.pose.pose.position.y - self.l_COM_self * np.sin(self.state[2])
        rviz_message.pose.orientation = vicon_msg.pose.pose.orientation

        # frame data is necessary for rviz
        rviz_message.header.frame_id = 'map'
        self.pub_rviz_vehicle_visualization.publish(rviz_message)






if __name__ == '__main__':
    try:
        rospy.init_node('Vehicles_Integrator_node' , anonymous=True)

        dt_int = 0.01
        vehicle_model = kinematic_bicycle
        disturbance = None
        

        #vehicle 1         #x y theta vx vy w
        initial_state_1 = [0, 0, 0, 0.0, 0, 0]
        car_number_1 = 1
        actuator_dynamics = False
        disturbance = False
        vehicle_1_integrator = Forward_intergrate_vehicle(car_number_1, vehicle_model, initial_state_1,
                                                           dt_int,actuator_dynamics,disturbance)


        vehicles_list = [vehicle_1_integrator]


        #set up GUI manager
        Forward_intergrate_GUI_manager_obj = Forward_intergrate_GUI_manager(vehicles_list)

        # forwards integrate
        #rate = rospy.Rate(1 / dt_int)
        while not rospy.is_shutdown():
            # forwards integrate all vehicles
            for i in range(len(vehicles_list)):
                # get time now
                rostime_begin_loop = rospy.get_rostime()
                vehicles_list[i].forward_integrate_1_timestep(rostime_begin_loop,dt_int)
                rostime_finished_loop = rospy.get_rostime()
                #evalaute time needed to do the loop and print
                time_for_loop = rostime_finished_loop - rostime_begin_loop
                #print('time for loop: ' + str(time_for_loop.to_sec()))

            # if you have extra time, wait until rate to keep the loop at the desired rate
            if time_for_loop.to_sec() < dt_int:
                # wait by the difference
                rospy.sleep(dt_int - time_for_loop.to_sec())


    except rospy.ROSInterruptException:
        pass
