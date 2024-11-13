# This file contains the function definitions for the DART simulator package
# The dynamic model is defined here, yet it is a copy-paste from the System_identification_data_processing folder.
# This is not best practice but at least this package is then self contained and easy to get to work.

import numpy as np
import os

def directly_measured_model_parameters():
    # from vicon system measurements
    theta_correction = 0.00768628716468811 # error between vehicle axis and vicon system reference axis
    lr_reference = 0.115  #0.11650    # (measureing it wit a tape measure it's 0.1150) reference point location taken by the vicon system measured from the rear wheel
    l_lateral_shift_reference = -0.01 # the reference point is shifted laterally by this amount 
    #COM_positon = 0.084 #0.09375 #centre of mass position measured from the rear wheel

    # car parameters
    l = 0.1735 # [m]length of the car (from wheel to wheel)
    m = 1.580 # mass [kg]
    m_front_wheel = 0.847 #[kg] mass pushing down on the front wheel
    m_rear_wheel = 0.733 #[kg] mass pushing down on the rear wheel


    COM_positon = l / (1+m_rear_wheel/m_front_wheel)
    lr = COM_positon
    lf = l-lr
    # Automatically adjust following parameters according to tweaked values
    l_COM = lr_reference - COM_positon

    #lateral measurements
    l_width = 0.08 # width of the car is 8 cm
    m_left_wheels = 0.794 # mass pushing down on the left wheels
    m_right_wheels = 0.805 # mass pushing down on the right wheels
    # so ok the centre of mass is pretty much in the middle of the car so won't add this to the derivations


    Jz = 1/12 * m *(l**2+l_width**2) #0.006513 # Moment of inertia of uniform rectangle of shape 0.1735 x 0.8 NOTE this is an approximation cause the mass is not uniformly distributed
    return [theta_correction, l_COM, l_lateral_shift_reference ,lr, lf, Jz, m,m_front_wheel,m_rear_wheel]


def model_parameters():
    # collect fitted model parameters here so that they can be easily accessed

    # full velocity range
    # motor parameters
    a_m =  25.35849952697754
    b_m =  4.815326690673828
    c_m =  -0.16377617418766022
    time_C_m =  0.0843319296836853
    # friction parameters
    a_f =  1.2659882307052612
    b_f =  7.666370391845703
    c_f =  0.7393041849136353
    d_f =  -0.11231517791748047

    # steering angle curve --from fitting on vicon data
    a_s =  1.392930030822754
    b_s =  0.36576229333877563
    c_s =  0.0029959678649902344 - 0.03 # littel adjustment to allign the tire curves
    d_s =  0.5147881507873535
    e_s =  1.0230425596237183


    # Front wheel parameters:
    d_t_f =  -0.8406859636306763
    c_t_f =  0.8407371044158936
    b_t_f =  8.598039627075195
    # Rear wheel parameters:
    d_t_r =  -0.8546739816665649
    c_t_r =  0.959108829498291
    b_t_r =  11.54928207397461


    #additional friction due to steering angle
    # Friction due to steering parameters:
    a_stfr =  -0.11826395988464355
    b_stfr =  5.915864944458008
    d_stfr =  0.22619032859802246
    e_stfr =  0.7793111801147461

    # steering dynamics
    k_stdn =  0.12851488590240479

    # pitch dynamics
    k_pitch =  0.14062348008155823
    w_natural_Hz_pitch =  2.7244157791137695



    return [a_m, b_m, c_m, time_C_m,
            a_f, b_f, c_f, d_f,
            a_s, b_s, c_s, d_s, e_s,
            d_t_f, c_t_f, b_t_f,d_t_r, c_t_r, b_t_r,
            a_stfr, b_stfr,d_stfr,e_stfr,
            k_stdn,
            k_pitch,w_natural_Hz_pitch]



class model_functions():
    # load model parameters
    [theta_correction_self, l_COM_self, l_lateral_shift_reference_self ,
     lr_self, lf_self, Jz_self, m_self,m_front_wheel_self,m_rear_wheel_self] = directly_measured_model_parameters()

    [a_m_self, b_m_self, c_m_self, d_m_self,
    a_f_self, b_f_self, c_f_self, d_f_self,
    a_s_self, b_s_self, c_s_self, d_s_self, e_s_self,
    d_t_f_self, c_t_f_self, b_t_f_self,d_t_r_self, c_t_r_self, b_t_r_self,
    a_stfr_self, b_stfr_self,d_stfr_self,e_stfr_self,
    k_stdn_self,k_pitch_self,w_natural_Hz_pitch_self] = model_parameters()

    def __init__(self):
        # this is just a class to collect all the functions that are used to model the dynamics
        pass

    def minmax_scale_hm(self,min,max,normalized_value):
        # normalized value should be between 0 and 1
        return min + normalized_value * (max-min)

    def steering_2_steering_angle(self,steering_command,a_s,b_s,c_s,d_s,e_s):

        w_s = 0.5 * (np.tanh(30*(steering_command+c_s))+1)
        steering_angle1 = b_s * np.tanh(a_s * (steering_command + c_s))
        steering_angle2 = d_s * np.tanh(e_s * (steering_command + c_s))
        steering_angle = (w_s)*steering_angle1+(1-w_s)*steering_angle2
        return steering_angle
    
    def rolling_friction(self,vx,a_f,b_f,c_f,d_f):
        F_rolling = - ( a_f * np.tanh(b_f  * vx) + c_f * vx + d_f * vx**2 )
        return F_rolling
    

    def motor_force(self,throttle_filtered,v,a_m,b_m,c_m):
        w_m = 0.5 * (np.tanh(100*(throttle_filtered+c_m))+1)
        Fx =  (a_m - b_m * v) * w_m * (throttle_filtered+c_m)
        return Fx
    
    def evaluate_slip_angles(self,vx,vy,w,lf,lr,steer_angle):
        vy_wheel_f,vy_wheel_r = self.evalaute_wheel_lateral_velocities(vx,vy,w,steer_angle,lf,lr)

        # do the same but for numpy
        vx_wheel_f = np.cos(-steer_angle) * vx - np.sin(-steer_angle)*(vy + lf*w)

        Vx_correction_term_f = 1 * np.exp(-3*vx_wheel_f**2) 
        Vx_correction_term_r = 1 *  np.exp(-3*vx**2) 

        Vx_f = vx_wheel_f + Vx_correction_term_f
        Vx_r = vx + Vx_correction_term_r
        
        alpha_f = np.arctan2(vy_wheel_f,Vx_f)
        alpha_r = np.arctan2(vy_wheel_r,Vx_r)
            
        return alpha_f,alpha_r
    
    def lateral_forces_activation_term(self,vx):
        return np.tanh(100 * vx**2)

    def lateral_tire_force(self,alpha,d_t,c_t,b_t,m_wheel):

        F_y = m_wheel * 9.81 * d_t * np.sin(c_t * np.arctan(b_t * alpha))
        return F_y 
    

    def evalaute_wheel_lateral_velocities(self,vx,vy,w,steer_angle,lf,lr):

        Vy_wheel_f = - np.sin(steer_angle) * vx + np.cos(steer_angle)*(vy + lf*w) 
        Vy_wheel_r = vy - lr*w
        return Vy_wheel_f,Vy_wheel_r
    

    def F_friction_due_to_steering(self,steer_angle,vx,a,b,d,e):        # evaluate forward force

        friction_term = a + (b * steer_angle * np.tanh(30 * steer_angle))
        vx_term =  - (0.5+0.5 *np.tanh(20*(vx-0.3))) * (e + d * (vx-0.5))

        return  vx_term * friction_term


    
    def solve_rigid_body_dynamics(self,vx,vy,w,steer_angle,Fx_front,Fx_rear,Fy_wheel_f,Fy_wheel_r,lf,lr,m,Jz):

        # evaluate centripetal acceleration
        a_cent_x = + w * vy
        a_cent_y = - w * vx

        # evaluate body forces
        Fx_body =  Fx_front*(np.cos(steer_angle))+ Fx_rear + Fy_wheel_f * (-np.sin(steer_angle))

        Fy_body =  Fx_front*(np.sin(steer_angle)) + Fy_wheel_f * (np.cos(steer_angle)) + Fy_wheel_r

        M       = Fx_front * (+np.sin(steer_angle)*lf) + Fy_wheel_f * (np.cos(steer_angle)*lf)+\
                Fy_wheel_r * (-lr)
        
        acc_x = Fx_body/m + a_cent_x
        acc_y = Fy_body/m + a_cent_y
        acc_w = M/Jz
        
        return acc_x,acc_y,acc_w
    

 
    def critically_damped_2nd_order_dynamics_numpy(self,x_dot,x,forcing_term,w_Hz):
        z = 1 # critically damped system
        w_natural = w_Hz * 2 * np.pi # convert to rad/s
        x_dot_dot = w_natural ** 2 * (forcing_term - x) - 2* w_natural * z * x_dot
        return x_dot_dot


    def produce_past_action_coefficients_1st_oder(self,C,length,dt):

        k_vec = np.zeros((length,1))
        for i in range(length):
            k_vec[i] = self.impulse_response_1st_oder(i*dt,C) 
        k_vec = k_vec * dt # the dt is really important to get the amplitude right
        return k_vec 


    def impulse_response_1st_oder(self,t,C):
        return np.exp(-t/C)*1/C



    def produce_past_action_coefficients_1st_oder_step_response(self,C,length,dt):

            k_vec = np.zeros((length,1))
            for i in range(1,length): # the first value is zero because it has not had time to act yet
                k_vec[i] = self.step_response_1st_oder(i*dt,C) - self.step_response_1st_oder((i-1)*dt,C)  
        
            return k_vec 
    

    def step_response_1st_oder(self,t,C):

        return 1 - np.exp(-t/C)
        
    def continuous_time_1st_order_dynamics(self,x,forcing_term,C):
        x_dot = 1/C * (forcing_term - x)
        return x_dot
    



def load_SVGPModel_actuator_dynamics_analytic(folder_path,evalaute_cov_tag):
    svgp_params_path = folder_path + '/SVGP_saved_parameters/'

    # Define the parameter names for each dimension (x, y, w)
    param_names = ['m', 'middle', 'L_inv', 'right_vec', 'inducing_locations', 'outputscale', 'lengthscale']
    dimensions = ['x', 'y', 'w']

    # Initialize an empty dictionary to store all parameters
    svgp_params = {}

    # Loop through each dimension and parameter name to load the .npy files
    for dim in dimensions:
        svgp_params[dim] = {}
        for param in param_names:
            file_path = os.path.join(svgp_params_path, f"{param}_{dim}.npy")
            if os.path.exists(file_path):
                svgp_params[dim][param] = np.load(file_path)
                #print(f"Loaded {param}_{dim}: shape {svgp_params[dim][param].shape}")
            else:
                print(f"Warning: {param}_{dim}.npy not found in {svgp_params_path}")

    # load time delay parameters
    time_delay_parameters_path = folder_path + '/SVGP_saved_parameters/time_delay_parameters.npy'
    time_delay_parameters = np.load(time_delay_parameters_path)
    actuator_time_delay_fitting_tag = time_delay_parameters[0]
    n_past_actions = time_delay_parameters[1]
    dt_svgp = time_delay_parameters[2]

    
    # now build the models
    model_vx = SVGP_analytic(svgp_params['x']['outputscale'],
                             svgp_params['x']['lengthscale'],
                             svgp_params['x']['inducing_locations'],
                             svgp_params['x']['right_vec'],
                             svgp_params['x']['L_inv'],
                             evalaute_cov_tag)
    model_vx.actuator_time_delay_fitting_tag = actuator_time_delay_fitting_tag
    model_vx.n_past_actions = n_past_actions
    model_vx.dt = dt_svgp

    model_vy = SVGP_analytic(svgp_params['y']['outputscale'],
                                svgp_params['y']['lengthscale'],
                                svgp_params['y']['inducing_locations'],
                                svgp_params['y']['right_vec'],
                                svgp_params['y']['L_inv'],
                                evalaute_cov_tag)
    model_vy.actuator_time_delay_fitting_tag = actuator_time_delay_fitting_tag
    model_vy.n_past_actions = n_past_actions
    model_vy.dt = dt_svgp
    
    model_w = SVGP_analytic(svgp_params['w']['outputscale'],
                                svgp_params['w']['lengthscale'],
                                svgp_params['w']['inducing_locations'],
                                svgp_params['w']['right_vec'],
                                svgp_params['w']['L_inv'],
                                evalaute_cov_tag)
    model_w.actuator_time_delay_fitting_tag = actuator_time_delay_fitting_tag
    model_w.n_past_actions = n_past_actions
    model_w.dt = dt_svgp

    return model_vx,model_vy,model_w


class SVGP_analytic():
    def __init__(self,outputscale,lengthscale,inducing_locations,right_vec,L_inv,evalaute_cov_tag):

        self.outputscale = outputscale
        self.lengthscale = lengthscale
        self.inducing_locations = inducing_locations
        self.right_vec = right_vec
        self.L_inv = L_inv
        self.evalaute_cov_tag = evalaute_cov_tag

    def forward(self, x_star):
        #make x_star into a 5 x 1 array
        x_star = np.expand_dims(x_star, axis=0)
        kXZ = rebuild_Kxy_RBF_vehicle_dynamics(x_star,np.squeeze(self.inducing_locations),self.outputscale,self.lengthscale)

        # calculate mean and covariance for x
        mean = kXZ @ self.right_vec
        if self.evalaute_cov_tag:
            # calculate covariance
            X = self.L_inv @ kXZ.T
            KXX = RBF_kernel_rewritten(x_star[0],x_star[0],self.outputscale,self.lengthscale)
            cov = KXX + X.T @ self.middle @ X
        else:
            cov = 0

        return mean[0], cov
    

def rebuild_Kxy_RBF_vehicle_dynamics(X,Y,outputscale,lengthscale):
    n = X.shape[0]
    m = Y.shape[0]
    KXY = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            KXY[i,j] = RBF_kernel_rewritten(X[i,:],Y[j,:],outputscale,lengthscale)
    return KXY


def RBF_kernel_rewritten(x,y,outputscale,lengthscale):
    exp_arg = np.zeros(len(lengthscale))
    for i in range(len(lengthscale)):
        exp_arg[i] = (x[i]-y[i])**2/lengthscale[i]**2
    return outputscale * np.exp(-0.5*np.sum(exp_arg))

