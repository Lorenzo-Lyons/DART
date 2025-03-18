import os
import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
from colorama import Fore, Style, Back

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

        if torch.is_tensor(steering_command):
            w_s = 0.5 * (torch.tanh(30*(steering_command+c_s))+1)
            steering_angle1 = b_s * torch.tanh(a_s * (steering_command + c_s)) 
            steering_angle2 = d_s * torch.tanh(e_s * (steering_command + c_s))
            steering_angle = (w_s)*steering_angle1+(1-w_s)*steering_angle2 
        else: # use numpy implementation
            w_s = 0.5 * (np.tanh(30*(steering_command+c_s))+1)
            steering_angle1 = b_s * np.tanh(a_s * (steering_command + c_s))
            steering_angle2 = d_s * np.tanh(e_s * (steering_command + c_s))
            steering_angle = (w_s)*steering_angle1+(1-w_s)*steering_angle2
        return steering_angle
    
    def rolling_friction(self,vx,a_f,b_f,c_f,d_f):
        if torch.is_tensor(vx):
            F_rolling = - ( a_f * torch.tanh(b_f  * vx) + c_f * vx + d_f * vx**2 )
        else:
            F_rolling = - ( a_f * np.tanh(b_f  * vx) + c_f * vx + d_f * vx**2 )
        return F_rolling
    

    def motor_force(self,throttle_filtered,v,a_m,b_m,c_m):
        if torch.is_tensor(throttle_filtered):
            w_m = 0.5 * (torch.tanh(100*(throttle_filtered+c_m))+1)
            Fx =  (a_m - b_m * v) * w_m * (throttle_filtered+c_m)
        else:
            w_m = 0.5 * (np.tanh(100*(throttle_filtered+c_m))+1)
            Fx =  (a_m - b_m * v) * w_m * (throttle_filtered+c_m)
        return Fx
    
    def evaluate_slip_angles(self,vx,vy,w,lf,lr,steer_angle):
        vy_wheel_f,vy_wheel_r = self.evalaute_wheel_lateral_velocities(vx,vy,w,steer_angle,lf,lr)

        if torch.is_tensor(vx):
            steer_angle_tensor = steer_angle * torch.Tensor([1]).cuda()
            vx_wheel_f = torch.cos(-steer_angle_tensor) * vx - torch.sin(-steer_angle_tensor)*(vy + lf*w)
            
            Vx_correction_term_f = 1 * torch.exp(-3*vx_wheel_f**2)
            Vx_correction_term_r = 1 * torch.exp(-3*vx**2)

            Vx_f = vx_wheel_f + Vx_correction_term_f
            Vx_r = vx + Vx_correction_term_r

            # evaluate slip angles
            alpha_f = torch.atan2(vy_wheel_f, Vx_f) 
            alpha_r = torch.atan2(vy_wheel_r, Vx_r)
        else:
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
        if torch.is_tensor(vx):
            return torch.tanh(100 * vx**2)
        else:
            return np.tanh(100 * vx**2)

    def lateral_tire_force(self,alpha,d_t,c_t,b_t,m_wheel):
        if torch.is_tensor(alpha):
            F_y = m_wheel * 9.81 * d_t * torch.sin(c_t * torch.arctan(b_t * alpha)) 
        else:
            F_y = m_wheel * 9.81 * d_t * np.sin(c_t * np.arctan(b_t * alpha))
        return F_y 
    

    def evalaute_wheel_lateral_velocities(self,vx,vy,w,steer_angle,lf,lr):
        if torch.is_tensor(vx):
            Vy_wheel_f = - torch.sin(steer_angle) * vx + torch.cos(steer_angle)*(vy + lf*w) 
            Vy_wheel_r = vy - lr*w
        else:
            Vy_wheel_f = - np.sin(steer_angle) * vx + np.cos(steer_angle)*(vy + lf*w) 
            Vy_wheel_r = vy - lr*w
        return Vy_wheel_f,Vy_wheel_r
    

    def F_friction_due_to_steering(self,steer_angle,vx,a,b,d,e):        # evaluate forward force
        if torch.is_tensor(steer_angle):
            friction_term = a + (b * steer_angle * torch.tanh(30 * steer_angle)) 
            vx_term =  - (0.5+0.5 *torch.tanh(20*(vx-0.3))) * (e + d * (vx-0.5))  
        else:
                friction_term = a + (b * steer_angle * np.tanh(30 * steer_angle))
                vx_term =  - (0.5+0.5 *np.tanh(20*(vx-0.3))) * (e + d * (vx-0.5))

        return  vx_term * friction_term


    
    def solve_rigid_body_dynamics(self,vx,vy,w,steer_angle,Fx_front,Fx_rear,Fy_wheel_f,Fy_wheel_r,lf,lr,m,Jz):
        if torch.is_tensor(vx):
            # evaluate centripetal acceleration
            a_cent_x = + w * vy  # x component of ac_centripetal
            a_cent_y = - w * vx  # y component of ac_centripetal

            # evaluate body forces
            Fx_body =  Fx_front*(torch.cos(steer_angle))+ Fx_rear + Fy_wheel_f * (-torch.sin(steer_angle))

            Fy_body =  Fx_front*(torch.sin(steer_angle)) + Fy_wheel_f * (torch.cos(steer_angle)) + Fy_wheel_r

            M       = Fx_front * (+torch.sin(steer_angle)*lf) + Fy_wheel_f * (torch.cos(steer_angle)*lf)+\
                    Fy_wheel_r * (-lr) 
            
            acc_x = Fx_body/m + a_cent_x
            acc_y = Fy_body/m + a_cent_y
            acc_w = M/Jz
        else:
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
    


    def produce_past_action_coefficients_2nd_oder_critically_damped(self,w_natural_Hz,length):
        # Generate the k coefficients for past actions
        #[d,c,b,damping,w_natural] = self.transform_parameters_norm_2_real()
        k_vec = torch.zeros((length,1)).cuda()
        k_dev_vec = torch.zeros((length,1)).cuda()
        for i in range(length):
            k_vec[i], k_dev_vec[i] = self.impulse_response_2n_oder_critically_damped(i*self.dt,w_natural_Hz) # 
        # the dt is really important to get the amplitude right
        k_vec = k_vec * self.dt
        k_dev_vec = k_dev_vec * self.dt
        return k_vec.double() ,  k_dev_vec.double()   


    def impulse_response_2n_oder_critically_damped(self,t,w_natural_Hz):
        #second order impulse response
        #[d,c,b,damping,w_natural] = self.transform_parameters_norm_2_real()
        w = w_natural_Hz * 2 *np.pi # convert to rad/s
        f = w**2 * t * torch.exp(-w*t)
        f_dev = w**2 * (torch.exp(-w*t)-w*t*torch.exp(-w*t)) 
        return f ,f_dev
    
    def critically_damped_2nd_order_dynamics_numpy(self,x_dot,x,forcing_term,w_Hz):
        z = 1 # critically damped system
        w_natural = w_Hz * 2 * np.pi # convert to rad/s
        x_dot_dot = w_natural ** 2 * (forcing_term - x) - 2* w_natural * z * x_dot
        return x_dot_dot


    def produce_past_action_coefficients_1st_oder(self,C,length,dt):
        if torch.is_tensor(C):
            k_vec = torch.zeros((length,1)).cuda()
            for i in range(length):
                k_vec[i] = self.impulse_response_1st_oder(i*dt,C) 
            k_vec = k_vec * dt # the dt is really important to get the amplitude right
            return k_vec.double()
        
        else:
            k_vec = np.zeros((length,1))
            for i in range(length):
                k_vec[i] = self.impulse_response_1st_oder(i*dt,C) 
            k_vec = k_vec * dt # the dt is really important to get the amplitude right
            return k_vec 


    def impulse_response_1st_oder(self,t,C):
        if torch.is_tensor(C):
            return torch.exp(-t/C)*1/C
        else:
            return np.exp(-t/C)*1/C



    def produce_past_action_coefficients_1st_oder_step_response(self,C,length,dt):
            if torch.is_tensor(C): # torch implementation
                k_vec = torch.zeros((length,1))#.cuda()
                for i in range(1,length):
                    k_vec[i] = self.step_response_1st_oder(i*dt,C) - self.step_response_1st_oder((i-1)*dt,C)
                k_vec = k_vec.double()
                # move to cuda if C was in cuda
                if C.is_cuda:
                    k_vec = k_vec.cuda()

            else: # numpy implementation
                k_vec = np.zeros((length,1))
                for i in range(1,length): # the first value is zero because it has not had time to act yet
                    k_vec[i] = self.step_response_1st_oder(i*dt,C) - self.step_response_1st_oder((i-1)*dt,C)  
            
            return k_vec 
    

    def step_response_1st_oder(self,t,C):
        if torch.is_tensor(C):
            return 1 - torch.exp(-t/C)
        else:
            return 1 - np.exp(-t/C)
        
    def continuous_time_1st_order_dynamics(self,x,forcing_term,C):
        x_dot = 1/C * (forcing_term - x)
        return x_dot


    def dynamic_bicycle(self, th, st, vx, vy, w ):  
        # this function takes the state input:z = [th st vx vy w] ---> output: [acc_x,acc_y,acc_w] in the car body frame


        #evaluate steering angle 
        steering_angle = self.steering_2_steering_angle(st,self.a_s_self,self.b_s_self,self.c_s_self,self.d_s_self,self.e_s_self)

        # # evaluate longitudinal forces
        Fx_wheels = + self.motor_force(th,vx,self.a_m_self,self.b_m_self,self.c_m_self)\
                    + self.rolling_friction(vx,self.a_f_self,self.b_f_self,self.c_f_self,self.d_f_self)\
                    + self.F_friction_due_to_steering(steering_angle,vx,self.a_stfr_self,self.b_stfr_self,self.d_stfr_self,self.e_stfr_self)

        c_front = (self.m_front_wheel_self)/self.m_self
        c_rear = (self.m_rear_wheel_self)/self.m_self

        # redistribute Fx to front and rear wheels according to normal load
        Fx_front = Fx_wheels * c_front
        Fx_rear = Fx_wheels * c_rear

        #evaluate slip angles
        alpha_f,alpha_r = self.evaluate_slip_angles(vx,vy,w,self.lf_self,self.lr_self,steering_angle)

        #lateral forces
        Fy_wheel_f = self.lateral_tire_force(alpha_f,self.d_t_f_self,self.c_t_f_self,self.b_t_f_self,self.m_front_wheel_self)
        Fy_wheel_r = self.lateral_tire_force(alpha_r,self.d_t_r_self,self.c_t_r_self,self.b_t_r_self,self.m_rear_wheel_self)

        acc_x,acc_y,acc_w = self.solve_rigid_body_dynamics(vx,vy,w,steering_angle,Fx_front,Fx_rear,Fy_wheel_f,Fy_wheel_r,self.lf_self,self.lr_self,self.m_self,self.Jz_self)
        
        return acc_x,acc_y,acc_w




def get_data(folder_path):
    import csv
    import os

    # This function gets (or produces) the merged data files from the specified folder
    print('Getting data')
    print('Looking for file " merged_files.csv "  in folder "', folder_path,'"')

    file_name = 'merged_files.csv'
    # Check if the CSV file exists in the folder
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path) and os.path.isfile(file_path):
        print('The CSV file exists in the specified folder.')

    else:
        print('The CSV file does not already exist in the specified folder. Proceding with file generation.')
        merge_data_files_from_a_folder(folder_path)

    #recording_name_train = file_name
    df = pd.read_csv(file_path)
    print('Raw data succesfully loaded.')
    return df



def merge_data_files_from_a_folder(folder_path):
    #this method creates a single file from all .csv files in the specified folder

    # Output file name and path
    file_name = 'merged_files.csv'
    output_file_path = folder_path + '/' + file_name

    # Get all CSV file paths in the folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    csv_files.sort(key=lambda x: os.path.basename(x))

    dataframes = []
    timing_offset = 0

    # Read each CSV file and store it in the dataframes list
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        #sometimes the files have some initial lines where all values are zero, so just remove them
        df = df[df['elapsed time sensors'] != 0.0]

        # set throttle to 0 when safety is off
        df['throttle'][df['safety_value'] == 0.0] = 0.0

        # reset time in each file to start from zero
        df['elapsed time sensors'] -= df['elapsed time sensors'].iloc[0]
        df['elapsed time sensors'] += timing_offset

        

        if 'vicon time' in df.keys():
            df['vicon time'] -= df['vicon time'].iloc[0]
            df['vicon time'] += timing_offset
            dt = np.average(df['vicon time'].diff().to_numpy()[1:]) # evaluate dt
            timing_offset = df['vicon time'].iloc[-1] + dt 
            # stitch position together so to avoid instantaneous change of position
            if dataframes:
                df['vicon x'] = df['vicon x'] - df['vicon x'].iloc[0]
                df['vicon y'] = df['vicon y'] - df['vicon y'].iloc[0]

                # now x and y must be rotated to allign with the previous file's last orientation
                theta = dataframes[-1]['vicon yaw'].iloc[-1] - df['vicon yaw'].iloc[0]
                # Compute the new x and y coordinates after rotation
                rotated_x = df['vicon x'].to_numpy() * np.cos(theta) - df['vicon y'].to_numpy() * np.sin(theta)
                rotated_y = df['vicon x'].to_numpy() * np.sin(theta) + df['vicon y'].to_numpy() * np.cos(theta)

                # this matches up the translation
                df['vicon x'] = rotated_x + dataframes[-1]['vicon x'].iloc[-1]
                df['vicon y'] = rotated_y + dataframes[-1]['vicon y'].iloc[-1]

                #not stich together the rotation angle
                df['vicon yaw'] = df['vicon yaw'] + theta #- df['vicon yaw'].iloc[0] + dataframes[-1]['vicon yaw'].iloc[-1]
                # correct yaw that may now be less than pi
                #df['vicon yaw'] = (df['vicon yaw'] + np.pi) % (2 * np.pi) - np.pi
        else:
            #update timing offset
            #extract safety off data and fix issues with timing
            dt = np.average(df['elapsed time sensors'].diff().to_numpy()[1:]) # evaluate dt
            timing_offset = df['elapsed time sensors'].iloc[-1] + dt # each file will have a dt timegap between it and the next file
        


        
        dataframes.append(df)

    # Concatenate all DataFrames into a single DataFrame vertically
    merged_df = pd.concat(dataframes, axis=0, ignore_index=True)

    #write merged csv file
    merged_df.to_csv(output_file_path, index=False)

    print('Merging complete. Merged file saved as:', output_file_path)
    return output_file_path #, num_lines



def evaluate_delay(signal1,signal2):
    # outputs delay expressed in vector index jumps
    # we assume that the signals are arrays of the same length
    if len(signal1) == len(signal2):
    
        # Use numpy's correlate function to find cross-correlation
        cross_corr = np.correlate(signal1, signal2, mode='full')
        #the length of the cross_corr vector will be N + N - 1
        # in position N you find the cross correlation for 0 delay
        # signal 1 is kept still and signal 2 is moved across. 
        # So if signal 2 is a delayed version of signal 1, the maximum
        # value of the cross correlation will accur before position N. (N means no delay)

        # Find the index of the maximum correlation
        delay_indexes = (len(signal1)) - (np.argmax(cross_corr)+1)  # plus one is needed cause np.argmax gives you the index of where that is

        return delay_indexes
    else:
        print('signals not of the same length! Stopping delay evaluation')


def process_raw_data_steering(df):
    
    # evaluate measured steering angle by doing inverse of kinematic bicycle model (only when velocity is higher than 0.8 m/s)
    # Note that dataset should not contain high velocities since the kinematic bicycle model will fail, and measured steering angle would be wrong
    L = 0.175 # distance between front and rear axels
    elapsed_time_vec = df['elapsed time sensors'][df['vel encoder'] > 0.8].to_numpy()
    #steering_delayed = df['steering delayed'][df['vel encoder'] > 0.8].to_numpy()
    steering = df['steering'][df['vel encoder'] > 0.8].to_numpy()

    vel_encoder = df['vel encoder'][df['vel encoder'] > 0.8].to_numpy()
    w_vec = df['W (IMU)'][df['vel encoder'] > 0.8].to_numpy()
    steering_angle= np.arctan2(w_vec * L ,  vel_encoder) 

    d = {'elapsed time sensors': elapsed_time_vec,
        #'W (IMU)': w_vec,
        'steering angle': steering_angle,
        #'steering delayed' : steering_delayed,
        'steering' : steering}

    df_steering_angle = pd.DataFrame(data=d)

    return df_steering_angle



def throttle_dynamics_data_processing(df_raw_data):
    mf = model_functions() # instantiate the model functions object

    dt = df_raw_data['vicon time'].diff().mean()  # Calculate the average time step
    th = 0
    filtered_throttle = np.zeros(df_raw_data.shape[0])
    # Loop through the data to compute the predicted steering angles

    ground_truth_refinement = 100 # this is used to integrate the steering angle with a higher resolution to avoid numerical errors
    for t in range(1, len(filtered_throttle)):
        # integrate ground trough with a much higher dt to have better numerical accuracy
        for k in range(ground_truth_refinement):
            th_dot = mf.continuous_time_1st_order_dynamics(th,df_raw_data['throttle'].iloc[t-1],mf.d_m_self)
            th += dt/ground_truth_refinement * th_dot
        filtered_throttle[t] = th

    return filtered_throttle


def steering_dynamics_data_processing(df_raw_data):
    mf = model_functions() # instantiate the model functions object

    # -------------------  forard integrate the steering signal  -------------------
    dt = df_raw_data['vicon time'].diff().mean()  # Calculate the average time step

    #steering = df_raw_data['steering'].to_numpy()

    # Initialize variables for the steering prediction
    st = 0
    st_vec = np.zeros(df_raw_data.shape[0])
    st_vec_angle_vec = np.zeros(df_raw_data.shape[0])

    # Loop through the data to compute the predicted steering angles
    for t in range(1, df_raw_data.shape[0]):
        ground_truth_refinement = 100 # this is used to integrate the steering angle with a higher resolution to avoid numerical errors
        for k in range(ground_truth_refinement):
            st_dot = mf.continuous_time_1st_order_dynamics(st,df_raw_data['steering'].iloc[t-1],mf.k_stdn_self) 
            # Update the steering value with the time step
            st += st_dot * dt/ground_truth_refinement

        # Compute the steering angle using the two models with weights
        steering_angle = mf.steering_2_steering_angle(st, mf.a_s_self,
                                                          mf.b_s_self,
                                                          mf.c_s_self,
                                                          mf.d_s_self,
                                                          mf.e_s_self)

        # Store the predicted steering angle
        st_vec_angle_vec[t] = steering_angle
        st_vec[t] = st

    return st_vec_angle_vec, st_vec



def plot_raw_data(df):
    plotting_time_vec = df['elapsed time sensors'].to_numpy()

    fig1, ((ax0, ax1, ax2)) = plt.subplots(3, 1, figsize=(10, 6), constrained_layout=True)
    ax0.set_title('dt check')
    ax0.plot(np.diff(df['elapsed time sensors']), label="dt", color='gray')
    ax0.set_ylabel('dt [s]')
    ax0.set_xlabel('data point')
    ax0.legend()

    # plot raw data velocity vs throttle
    ax1.set_title('Raw Velocity vs throttle')
    ax1.plot(plotting_time_vec, df['vel encoder'].to_numpy(), label="V encoder [m/s]", color='dodgerblue')
    ax1.plot(plotting_time_vec, df['throttle'].to_numpy(), label="throttle raw []", color='gray')
    # Create a background where the safety is disingaged
    mask = np.array(df['safety_value']) == 1
    ax1.fill_between(plotting_time_vec, ax1.get_ylim()[0], ax1.get_ylim()[1], where=mask, color='gray', alpha=0.1, label='safety value disingaged')
    ax1.set_xlabel('time [s]')
    ax1.legend()

    # plot raw data w vs steering
    ax2.set_title('Raw Omega')
    ax2.plot(plotting_time_vec, df['W (IMU)'].to_numpy(),label="omega IMU raw data [rad/s]", color='orchid')
    ax2.plot(plotting_time_vec, df['steering'].to_numpy(),label="steering raw []", color='pink') 
    ax2.fill_between(plotting_time_vec, ax2.get_ylim()[0], ax2.get_ylim()[1], where=mask, color='gray', alpha=0.1, label='safety value disingaged')
    ax2.set_xlabel('time [s]')
    ax2.legend()
    return ax0,ax1,ax2







# once finished working on this move it to where the data processing is done
def process_rosbag_data(bag_file):
    import rosbag
        # Initialize a list to store data
    data = []

    # Open the bag file
    with rosbag.Bag(bag_file, 'r') as bag:
        # Iterate through the messages in the specified topic
        for topic, msg, t in bag.read_messages(topics=['/vicon/jetracer1']):
            # Extract data from the PoseWithCovarianceStamped message
            timestamp = t.to_sec()  # Timestamp in seconds
            
            # Extract position (x, y, z)
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            
            # Extract orientation (quaternion x, y, z, w)
            qx = msg.pose.pose.orientation.x
            qy = msg.pose.pose.orientation.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w
            
            # Convert quaternion to yaw angle (radians)
            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
            yaw = np.arctan2(siny_cosp, cosy_cosp)

            # Add the data to the list 
            data.append([timestamp, x, y, yaw])
    # Convert the data to a DataFrame
    columns = ["vicon time", "vicon x", "vicon y", "vicon yaw"]
    df_vicon = pd.DataFrame(data, columns=columns)

    # get throttle data
    data = []
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=['/throttle_complete_stamp_1']):
            timestamp = t.to_sec()
            t1 = msg.header1.stamp.to_sec()
            t2 = msg.header2.stamp.to_sec()
            t3 = msg.header3.stamp.to_sec()
            throttle = msg.data
            data.append([timestamp,t1,t2,t3,throttle])
    columns = ["time",'time 1','time 2','time 3', "throttle"]
    df_throttle = pd.DataFrame(data, columns=columns)

    # get steering data
    data = []
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=['/steering_complete_stamp_1']):
            timestamp = t.to_sec()
            t1 = msg.header1.stamp.to_sec()
            t2 = msg.header2.stamp.to_sec()
            t3 = msg.header3.stamp.to_sec()
            steering = msg.data
            data.append([timestamp,t1,t2,t3,steering])
    columns = ["time",'time 1','time 2','time 3', "steering"]
    df_steering = pd.DataFrame(data, columns=columns)


    # get safety off data
    data = []
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=['/safety_value']):
            timestamp = t.to_sec()
            safety_value = msg.data
            data.append([timestamp,safety_value])
    columns = ["time","safety_value"]
    df_safety = pd.DataFrame(data, columns=columns)

    # assign t2 as the middle point between t1 and t3
    # this is a bit of an approximation but it should be acceptable
    t2_th_reconstructed = 0.5 * (df_throttle['time 1']+df_throttle['time 3']).to_numpy()
    t2_st_reconstructed = 0.5 * (df_steering['time 1']+df_steering['time 3']).to_numpy()

    
    # now reallign the input data with the state data coming from the vicon
    from scipy.interpolate import interp1d

    # Define FOH interpolation functions
    throttle_foh = interp1d(t2_th_reconstructed, df_throttle['throttle'].to_numpy(),
                            kind='previous', bounds_error=False, fill_value=(df_throttle['throttle'].iloc[0], df_throttle['throttle'].iloc[-1]))

    steering_foh = interp1d(t2_st_reconstructed, df_steering['steering'].to_numpy(),
                            kind='previous', bounds_error=False, fill_value=(df_steering['steering'].iloc[0], df_steering['steering'].iloc[-1]))

    safety_value_foh = interp1d(df_safety['time'].to_numpy(), df_safety['safety_value'].to_numpy(),
                                kind='previous', bounds_error=False, fill_value=(df_safety['safety_value'].iloc[0], df_safety['safety_value'].iloc[-1]))

    # Resample using First-Order Hold
    throttle_resampled = throttle_foh(df_vicon['vicon time'].to_numpy())
    steering_resampled = steering_foh(df_vicon['vicon time'].to_numpy())
    safety_value_resampled = safety_value_foh(df_vicon['vicon time'].to_numpy())

    
    
    # add to vicon df
    df_vicon['throttle'] = throttle_resampled
    df_vicon['steering'] = steering_resampled
    df_vicon['safety_value'] = safety_value_resampled

    # set throttle to zero when safety is off
    df_vicon['throttle'][safety_value_resampled != 1] = 0.0



    # plot the throttle data   (USED FOR DEBUGGIN PURPOSES)
    # # # 2 subplots 
    # # fig, ax = plt.subplots(2, 2, figsize=(8, 18), sharex=True)

    # # # Access individual subplots correctly
    # # ax1 = ax[0, 0]  # Top-left
    # # ax2 = ax[0, 1]  # Top-right
    # # ax3 = ax[1, 0]  # Bottom-left
    # # ax4 = ax[1, 1]  # Bottom-right

    # # ax1.plot(df_throttle['time 1'].to_numpy(),df_throttle['throttle'].to_numpy(),label='sent throttle from laptop')
    # # ax1.plot(df_throttle['time 2'].to_numpy(),df_throttle['throttle'].to_numpy(),label='car received throttle')
    # # ax1.plot(df_throttle['time 3'].to_numpy(),df_throttle['throttle'].to_numpy(),label='laptop received throttle back from car')
    # # ax1.plot(df_throttle['time'].to_numpy(),df_throttle['throttle'].to_numpy(),label='msg timestamp from bag',linestyle='--')
    # # ax1.set_title('throttle data')

    # # # plot time difference between the car and the laptop
    # # ax2.plot(df_throttle['time'].to_numpy(),df_throttle['time 3'].to_numpy()-df_throttle['time 1'].to_numpy(),label='laptop round trip')
    # # ax2.legend()

    # # # plot the steering data
    # # ax3.plot(df_steering['time 1'].to_numpy(),df_steering['steering'].to_numpy(),label='sent steering from laptop')
    # # ax3.plot(df_steering['time 2'].to_numpy(),df_steering['steering'].to_numpy(),label='car received steering')
    # # ax3.plot(df_steering['time 3'].to_numpy(),df_steering['steering'].to_numpy(),label='laptop received steering back from car')
    # # ax3.plot(df_steering['time'].to_numpy(),df_steering['steering'].to_numpy(),label='msg timestamp from bag',linestyle='--')
    # # ax3.set_title('steering data')
    
    # # # plot time difference between the car and the laptop
    # # ax4.plot(df_steering['time'].to_numpy(),df_steering['time 3'].to_numpy()-df_steering['time 1'].to_numpy(),label='laptop round trip')
    # # ax4.legend()

    # # # add to plot
    # # ax1.plot(t2_th_reconstructed,df_throttle['throttle'].to_numpy(),label='reconstructed t2')
    # # ax3.plot(t2_st_reconstructed,df_steering['steering'].to_numpy(),label='reconstructed t2')
    # # # add to plot
    # # ax1.plot(df_vicon['time'].to_numpy(),throttle_resampled,label='realligned throttle')
    # # ax3.plot(df_vicon['time'].to_numpy(),steering_resampled,label='realligned steering')

    # # ax1.legend()
    # # ax3.legend()
    # # plt.show()
    return df_vicon











def process_vicon_data_kinematics(df,steps_shift,using_rosbag_data=False,bag_file=None):
    print('Processing kinematics data')
    mf = model_functions()


    # resampling the robot data to have the same time as the vicon data
    if using_rosbag_data==False:
        from scipy.interpolate import interp1d

        # Step 1: Identify sensor time differences and extract sensor checkpoints
        sensor_time_diff = df['elapsed time sensors'].diff()

        # Times where sensor values change more than 0.01s (100Hz -> 10Hz)
        sensor_time = df['elapsed time sensors'][sensor_time_diff > 0.01].to_numpy()
        steering_at_checkpoints = df['steering'][sensor_time_diff > 0.01].to_numpy()

        # Step 2: Interpolate using Zero-Order Hold
        zoh_interp = interp1d(sensor_time, steering_at_checkpoints, kind='previous', bounds_error=False, fill_value="extrapolate")

        # Step 3: Apply interpolation to 'vicon time'
        df['steering'] = zoh_interp(df['vicon time'].to_numpy())

    else:
        df = process_rosbag_data(bag_file)

    
    # robot2vicon_delay = 0 # samples delay between the robot and the vicon data # very important to get it right (you can see the robot reacting to throttle and steering inputs before they have happened otherwise)
    # # this is beacause the lag between vicon-->laptop, and robot-->laptop is different. (The vicon data arrives sooner)

    # # there is a timedelay between robot and vicon system. Ideally the right way to do this would be to shift BACKWARDS in time the robot data.
    # # but equivalently we can shift FORWARDS in time the vicon data. This is done by shifting the vicon time backwards by the delay time.
    # # This is ok since we just need the data to be consistent. but be aware of this
    # df['vicon x'] = df['vicon x'].shift(+robot2vicon_delay)
    # df['vicon y'] = df['vicon y'].shift(+robot2vicon_delay)
    # df['vicon yaw'] = df['vicon yaw'].shift(+robot2vicon_delay)
    # # account for fisrt values that will be NaN
    # df['vicon x'].iloc[:robot2vicon_delay] = df['vicon x'].iloc[robot2vicon_delay]
    # df['vicon y'].iloc[:robot2vicon_delay] = df['vicon y'].iloc[robot2vicon_delay]
    # df['vicon yaw'].iloc[:robot2vicon_delay] = df['vicon yaw'].iloc[robot2vicon_delay]


    #  ---  relocating reference point to the centre of mass  ---
    df['vicon x'] = df['vicon x'] - mf.l_COM_self*np.cos(df['vicon yaw']) - mf.l_lateral_shift_reference_self*np.cos(df['vicon yaw']+np.pi/2)
    df['vicon y'] = df['vicon y'] - mf.l_COM_self*np.sin(df['vicon yaw']) - mf.l_lateral_shift_reference_self*np.sin(df['vicon yaw']+np.pi/2)
    # -----------------------------------------------------------


    # -----     KINEMATICS      ------
    df['unwrapped yaw'] = unwrap_hm(df['vicon yaw'].to_numpy()) + mf.theta_correction_self


    # --- evaluate first time derivative ---

    shifted_time0 = df['vicon time'].shift(+steps_shift)
    shifted_x0 = df['vicon x'].shift(+steps_shift)
    shifted_y0 = df['vicon y'].shift(+steps_shift)
    shifted_yaw0 = df['unwrapped yaw'].shift(+steps_shift)

    shifted_time2 = df['vicon time'].shift(-steps_shift)
    shifted_x2 = df['vicon x'].shift(-steps_shift)
    shifted_y2 = df['vicon y'].shift(-steps_shift)
    shifted_yaw2 = df['unwrapped yaw'].shift(-steps_shift)


    # Finite differences
    df['vx_abs_filtered'] = (shifted_x2 - shifted_x0) / (shifted_time2 - shifted_time0)
    df['vy_abs_filtered'] = (shifted_y2 - shifted_y0) / (shifted_time2 - shifted_time0)
    df['w']  = (shifted_yaw2 - shifted_yaw0) / (shifted_time2 - shifted_time0)

    # Handle the last 5 elements (they will be NaN due to the shift)
    df['vx_abs_filtered'].iloc[-steps_shift:] = 0
    df['vy_abs_filtered'].iloc[-steps_shift:] = 0
    df['w'].iloc[-steps_shift:] = 0

    df['vx_abs_filtered'].iloc[:steps_shift] = 0
    df['vy_abs_filtered'].iloc[:steps_shift] = 0
    df['w'].iloc[:steps_shift] = 0


    # --- evalaute second time derivative ---
    # Shifted values for steps_shift indices ahead
    shifted_vx0 = df['vx_abs_filtered'].shift(+steps_shift)
    shifted_vy0 = df['vy_abs_filtered'].shift(+steps_shift)
    shifted_w0 = df['w'].shift(+steps_shift)

    shifted_vx2 = df['vx_abs_filtered'].shift(-steps_shift)
    shifted_vy2 = df['vy_abs_filtered'].shift(-steps_shift)
    shifted_w2 = df['w'].shift(-steps_shift)

    # Calculate the finite differences for acceleration
    df['ax_abs_filtered_more'] = (shifted_vx2 - shifted_vx0) / (shifted_time2 - shifted_time0)
    df['ay_abs_filtered_more'] = (shifted_vy2 - shifted_vy0) / (shifted_time2 - shifted_time0)
    df['acc_w'] = (shifted_w2 - shifted_w0) / (shifted_time2 - shifted_time0)

    # Handle the last 5 elements (they will be NaN due to the shift)
    df['ax_abs_filtered_more'].iloc[-steps_shift:] = 0
    df['ay_abs_filtered_more'].iloc[-steps_shift:] = 0
    df['acc_w'].iloc[-steps_shift:] = 0

    df['ax_abs_filtered_more'].iloc[:steps_shift] = 0
    df['ay_abs_filtered_more'].iloc[:steps_shift] = 0
    df['acc_w'].iloc[:steps_shift] = 0


    # --- convert velocity and acceleration into body frame ---
    vx_body_vec = np.zeros(df.shape[0])
    vy_body_vec = np.zeros(df.shape[0])
    ax_body_vec_nocent = np.zeros(df.shape[0])
    ay_body_vec_nocent = np.zeros(df.shape[0])

    for i in range(df.shape[0]):
        rot_angle =  - df['unwrapped yaw'].iloc[i] # from global to body you need to rotate by -theta!

        R     = np.array([[ np.cos(rot_angle), -np.sin(rot_angle)],
                          [ np.sin(rot_angle),  np.cos(rot_angle)]])
        

        vxvy = np.expand_dims(np.array(df[['vx_abs_filtered','vy_abs_filtered']].iloc[i]),1)
        axay = np.expand_dims(np.array(df[['ax_abs_filtered_more','ay_abs_filtered_more']].iloc[i]),1)

        vxvy_body = R @ vxvy
        axay_nocent = R @ axay

        vx_body_vec[i],vy_body_vec[i] = vxvy_body[0], vxvy_body[1]
        ax_body_vec_nocent[i],ay_body_vec_nocent[i] = axay_nocent[0], axay_nocent[1]

    df['vx body'] = vx_body_vec
    df['vy body'] = vy_body_vec

    df['ax body no centrifugal'] = ax_body_vec_nocent
    df['ay body no centrifugal'] = ay_body_vec_nocent

    # add acceleration in own body frame
    accx_cent = + df['vy body'].to_numpy() * df['w'].to_numpy() 
    accy_cent = - df['vx body'].to_numpy() * df['w'].to_numpy()

    # add centrifugal forces to df
    df['ax body'] = accx_cent + df['ax body no centrifugal'].to_numpy()
    df['ay body'] = accy_cent + df['ay body no centrifugal'].to_numpy()
    return df





def process_raw_vicon_data(df,steps_shift):
    print('Processing dynamics data')

    mf = model_functions() # instantiate the model functions object

    # process kinematics from vicon data
    #df = process_vicon_data_kinematics(df,steps_shift,theta_correction, l_COM, l_lateral_shift_reference)

    # Evaluate steering angle and slip angles as they can be useful to tweak the parameters relative to the measuring system
    
    # evaluate steering angle if it is not provided
    # if 'steering angle' in df.columns:
    #     steering_angle = df['steering angle'].to_numpy()
    # else:
    steering_angle = mf.steering_2_steering_angle(df['steering'].to_numpy(),
                                                            mf.a_s_self,
                                                            mf.b_s_self,
                                                            mf.c_s_self,
                                                            mf.d_s_self,
                                                            mf.e_s_self)
    df['steering angle'] = steering_angle 


    # if the provided data has the forward euler integrated inputs i.e. it has the input dynamics data, then use that instead
    if 'steering angle filtered' in df.columns:
        steering_angle = df['steering angle filtered'].to_numpy()
    else:
        steer_angle = df['steering angle'].to_numpy()






    df['Vx_wheel_front'] =  np.cos(-steering_angle) * df['vx body'].to_numpy() - np.sin(-steering_angle)*(df['vy body'].to_numpy() + mf.lf_self*df['w'].to_numpy())
    
    # evaluate slip angles
    a_slip_f, a_slip_r = mf.evaluate_slip_angles(df['vx body'].to_numpy(),df['vy body'].to_numpy(),df['w'].to_numpy(),mf.lf_self,mf.lr_self,steering_angle)

    # add new columns
    df['slip angle front'] = a_slip_f
    df['slip angle rear'] = a_slip_r


    # -----     DYNAMICS      ------
    # evaluate forces in body frame starting from the ones in the absolute frame
    Fx_wheel_vec = np.zeros(df.shape[0])
    Fy_r_wheel_vec = np.zeros(df.shape[0])
    Fy_f_wheel_vec = np.zeros(df.shape[0])

    # evalauting lateral velocities on wheels
    V_y_f_wheel = np.zeros(df.shape[0])

    # evaluate lateral forces from lateral and yaw dynamics
    for i in range(0,df.shape[0]):

        # ax body no centrifugal are just the forces rotated by the yaw angle
        b = np.array([df['ax body no centrifugal'].iloc[i]*mf.m_self,
                      df['ay body no centrifugal'].iloc[i]*mf.m_self,
                     (df['acc_w'].iloc[i])*mf.Jz_self]) 
        
        steer_angle = steering_angle[i] # df['steering angle'].iloc[i]
        
        # accounting for static load partitioning on Fx
        c_front = (mf.m_front_wheel_self)/mf.m_self
        c_rear = (mf.m_rear_wheel_self)/mf.m_self

        A = np.array([[+c_front * np.cos(steer_angle) + c_rear * 1,-np.sin(steer_angle)     , 0],
                      [+c_front * np.sin(steer_angle)             ,+np.cos(steer_angle)     , 1],
                      [+c_front * mf.lf_self * np.sin(steer_angle)        , mf.lf_self * np.cos(steer_angle),-mf.lr_self]])
        
        [Fx_i_wheel, Fy_f_wheel, Fy_r_wheel] = np.linalg.solve(A, b)

        Fx_wheel_vec[i]   = Fx_i_wheel
        Fy_f_wheel_vec[i] = Fy_f_wheel
        Fy_r_wheel_vec[i] = Fy_r_wheel
        

        # evaluate wheel lateral velocities
        V_y_f_wheel[i] = np.cos(steer_angle)*(df['vy body'].to_numpy()[i] + mf.lf_self*df['w'].to_numpy()[i]) - np.sin(steer_angle) * df['vx body'].to_numpy()[i]
    V_y_r_wheel = df['vy body'].to_numpy() - mf.lr_self*df['w'].to_numpy()

    # add new columns
    df['Fx wheel'] = Fx_wheel_vec  # this is the force on a single wheel
    df['Fy front wheel'] = Fy_f_wheel_vec
    df['Fy rear wheel'] = Fy_r_wheel_vec
    df['V_y front wheel'] = V_y_f_wheel
    df['V_y rear wheel'] = V_y_r_wheel

    return df


def unwrap_hm(x):  # this function is used to unwrap the angles
    if isinstance(x, (int, float)):
        return np.unwrap([x])[0]
    elif isinstance(x, np.ndarray):
        return np.unwrap(x)
    else:
        raise ValueError("Invalid input type. Expected 'float', 'int', or 'numpy.ndarray'.")


# def generate_tensor_past_actions(df, n_past_actions, refinement_factor, key_to_repeat):
#     # Initialize a list to store the refined past action values
#     refined_past_actions = []
    
#     # Iterate over the past actions and create the refined past actions directly
#     for i in range(0, n_past_actions):
#         # Shift the values for each past action step
#         past_action = df[key_to_repeat].shift(i, fill_value=0)
        
#         # Refine the action values by zero-order hold and append them to the refined list
#         for k in range(refinement_factor):
#             refined_past_actions.append(past_action)

#     # Convert the refined past actions list into a numpy array (or tensor)
#     refined_past_actions_matrix = np.stack(refined_past_actions, axis=1)
    
#     # Convert the matrix into a tensor and move it to the GPU (if available)
#     train_x = torch.tensor(refined_past_actions_matrix) #.cuda()

#     return train_x


def generate_tensor_past_actions(df, n_past_actions, key_to_repeat):
    # ordered left to right from the most recent to the oldest
    #[t0 t-1 t-2 t-3 ... t-n]

    # Initialize a list to store the past action values
    past_actions = []
    
    # Iterate over the past actions and create the past action values directly
    for i in range(n_past_actions):
        # Shift the values for each past action step
        past_action = df[key_to_repeat].shift(i, fill_value=0)
        past_actions.append(past_action)

    # Convert the past actions list into a numpy array (or tensor)
    past_actions_matrix = np.stack(past_actions, axis=1)
    
    # Convert the matrix into a tensor
    train_x = torch.tensor(past_actions_matrix)
    
    return train_x



def plot_kinemaitcs_data(df):
    # plot vicon data filtering process
    plotting_time_vec = df['vicon time'].to_numpy()

    fig1, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    ax1.set_title('velocity x')
    #ax1.plot(plotting_time_vec, df['vx_abs_raw'].to_numpy(), label="vicon abs vx raw", color='k')
    ax1.plot(plotting_time_vec, df['vx body'].to_numpy(), label="vx body", color='dodgerblue')
    ax1.legend()

    ax4.set_title('acceleration x')
    ax4.plot(plotting_time_vec, df['ax body'].to_numpy(), label="ax body", color='dodgerblue',alpha=0.3)
    ax4.legend()


    ax2.set_title('velocity y')
    ax2.plot(plotting_time_vec, df['vy body'].to_numpy(), label="vy body", color='orangered')
    ax2.legend()

    ax5.set_title('acceleration y')
    ax5.plot(plotting_time_vec, df['ay body'].to_numpy(), label="ay body", color='orangered',alpha=0.3)
    ax5.legend()


    ax3.set_title('velocity yaw')
    #ax3.plot(plotting_time_vec, df['w_abs_raw'].to_numpy(), label="vicon w raw", color='k')
    ax3.plot(plotting_time_vec, df['w'].to_numpy(), label="w", color='slateblue')
    ax3.legend()

    ax6.set_title('acceleration yaw')
    #ax6.plot(plotting_time_vec, df['aw_abs_raw'].to_numpy(), label="vicon aw raw", color='k')
    #ax6.plot(plotting_time_vec, df['aw_abs_filtered'].to_numpy(), label="vicon aw filtered", color='k')
    ax6.plot(plotting_time_vec, df['acc_w'].to_numpy(), label="acc w", color='slateblue',alpha=0.3)
    ax6.legend()

    ax_vx = ax1
    ax_vy = ax2
    ax_w = ax3
    ax_acc_x = ax4
    ax_acc_y = ax5
    ax_acc_w = ax6





    # plot raw vicon data
    fig1, ((ax1, ax2, ax3 , ax4)) = plt.subplots(4, 1, figsize=(10, 6), constrained_layout=True)
    # get axis for velocity data
    ax_vx2 = ax2
    ax_w2 = ax3

    ax1.set_title('Velocity data')
    #ax1.plot(plotting_time_vec, df['vx_abs'].to_numpy(), label="Vx abs data", color='lightblue')
    #ax1.plot(plotting_time_vec, df['vy_abs'].to_numpy(), label="Vy abs data", color='rosybrown')
    ax1.plot(plotting_time_vec, df['vx body'].to_numpy(), label="Vx body", color='dodgerblue')
    ax1.plot(plotting_time_vec, df['vy body'].to_numpy(), label="Vy body", color='orangered')
    ax1.legend()

    # plot body frame data time history
    ax2.set_title('Vx data raw vicon')
    ax2.plot(plotting_time_vec, df['throttle'].to_numpy(), label="Throttle",color='gray', alpha=1)
    ax2.plot(plotting_time_vec, df['vx body'].to_numpy(), label="Vx body frame",color='dodgerblue')
    #plot safety value
    mask = np.array(df['safety_value']) == 1
    ax2.fill_between(plotting_time_vec, ax2.get_ylim()[0], ax2.get_ylim()[1], where=mask, color='gray', alpha=0.1, label='safety value disingaged')
    #ax2.plot(plotting_time_vec,df['safety_value'].to_numpy(),label='safety value',color='k')
    if 'vel encoder' in df.columns:
        ax2.plot(plotting_time_vec, df['vel encoder'].to_numpy(),label="Velocity Encoder raw", color='indigo')
    if 'Vx_wheel_front' in df.columns:
        ax2.plot(plotting_time_vec, df['Vx_wheel_front'].to_numpy(), label="Vx front wheel",color='navy')
    

    
    ax2.legend()
    # plot omega data time history
    ax3.set_title('Omega data time history')
    ax3.plot(plotting_time_vec, df['steering'].to_numpy(),label="steering input raw data", color='pink') #  -17 / 180 * np.pi * 
    if 'W IMU' in df.columns:
        ax3.plot(plotting_time_vec, df['W IMU'].to_numpy(),label="omega IMU raw data", color='orchid')
    #ax3.plot(plotting_time_vec, df['w_abs'].to_numpy(), label="omega opti", color='lightblue')
    ax3.plot(plotting_time_vec, df['w'].to_numpy(), label="omega opti filtered",color='slateblue')
    ax3.legend()

    ax4.set_title('x - y - theta time history')
    ax4.plot(plotting_time_vec, df['vicon x'].to_numpy(), label="x opti",color='slateblue')
    ax4.plot(plotting_time_vec, df['vicon y'].to_numpy(), label="y opti",color='orangered')
    ax4.plot(plotting_time_vec, df['unwrapped yaw'].to_numpy(), label="unwrapped theta",color='yellowgreen')
    ax4.plot(plotting_time_vec, df['vicon yaw'].to_numpy(), label="theta raw data", color='darkgreen')
    ax4.legend()

    return ax_vx,ax_vy, ax_w, ax_acc_x,ax_acc_y,ax_acc_w,ax_vx2,ax_w2



def plot_vicon_data(df):
    
    ax_vx,ax_vy, ax_w, ax_acc_x,ax_acc_y,ax_acc_w = plot_kinemaitcs_data(df)

    plotting_time_vec = df['vicon time'].to_numpy()

    # plot slip angles
    fig2, ((ax1, ax2, ax3)) = plt.subplots(3, 1, figsize=(10, 6), constrained_layout=True)
    ax1.set_title('slip angle front')
    ax1.plot(plotting_time_vec, df['slip angle front'].to_numpy(), label="slip angle front", color='peru')
    ax1.plot(plotting_time_vec, df['slip angle rear'].to_numpy(), label="slip angle rear", color='darkred')
    # ax1.plot(plotting_time_vec, df['acc_w'].to_numpy(), label="acc w", color='slateblue')
    # ax1.plot(plotting_time_vec, df['vy body'].to_numpy(), label="Vy body", color='orangered')
    # ax1.plot(plotting_time_vec, df['vx body'].to_numpy(), label="Vx body", color='dodgerblue')
    ax1.legend()

    ax2.set_title('Wheel lateral velocities')
    ax2.plot(plotting_time_vec, df['V_y front wheel'].to_numpy(), label="V_y rear wheel", color='peru')
    ax2.plot(plotting_time_vec, df['V_y rear wheel'].to_numpy(), label="V_y front wheel", color='darkred')
    ax2.legend()


    ax3.set_title('Normalized Steering and acc W')
    ax3.plot(plotting_time_vec, df['acc_w'].to_numpy()/df['acc_w'].max(), label="acc w normalized", color='slateblue')
    ax3.plot(plotting_time_vec, df['steering'].to_numpy()/df['steering'].max(), label="steering normalized", color='purple')
    #ax3.plot(df['vicon time'].to_numpy(),df['steering angle time delayed'].to_numpy()/df['steering angle time delayed'].max(),label='steering angle time delayed normalized',color='k')
    ax3.legend()


    # instantiate the model functions object to instantiate the fitted model parameters
    mf = model_functions() 




    # plot Wheel velocity vs force data
    fig1, ((ax_wheel_f_alpha,ax_wheel_r_alpha)) = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)
    # determine x limits
    x_lim_alpha = [np.min([df['slip angle rear'].min(),df['slip angle front'].min()]),
             np.max([df['slip angle rear'].max(),df['slip angle front'].max()])]
    
    # evaluate wheel curve
    slip_angles_to_plot = np.linspace(x_lim_alpha[0],x_lim_alpha[1],100)
    wheel_curve_f = mf.lateral_tire_force(slip_angles_to_plot,
                                              mf.d_t_f_self,
                                              mf.c_t_f_self,
                                              mf.b_t_f_self,
                                              mf.m_front_wheel_self)



    wheel_curve_r = mf.lateral_tire_force(slip_angles_to_plot,
                                              mf.d_t_r_self,
                                              mf.c_t_r_self,
                                              mf.b_t_r_self,
                                              mf.m_rear_wheel_self)

    
    y_lim_alpha = [np.min([df['Fy front wheel'].min(),df['Fy rear wheel'].min()]),
                   np.max([df['Fy front wheel'].max(),df['Fy rear wheel'].max()])]
    
    #color_code_label = 'steering'
    color_code_label = 'ax body'
    #color_code_label = 'ay body'
    cmap = 'Spectral'
    #cmap = 'plasma'

    c_front = df[color_code_label].to_numpy()

    scatter_front = ax_wheel_f_alpha.scatter(df['slip angle front'].to_numpy(),df['Fy front wheel'].to_numpy(),label='front wheel',c=c_front,cmap=cmap,s=3) #df['vel encoder'].to_numpy()- 

    cbar1 = fig1.colorbar(scatter_front, ax=ax_wheel_f_alpha)
    cbar1.set_label(color_code_label)  # Label the colorbar  'vel encoder-vx body'

    #ax_wheel_f.scatter(df['V_y rear wheel'].to_numpy(),df['Fy rear wheel'].to_numpy(),label='rear wheel',color='darkred',s=3)
    scatter_rear = ax_wheel_r_alpha.scatter(df['slip angle rear'].to_numpy(),df['Fy rear wheel'].to_numpy(),label='rear wheel',c=c_front,cmap=cmap,s=3)

    #add wheel curve
    ax_wheel_f_alpha.plot(slip_angles_to_plot,wheel_curve_f,color='silver',label='Tire model',linewidth=4,linestyle='--')
    ax_wheel_r_alpha.plot(slip_angles_to_plot,wheel_curve_r,color='silver',label='Tire model',linewidth=4,linestyle='--')


    ax_wheel_f_alpha.scatter(np.array([0.0]),np.array([0.0]),color='orangered',label='zero',marker='+', zorder=20) # plot zero as an x 
    ax_wheel_r_alpha.scatter(np.array([0.0]),np.array([0.0]),color='orangered',label='zero',marker='+', zorder=20) # plot zero as an x

    ax_wheel_r_alpha.set_xlabel('slip angle [rad]')
    ax_wheel_r_alpha.set_ylabel('Fy')
    ax_wheel_r_alpha.set_xlim(x_lim_alpha[0],x_lim_alpha[1])
    ax_wheel_r_alpha.set_ylim(y_lim_alpha[0],y_lim_alpha[1])
    ax_wheel_r_alpha.legend()


    ax_wheel_f_alpha.set_xlabel('slip angle [rad]') 
    ax_wheel_f_alpha.set_ylabel('Fy')
    ax_wheel_f_alpha.set_xlim(x_lim_alpha[0],x_lim_alpha[1])
    ax_wheel_f_alpha.set_ylim(y_lim_alpha[0],y_lim_alpha[1])
    ax_wheel_f_alpha.legend()
    ax_wheel_f_alpha.set_title('Wheel lateral forces')
    #colorbar = fig1.colorbar(scatter, label='steering angle time delayed derivative')
 















    # plot dt data to check no jumps occur
    fig1, ((ax1)) = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    ax1.plot(df['vicon time'].to_numpy(),df['vicon time'].diff().to_numpy())
    ax1.set_title('time steps')

    # plot acceleration data
    fig1, ((ax1, ax2, ax3),(ax_acc_x_body, ax_acc_y_body, ax_acc_w)) = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    ax1.plot(df['vicon time'].to_numpy(), df['ax body no centrifugal'].to_numpy(),label='acc x absolute measured in body frame',color = 'dodgerblue')
    ax1.set_xlabel('time [s]')
    ax1.set_title('X_ddot @ R(yaw)')
    ax1.legend()

    ax2.plot(df['vicon time'].to_numpy(), df['ay body no centrifugal'].to_numpy(),label='acc y absolute measured in body frame',color = 'orangered')
    ax2.set_xlabel('time [s]')
    ax2.set_title('Y_ddot @ R(yaw)')
    ax2.legend()

    ax3.plot(df['vicon time'].to_numpy(), df['acc_w'].to_numpy(),label='dt',color = 'slateblue')
    ax3.set_xlabel('time [s]')
    ax3.set_title('Acc w')
    ax3.legend()

    # plot accelerations in the body frame
    ax_acc_x_body.plot(df['vicon time'].to_numpy(), df['ax body'].to_numpy(),label='acc x in body frame',color = 'dodgerblue')
    ax_acc_x_body.set_xlabel('time [s]')
    ax_acc_x_body.set_title('X_ddot @ R(yaw) + cent')
    ax_acc_x_body.legend()

    ax_acc_y_body.plot(df['vicon time'].to_numpy(), df['ay body'].to_numpy(),label='acc y in body frame',color = 'orangered')
    ax_acc_y_body.set_xlabel('time [s]')
    ax_acc_y_body.set_title('Y_ddot @ R(yaw) + cent')
    ax_acc_y_body.legend()

    ax_acc_w.plot(df['vicon time'].to_numpy(), df['acc_w'].to_numpy(),label='acc w',color = 'slateblue')
    ax_acc_w.set_xlabel('time [s]')
    ax_acc_w.set_title('Acc w')
    ax_acc_w.legend()




    # plot x-y trajectory
    plt.figure()
    plt.plot(df['vicon x'].to_numpy(),df['vicon y'].to_numpy())
    plt.title('x-y trajectory')

    # plot the steering angle time delayed vs W  Usefull to get the steering delay right
    plt.figure()
    plt.title('steering angle time delayed vs W nomalized')
    plt.plot(df['vicon time'].to_numpy(),df['steering angle'].to_numpy()/df['steering angle'].max(),label='steering angle normalized')
    #plt.plot(df['vicon time'].to_numpy(),df['steering angle time delayed'].to_numpy()/df['steering angle time delayed'].max(),label='steering angle time delayed normalized')
    plt.plot(df['vicon time'].to_numpy(),df['w'].to_numpy()/df['w'].max(),label='w filtered normalized')
    plt.legend()


    #plot wheel force saturation
    # plot acceleration data
    # evaluate total wheel forces abs value
    Fy_f_wheel_abs = (df['Fy front wheel'].to_numpy()**2 + df['Fx wheel'].to_numpy()**2)**0.5
    Fy_r_wheel_abs = (df['Fy rear wheel'].to_numpy()**2 + df['Fx wheel'].to_numpy()**2)**0.5

    wheel_slippage = np.abs(df['vel encoder'].to_numpy() - df['vx body'].to_numpy())

    fig1, ((ax_total_force_front,ax_total_force_rear)) = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    ax_total_force_front.plot(df['vicon time'].to_numpy(), Fy_f_wheel_abs,label='Total wheel force front',color = 'peru')
    ax_total_force_front.plot(df['vicon time'].to_numpy(), wheel_slippage,label='longitudinal slippage',color = 'gray')
    ax_total_force_front.plot(df['vicon time'].to_numpy(), df['ax body no centrifugal'].to_numpy(),label='longitudinal acceleration',color = 'dodgerblue')
    ax_total_force_front.plot(df['vicon time'].to_numpy(), df['ay body no centrifugal'].to_numpy(),label='lateral acceleration',color = 'orangered')
    ax_total_force_front.set_xlabel('time [s]')
    ax_total_force_front.set_title('Front total wheel force')
    ax_total_force_front.legend()

    ax_total_force_rear.plot(df['vicon time'].to_numpy(), Fy_r_wheel_abs,label='Total wheel force rear',color = 'darkred')
    ax_total_force_rear.plot(df['vicon time'].to_numpy(), wheel_slippage,label='longitudinal slippage',color = 'gray')
    ax_total_force_rear.plot(df['vicon time'].to_numpy(), df['ax body no centrifugal'].to_numpy(),label='longitudinal acceleration',color = 'dodgerblue')
    ax_total_force_rear.plot(df['vicon time'].to_numpy(), df['ay body no centrifugal'].to_numpy(),label='lateral acceleration',color = 'orangered')
    ax_total_force_rear.set_xlabel('time [s]')
    ax_total_force_rear.set_title('Rear total wheel force')
    ax_total_force_rear.legend()

    # plotting forces
    fig1, ((ax_lat_force,ax_long_force)) = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    accx_cent = + df['vy body'].to_numpy() * df['w'].to_numpy() 
    accy_cent = - df['vx body'].to_numpy() * df['w'].to_numpy() 
    ax_lat_force.plot(df['vicon time'].to_numpy(), df['Fy front wheel'].to_numpy(),label='Fy front measured',color = 'peru')
    ax_lat_force.plot(df['vicon time'].to_numpy(), df['Fy rear wheel'].to_numpy(),label='Fy rear measured',color = 'darkred')
    ax_lat_force.plot(df['vicon time'].to_numpy(), wheel_slippage,label='longitudinal slippage',color = 'gray')
    ax_lat_force.plot(df['vicon time'].to_numpy(), accx_cent + df['ax body no centrifugal'].to_numpy(),label='longitudinal acceleration (with cent))',color = 'dodgerblue')
    ax_lat_force.plot(df['vicon time'].to_numpy(), accy_cent + df['ay body no centrifugal'].to_numpy(),label='lateral acceleration (with cent)',color = 'orangered')
    ax_lat_force.set_xlabel('time [s]')
    ax_lat_force.set_title('Lateral wheel forces')
    ax_lat_force.legend()

    ax_long_force.plot(df['vicon time'].to_numpy(), df['Fx wheel'].to_numpy(),label='longitudinal forces',color = 'dodgerblue')
    ax_long_force.plot(df['vicon time'].to_numpy(), wheel_slippage,label='longitudinal slippage',color = 'gray')
    ax_long_force.set_xlabel('time [s]')
    ax_long_force.set_title('Longitudinal wheel force')
    ax_long_force.legend()



    return ax_wheel_f_alpha,ax_wheel_r_alpha,ax_total_force_front,\
ax_total_force_rear,ax_lat_force,ax_long_force,\
ax_acc_x_body,ax_acc_y_body,ax_acc_w







class friction_curve_model(torch.nn.Sequential,model_functions):
    def __init__(self):
        super(friction_curve_model, self).__init__()
        # initialize parameters NOTE that the initial values should be [0,1], i.e. they should be the normalized value.
        self.register_parameter(name='a', param=torch.nn.Parameter(torch.Tensor([0.5]).cuda()))
        self.register_parameter(name='b', param=torch.nn.Parameter(torch.Tensor([0.5]).cuda()))
        self.register_parameter(name='c', param=torch.nn.Parameter(torch.Tensor([0.5]).cuda()))
        self.register_parameter(name='d', param=torch.nn.Parameter(torch.Tensor([0.5]).cuda()))


    def transform_parameters_norm_2_real(self):
        # Normalizing the fitting parameters is necessary to handle parameters that have different orders of magnitude.
        # This method converts normalized values to real values. I.e. maps from [0,1] --> [min_val, max_val]
        # so every parameter is effectively constrained to be within a certain range.
        # where min_val max_val are set here in this method as the first arguments of minmax_scale_hm

        constraint_weights = torch.nn.Hardtanh(0, 1) # this constraint will make sure that the parmeter is between 0 and 1

        #friction curve = -  a * tanh(b  * v) - v * c
        a = self.minmax_scale_hm(0.1,3.0,constraint_weights(self.a))
        b = self.minmax_scale_hm(1,100,constraint_weights(self.b))
        c = self.minmax_scale_hm(0.01,2,constraint_weights(self.c))
        d = self.minmax_scale_hm(-0.2,0.2,constraint_weights(self.d))
        return [a,b,c,d]
  
    def minmax_scale_hm(self,min,max,normalized_value):
    # normalized value should be between 0 and 1
        return min + normalized_value * (max-min)
    
    def forward(self, train_x):  # this is the model that will be fitted
        # --- friction evalaution
        [a,b,c,d] = self.transform_parameters_norm_2_real()
        return self.rolling_friction(train_x,a,b,c,d)




class motor_and_friction_model(torch.nn.Sequential,model_functions):
    def __init__(self,n_previous_throttle,dt,fit_friction_flag):
        super(motor_and_friction_model, self).__init__()
        # define number of past throttle actions to keep use
        self.n_previous_throttle = n_previous_throttle
        self.dt = dt
        self.fit_friction_flag = fit_friction_flag


        # initialize parameters NOTE that the initial values should be [0,1], i.e. they should be the normalized value.
        # motor parameters
        self.register_parameter(name='a_m', param=torch.nn.Parameter(torch.Tensor([0.5]).cuda()))
        self.register_parameter(name='b_m', param=torch.nn.Parameter(torch.Tensor([0.5]).cuda()))
        self.register_parameter(name='c_m', param=torch.nn.Parameter(torch.Tensor([0.5]).cuda()))
        self.register_parameter(name='time_C_m', param=torch.nn.Parameter(torch.Tensor([0.5]).cuda()))
        # friction parameters
        self.register_parameter(name='a_f', param=torch.nn.Parameter(torch.Tensor([0.5]).cuda()))
        self.register_parameter(name='b_f', param=torch.nn.Parameter(torch.Tensor([0.5]).cuda()))
        self.register_parameter(name='c_f', param=torch.nn.Parameter(torch.Tensor([0.5]).cuda()))
        self.register_parameter(name='d_f', param=torch.nn.Parameter(torch.Tensor([0.5]).cuda()))

    def transform_parameters_norm_2_real(self):
        # Normalizing the fitting parameters is necessary to handle parameters that have different orders of magnitude.
        # This method converts normalized values to real values. I.e. maps from [0,1] --> [min_val, max_val]
        # so every parameter is effectively constrained to be within a certain range.
        # where min_val max_val are set here in this method as the first arguments of minmax_scale_hm
        constraint_weights = torch.nn.Hardtanh(0, 1) # this constraint will make sure that the parmeter is between 0 and 1

        # motor curve F= (a - v * b) * w * (throttle+c) : w = 0.5 * (torch.tanh(100*(throttle+c))+1)
        a_m = self.minmax_scale_hm(0,45,constraint_weights(self.a_m))
        b_m = self.minmax_scale_hm(0,15,constraint_weights(self.b_m))
        c_m = self.minmax_scale_hm(-0.3,0,constraint_weights(self.c_m))
        time_C_m = self.minmax_scale_hm(0.0001,0.5,constraint_weights(self.time_C_m))

        # friction curve F= -  a * tanh(b  * v) - v * c
        a_f = self.minmax_scale_hm(0.1,3.0,constraint_weights(self.a_f))
        b_f = self.minmax_scale_hm(1,100,constraint_weights(self.b_f))
        c_f = self.minmax_scale_hm(0.01,2,constraint_weights(self.c_f))
        d_f = self.minmax_scale_hm(-0.2,0.2,constraint_weights(self.d_f))

        return [a_m,b_m,c_m,time_C_m,a_f,b_f,c_f,d_f]
    
        
    def minmax_scale_hm(self,min,max,normalized_value):
    # normalized value should be between 0 and 1
        return min + normalized_value * (max-min)
    
    def forward(self, train_x):  # this is the model that will be fitted
        v = torch.unsqueeze(train_x[:,0],1)
        throttle_mat = train_x[:,1:]

        # evaluate motor force as a function of the throttle
        [a_m,b_m,c_m,time_C_m,a_f,b_f,c_f,d_f] = self.transform_parameters_norm_2_real()

        #k_vec = self.produce_past_action_coefficients_1st_oder_step_response(time_C_m,self.n_previous_throttle,self.dt)

        k_vec_base = self.produce_past_action_coefficients_1st_oder_step_response(time_C_m,self.n_previous_throttle+1,self.dt) # 
        # using average between current and following coefficient
        #k_vec = 0.5 * (k_vec_base[0:-1] + k_vec_base[1:])
        k_vec = k_vec_base[1:]

        filtered_throttle_model = throttle_mat @ k_vec

        if self.fit_friction_flag:
            Friction = self.rolling_friction(v,a_f,b_f,c_f,d_f)
        else:
            Friction = self.rolling_friction(v,self.a_f_self,self.b_f_self,self.c_f_self,self.d_f_self)
        
        
        Fx = self.motor_force(filtered_throttle_model,v,a_m,b_m,c_m) + Friction

        return Fx, filtered_throttle_model, k_vec


class steering_curve_model(torch.nn.Sequential,model_functions):
    def __init__(self):
        super(steering_curve_model, self).__init__()
        self.register_parameter(name='a_s', param=torch.nn.Parameter(torch.Tensor([0.5])))
        self.register_parameter(name='b_s', param=torch.nn.Parameter(torch.Tensor([0.5])))
        self.register_parameter(name='c_s', param=torch.nn.Parameter(torch.Tensor([0.5])))
        self.register_parameter(name='d_s', param=torch.nn.Parameter(torch.Tensor([0.5])))
        self.register_parameter(name='e_s', param=torch.nn.Parameter(torch.Tensor([0.5])))

    def transform_parameters_norm_2_real(self):
        # Normalizing the fitting parameters is necessary to handle parameters that have different orders of magnitude.
        # This method converts normalized values to real values. I.e. maps from [0,1] --> [min_val, max_val]
        # so every parameter is effectively constrained to be within a certain range.
        # where min_val max_val are set here in this method as the first arguments of minmax_scale_hm
        constraint_weights = torch.nn.Hardtanh(0, 1) # this constraint will make sure that the parmeter is between 0 and 1

        a_s = self.minmax_scale_hm(0.1,5,constraint_weights(self.a_s))
        b_s = self.minmax_scale_hm(0.2,0.6,constraint_weights(self.b_s))
        c_s = self.minmax_scale_hm(-0.1,0.1,constraint_weights(self.c_s))

        d_s = self.minmax_scale_hm(0.2,0.6,constraint_weights(self.d_s))
        e_s = self.minmax_scale_hm(0.1,5,constraint_weights(self.e_s))
        return [a_s,b_s,c_s,d_s,e_s]
        
    def minmax_scale_hm(self,min,max,normalized_value):
    # normalized value should be between 0 and 1
        return min + normalized_value * (max-min)

    def forward(self, steering_command):
        [a_s,b_s,c_s,d_s,e_s] = self.transform_parameters_norm_2_real()
        steering_angle = self.steering_2_steering_angle(steering_command,a_s,b_s,c_s,d_s,e_s)
        return steering_angle
     

class pacejka_tire_model(torch.nn.Sequential,model_functions):
    def __init__(self):
        super(pacejka_tire_model, self).__init__()

        # initialize parameters NOTE that the initial values should be [0,1], i.e. they should be the normalized value.
        self.register_parameter(name='d_f', param=torch.nn.Parameter(torch.Tensor([0.5]).cuda()))
        self.register_parameter(name='c_f', param=torch.nn.Parameter(torch.Tensor([0.5]).cuda()))
        self.register_parameter(name='b_f', param=torch.nn.Parameter(torch.Tensor([0.5]).cuda()))


        self.register_parameter(name='d_r', param=torch.nn.Parameter(torch.Tensor([0.5]).cuda()))
        self.register_parameter(name='c_r', param=torch.nn.Parameter(torch.Tensor([0.5]).cuda()))
        self.register_parameter(name='b_r', param=torch.nn.Parameter(torch.Tensor([0.5]).cuda()))

    def transform_parameters_norm_2_real(self):
        # Normalizing the fitting parameters is necessary to handle parameters that have different orders of magnitude.
        # This method converts normalized values to real values. I.e. maps from [0,1] --> [min_val, max_val]
        # so every parameter is effectively constrained to be within a certain range.
        # where min_val max_val are set here in this method as the first arguments of minmax_scale_hm

        constraint_weights = torch.nn.Hardtanh(0, 1) # this constraint will make sure that the parmeter is between 0 and 1
        
        d_f = self.minmax_scale_hm(0,-2,constraint_weights(self.d_f))
        c_f = self.minmax_scale_hm(0,2,constraint_weights(self.c_f))
        b_f = self.minmax_scale_hm(0.01,20,constraint_weights(self.b_f))
        #e_f = self.minmax_scale_hm(-1,1,constraint_weights(self.e_f))

        # rear tire
        d_r = self.minmax_scale_hm(0,-2,constraint_weights(self.d_r))
        c_r = self.minmax_scale_hm(0,2,constraint_weights(self.c_r))
        b_r = self.minmax_scale_hm(0.01,20,constraint_weights(self.b_r))
        #e_r = self.minmax_scale_hm(-1,1,constraint_weights(self.e_r))

        return [d_f,c_f,b_f,d_r,c_r,b_r]
        
    def minmax_scale_hm(self,min,max,normalized_value):
    # normalized value should be between 0 and 1
        return min + normalized_value * (max-min)
    
    def forward(self, train_x):  # this is the model that will be fitted
        alpha_front = torch.unsqueeze(train_x[:,0],1)
        alpha_rear  = torch.unsqueeze(train_x[:,1],1) 
    
        [d_t_f,c_t_f,b_t_f,d_t_r,c_t_r,b_t_r] = self.transform_parameters_norm_2_real() 
        # evalaute lateral tire force

        F_y_f = self.lateral_tire_force(alpha_front,d_t_f,c_t_f,b_t_f,self.m_front_wheel_self) # adding front-rear nominal loading
        F_y_r = self.lateral_tire_force(alpha_rear,d_t_r,c_t_r,b_t_r,self.m_rear_wheel_self) 

        return F_y_f,F_y_r


class pacejka_tire_model_pitch(torch.nn.Sequential,model_functions):
    def __init__(self,n_past_actions,dt):
        super(pacejka_tire_model_pitch, self).__init__()

        self.dt = dt
        self.n_past_actions = n_past_actions


        # initialize parameters NOTE that the initial values should be [0,1], i.e. they should be the normalized value.
        self.register_parameter(name='k_pitch', param=torch.nn.Parameter(torch.Tensor([0.5]).cuda()))
        self.register_parameter(name='w_natural_Hz_pitch', param=torch.nn.Parameter(torch.Tensor([0.5]).cuda()))

    def transform_parameters_norm_2_real(self):
        # Normalizing the fitting parameters is necessary to handle parameters that have different orders of magnitude.
        # This method converts normalized values to real values. I.e. maps from [0,1] --> [min_val, max_val]
        # so every parameter is effectively constrained to be within a certain range.
        # where min_val max_val are set here in this method as the first arguments of minmax_scale_hm

        constraint_weights = torch.nn.Hardtanh(0, 1) # this constraint will make sure that the parmeter is between 0 and 1
        
        k_pitch = self.minmax_scale_hm(0,0.5,constraint_weights(self.k_pitch))
        w_natural_Hz_pitch = self.minmax_scale_hm(0,5,constraint_weights(self.w_natural_Hz_pitch))

        return [k_pitch,w_natural_Hz_pitch]
        
    def minmax_scale_hm(self,min,max,normalized_value):
        # normalized value should be between 0 and 1
        return min + normalized_value * (max-min)
    
    def forward(self, train_x):  # this is the model that will be fitted
        alpha_front = torch.unsqueeze(train_x[:,0],1)
        alpha_rear  = torch.unsqueeze(train_x[:,1],1)
        acc_x = torch.unsqueeze(train_x[:,2],1) # this is useful in intial phase where you are looking if there is an influence of the acceleration on the tire forces in the first place

        [k_pitch,w_natural_Hz_pitch] = self.transform_parameters_norm_2_real() 

        past_acc_x = train_x[:,3:]

        # #produce past action coefficients
        k_vec_pitch,k_dev_vec_pitch = self.produce_past_action_coefficients_2nd_oder_critically_damped(w_natural_Hz_pitch,self.n_past_actions) # 

        # # convert to rad/s
        w_natural_pitch = w_natural_Hz_pitch * 2 *np.pi

        # # pitch dynamics
        c_pitch = 2 * w_natural_pitch 
        k_pitch_dynamics = w_natural_pitch**2
        normal_force_non_scaled = past_acc_x @ k_vec_pitch + c_pitch/k_pitch_dynamics * past_acc_x @ k_dev_vec_pitch # this is the non-scaled response (we don't know the magnitude of the input)
    
        acc_x_filtered = past_acc_x @ k_vec_pitch # for later plotting purpouses

        # evaluate influence coefficients based on equilibrium of moments
        l_tilde = -0.5*self.lf_self**2-0.5*self.lr_self**2-self.lf_self*self.lr_self
        l_star = (self.lf_self-self.lr_self)/2
        #z_COM = 0.07 #  

        k_pitch_front = k_pitch * (+self.lf_self + l_star)/l_tilde  / 9.81 # covert to Kg force
        #k_pitch_rear =  k_pitch * (-self.lr + l_star)/l_tilde  / 9.81 # covert to Kg force

        acc_x_term_f = normal_force_non_scaled * k_pitch_front  #* torch.tanh(-k_roll * acc_x**2) # activates it after 1.5 m/s^2
        #acc_x_term_r = acc_x * k_pitch_rear  #D_m_r = -acc_x * k_pitch_rear  + k_roll * acc_y

        # evalaute lateral tire force
        F_y_f = self.lateral_tire_force(alpha_front,self.d_t_f_self,self.c_t_f_self,self.b_t_f_self,self.m_front_wheel_self + acc_x_term_f) # adding front-rear nominal loading
        F_y_r = self.lateral_tire_force(alpha_rear,self.d_t_r_self,self.c_t_r_self,self.b_t_r_self,self.m_rear_wheel_self) #+ D_m_r

        return F_y_f,F_y_r, acc_x_filtered



class steering_dynamics_model_NN(torch.nn.Module):
    def __init__(self,input_size, output_size):

        super(steering_dynamics_model_NN, self).__init__()
        self.linear_layer = torch.nn.Linear(input_size, output_size)
        self.activation = torch.nn.Tanh()

    def forward(self,train_x):
        out = self.linear_layer(train_x)
        #out_activation = self.activation(out)

        return out #* out_activation


class steering_dynamics_model_first_order(torch.nn.Sequential,model_functions):
    def __init__(self,n_past_actions,dt):
        super(steering_dynamics_model_first_order, self).__init__()
        self.n_past_actions = n_past_actions
        self.dt = dt

        # initialize parameters NOTE that the initial values should be [0,1], i.e. they should be the normalized value.
        #steering dynamics parameters
        self.register_parameter(name='k', param=torch.nn.Parameter(torch.Tensor([0.5]).cuda()))

        self.constraint_weights = torch.nn.Hardtanh(0, 1)

    def transform_parameters_norm_2_real(self):
        k = self.minmax_scale_hm(0,0.3,self.constraint_weights(self.k))
        return [k]
    
    def forward(self,train_x):

        # extract parameters
        [k] = self.transform_parameters_norm_2_real()

        # #produce past action coefficients
        #k_vec = self.produce_past_action_coefficients_1st_oder(k,self.n_past_actions) # 
        k_vec = self.produce_past_action_coefficients_1st_oder_step_response(k,self.n_past_actions,self.dt)


        steering = train_x @ k_vec

        steering_angle = self.steering_2_steering_angle(steering,
                                                        self.a_s_self,
                                                        self.b_s_self,
                                                        self.c_s_self,
                                                        self.d_s_self,
                                                        self.e_s_self)

        return steering_angle
    



class steering_friction_model(torch.nn.Sequential,model_functions):
    def __init__(self):
        
        super(steering_friction_model, self).__init__()
        # initialize parameters NOTE that the initial values should be [0,1], i.e. they should be the normalized value.
        self.register_parameter(name='a_stfr', param=torch.nn.Parameter(torch.Tensor([0.5]).cuda()))
        self.register_parameter(name='b_stfr', param=torch.nn.Parameter(torch.Tensor([0.5]).cuda()))
        self.register_parameter(name='d_stfr', param=torch.nn.Parameter(torch.Tensor([0.5]).cuda()))
        self.register_parameter(name='e_stfr', param=torch.nn.Parameter(torch.Tensor([0.5]).cuda()))



    def transform_parameters_norm_2_real(self):
        # Normalizing the fitting parameters is necessary to handle parameters that have different orders of magnitude.
        # This method converts normalized values to real values. I.e. maps from [0,1] --> [min_val, max_val]
        # so every parameter is effectively constrained to be within a certain range.
        # where min_val max_val are set here in this method as the first arguments of minmax_scale_hm

        constraint_weights = torch.nn.Hardtanh(0, 1) # this constraint will make sure that the parmeter is between 0 and 1

        #friction curve F= -  a * tanh(b  * v) - v * c
        a_stfr = self.minmax_scale_hm(-1,0,constraint_weights(self.a_stfr))
        b_stfr = self.minmax_scale_hm(0,15,constraint_weights(self.b_stfr))

        e_stfr = self.minmax_scale_hm(0.5,1.5,constraint_weights(self.e_stfr))
        d_stfr = self.minmax_scale_hm(-2,2,constraint_weights(self.d_stfr))

        return [a_stfr,b_stfr,d_stfr,e_stfr]
 
        
    def minmax_scale_hm(self,min,max,normalized_value):
    # normalized value should be between 0 and 1
        return min + normalized_value * (max-min)

    
    def forward(self, train_x):  # this is the model that will be fitted
        
        #returns vx_dot,vy_dot,w_dot in the vehicle body frame
        # train_x = [ 'vx body', 'vy body', 'w', 'throttle' ,'steering angle'
        vx = torch.unsqueeze(train_x[:,0],1)
        vy = torch.unsqueeze(train_x[:,1],1) 
        w = torch.unsqueeze(train_x[:,2],1) 
        throttle = torch.unsqueeze(train_x[:,3],1) 
        steer = torch.unsqueeze(train_x[:,4],1) 

        # convert steering to steering angle
        steer_angle = self.steering_2_steering_angle(steer,self.a_s_self,self.b_s_self,self.c_s_self,self.d_s_self,self.e_s_self)

        [a_stfr,b_stfr,d_stfr,e_stfr] = self.transform_parameters_norm_2_real()

        # # evaluate longitudinal forces
        Fx_wheels = + self.motor_force(throttle,vx,self.a_m_self,self.b_m_self,self.c_m_self)\
                    + self.rolling_friction(vx,self.a_f_self,self.b_f_self,self.c_f_self,self.d_f_self)\
                    + self.F_friction_due_to_steering(steer_angle,vx,a_stfr,b_stfr,d_stfr,e_stfr)

        c_front = (self.m_front_wheel_self)/self.m_self
        c_rear = (self.m_rear_wheel_self)/self.m_self

        # redistribute Fx to front and rear wheels according to normal load
        Fx_front = Fx_wheels * c_front
        Fx_rear = Fx_wheels * c_rear

        #evaluate slip angles
        alpha_f,alpha_r = self.evaluate_slip_angles(vx,vy,w,self.lf_self,self.lr_self,steer_angle)

        #lateral forces
        Fy_wheel_f = self.lateral_tire_force(alpha_f,self.d_t_f_self,self.c_t_f_self,self.b_t_f_self,self.m_front_wheel_self)
        Fy_wheel_r = self.lateral_tire_force(alpha_r,self.d_t_r_self,self.c_t_r_self,self.b_t_r_self,self.m_rear_wheel_self)

        acc_x,acc_y,acc_w = self.solve_rigid_body_dynamics(vx,vy,w,steer_angle,Fx_front,Fx_rear,Fy_wheel_f,Fy_wheel_r,self.lf_self,self.lr_self,self.m_self,self.Jz_self)

        return acc_x,acc_y,acc_w












def plot_motor_friction_curves(df,acceleration_curve_model_obj,fitting_friction):

    #plot motor characteristic curve
    tau_vec = torch.unsqueeze(torch.linspace(-1,1,100),1).cuda()
    v_vec = torch.unsqueeze(torch.linspace(0,df['vel encoder smoothed'].max(),100),1).cuda()
    data_vec = torch.cat((tau_vec, v_vec), 1)

    
    #plot friction curve
    friction_vec = acceleration_curve_model_obj.friction_curve(v_vec).detach().cpu().numpy()

    fig1, ((ax1)) = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    ax1.plot(v_vec.cpu().numpy(),friction_vec,label = 'Friction curve',zorder=20,color='orangered',linewidth=5)
    ax1.set_xlabel('velocity [m\s]')
    ax1.set_ylabel('[N]')
    ax1.set_title('Friction curve')
    #ax1.grid()

    if fitting_friction:
        return ax1
    
    else:
        # also plot motor curve
        Fx_vec = acceleration_curve_model_obj.motor_curve(data_vec).detach().cpu().numpy()

        fig1, ((ax2)) = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
        ax2.plot(tau_vec.cpu().numpy(),Fx_vec,label = 'Th curve')
        ax2.set_title('Motor curve curve')
        ax2.set_xlabel('Throttle')
        ax2.set_ylabel('[N]')
        ax2.grid()

        return (ax1,ax2)
    






def produce_long_term_predictions(input_data, forward_function,prediction_window,jumps,forward_propagate_indexes):

    # forward_function is what will perform acc_x, acc_y, acc_w = forward_function(vx,vy,w,steering,throttle)

    # plotting long term predictions on data
    # each prediction window starts from a data point and then the quantities are propagated according to the provided model,
    # so they are not tied to the Vx Vy W data in any way. Though the throttle and steering inputs are taken from the data of course.

    # --- plot fitting results ---
    # input_data = 'vicon time', 'vx body', 'vy body', 'w', 'throttle filtered' ,'steering filtered', 'throttle' ,'steering','vicon x','vicon y','vicon yaw'

    #prepare tuple containing the long term predictions
    n_states = 5
    n_inputs = 2

    states_list = list(range(1,n_states+1))

    long_term_preds = ()
    


    # iterate through each prediction window
    print('------------------------------')
    print('producing long term predictions')
    from tqdm import tqdm
    tqdm_obj = tqdm(range(0,input_data.shape[0],jumps), desc="long term preds", unit="pred")

    for i in tqdm_obj:
        

        #reset couner
        k = 0
        elpsed_time_long_term_pred = 0

        # set up initial states
        long_term_pred = np.expand_dims(input_data[i, :],0)


        # iterate through time indexes of each prediction window
        while elpsed_time_long_term_pred < prediction_window and k + i + 1 < len(input_data):
            #store time values
            #long_term_pred[k+1,0] = input_data[k+i, 0] 

            dt = input_data[i + k + 1, 0] - input_data[i + k, 0]
            elpsed_time_long_term_pred = elpsed_time_long_term_pred + dt

            #produce propagated state
            state_action_k = long_term_pred[k,1:n_states+n_inputs + 1]
            
            # run it through the model
            accelerations = forward_function(state_action_k) # absolute accelerations in the current vehicle frame of reference
            
            # evaluate new state
            new_state_new_frame_candidate = long_term_pred[k,1:n_states+1] + accelerations * dt 

            # Initialize the new state
            new_state_new_frame = np.zeros(n_states)

            # Forward propagate the quantities (vx, vy, w)
            for idx in states_list: 
                if idx in forward_propagate_indexes:
                    new_state_new_frame[idx-1] = new_state_new_frame_candidate[idx-1]
                else:
                    new_state_new_frame[idx-1] = input_data[i + k + 1, idx] # no mius one here because the first entry is time

        

            # forward propagate x y yaw state
            x_index = n_states+n_inputs+1
            y_index = n_states+n_inputs+2
            yaw_index = n_states+n_inputs+3

            rot_angle = long_term_pred[k,yaw_index] # extract yaw angle
            R = np.array([
                [np.cos(rot_angle), -np.sin(rot_angle), 0],
                [np.sin(rot_angle), np.cos(rot_angle), 0],
                [0, 0, 1]
            ])

            # absolute velocities
            abs_vxvyw = R @ np.array([long_term_pred[k,1],long_term_pred[k,2],long_term_pred[k,3]])

            # propagate x y yaw according to the previous state
            new_xyyaw = np.array([long_term_pred[k,x_index],long_term_pred[k,y_index],long_term_pred[k,yaw_index]]) + abs_vxvyw * dt

            # put everything together
            current_time_index = i + k + 1
            new_row = np.array([input_data[current_time_index, 0], # time
                                *new_state_new_frame,
                                input_data[current_time_index,n_states+1], # throttle input
                                input_data[current_time_index,n_states+2], # steering input
                                *new_xyyaw])
            
            long_term_pred = np.vstack([long_term_pred, new_row])

            # update k
            k = k + 1

        long_term_preds += (long_term_pred,)  

    return long_term_preds









class dyn_model_culomb_tires(model_functions):
    def __init__(self,steering_friction_flag,pitch_dynamics_flag):

        self.pitch_dynamics_flag = pitch_dynamics_flag
        self.steering_friction_flag = steering_friction_flag


        if self.pitch_dynamics_flag:
            self.w_natural_pitch = self.w_natural_Hz_pitch_self * 2 *np.pi
            self.c_pitch = 2 * self.w_natural_pitch 
            self.k_pitch_dynamics = self.w_natural_pitch**2

            # evaluate influence coefficients based on equilibrium of moments
            l_tilde = -0.5*self.lf_self**2-0.5*self.lr_self**2-self.lf_self*self.lr_self
            l_star = (self.lf_self-self.lr_self)/2
            self.k_pitch_front = self.k_pitch_self * (+self.lf_self + l_star)/l_tilde  / 9.81 



    def forward(self, state_action):
        #returns vx_dot,vy_dot,w_dot in the vehicle body frame
        #state_action = [vx,vy,w,throttle,steer,pitch,pitch_dot,roll,roll_dot]
        vx = state_action[0]
        vy = state_action[1]
        w = state_action[2]
        throttle = state_action[3]
        steering = state_action[4]
        throttle_command = state_action[5]
        steering_command = state_action[6]


        # # convert steering to steering angle
        steer_angle = self.steering_2_steering_angle(steering,self.a_s_self,self.b_s_self,self.c_s_self,self.d_s_self,self.e_s_self)

        # if using pitch dynamics account for the extra load on front tires (you also need to enable the input dynamics)
        if self.pitch_dynamics_flag:
            pitch_dot = state_action[7]
            pitch = state_action[8]
            # evaluate pitch contribution
            normal_force_non_scaled = pitch + self.c_pitch/self.k_pitch_dynamics * pitch_dot
            additional_load_front = self.k_pitch_front * normal_force_non_scaled
        else:
            additional_load_front = 0



    
        # # evaluate longitudinal forces
        Fx_wheels = + self.motor_force(throttle,vx,self.a_m_self,self.b_m_self,self.c_m_self)\
                    + self.rolling_friction(vx,self.a_f_self,self.b_f_self,self.c_f_self,self.d_f_self)
        # add extra friction due to steering
        if self.steering_friction_flag:
            Fx_wheels += self.F_friction_due_to_steering(steer_angle,vx,self.a_stfr_self,self.b_stfr_self,self.d_stfr_self,self.e_stfr_self)



        c_front = (self.m_front_wheel_self)/self.m_self
        c_rear = (self.m_rear_wheel_self)/self.m_self

        # redistribute Fx to front and rear wheels according to normal load
        Fx_front = Fx_wheels * c_front
        Fx_rear = Fx_wheels * c_rear

        #evaluate slip angles
        alpha_f,alpha_r = self.evaluate_slip_angles(vx,vy,w,self.lf_self,self.lr_self,steer_angle)

        #lateral forces
        Fy_wheel_f = self.lateral_tire_force(alpha_f,self.d_t_f_self,self.c_t_f_self,self.b_t_f_self,self.m_front_wheel_self + additional_load_front)
        Fy_wheel_r = self.lateral_tire_force(alpha_r,self.d_t_r_self,self.c_t_r_self,self.b_t_r_self,self.m_rear_wheel_self)

        acc_x,acc_y,acc_w = self.solve_rigid_body_dynamics(vx,vy,w,steer_angle,Fx_front,Fx_rear,Fy_wheel_f,Fy_wheel_r,self.lf_self,self.lr_self,self.m_self,self.Jz_self)


        # evaluate input dynamics
        throttle_dot = self.continuous_time_1st_order_dynamics(throttle,throttle_command,self.d_m_self)
        steering_dot = self.continuous_time_1st_order_dynamics(steering,steering_command,self.k_stdn_self)



        if self.pitch_dynamics_flag:
            # solve pitch dynamics
            pitch_dot_dot = self.critically_damped_2nd_order_dynamics_numpy(pitch_dot,pitch,acc_x,self.w_natural_Hz_pitch_self)




        if self.pitch_dynamics_flag:
            return np.array([acc_x,acc_y,acc_w,throttle_dot,steering_dot, pitch_dot_dot, pitch_dot])
        else:
            return np.array([acc_x,acc_y,acc_w,throttle_dot,steering_dot])
    






class dyn_model_culomb_tires_pitch(dyn_model_culomb_tires):
    def __init__(self,dyn_model_culomb_tires_obj):
        self.dyn_model_culomb_tires_obj = dyn_model_culomb_tires_obj

        



    def forward(self,state_action):
        # forward the usual dynamic model
        #   'vx body',      0
        #   'vy body',      1
        #   'w',        2
        #   'throttle integrated' ,  3
        #   'steering integrated',   4
        #   'pitch dot',    5
        #   'pitch',        6
        #   'throttle',     7
        #   'steering',     8



        state_action_base_model = state_action[:7]
        [acc_x,acc_y,acc_w, pitch_dot_dot, pitch_dot] = self.dyn_model_culomb_tires_obj.forward(state_action_base_model) # forward base model

        # extract axuliary states
        throttle = state_action[3]
        steering = state_action[4]

        throttle_command = state_action[7]
        steering_command = state_action[8]


        # forwards integrate steering and throttle commands
        throttle_time_constant = 0.1 * self.d_m_self / (1 + self.d_m_self) # converting from discrete time to continuous time
        throttle_dot = (throttle_command - throttle) / throttle_time_constant

        # steering dynamics
        st_dot = (steering_command - steering) / 0.01 * self.k_stdn_self

        return [acc_x,acc_y,acc_w, pitch_dot_dot, pitch_dot,throttle_dot,st_dot]






def produce_long_term_predictions_full_model(input_data, model,prediction_window,jumps,forward_propagate_indexes):
    # plotting long term predictions on data
    # each prediction window starts from a data point and then the quantities are propagated according to the provided model,
    # so they are not tied to the Vx Vy W data in any way. Though the throttle and steering inputs are taken from the data of course.

    # --- plot fitting results ---
    # input_data = ['vicon time',   0
                #   'vx body',      1
                #   'vy body',      2
                #   'w',        3
                #   'throttle integrated' ,  4
                #   'steering integrated',   5
                #   'pitch dot',    6
                #   'pitch',        7
                #   'throttle',     8
                #   'steering',     9
                #   'vicon x',      10
                #   'vicon y',      11
                #   'vicon yaw']    12

    #prepare tuple containing the long term predictions
    long_term_preds = ()
    

    # iterate through each prediction window
    print('------------------------------')
    print('producing long term predictions')
    from tqdm import tqdm
    tqdm_obj = tqdm(range(0,input_data.shape[0],jumps), desc="long term preds", unit="pred")

    for i in tqdm_obj:
        
        #reset couner
        k = 0
        elpsed_time_long_term_pred = 0

        # set up initial positions
        long_term_pred = np.expand_dims(input_data[i, :],0)


        # iterate through time indexes of each prediction window
        while elpsed_time_long_term_pred < prediction_window and k + i + 1 < len(input_data):
            #store time values
            #long_term_pred[k+1,0] = input_data[k+i, 0] 
            dt = input_data[i + k + 1, 0] - input_data[i + k, 0]
            elpsed_time_long_term_pred = elpsed_time_long_term_pred + dt

            #produce propagated state
            state_action_k = long_term_pred[k,1:10]
            
            # run it through the model (forward the full model)
            accelerations = model.forward(state_action_k) # absolute accelerations in the current vehicle frame of reference
            



            # evaluate new state
            new_state_new_frame = np.zeros(7)

            for prop_index in range(1,7):
                # chose quantities to forward propagate
                if prop_index in forward_propagate_indexes:
                    new_state_new_frame[prop_index-1] = long_term_pred[k,prop_index] + accelerations[prop_index-1] * dt 
                else:
                    new_state_new_frame[prop_index-1] = input_data[i+k+1, prop_index]


            # forward propagate x y yaw state
            rot_angle = long_term_pred[k,12]
            R = np.array([
                [np.cos(rot_angle), -np.sin(rot_angle), 0],
                [np.sin(rot_angle), np.cos(rot_angle), 0],
                [0, 0, 1]
            ])

            # absolute velocities from previous time instant
            abs_vxvyw = R @ np.array([long_term_pred[k,1],long_term_pred[k,2],long_term_pred[k,3]])


            # propagate x y yaw according to the previous state
            new_xyyaw = np.array([long_term_pred[k,10],long_term_pred[k,11],long_term_pred[k,12]]) + abs_vxvyw * dt

            # put everything together
            new_row = np.array([input_data[i + k + 1, 0],*new_state_new_frame,input_data[i+k+1,8],input_data[i+k+1,9],*new_xyyaw])
            long_term_pred = np.vstack([long_term_pred, new_row])

            # update k
            k = k + 1

        long_term_preds += (long_term_pred,)  

    return long_term_preds











import torch
import gpytorch
import tqdm
import random
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy











# SVGP 
class SVGPModel_actuator_dynamics(ApproximateGP,model_functions):
    def __init__(self,inducing_points):
        n_inputs = 5 # how many inputs will be given to the SVGP after time delay fitting has happened
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(SVGPModel_actuator_dynamics, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=n_inputs))


    def setup_time_delay_fitting(self,actuator_time_delay_fitting_tag,n_past_actions,dt):
        # time filtering related
        n_past_actions = int(n_past_actions)
        self.actuator_time_delay_fitting_tag = actuator_time_delay_fitting_tag
        self.n_past_actions = n_past_actions
        self.dt = dt

        if self.actuator_time_delay_fitting_tag == 1:
            # add time constant parameters
            self.register_parameter('time_C_throttle', torch.nn.Parameter(torch.Tensor([0.5]).cuda()))
            self.register_parameter('time_C_steering', torch.nn.Parameter(torch.Tensor([0.5]).cuda()))
            self.constraint_weights = torch.nn.Hardtanh(0, 1) # this constraint will make sure that the parmeter is between 0 and 1
            
        elif self.actuator_time_delay_fitting_tag == 2: # define linear layer for time delay fitting
            # self.raw_weights_throttle = torch.nn.Parameter(torch.randn(1, n_past_actions) * 1)
            # self.raw_weights_steering = torch.nn.Parameter(torch.randn(1, n_past_actions) * 1)
            self.raw_weights_throttle = torch.nn.Parameter(torch.ones(1, n_past_actions) * 0.5)
            self.raw_weights_steering = torch.nn.Parameter(torch.ones(1, n_past_actions) * 0.5)

    def transform_parameters_norm_2_real(self):
        
        time_C_throttle = self.minmax_scale_hm(0.001,0.1,self.constraint_weights(self.time_C_throttle))
        time_C_steering = self.minmax_scale_hm(0.001,0.1,self.constraint_weights(self.time_C_steering))
        return [time_C_throttle,time_C_steering] 
    
    def constrained_linear_layer(self, raw_weights):
        # Apply softplus to make weights positive
        positive_weights = torch.nn.functional.softplus(raw_weights)
        # Normalize the weights along the last dimension so they sum to 1 for each output unit
        normalized_weights = positive_weights / positive_weights.sum(dim=1, keepdim=True)
        # Apply the weights to the input
        return normalized_weights

    def return_likelyhood_optimizer_objects(self,learning_rate,fit_likelihood_noise_tag,raw_likelihood_noise):
        likelihood = gpytorch.likelihoods.GaussianLikelihood() # gaussian likelihood
        if fit_likelihood_noise_tag:
            optimizer = torch.optim.AdamW([{'params': self.parameters()}, {'params': likelihood.parameters()},], lr=learning_rate)
        else:
            likelihood.noise_covar.raw_noise.data = torch.tensor([raw_likelihood_noise]).cuda()
            optimizer = torch.optim.AdamW([{'params': self.parameters()}], lr=learning_rate)
        return likelihood, optimizer


    def forward(self, x_4_model):
        # produce the inputs for the model
        #x_4_model, weights_throttle, weights_steering = self.produce_th_st_4_model(x)
        # feed them to the model
        mean_x = self.mean_module(x_4_model)
        covar_x = self.covar_module(x_4_model)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    def produce_th_st_4_model(self,x):
        # extract throttle and steering inputs
        if self.actuator_time_delay_fitting_tag == 1 or self.actuator_time_delay_fitting_tag == 2: # extract inputs
            throttle_past_actions = x[:,3:self.n_past_actions+3] # extract throttle past actions
            steering_past_actions = x[:,3 + self.n_past_actions :] # extract steering past actions

            if self.actuator_time_delay_fitting_tag == 1: # using physics informed
                [time_C_throttle,time_C_steering] = self.transform_parameters_norm_2_real()
                weights_throttle = self.produce_past_action_coefficients_1st_oder_step_response(time_C_throttle,self.n_past_actions,self.dt).float()
                weights_steering = self.produce_past_action_coefficients_1st_oder_step_response(time_C_steering,self.n_past_actions,self.dt).float()

            elif self.actuator_time_delay_fitting_tag == 2: # using linear layer
                weights_throttle = self.constrained_linear_layer(self.raw_weights_throttle).t()
                weights_steering = self.constrained_linear_layer(self.raw_weights_steering).t()

            throttle = torch.matmul(throttle_past_actions, weights_throttle)
            steering = torch.matmul(steering_past_actions, weights_steering)

            x_4_model = torch.cat((x[:,:3],throttle,steering),1) # concatenate the inputs
            return x_4_model, weights_throttle, weights_steering
        else:
            x_4_model = x
            return x_4_model,[],[]







# SVGP 
class SVGP_submodel_actuator_dynamics(ApproximateGP):
    # the single SVGP used to model either vx, vy or w
    def __init__(self,inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(SVGP_submodel_actuator_dynamics, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.size(1)))

    def forward(self, x_4_model):

        mean_x = self.mean_module(x_4_model)
        covar_x = self.covar_module(x_4_model)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



class SVGP_unified_model(torch.nn.Sequential,model_functions):
    def __init__(self,inducing_points,n_past_actions,dt,actuator_time_delay_fitting_tag,submodel_vx,submodel_vy,submodel_vw):
        
        super().__init__() # this will also add the same parameters to this class

        # instantiate the 3 models
        self.model_vx = submodel_vx
        self.model_vy = submodel_vy
        self.model_w = submodel_vw
        

        # produce likelihood objects 
        self.likelihood_vx = gpytorch.likelihoods.GaussianLikelihood() 
        self.likelihood_vy = gpytorch.likelihoods.GaussianLikelihood() 
        self.likelihood_w  = gpytorch.likelihoods.GaussianLikelihood() 

        # add time delay fitting parameters
        self.setup_time_delay_fitting(actuator_time_delay_fitting_tag,n_past_actions,dt)

        self.Hardsigmoid = torch.nn.Hardsigmoid()



    def setup_time_delay_fitting(self,actuator_time_delay_fitting_tag,n_past_actions,dt):
        # time filtering related
        n_past_actions = int(n_past_actions)
        self.actuator_time_delay_fitting_tag = actuator_time_delay_fitting_tag
        self.n_past_actions = n_past_actions
        self.dt = dt

        if self.actuator_time_delay_fitting_tag == 1:
            # add time constant parameters
            self.register_parameter('time_C_throttle', torch.nn.Parameter(torch.Tensor([0.5]).cuda()))
            self.register_parameter('time_C_steering', torch.nn.Parameter(torch.Tensor([0.5]).cuda()))
            self.constraint_weights = torch.nn.Hardtanh(0, 1) # this constraint will make sure that the parmeter is between 0 and 1
            
        elif self.actuator_time_delay_fitting_tag == 2: # define linear layer for time delay fitting
            # self.raw_weights_throttle = torch.nn.Parameter(torch.randn(1, n_past_actions) * 1)
            # self.raw_weights_steering = torch.nn.Parameter(torch.randn(1, n_past_actions) * 1)
            # put to cuda if available
            if torch.cuda.is_available():
                self.raw_weights_throttle = torch.nn.Parameter(torch.ones(1, n_past_actions).cuda() * 0.5)
                self.raw_weights_steering = torch.nn.Parameter(torch.ones(1, n_past_actions).cuda() * 0.5)
            else:
                self.raw_weights_throttle = torch.nn.Parameter(torch.ones(1, n_past_actions) * 0.5)
                self.raw_weights_steering = torch.nn.Parameter(torch.ones(1, n_past_actions) * 0.5)


    def forward(self, x):
        # extract throttle and steering inputs
        x_4_model, weights_throttle, weights_steering , non_normalized_w_th, non_normalized_w_st = self.produce_th_st_4_model(x)
        output_vx = self.model_vx(x_4_model)
        output_vy = self.model_vy(x_4_model)
        output_w = self.model_w(x_4_model)

        return output_vx,output_vy,output_w,weights_throttle,weights_steering, non_normalized_w_th, non_normalized_w_st
        

    def produce_th_st_4_model(self,x):
        # extract throttle and steering inputs
        if self.actuator_time_delay_fitting_tag == 1 or self.actuator_time_delay_fitting_tag == 2: # extract inputs
            throttle_past_actions = x[:,3:self.n_past_actions+3] # extract throttle past actions
            steering_past_actions = x[:,3 + self.n_past_actions :] # extract steering past actions

            if self.actuator_time_delay_fitting_tag == 1: # using physics informed
                [time_C_throttle,time_C_steering] = self.transform_parameters_norm_2_real()
                weights_throttle = self.produce_past_action_coefficients_1st_oder_step_response(time_C_throttle,self.n_past_actions,self.dt).float()
                weights_steering = self.produce_past_action_coefficients_1st_oder_step_response(time_C_steering,self.n_past_actions,self.dt).float()

            elif self.actuator_time_delay_fitting_tag == 2: # using linear layer
                weights_throttle, non_normalized_w_th = self.constrained_linear_layer(self.raw_weights_throttle)
                weights_steering, non_normalized_w_st = self.constrained_linear_layer(self.raw_weights_steering)
                # transpose
                weights_throttle = weights_throttle.t()
                weights_steering = weights_steering.t()
                non_normalized_w_th = non_normalized_w_th.t()
                non_normalized_w_st = non_normalized_w_st.t()

            throttle = throttle_past_actions @ weights_throttle #torch.matmul(throttle_past_actions, weights_throttle)
            steering = steering_past_actions @ weights_steering #torch.matmul(steering_past_actions, weights_steering)

            x_4_model = torch.cat((x[:,:3],throttle,steering),1) # concatenate the inputs
            return x_4_model, weights_throttle, weights_steering, non_normalized_w_th, non_normalized_w_st
        else:
            x_4_model = x
            return x_4_model,[],[],[],[]
        


    def constrained_linear_layer(self, raw_weights):
        # Apply softplus to make weights positive
        #positive_weights = torch.nn.functional.softplus(raw_weights)
        # pass raw weights through sigmoid
        positive_weights = self.Hardsigmoid(raw_weights)
        
        # Normalize the weights along the last dimension so they sum to 1 for each output unit
        normalized_weights = positive_weights / positive_weights.sum(dim=1, keepdim=True)
        # Apply the weights to the input
        return normalized_weights, positive_weights
    

    def train_model(self,num_epochs,learning_rate,train_x, train_y_vx, train_y_vy, train_y_w):
        
        # start fitting
        # make contiguous (not sure why)
        train_x = train_x.contiguous()
        train_y_vx = train_y_vx.contiguous()
        train_y_vy = train_y_vy.contiguous()
        train_y_w = train_y_w.contiguous()

        # define batches for training (each bach will be used to perform a gradient descent step in each iteration. So toal parameters updates are Epochs*n_batches)
        from torch.utils.data import TensorDataset, DataLoader
        train_dataset = TensorDataset(train_x, train_y_vx, train_y_vy, train_y_w) # 

        # define data loaders
        train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True) #  batch_size=250

        #set to training mode
        self.model_vx.train()
        self.model_vy.train()
        self.model_w.train()
        self.likelihood_vx.train()
        self.likelihood_vy.train()
        self.likelihood_w.train()


        # Set up loss object. We're using the VariationalELBO
        mll_vx = gpytorch.mlls.VariationalELBO(self.likelihood_vx, self.model_vx, num_data=train_y_vx.size(0))
        mll_vy = gpytorch.mlls.VariationalELBO(self.likelihood_vy, self.model_vy, num_data=train_y_vy.size(0))
        mll_w = gpytorch.mlls.VariationalELBO(self.likelihood_w, self.model_w, num_data=train_y_w.size(0))

        loss_2_print_vx_vec = []
        loss_2_print_vy_vec = []
        loss_2_print_w_vec = []
        loss_2_print_weights = []
        total_loss_vec = []


        #optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        #optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        # Print the parameters with their names and shapes
        print('Parameters to optimize:')
        for name, param in self.named_parameters():
            print(f"Parameter name: {name}, Shape: {param.shape}")




        # start training (tqdm is just to show the loading bar)
        bar_format=bar_format=f"{Fore.GREEN}{{l_bar}}{Fore.GREEN}{{bar}}{Style.RESET_ALL}{Fore.GREEN}{{r_bar}}{Style.RESET_ALL}"
        epochs_iter = tqdm.tqdm(range(num_epochs), desc=f"{Fore.GREEN}Epochs", leave=True, bar_format=bar_format)
        
        for i in epochs_iter: 
            #torch.cuda.empty_cache()  # Releases unused cached memory

            # Within each iteration, we will go over each minibatch of data
            minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False) 

            for x_batch, y_batch_vx ,y_batch_vy, y_batch_w in minibatch_iter: # 
                # Zero backprop gradients
                optimizer.zero_grad()  # Clear previous gradients

                # Forward pass
                output_vx, output_vy, output_w, weights_throttle, weights_steering, non_normalized_w_th, non_normalized_w_st = self(x_batch)
                
                # Calculate individual losses
                #q_weights = 0.01
                #q_dev_weights = 1
                # penalize weights that are further in the past
                # define increasing values 
                #weights_of_weights = torch.flip(torch.unsqueeze(torch.arange(1, self.n_past_actions+1).float().cuda(),0) / self.n_past_actions,[1])
                #weights_of_weights = torch.ones(1,self.n_past_actions).cuda()


                # diff_weights_throttle = torch.diff(torch.squeeze(weights_throttle))
                # diff_weights_steering = torch.diff(torch.squeeze(weights_steering))
                # th_weights_squared = (1+non_normalized_w_th)**2
                # st_weights_squared = (1+non_normalized_w_st)**2

                # loss_weights =  + q_dev_weights * ((torch.mean(diff_weights_throttle**2) + torch.mean(diff_weights_steering**2)))\
                #                 - q_weights * (torch.mean(th_weights_squared)-1 + torch.mean(st_weights_squared)-1)
                
                # trying to enforce the weights to stick together
                weights_loss_scale = 0.05 # sale the loss equally to avoid it being dominant 0.05
                q_var_weights = 1
                q_dev_weights = 50
                q_weights = 0.001


                time_vec = torch.arange(0,self.n_past_actions).float().cuda() # this is the index of the time delay, but ok just multiply by dt to get the time
                w_th_times_time = time_vec * torch.squeeze(non_normalized_w_th)
                w_st_times_time = time_vec * torch.squeeze(non_normalized_w_st)

                mean_time_delay_th = torch.mean(w_th_times_time)
                mean_time_delay_st = torch.mean(w_st_times_time)

                var_th = torch.mean((w_th_times_time - mean_time_delay_th)**2)
                var_st = torch.mean((w_st_times_time - mean_time_delay_st)**2)

                diff_weights_throttle = torch.diff(torch.squeeze(non_normalized_w_th))
                diff_weights_steering = torch.diff(torch.squeeze(non_normalized_w_st))

                #th_weights_squared = (1+non_normalized_w_th)**2
                #st_weights_squared = (1+non_normalized_w_st)**2




                loss_weights = q_var_weights * (var_th + var_st)+\
                             + q_dev_weights * torch.sum(diff_weights_throttle**2 + diff_weights_steering**2)\
                             + q_weights * torch.sum(    (w_th_times_time/self.n_past_actions)**2 + (w_st_times_time/self.n_past_actions)**2    )
                             #- q_weights * (torch.mean(th_weights_squared)-1 + torch.mean(st_weights_squared)-1)

                #scale the loss
                loss_weights = weights_loss_scale * loss_weights


                loss_vx = -mll_vx(output_vx, y_batch_vx[:,0])
                loss_vy = -mll_vy(output_vy, y_batch_vy[:,0])
                loss_w = -mll_w(output_w, y_batch_w[:,0])

                # Combine all losses
                total_loss =  loss_weights + loss_vx + loss_w + loss_vy
                
                # Print the current loss for vx (or use other loss types if needed)
                minibatch_iter.set_postfix(loss=total_loss.item())

                # Backward pass (compute gradients for the total loss)
                total_loss.backward()

                # Update parameters using the optimizer
                optimizer.step()


            loss_2_print_vx_vec = [*loss_2_print_vx_vec, loss_vx.item()]
            loss_2_print_vy_vec = [*loss_2_print_vy_vec, loss_vy.item()]
            loss_2_print_w_vec = [*loss_2_print_w_vec, loss_w.item()]
            loss_2_print_weights = [*loss_2_print_weights, loss_weights.item()]
            total_loss_vec = [*total_loss_vec, total_loss.item()]

            
        #plot loss functions
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(loss_2_print_vx_vec,label='loss vx',color='dodgerblue') 
        ax.plot(loss_2_print_vy_vec,label='loss vy',color='orangered')
        ax.plot(loss_2_print_w_vec,label='loss w',color='orchid')
        ax.plot(total_loss_vec,label='total loss',color='k')
        ax.plot(loss_2_print_weights,label='loss weights',color='lime')
        ax.legend()


    def save_model(self,folder_path_SVGP_params,actuator_time_delay_fitting_tag,n_past_actions,dt):
        # SAve model parameters


        # analytical version of the model [necessary for solver implementation]
        # rebuild SVGP using m and S 
        inducing_locations_x = self.model_vx.variational_strategy.inducing_points.cpu().detach().numpy()
        outputscale_x = self.model_vx.covar_module.outputscale.item()
        lengthscale_x = self.model_vx.covar_module.base_kernel.lengthscale.cpu().detach().numpy()[0]

        inducing_locations_y = self.model_vy.variational_strategy.inducing_points.cpu().detach().numpy()
        outputscale_y = self.model_vy.covar_module.outputscale.item()
        lengthscale_y = self.model_vy.covar_module.base_kernel.lengthscale.cpu().detach().numpy()[0]

        inducing_locations_w = self.model_w.variational_strategy.inducing_points.cpu().detach().numpy()
        outputscale_w = self.model_w.covar_module.outputscale.item()
        lengthscale_w = self.model_w.covar_module.base_kernel.lengthscale.cpu().detach().numpy()[0]


        KZZ_x = rebuild_Kxy_RBF_vehicle_dynamics(np.squeeze(inducing_locations_x),np.squeeze(inducing_locations_x),outputscale_x,lengthscale_x)
        KZZ_y = rebuild_Kxy_RBF_vehicle_dynamics(np.squeeze(inducing_locations_y),np.squeeze(inducing_locations_y),outputscale_y,lengthscale_y)
        KZZ_w = rebuild_Kxy_RBF_vehicle_dynamics(np.squeeze(inducing_locations_w),np.squeeze(inducing_locations_w),outputscale_w,lengthscale_w)


        # call prediction module on inducing locations
        n_inducing_points = inducing_locations_x.shape[0]
        jitter_term = 0.0001 * np.eye(n_inducing_points)  # this is very important for numerical stability


        preds_zz_x = self.model_vx(self.model_vx.variational_strategy.inducing_points)
        preds_zz_y = self.model_vy(self.model_vy.variational_strategy.inducing_points)
        preds_zz_w = self.model_w( self.model_w.variational_strategy.inducing_points)

        m_x = preds_zz_x.mean.detach().cpu().numpy() # model.variational_strategy.variational_distribution.mean.detach().cpu().numpy()  #
        S_x = self.model_vx.variational_strategy.variational_distribution.covariance_matrix.detach().cpu().numpy()  # preds_zz.covariance_matrix.detach().cpu().numpy() # 

        m_y = preds_zz_y.mean.detach().cpu().numpy() # model.variational_strategy.variational_distribution.mean.detach().cpu().numpy()  #
        S_y = self.model_vy.variational_strategy.variational_distribution.covariance_matrix.detach().cpu().numpy()  

        m_w = preds_zz_w.mean.detach().cpu().numpy() # model.variational_strategy.variational_distribution.mean.detach().cpu().numpy()  #
        S_w = self.model_w.variational_strategy.variational_distribution.covariance_matrix.detach().cpu().numpy()  

        # Compute the covariance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX

        # solve the pre-post multiplication block

        # Define a lower triangular matrix L and a matrix B
        L_inv_x = np.linalg.inv(np.linalg.cholesky(KZZ_x + jitter_term))
        #KZZ_inv_x = np.linalg.inv(KZZ_x + jitter_term)
        right_vec_x = np.linalg.solve(KZZ_x + jitter_term, m_x)
        middle_x = S_x - np.eye(n_inducing_points)

        L_inv_y = np.linalg.inv(np.linalg.cholesky(KZZ_y + jitter_term))
        #KZZ_inv_y = np.linalg.inv(KZZ_y + jitter_term)
        right_vec_y = np.linalg.solve(KZZ_y + jitter_term, m_y)
        middle_y = S_y - np.eye(n_inducing_points)

        L_inv_w = np.linalg.inv(np.linalg.cholesky(KZZ_w + jitter_term))
        #KZZ_inv_w = np.linalg.inv(KZZ_w + jitter_term)
        right_vec_w = np.linalg.solve(KZZ_w + jitter_term, m_w)
        middle_w = S_w - np.eye(n_inducing_points)


        # save quantities to use them later in a solver
        np.save(folder_path_SVGP_params+'m_x.npy', m_x)
        np.save(folder_path_SVGP_params+'middle_x.npy', middle_x)
        np.save(folder_path_SVGP_params+'L_inv_x.npy', L_inv_x)
        np.save(folder_path_SVGP_params+'right_vec_x.npy', right_vec_x)
        np.save(folder_path_SVGP_params+'inducing_locations_x.npy', inducing_locations_x)
        np.save(folder_path_SVGP_params+'outputscale_x.npy', outputscale_x)
        np.save(folder_path_SVGP_params+'lengthscale_x.npy', lengthscale_x)

        np.save(folder_path_SVGP_params+'m_y.npy', m_y)
        np.save(folder_path_SVGP_params+'middle_y.npy', middle_y)
        np.save(folder_path_SVGP_params+'L_inv_y.npy', L_inv_y)
        np.save(folder_path_SVGP_params+'right_vec_y.npy', right_vec_y)
        np.save(folder_path_SVGP_params+'inducing_locations_y.npy', inducing_locations_y)
        np.save(folder_path_SVGP_params+'outputscale_y.npy', outputscale_y)
        np.save(folder_path_SVGP_params+'lengthscale_y.npy', lengthscale_y)

        np.save(folder_path_SVGP_params+'m_w.npy', m_w)
        np.save(folder_path_SVGP_params+'middle_w.npy', middle_w)
        np.save(folder_path_SVGP_params+'L_inv_w.npy', L_inv_w)
        np.save(folder_path_SVGP_params+'right_vec_w.npy', right_vec_w)
        np.save(folder_path_SVGP_params+'inducing_locations_w.npy', inducing_locations_w)
        np.save(folder_path_SVGP_params+'outputscale_w.npy', outputscale_w)
        np.save(folder_path_SVGP_params+'lengthscale_w.npy', lengthscale_w)

        # save SVGP models in torch format
        # save the time delay realated parameters
        #time_delay_parameters = np.array([actuator_time_delay_fitting_tag,n_past_actions,dt])
        #np.save(folder_path_SVGP_params + 'time_delay_parameters.npy', time_delay_parameters)
        np.save(folder_path_SVGP_params + 'actuator_time_delay_fitting_tag.npy', actuator_time_delay_fitting_tag)
        np.save(folder_path_SVGP_params + 'n_past_actions.npy', n_past_actions)
        np.save(folder_path_SVGP_params + 'dt.npy', dt)


        # save weights
        if actuator_time_delay_fitting_tag == 2:
            # save weights
            #raw_weights_throttle = self.raw_weights_throttle.detach().cpu().numpy()
            #raw_weights_steering = self.raw_weights_steering.detach().cpu().numpy()
            # evaluate the weights
            weights_throttle, non_normalized_w_th = self.constrained_linear_layer(self.raw_weights_throttle)
            weights_throttle = weights_throttle.t().detach().cpu().numpy()
            weights_steering, non_normalized_w_st = self.constrained_linear_layer(self.raw_weights_steering)
            weights_steering = weights_steering.t().detach().cpu().numpy()

            np.save(folder_path_SVGP_params + 'weights_throttle.npy', weights_throttle)
            np.save(folder_path_SVGP_params + 'weights_steering.npy', weights_steering)

        # vx
        model_path_vx = folder_path_SVGP_params + 'svgp_model_vx.pth'
        likelihood_path_vx = folder_path_SVGP_params + 'svgp_likelihood_vx.pth'
        torch.save(self.model_vx.state_dict(), model_path_vx)
        torch.save(self.likelihood_vx.state_dict(), likelihood_path_vx)
        # vy
        model_path_vy = folder_path_SVGP_params + 'svgp_model_vy.pth'
        likelihood_path_vy = folder_path_SVGP_params + 'svgp_likelihood_vy.pth'
        torch.save(self.model_vy.state_dict(), model_path_vy)
        torch.save(self.likelihood_vy.state_dict(), likelihood_path_vy)
        # w
        model_path_w = folder_path_SVGP_params + 'svgp_model_w.pth'
        likelihood_path_w = folder_path_SVGP_params + 'svgp_likelihood_w.pth'
        torch.save(self.model_w.state_dict(), model_path_w)
        torch.save(self.likelihood_w.state_dict(), likelihood_path_w)

        print('------------------------------')
        print('--- saved model parameters ---')
        print('------------------------------')
        print('saved parameters in folder: ', folder_path_SVGP_params)


# put SVGP unified analytical here
class SVGP_unified_analytic:
        def __init__(self):
            pass
        def load_parameters(self, folder_path):
            print('SVGP unified model with actuator dynamics')
            print('Loading SVGP saved parameters from folder:', folder_path)

            # Define the parameter names for each dimension (x, y, w)
            param_names = ['m', 'middle', 'L_inv', 'right_vec', 'inducing_locations', 'outputscale', 'lengthscale','max_stdev']
            dimensions = ['x', 'y', 'w']

            # Initialize an empty dictionary to store all parameters
            svgp_params = {}

            # Loop through each dimension and parameter name to load the .npy files
            print('')
            print('Loading SVGP saved parameters from folder:', folder_path)
            print('')

            # load actuator dynamics parameters
            setattr(self, "dt", np.load(os.path.join(folder_path, "dt.npy")))
            self.dt = self.dt.item() # convert to item to set it as a scalar
            setattr(self, "n_past_actions", np.load(os.path.join(folder_path, "n_past_actions.npy")))
            setattr(self, "actuator_time_delay_fitting_tag", np.load(os.path.join(folder_path, "actuator_time_delay_fitting_tag.npy")))
            if self.actuator_time_delay_fitting_tag==2:
                setattr(self, "weights_throttle", np.load(os.path.join(folder_path, "weights_throttle.npy")))
                setattr(self, "weights_steering", np.load(os.path.join(folder_path, "weights_steering.npy")))



            for dim in dimensions:
                svgp_params[dim] = {}
                for param in param_names:
                    file_path = os.path.join(folder_path, f"{param}_{dim}.npy")
                    if os.path.exists(file_path):
                        svgp_params[dim][param] = np.load(file_path)
                        # assign to self
                        setattr(self, f"{param}_{dim}", svgp_params[dim][param])
                        print(f"Loaded {param}_{dim}: shape {svgp_params[dim][param].shape}")
                    else:
                        print(f"Warning: {param}_{dim}.npy not found in {folder_path}")
            print('')

        def predictive_mean_only(self,x_star):
            # x_star = [th st vx vy w]
            # output is acc_vx, acc_vy, acc_w

            kXZ_x = rebuild_Kxy_RBF_vehicle_dynamics(x_star,np.squeeze(self.inducing_locations_x),self.outputscale_x,self.lengthscale_x)
            kXZ_y = rebuild_Kxy_RBF_vehicle_dynamics(x_star,np.squeeze(self.inducing_locations_y),self.outputscale_y,self.lengthscale_y)
            kXZ_w = rebuild_Kxy_RBF_vehicle_dynamics(x_star,np.squeeze(self.inducing_locations_w),self.outputscale_w,self.lengthscale_w)
            # prediction
            mean_x = kXZ_x @ self.right_vec_x
            mean_y = kXZ_y @ self.right_vec_y
            mean_w = kXZ_w @ self.right_vec_w

            return mean_x, mean_y, mean_w
        
        def forward_4_long_term_prediction(self,state_action):
            # state_action = [vx, vy, w, throttle_filtered, steering_filtered, throttle, steering]
            x_star = np.expand_dims(state_action[:5],0)
            mean_x, mean_y, mean_w = self.predictive_mean_only(x_star)
            accelerations = np.array([mean_x.item(), mean_y.item(), mean_w.item(),0.0,0.0]) # last two dummy values for the th and st dynamics that are computed offline
            return accelerations


        
        def predictive_mean_cov(self,x_star):
            # x_star = [th st vx vy w]
            # output is acc_vx, acc_vy, acc_w

            # x
            kXZ_x = rebuild_Kxy_RBF_vehicle_dynamics(x_star,np.squeeze(self.inducing_locations_x),self.outputscale_x,self.lengthscale_x)
            X_x = self.L_inv_x @ kXZ_x.T
            KXX_x = RBF_kernel_rewritten(x_star[0],x_star[0],self.outputscale_x,self.lengthscale_x)
            # prediction
            mean_x = kXZ_x @ self.right_vec_x
            cov_mS_x = KXX_x + X_x.T @ self.middle_x @ X_x

            # y
            kXZ_y = rebuild_Kxy_RBF_vehicle_dynamics(x_star,np.squeeze(self.inducing_locations_y),self.outputscale_y,self.lengthscale_y)
            X_y = self.L_inv_y @ kXZ_y.T
            KXX_y = RBF_kernel_rewritten(x_star[0],x_star[0],self.outputscale_y,self.lengthscale_y)
            # prediction
            mean_y = kXZ_y @ self.right_vec_y
            cov_mS_y = KXX_y + X_y.T @ self.middle_y @ X_y

            # w
            kXZ_w = rebuild_Kxy_RBF_vehicle_dynamics(x_star,np.squeeze(self.inducing_locations_w),self.outputscale_w,self.lengthscale_w)
            X_w = self.L_inv_w @ kXZ_w.T
            KXX_w = RBF_kernel_rewritten(x_star[0],x_star[0],self.outputscale_w,self.lengthscale_w)
            # prediction
            mean_w = kXZ_w @ self.right_vec_w
            cov_mS_w = KXX_w + X_w.T @ self.middle_w @ X_w

        
            return mean_x, mean_y, mean_w, cov_mS_x, cov_mS_y, cov_mS_w



class dynamic_bicycle_actuator_delay_fitting(torch.nn.Sequential,model_functions):
    def __init__(self,n_past_actions,dt):
        
        super().__init__() # this will also add the same parameters to this class
        self.n_past_actions = n_past_actions
        self.dt = dt

        # default initialization of constant weights
        if torch.cuda.is_available():
            self.raw_weights_throttle = torch.nn.Parameter(torch.ones(1, n_past_actions).cuda()*0.5)
            self.raw_weights_steering = torch.nn.Parameter(torch.ones(1, n_past_actions).cuda()*0.5)
        else:
            self.raw_weights_throttle = torch.nn.Parameter(torch.ones(1, n_past_actions)*0.5)
            self.raw_weights_steering = torch.nn.Parameter(torch.ones(1, n_past_actions)*0.5)

        self.Hardsigmoid = torch.nn.Hardsigmoid()

    def constrained_linear_layer(self, raw_weights):
        # Apply softplus to make weights positive
        #positive_weights = torch.nn.functional.softplus(raw_weights)
        # pass raw weights through sigmoid
        
        #positive_weights = self.Hardsigmoid(raw_weights)
        # just square up the weights
        positive_weights = raw_weights**2
        #positive_weights = torch.abs(raw_weights)
        
        # Normalize the weights along the last dimension so they sum to 1 for each output unit
        normalized_weights = positive_weights / positive_weights.sum(dim=1, keepdim=True)
        # Apply the weights to the input
        return normalized_weights, positive_weights


    def produce_th_st_4_model(self,x):
        # extract throttle and steering inputs

        throttle_past_actions = x[:,3:self.n_past_actions+3] # extract throttle past actions
        steering_past_actions = x[:,3 + self.n_past_actions :] # extract steering past actions

        weights_throttle, non_normalized_w_th = self.constrained_linear_layer(self.raw_weights_throttle)
        weights_steering, non_normalized_w_st = self.constrained_linear_layer(self.raw_weights_steering)

        # transpose
        weights_throttle = weights_throttle.t()
        weights_steering = weights_steering.t()
        non_normalized_w_th = non_normalized_w_th.t()
        non_normalized_w_st = non_normalized_w_st.t()

        throttle = throttle_past_actions @ weights_throttle #torch.matmul(throttle_past_actions, weights_throttle)
        steering = steering_past_actions @ weights_steering #torch.matmul(steering_past_actions, weights_steering)




        x_4_model = torch.cat((x[:,:3],throttle,steering),1) # concatenate the inputs
        return x_4_model, weights_throttle, weights_steering, non_normalized_w_th, non_normalized_w_st


    def forward(self, x):
        # extract throttle and steering inputs
        x_4_model, weights_throttle, weights_steering , non_normalized_w_th, non_normalized_w_st = self.produce_th_st_4_model(x)
        # replace with dynamic_bycicle model
        vx = x_4_model[:,0]
        vy = x_4_model[:,1]
        w = x_4_model[:,2]
        th = x_4_model[:,3]
        st = x_4_model[:,4]

        acc_x, acc_y, acc_w = self.dynamic_bicycle(th, st, vx, vy, w)

        return acc_x, acc_y, acc_w,weights_throttle,weights_steering, non_normalized_w_th, non_normalized_w_st

    def train_model(self,num_epochs,learning_rate,train_x, train_y_vx, train_y_vy, train_y_w, live_plot_weights,train_th,train_st):
        
        # start fitting
        # make contiguous (not sure why)
        train_x = train_x.contiguous()
        train_y_vx = train_y_vx.contiguous()
        train_y_vy = train_y_vy.contiguous()
        train_y_w = train_y_w.contiguous()

        # define batches for training (each bach will be used to perform a gradient descent step in each iteration. So toal parameters updates are Epochs*n_batches)
        from torch.utils.data import TensorDataset, DataLoader
        train_dataset = TensorDataset(train_x, train_y_vx, train_y_vy, train_y_w) # 

        # define data loaders
        train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True) #  batch_size=250

        # Set up loss object. We're using the VariationalELBO
        mse_loss = torch.nn.MSELoss()

        loss_2_print_vx_vec = []
        loss_2_print_vy_vec = []
        loss_2_print_w_vec = []
        loss_2_print_weights = []
        total_loss_vec = []

        #optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-3)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        #optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)




        # Print the parameters with their names and shapes
        print('Parameters to optimize:')
        for name, param in self.named_parameters():
            print(f"Parameter name: {name}, Shape: {param.shape}")

        import matplotlib.pyplot as plt
        if live_plot_weights:
            
            from IPython.display import display, clear_output
            # Define time axis
            time_axis = np.linspace(0, self.n_past_actions * self.dt, self.n_past_actions)
            # Initialize plot
            plt.ion()  # Interactive mode on
            fig, ax1 = plt.subplots(figsize=(10, 5))  # Width = 10, Height = 5


            line_th, = ax1.plot([], [], color='dodgerblue', label='Throttle Weights')
            line_st, = ax1.plot([], [], color='orangered', label='Steering Weights')

            ax1.set_xlabel("Time Delay [s]")
            ax1.set_ylabel("Weight")
            ax1.legend()
            ax1.set_title("Live Plot of Weights Over Time")
            # set the limits to 0 1
            #ax1.set_ylim(0, 1.3)

            # plot initial weights
            weights_throttle, non_normalized_w_th = self.constrained_linear_layer(self.raw_weights_throttle)
            weights_steering, non_normalized_w_st = self.constrained_linear_layer(self.raw_weights_steering)

            # weights for plotting
            # Convert to numpy for plotting
            weights_th_numpy = non_normalized_w_th.detach().cpu().numpy().squeeze()
            weights_st_numpy = non_normalized_w_st.detach().cpu().numpy().squeeze()

            # Live update plot every few iterations
            line_th.set_xdata(time_axis)
            line_th.set_ydata(weights_th_numpy)
            line_st.set_xdata(time_axis)
            line_st.set_ydata(weights_st_numpy)

            #ax1.relim()  # Recalculate limits
            ax1.autoscale_view(True, True, True)  # Autoscale axes
            display(fig)
            fig.canvas.manager.window.raise_()
            plt.pause(1)











        # start training (tqdm is just to show the loading bar)
        bar_format=bar_format=f"{Fore.GREEN}{{l_bar}}{Fore.GREEN}{{bar}}{Style.RESET_ALL}{Fore.GREEN}{{r_bar}}{Style.RESET_ALL}"
        epochs_iter = tqdm.tqdm(range(num_epochs), desc=f"{Fore.GREEN}Epochs", leave=True, bar_format=bar_format)
        
        for i in epochs_iter: 
            #torch.cuda.empty_cache()  # Releases unused cached memory

            # Within each iteration, we will go over each minibatch of data
            minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False) 

            for x_batch, y_batch_vx ,y_batch_vy, y_batch_w in minibatch_iter: # 



                # Zero backprop gradients
                optimizer.zero_grad()  # Clear previous gradients

                # Forward pass
                acc_x, acc_y, acc_w, weights_throttle, weights_steering, non_normalized_w_th, non_normalized_w_st = self(x_batch)
                


                # Calculate individual losses
 
                
                # trying to enforce the weights to stick together
                weights_loss_scale = 10 # sale the loss equally to avoid it being dominant 0.05
                q_var_weights = 10
                q_dev_weights = 0.1
                q_weights = 1

                time_vec = torch.arange(0,self.n_past_actions).float().cuda() # this is the index of the time delay, but ok just multiply by dt to get the time
                w_th_times_time = time_vec * torch.squeeze(non_normalized_w_th)
                w_st_times_time = time_vec * torch.squeeze(non_normalized_w_st)

                mean_time_delay_th = torch.mean(w_th_times_time)
                mean_time_delay_st = torch.mean(w_st_times_time)

                var_th = torch.mean((w_th_times_time - mean_time_delay_th)**2)
                var_st = torch.mean((w_st_times_time - mean_time_delay_st)**2)

                diff_weights_throttle = torch.diff(torch.squeeze(non_normalized_w_th))
                diff_weights_steering = torch.diff(torch.squeeze(non_normalized_w_st))

                # for discouraging weights far from the time now
                weights_of_weights = torch.arange(10,10+self.n_past_actions).float().cuda() # this is the index of the time delay, but ok just multiply by dt to get the time
                w_th_weighted = weights_of_weights * torch.squeeze(non_normalized_w_th)
                w_st_weighted  = weights_of_weights * torch.squeeze(non_normalized_w_st)



                loss_weights_th = - 0.1 * q_weights * torch.norm(weights_throttle, p=float('inf')) # q_weights * torch.mean((1+weights_throttle)**2) 
                loss_weights_st = - 0.1 * q_weights * torch.norm(weights_steering, p=float('inf')) # q_weights * torch.mean((1+weights_throttle)**2) 

                                #+ q_dev_weights * torch.sum(diff_weights_throttle**2 + diff_weights_steering**2)\
                #+ q_weights * torch.sum( (w_th_weighted/self.n_past_actions )**2 + (w_st_weighted/self.n_past_actions )**2    ) # /self.n_past_actions 
                #
                                
                                #+q_var_weights * (var_th + var_st)
                             
                             #- q_weights * (torch.mean(th_weights_squared)-1 + torch.mean(st_weights_squared)-1)

                #scale the loss
                loss_weights_th = weights_loss_scale * loss_weights_th
                loss_weights_st = weights_loss_scale * loss_weights_st


                loss_vx = mse_loss(acc_x, y_batch_vx[:,0])
                loss_vy = mse_loss(acc_y, y_batch_vy[:,0])
                loss_w =  mse_loss(acc_w, y_batch_w[:,0])

                # Combine all losses
                if train_st==True and train_th==True:
                    total_loss =  loss_weights_th + loss_weights_st + loss_vx + loss_w + loss_vy
                elif train_st==True and train_th==False:
                    total_loss =  loss_weights_st + loss_w + loss_vy
                elif train_st==False and train_th==True:
                    total_loss =  loss_weights_th + loss_vx 
                #total_loss =   loss_vx + loss_weights_th
                #total_loss = loss_vy + loss_w + loss_weights_st
                
                # Print the current loss for vx (or use other loss types if needed)
                minibatch_iter.set_postfix(loss=total_loss.item())

                # Backward pass (compute gradients for the total loss)
                total_loss.backward()

                # Update parameters using the optimizer
                optimizer.step()


            # weights for plotting
            if live_plot_weights:
                # Convert to numpy for plotting
                weights_th_numpy = weights_throttle.detach().cpu().numpy().squeeze()
                weights_st_numpy = weights_steering.detach().cpu().numpy().squeeze()

                # Live update plot every few iterations
                if len(weights_th_numpy) == len(time_axis):  # Ensure matching dimensions
                    clear_output(wait=True)
                    line_th.set_xdata(time_axis)
                    line_th.set_ydata(weights_th_numpy)
                    line_st.set_xdata(time_axis)
                    line_st.set_ydata(weights_st_numpy)

                    ax1.relim()  # Recalculate limits
                    ax1.autoscale_view(True, True, True)  # Autoscale axes
                    ax1.set_title(f"Live Plot of Weights Over Time - Epoch {i + 1}")
                    #display(fig)
                    fig.canvas.manager.window.raise_()
                    
                    plt.pause(0.01)



            

            loss_2_print_vx_vec = [*loss_2_print_vx_vec, loss_vx.item()]
            loss_2_print_vy_vec = [*loss_2_print_vy_vec, loss_vy.item()]
            loss_2_print_w_vec = [*loss_2_print_w_vec, loss_w.item()]
            loss_2_print_weights = [*loss_2_print_weights, loss_weights_th.item()+loss_weights_st.item()]
            total_loss_vec = [*total_loss_vec, total_loss.item()]


        if live_plot_weights:
            plt.ioff()  # Turn off interactive mode
            plt.close(fig)  # Close the figure

        #plot loss functions
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(loss_2_print_vx_vec,label='loss vx',color='dodgerblue') 
        ax.plot(loss_2_print_vy_vec,label='loss vy',color='orangered')
        ax.plot(loss_2_print_w_vec,label='loss w',color='orchid')
        ax.plot(loss_2_print_weights,label='loss weights',color='lime')
        ax.plot(total_loss_vec,label='total loss',color='k')
        
        ax.legend()


    def save_model(self,folder_2_save_params,n_past_actions,dt,train_th,train_st):

        np.save(folder_2_save_params + 'n_past_actions.npy', n_past_actions)
        np.save(folder_2_save_params + 'dt.npy', dt)

        weights_throttle, non_normalized_w_th = self.constrained_linear_layer(self.raw_weights_throttle)
        weights_throttle = weights_throttle.t().detach().cpu().numpy()
        weights_steering, non_normalized_w_st = self.constrained_linear_layer(self.raw_weights_steering)
        weights_steering = weights_steering.t().detach().cpu().numpy()

        if train_th==True:
            np.save(folder_2_save_params + 'weights_throttle.npy', weights_throttle)
        if train_st==True:
            np.save(folder_2_save_params + 'weights_steering.npy', weights_steering)

        print('------------------------------')
        print('--- saved model parameters ---')
        print('------------------------------')
        print('saved parameters in folder: ', folder_2_save_params)



def load_SVGPModel_actuator_dynamics(folder_path):
    #load model paths
    model_path_vx = folder_path + '/svgp_model_vx.pth'
    model_path_vy = folder_path + '/svgp_model_vy.pth'
    model_path_w = folder_path + '/svgp_model_w.pth'

    #load inducing points
    inducing_points_vx = folder_path + '/inducing_locations_x.npy'
    inducing_points_vy = folder_path + '/inducing_locations_y.npy'
    inducing_points_w = folder_path + '/inducing_locations_w.npy'

    #load nupy arrays
    inducing_points_vx = np.load(inducing_points_vx)
    inducing_points_vy = np.load(inducing_points_vy)
    inducing_points_w = np.load(inducing_points_w)

    # convert to torch tensors
    inducing_points_vx = torch.from_numpy(inducing_points_vx).float()
    inducing_points_vy = torch.from_numpy(inducing_points_vy).float()
    inducing_points_w = torch.from_numpy(inducing_points_w).float()

    # instantiate the models
    model_vx = SVGPModel_actuator_dynamics(inducing_points_vx)
    model_vy = SVGPModel_actuator_dynamics(inducing_points_vy)
    model_w = SVGPModel_actuator_dynamics(inducing_points_w)

    #load time delay parameters
    time_delay_parameters_path = folder_path + '/time_delay_parameters.npy'
    time_delay_parameters = np.load(time_delay_parameters_path)

    #set up time delay parameters
    # set up time filtering
    actuator_time_delay_fitting_tag = time_delay_parameters[0]
    n_past_actions = time_delay_parameters[1]
    dt_svgp = time_delay_parameters[2]
    model_vx.setup_time_delay_fitting(actuator_time_delay_fitting_tag,n_past_actions,dt_svgp)
    model_vy.setup_time_delay_fitting(actuator_time_delay_fitting_tag,n_past_actions,dt_svgp)
    model_w.setup_time_delay_fitting(actuator_time_delay_fitting_tag,n_past_actions,dt_svgp)

    # load state dictionaries
    model_vx.load_state_dict(torch.load(model_path_vx))
    model_vy.load_state_dict(torch.load(model_path_vy))
    model_w.load_state_dict(torch.load(model_path_w))

    # set to evaluation mode
    model_vx.eval()
    model_vy.eval()
    model_w.eval()

    return model_vx,model_vy,model_w



# def load_SVGPModel_actuator_dynamics_analytic(folder_path):
#     svgp_params_path = folder_path + '/SVGP_saved_parameters/'

#     # Define the parameter names for each dimension (x, y, w)
#     param_names = ['m', 'middle', 'L_inv', 'right_vec', 'inducing_locations', 'outputscale', 'lengthscale']
#     dimensions = ['x', 'y', 'w']

#     # Initialize an empty dictionary to store all parameters
#     svgp_params = {}

#     # Loop through each dimension and parameter name to load the .npy files
#     for dim in dimensions:
#         svgp_params[dim] = {}
#         for param in param_names:
#             file_path = os.path.join(svgp_params_path, f"{param}_{dim}.npy")
#             if os.path.exists(file_path):
#                 svgp_params[dim][param] = np.load(file_path)
#                 #print(f"Loaded {param}_{dim}: shape {svgp_params[dim][param].shape}")
#             else:
#                 print(f"Warning: {param}_{dim}.npy not found in {svgp_params_path}")

#     # load time delay parameters
#     time_delay_parameters_path = folder_path + '/SVGP_saved_parameters/time_delay_parameters.npy'
#     time_delay_parameters = np.load(time_delay_parameters_path)
#     actuator_time_delay_fitting_tag = time_delay_parameters[0]
#     n_past_actions = time_delay_parameters[1]
#     dt_svgp = time_delay_parameters[2]

#     evalaute_cov_tag = False # dont evaluate covariance for now
#     # now build the models
#     model_vx = SVGP_analytic(svgp_params['x']['outputscale'],
#                              svgp_params['x']['lengthscale'],
#                              svgp_params['x']['inducing_locations'],
#                              svgp_params['x']['right_vec'],
#                              svgp_params['x']['L_inv'],
#                              evalaute_cov_tag)
#     model_vx.actuator_time_delay_fitting_tag = actuator_time_delay_fitting_tag
#     model_vx.n_past_actions = n_past_actions
#     model_vx.dt = dt_svgp

#     model_vy = SVGP_analytic(svgp_params['y']['outputscale'],
#                                 svgp_params['y']['lengthscale'],
#                                 svgp_params['y']['inducing_locations'],
#                                 svgp_params['y']['right_vec'],
#                                 svgp_params['y']['L_inv'],
#                                 evalaute_cov_tag)
#     model_vy.actuator_time_delay_fitting_tag = actuator_time_delay_fitting_tag
#     model_vy.n_past_actions = n_past_actions
#     model_vy.dt = dt_svgp
    
#     model_w = SVGP_analytic(svgp_params['w']['outputscale'],
#                                 svgp_params['w']['lengthscale'],
#                                 svgp_params['w']['inducing_locations'],
#                                 svgp_params['w']['right_vec'],
#                                 svgp_params['w']['L_inv'],
#                                 evalaute_cov_tag)
#     model_w.actuator_time_delay_fitting_tag = actuator_time_delay_fitting_tag
#     model_w.n_past_actions = n_past_actions
#     model_w.dt = dt_svgp

#     return model_vx,model_vy,model_w



def load_SVGPModel_actuator_dynamics_analytic(folder_path,evalaute_cov_tag):
    svgp_params_path = folder_path 

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
    time_delay_parameters_path = folder_path + '/time_delay_parameters.npy'
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


















# simple SVGP model 
class SVGPModel(ApproximateGP):
    def __init__(self,inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(SVGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# plot simple SVGP output
def plot_GP(ax,x,y,subset_indexes,model,likelihood,resolution,df):
    resolution = resolution * 0.99
    # move to cpu
    x = x.cpu()
    y = y.cpu()
    model = model.cpu()
    likelihood = likelihood.cpu()

    # plot subset with an orange circle
    ax.plot(df['vicon time'].to_numpy()[subset_indexes], y[subset_indexes].cpu().numpy(), 'o', color='orange',alpha=0.5,markersize=3)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred_likelyhood = likelihood(model(x))
        lower_likelyhood, upper_likelyhood = observed_pred_likelyhood.confidence_region()
        ax.plot(df['vicon time'].to_numpy(), observed_pred_likelyhood.mean.cpu().numpy(), 'k')
        ax.fill_between(df['vicon time'].to_numpy(), lower_likelyhood.cpu().numpy(), upper_likelyhood.cpu().numpy(), alpha=0.3)

        # obseved pred epistemic uncertainty
        observed_pred_model = model(x)
        lower_model, upper_model = observed_pred_model.confidence_region()
        ax.fill_between(df['vicon time'].to_numpy(), lower_model.cpu().numpy(), upper_model.cpu().numpy(), alpha=0.3, color='k')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Training Data')

    #move back to gpu
    x = x.cuda()
    y = y.cuda()
    model = model.cuda()
    likelihood = likelihood.cuda()




def train_SVGP_model(num_epochs,
                    train_x, train_y_vx, train_y_vy, train_y_w,
                    model_vx,model_vy,model_w,
                    likelihood_vx,likelihood_vy,likelihood_w,
                    optimizer_vx,optimizer_vy,optimizer_w):
    
    # start fitting
    # make contiguous (not sure why)
    train_x = train_x.contiguous()
    train_y_vx = train_y_vx.contiguous()
    train_y_vy = train_y_vy.contiguous()
    train_y_w = train_y_w.contiguous()

    # define batches for training (each bach will be used to perform a gradient descent step in each iteration. So toal parameters updates are Epochs*n_batches)
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset_vx = TensorDataset(train_x, train_y_vx)
    train_dataset_vy = TensorDataset(train_x, train_y_vy)
    train_dataset_w = TensorDataset(train_x, train_y_w)

    # define data loaders
    train_loader_vx = DataLoader(train_dataset_vx, batch_size=250, shuffle=True)
    train_loader_vy = DataLoader(train_dataset_vy, batch_size=250, shuffle=True)
    train_loader_w = DataLoader(train_dataset_w, batch_size=250, shuffle=True)


    # Assign training data to models just to have it all together for later plotting
    model_vx.train_x = train_x 
    model_vx.train_y_vx = train_y_vx

    model_vy.train_x = train_x 
    model_vy.train_y_vy = train_y_vy

    model_w.train_x = train_x 
    model_w.train_y_w = train_y_w


    #move to GPU for faster fitting
    if torch.cuda.is_available():
        model_vx = model_vx.cuda()
        model_vy = model_vy.cuda()
        model_w = model_w.cuda()
        likelihood_vx = likelihood_vx.cuda()
        likelihood_vy = likelihood_vy.cuda()
        likelihood_w = likelihood_w.cuda()

    #set to training mode
    model_vx.train()
    model_vy.train()
    model_w.train()
    likelihood_vx.train()
    likelihood_vy.train()
    likelihood_w.train()

    # Set up loss object. We're using the VariationalELBO
    mll_vx = gpytorch.mlls.VariationalELBO(likelihood_vx, model_vx, num_data=train_y_vx.size(0))#, beta=1)
    mll_vy = gpytorch.mlls.VariationalELBO(likelihood_vy, model_vy, num_data=train_y_vy.size(0))#, beta=1)
    mll_w = gpytorch.mlls.VariationalELBO(likelihood_w, model_w, num_data=train_y_w.size(0))#, beta=1)

    loss_2_print_vx_vec = []
    loss_2_print_vy_vec = []
    loss_2_print_w_vec = []

    # start training (tqdm is just to show the loading bar)
    bar_format=bar_format=f"{Fore.GREEN}{{l_bar}}{Fore.GREEN}{{bar}}{Style.RESET_ALL}{Fore.GREEN}{{r_bar}}{Style.RESET_ALL}"
    epochs_iter = tqdm.tqdm(range(num_epochs), desc=f"{Fore.GREEN}Epochs", leave=True, bar_format=bar_format)
    
    
    for i in epochs_iter: #range(num_epochs):
        torch.cuda.empty_cache()  # Releases unused cached memory


        # Within each iteration, we will go over each minibatch of data
        minibatch_iter_vx = tqdm.tqdm(train_loader_vx, desc="Minibatch vx", leave=False) # , disable=True
        minibatch_iter_vy = tqdm.tqdm(train_loader_vy, desc="Minibatch vy", leave=False) # , disable=True
        minibatch_iter_w  = tqdm.tqdm(train_loader_w,  desc="Minibatch w",  leave=False) # , disable=True

        for x_batch_vx, y_batch_vx in minibatch_iter_vx:

            optimizer_vx.zero_grad()
            output_vx = model_vx(x_batch_vx)
            loss_vx = -mll_vx(output_vx, y_batch_vx[:,0])
            minibatch_iter_vx.set_postfix(loss=loss_vx.item())
            loss_vx.backward()
            optimizer_vx.step()

        loss_2_print_vx_vec = [*loss_2_print_vx_vec, loss_vx.item()]

        for x_batch_vy, y_batch_vy in minibatch_iter_vy:
            optimizer_vy.zero_grad()
            output_vy = model_vy(x_batch_vy)
            loss_vy = -mll_vy(output_vy, y_batch_vy[:,0])
            minibatch_iter_vy.set_postfix(loss=loss_vy.item())
            loss_vy.backward()
            optimizer_vy.step()

        loss_2_print_vy_vec = [*loss_2_print_vy_vec, loss_vy.item()]

        for x_batch_w, y_batch_w in minibatch_iter_w:
            optimizer_w.zero_grad()
            output_w = model_w(x_batch_w)
            loss_w = -mll_w(output_w, y_batch_w[:,0])
            minibatch_iter_w.set_postfix(loss=loss_w.item())
            loss_w.backward()
            optimizer_w.step()

        loss_2_print_w_vec = [*loss_2_print_w_vec, loss_w.item()]
           
    #plot loss functions
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(loss_2_print_vx_vec,label='loss vx',color='dodgerblue') 
    ax.plot(loss_2_print_vy_vec,label='loss vy',color='orangered')
    ax.plot(loss_2_print_w_vec,label='loss w',color='orchid')
    ax.legend()


    #move to gpu for later evaluation
    model_vx = model_vx.cuda()
    model_vy = model_vy.cuda()
    model_w = model_w.cuda()

    return model_vx, model_vy, model_w, likelihood_vx, likelihood_vy, likelihood_w





#define orthogonally decoupled SVGP model
# Orthogonally decoupled SVGP
def make_orthogonal_vs(model,mean_inducing_points,covar_inducing_points):


    covar_variational_strategy = gpytorch.variational.VariationalStrategy(
        model, covar_inducing_points,
        gpytorch.variational.CholeskyVariationalDistribution(covar_inducing_points.size(-2)),
        learn_inducing_locations=True
    )

    variational_strategy = gpytorch.variational.OrthogonallyDecoupledVariationalStrategy(
        covar_variational_strategy, mean_inducing_points,
        gpytorch.variational.DeltaVariationalDistribution(mean_inducing_points.size(-2)),
    )
    return variational_strategy

class OrthDecoupledApproximateGP(ApproximateGP):
    def __init__(self,mean_inducing_points,covar_inducing_points):
        #variational_distribution = gpytorch.variational.DeltaVariationalDistribution(inducing_points.size(-2))
        variational_strategy = make_orthogonal_vs(self,mean_inducing_points,covar_inducing_points)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module =  gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=mean_inducing_points.size(dim=1)))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_decoupled_SVGP_model(learning_rate,num_epochs, train_x, train_y_vx, train_y_vy, train_y_w, n_inducing_points_mean,n_inducing_points_cov):
    
    # start fitting
    # make contiguous (not sure why)
    train_x = train_x.contiguous()
    train_y_vx = train_y_vx.contiguous()
    train_y_vy = train_y_vy.contiguous()
    train_y_w = train_y_w.contiguous()

    # define batches for training (each bach will be used to perform a gradient descent step in each iteration. So toal parameters updates are Epochs*n_batches)
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset_vx = TensorDataset(train_x, train_y_vx)
    train_dataset_vy = TensorDataset(train_x, train_y_vy)
    train_dataset_w = TensorDataset(train_x, train_y_w)

    # define data loaders
    train_loader_vx = DataLoader(train_dataset_vx, batch_size=250, shuffle=True)
    train_loader_vy = DataLoader(train_dataset_vy, batch_size=250, shuffle=True)
    train_loader_w = DataLoader(train_dataset_w, batch_size=250, shuffle=True)

    #choosing initial guess inducing points as a random subset of the training data
    random.seed(10) # set the seed so to have same points for every run
    # random selection of inducing points
    random_indexes_mean = random.choices(range(train_x.shape[0]), k=n_inducing_points_mean)
    inducing_points_mean = train_x[random_indexes_mean, :]
    inducing_points_mean = inducing_points_mean.to(torch.float32)

    random_indexes_cov = random.choices(range(inducing_points_mean.shape[0]), k=n_inducing_points_cov)
    inducing_points_cov = train_x[random_indexes_cov, :]


    #initialize models
    model_vx = OrthDecoupledApproximateGP(inducing_points_mean,inducing_points_cov)
    model_vy = OrthDecoupledApproximateGP(inducing_points_mean,inducing_points_cov)
    model_w = OrthDecoupledApproximateGP(inducing_points_mean,inducing_points_cov)

    # Assign training data to models just to have it all together for later plotting
    model_vx.train_x = train_x 
    model_vx.train_y_vx = train_y_vx

    model_vy.train_x = train_x 
    model_vy.train_y_vy = train_y_vy

    model_w.train_x = train_x 
    model_w.train_y_w = train_y_w


    #define likelyhood objects
    likelihood_vx = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_vy = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_w = gpytorch.likelihoods.GaussianLikelihood()
 


    #move to GPU for faster fitting
    if torch.cuda.is_available():
        model_vx = model_vx.cuda()
        model_vy = model_vy.cuda()
        model_w = model_w.cuda()
        likelihood_vx = likelihood_vx.cuda()
        likelihood_vy = likelihood_vy.cuda()
        likelihood_w = likelihood_w.cuda()

    #set to training mode
    model_vx.train()
    model_vy.train()
    model_w.train()
    likelihood_vx.train()
    likelihood_vy.train()
    likelihood_w.train()

    #set up optimizer and its options
    optimizer_vx = torch.optim.AdamW([{'params': model_vx.parameters()}, {'params': likelihood_vx.parameters()},], lr=learning_rate)
    optimizer_vy = torch.optim.AdamW([{'params': model_vy.parameters()}, {'params': likelihood_vy.parameters()},], lr=learning_rate)
    optimizer_w = torch.optim.AdamW([{'params': model_w.parameters()}, {'params': likelihood_w.parameters()},], lr=learning_rate)


    # Set up loss object. We're using the VariationalELBO
    mll_vx = gpytorch.mlls.VariationalELBO(likelihood_vx, model_vx, num_data=train_y_vx.size(0))#, beta=1)
    mll_vy = gpytorch.mlls.VariationalELBO(likelihood_vy, model_vy, num_data=train_y_vy.size(0))#, beta=1)
    mll_w = gpytorch.mlls.VariationalELBO(likelihood_w, model_w, num_data=train_y_w.size(0))#, beta=1)

    # start training (tqdm is just to show the loading bar)
    epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")



    loss_2_print_vx_vec = []
    loss_2_print_vy_vec = []
    loss_2_print_w_vec = []

    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        minibatch_iter_vx = tqdm.tqdm(train_loader_vx, desc="Minibatch vx", leave=False, disable=True)
        minibatch_iter_vy = tqdm.tqdm(train_loader_vy, desc="Minibatch vy", leave=False, disable=True)
        minibatch_iter_w  = tqdm.tqdm(train_loader_w,  desc="Minibatch w",  leave=False, disable=True)

        for x_batch_vx, y_batch_vx in minibatch_iter_vx:
            optimizer_vx.zero_grad()
            output_vx = model_vx(x_batch_vx)
            loss_vx = -mll_vx(output_vx, y_batch_vx[:,0])
            minibatch_iter_vx.set_postfix(loss=loss_vx.item())
            loss_vx.backward()
            optimizer_vx.step()

        loss_2_print_vx_vec = [*loss_2_print_vx_vec, loss_vx.item()]

        for x_batch_vy, y_batch_vy in minibatch_iter_vy:
            optimizer_vy.zero_grad()
            output_vy = model_vy(x_batch_vy)
            loss_vy = -mll_vy(output_vy, y_batch_vy[:,0])
            minibatch_iter_vy.set_postfix(loss=loss_vy.item())
            loss_vy.backward()
            optimizer_vy.step()

        loss_2_print_vy_vec = [*loss_2_print_vy_vec, loss_vy.item()]

        for x_batch_w, y_batch_w in minibatch_iter_w:
            optimizer_w.zero_grad()
            output_w = model_w(x_batch_w)
            loss_w = -mll_w(output_w, y_batch_w[:,0])
            minibatch_iter_w.set_postfix(loss=loss_w.item())
            loss_w.backward()
            optimizer_w.step()

        loss_2_print_w_vec = [*loss_2_print_w_vec, loss_w.item()]
           
    #plot loss functions
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(loss_2_print_vx_vec,label='loss vx',color='dodgerblue') 
    ax.plot(loss_2_print_vy_vec,label='loss vy',color='orangered')
    ax.plot(loss_2_print_w_vec,label='loss w',color='orchid')
    ax.legend()


    #move to gpu for later evaluation
    model_vx = model_vx.cuda()
    model_vy = model_vy.cuda()
    model_w = model_w.cuda()

    return model_vx, model_vy, model_w, likelihood_vx, likelihood_vy, likelihood_w






class dyn_model_SVGP_4_long_term_predictions():
    def __init__(self,model_vx,model_vy,model_w):

        self.model_vx = model_vx
        self.model_vy = model_vy
        self.model_w = model_w

    def forward(self, state_action):
        # input will have both inputs and input commands, so chose the right ones
        if self.model_vx.actuator_time_delay_fitting_tag == 0: # take unfiltered inputs
            #['vx body', 'vy body', 'w', 'throttle filtered' ,'steering filtered','throttle','steering']
            state_action_base_model = np.array([*state_action[:3],state_action[5],state_action[6]])
            throttle_dot = 0
            steering_dot = 0

        elif self.model_vx.actuator_time_delay_fitting_tag == 3 or self.model_vx.actuator_time_delay_fitting_tag ==2: # take filtered inputs
            #['vx body', 'vy body', 'w', 'throttle filtered' ,'steering filtered','throttle','steering']
            state_action_base_model = state_action[:5]
            throttle_dot = 0
            steering_dot = 0
            
        input = torch.unsqueeze(torch.Tensor(state_action_base_model),0)
        ax = self.model_vx(input).mean.detach().cpu().numpy()[0]
        ay = self.model_vy(input).mean.detach().cpu().numpy()[0]
        aw = self.model_w(input).mean.detach().cpu().numpy()[0]

        return np.array([ax,ay,aw,throttle_dot,steering_dot])




class dyn_model_SVGP_4_long_term_predictions_analytical():
    def __init__(self,model_vx,model_vy,model_w):

        self.model_vx = model_vx
        self.model_vy = model_vy
        self.model_w = model_w

    def forward(self, state_action):
        # input will have both inputs and input commands, so chose the right ones
        if self.model_vx.actuator_time_delay_fitting_tag == 0: # take unfiltered inputs
            #['vx body', 'vy body', 'w', 'throttle filtered' ,'steering filtered','throttle','steering']
            state_action_base_model = np.array([*state_action[:3],state_action[5],state_action[6]])
            throttle_dot = 0
            steering_dot = 0

        elif self.model_vx.actuator_time_delay_fitting_tag == 3: # take filtered inputs
            #['vx body', 'vy body', 'w', 'throttle filtered' ,'steering filtered','throttle','steering']
            state_action_base_model = state_action[:5]
            throttle_dot = 0
            steering_dot = 0
            
        ax, cov_x = self.model_vx.forward(state_action_base_model)
        ay, cov_y = self.model_vy.forward(state_action_base_model)
        aw, cov_w = self.model_w.forward(state_action_base_model)


        return np.array([ax,ay,aw,throttle_dot,steering_dot])


# class SVGP_analytic():
#     def __init__(self,outputscale,lengthscale,inducing_locations,right_vec,L_inv,evalaute_cov_tag):

#         self.outputscale = outputscale
#         self.lengthscale = lengthscale
#         self.inducing_locations = inducing_locations
#         self.right_vec = right_vec
#         self.L_inv = L_inv
#         self.evalaute_cov_tag = evalaute_cov_tag

#     def forward(self, x_star):
#         #make x_star into a 5 x 1 array
#         x_star = np.expand_dims(x_star, axis=0)
#         kXZ = rebuild_Kxy_RBF_vehicle_dynamics(x_star,np.squeeze(self.inducing_locations),self.outputscale,self.lengthscale)

#         # calculate mean and covariance for x
#         mean = kXZ @ self.right_vec
#         if self.evalaute_cov_tag:
#             # calculate covariance
#             X = self.L_inv @ kXZ.T
#             KXX = RBF_kernel_rewritten(x_star[0],x_star[0],self.outputscale,self.lengthscale)
#             cov = KXX + X.T @ self.middle @ X
#         else:
#             cov = 0

#         return mean[0], cov
    

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



