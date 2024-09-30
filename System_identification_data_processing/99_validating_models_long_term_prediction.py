from functions_for_data_processing import get_data, plot_raw_data, process_raw_vicon_data,plot_vicon_data\
,dyn_model_culomb_tires,produce_long_term_predictions,full_dynamic_model,model_parameters,throttle_dynamics,\
    process_vicon_data_kinematics,steering_dynamics,directly_measured_model_parameters
from matplotlib import pyplot as plt
import torch
import numpy as np
import pandas as pd
import glob
import os
from scipy.interpolate import CubicSpline
import matplotlib
font = {'family' : 'normal',
        'size'   : 10}

#matplotlib.rc('font', **font)

# This script is used to fit the tire model to the data collected using the vicon external tracking system with a 
# SIMPLE CULOMB friction tyre model.

# chose what stated to forward propagate (the others will be taken from the data, this can highlight individual parts of the model)
forward_propagate_indexes = [1,2,3] #[1,2,3] # 1 = vx, 2=vy, 3=w

# select data folder NOTE: this assumes that the current directory is DART
#folder_path = 'System_identification_data_processing/Data/90_model_validation_long_term_predictions'  # the battery was very low for this one
#folder_path = 'System_identification_data_processing/Data/91_model_validation_long_term_predictions_fast'
folder_path = 'System_identification_data_processing/Data/91_free_driving_16_sept_2024'
#folder_path = 'System_identification_data_processing/Data/91_free_driving_16_sept_2024_slow'
#folder_path = 'System_identification_data_processing/Data/free_driving_steer_rate_testing_16_sept_2024'

#folder_path = 'System_identification_data_processing/Data/81_throttle_ramps_only_steer03'
#folder_path = 'System_identification_data_processing/Data/circles_27_sept_2024'

[theta_correction, l_COM, l_lateral_shift_reference ,lr, lf, Jz, m,m_front_wheel,m_rear_wheel] = directly_measured_model_parameters()

# load model parameters
[a_m, b_m, c_m, d_m,
a_f, b_f, c_f, d_f,
a_s, b_s, c_s, d_s, e_s,
d_t_f, c_t_f, b_t_f,d_t_r, c_t_r, b_t_r,
a_stfr, b_stfr,d_stfr,e_stfr,f_stfr,g_stfr,
max_st_dot,fixed_delay_stdn,k_stdn,
w_natural_Hz_pitch,k_f_pitch,k_r_pitch,
w_natural_Hz_roll,k_f_roll,k_r_roll] = model_parameters()

#a_stfr = [] # shut down steering friction term


# the model gives you the derivatives of it's own states, so you can integrate them to get the states in the new time instant
dynamic_model = dyn_model_culomb_tires(m,m_front_wheel,m_rear_wheel,lr,lf,l_COM,Jz,
                 a_m,b_m,c_m,
                 a_f,b_f,c_f,d_f,
                 d_t_f, c_t_f, b_t_f,d_t_r, c_t_r, b_t_r,
                 a_stfr, b_stfr,d_stfr,e_stfr,f_stfr,g_stfr)






# process data

steps_shift = 5 # decide to filter more or less the vicon data

# df_raw_data = get_data(folder_path)

# # cut time
# #df_raw_data = df_raw_data[df_raw_data['vicon time']<235]

# df_kinematics = process_vicon_data_kinematics(df_raw_data,steps_shift,theta_correction, l_COM, l_lateral_shift_reference)
# df = process_raw_vicon_data(df_kinematics,steps_shift)


# # replace throttle with time integrated throttle
# filtered_throttle = throttle_dynamics(df_raw_data,d_m)
# df['throttle'] = filtered_throttle

# # add steering time integrated 
# st_vec_angle_optuna, st_vec_optuna = steering_dynamics(df_raw_data,a_s,b_s,c_s,d_s,e_s,max_st_dot,fixed_delay_stdn,k_stdn)
# # over-write the actual data with the forward integrated data
# df['steering angle'] = st_vec_angle_optuna
# df['steering'] = st_vec_optuna




# check if there is a processed vicon data file already
file_name = 'processed_vicon_data.csv'
# Check if the CSV file exists in the folder
file_path = os.path.join(folder_path, file_name)

if not os.path.isfile(file_path):
    df_raw_data = get_data(folder_path)
    # cut time
    #df_raw_data = df_raw_data[df_raw_data['vicon time']<235]

    # replace steering angle with time integated version
    # replace throttle with time integrated throttle
    filtered_throttle = throttle_dynamics(df_raw_data,d_m)
    df_raw_data['throttle'] = filtered_throttle

    # add steering time integrated 
    st_vec_angle_optuna, st_vec_optuna = steering_dynamics(df_raw_data,a_s,b_s,c_s,d_s,e_s,max_st_dot,fixed_delay_stdn,k_stdn)
    # over-write the actual data with the forward integrated data
    df_raw_data['steering angle'] = st_vec_angle_optuna
    df_raw_data['steering'] = st_vec_optuna

    # process kinematics and dynamics
    df_kinematics = process_vicon_data_kinematics(df_raw_data,steps_shift,theta_correction, l_COM, l_lateral_shift_reference)
    df = process_raw_vicon_data(df_kinematics,steps_shift)

    #save the processed data file
    df.to_csv(file_path, index=False)
    print(f"File '{file_path}' saved.")
else:
    print(f"File '{file_path}' already exists, loading data.")
    df = pd.read_csv(file_path)







# plot raw data
ax0,ax1,ax2 = plot_raw_data(df)

# plot vicon related data (longitudinal and lateral velocities, yaw rate related)
ax_wheel_f_alpha,ax_wheel_r_alpha,ax_total_force_front,\
ax_total_force_rear,ax_lat_force,ax_long_force,\
ax_acc_x_body,ax_acc_y_body,ax_acc_w = plot_vicon_data(df) 






# producing long term predictions
columns_to_extract = ['vicon time', 'vx body', 'vy body', 'w', 'throttle' ,'steering angle','vicon x','vicon y','vicon yaw']
input_data_long_term_predictions = df[columns_to_extract].to_numpy()
prediction_window = 1.5 # [s]
jumps = 25 #25
long_term_predictions = produce_long_term_predictions(input_data_long_term_predictions, dynamic_model,prediction_window,jumps,forward_propagate_indexes)

# plot long term predictions over real data
fig, ((ax10,ax11,ax12)) = plt.subplots(3, 1, figsize=(10, 6))
fig.subplots_adjust(top=0.995,
                    bottom=0.11,
                    left=0.095,
                    right=0.995,
                    hspace=0.345,
                    wspace=0.2)

time_vec_data = df['vicon time'].to_numpy()


# velocities
ax10.plot(time_vec_data,input_data_long_term_predictions[:,1],color='dodgerblue',label='vx',linewidth=4,linestyle='-')
ax10.set_xlabel('Time [s]')
ax10.set_ylabel('Vx body[m/s]')
ax10.legend()
ax10.set_title('Vx')


ax11.plot(time_vec_data,input_data_long_term_predictions[:,2],color='orangered',label='vy',linewidth=4,linestyle='-')
ax11.set_xlabel('Time [s]')
ax11.set_ylabel('Vy body[m/s]')
ax11.legend()
ax11.set_title('Vy')


ax12.plot(time_vec_data,input_data_long_term_predictions[:,3],color='orchid',label='w',linewidth=4,linestyle='-')
ax12.set_xlabel('Time [s]')
ax12.set_ylabel('W [rad/s]')
ax12.legend()
ax12.set_title('W')


# positions
fig, ((ax13,ax14,ax15)) = plt.subplots(3, 1, figsize=(10, 6))
fig.subplots_adjust(top=0.995,
                    bottom=0.11,
                    left=0.095,
                    right=0.995,
                    hspace=0.345,
                    wspace=0.2)



ax13.plot(time_vec_data,input_data_long_term_predictions[:,6],color='dodgerblue',label='x',linewidth=4,linestyle='-')
ax13.set_xlabel('time [s]')
ax13.set_ylabel('y [m]')
ax13.legend()
ax13.set_title('trajectory in the x-y plane')

ax14.plot(time_vec_data,input_data_long_term_predictions[:,7],color='orangered',label='y',linewidth=4,linestyle='-')
ax14.set_xlabel('time [s]')
ax14.set_ylabel('y [m]')
ax14.legend()
ax14.set_title('trajectory in the x-y plane')

ax15.plot(time_vec_data,input_data_long_term_predictions[:,8],color='orchid',label='yaw',linewidth=4,linestyle='-')
ax15.set_xlabel('time [s]')
ax15.set_ylabel('yaw [rad]')
ax15.legend()
ax15.set_title('vehicle yaw')


# trajectory
fig, ((ax16)) = plt.subplots(1, 1, figsize=(10, 6))
fig.subplots_adjust(top=0.995,
                    bottom=0.11,
                    left=0.095,
                    right=0.995,
                    hspace=0.345,
                    wspace=0.2)

ax16.plot(input_data_long_term_predictions[:,6],input_data_long_term_predictions[:,7],color='orange',label='trajectory',linewidth=4,linestyle='-')
ax16.set_xlabel('x [m]')
ax16.set_ylabel('y [m]')
ax16.legend()
ax16.set_title('vehicle trajectory in the x-y plane')




fig1, ((ax_acc_x,ax_acc_y,ax_acc_w)) = plt.subplots(3, 1, figsize=(10, 6), constrained_layout=True)

# plot longitudinal accleeartion
ax_acc_x.plot(df['vicon time'].to_numpy(),df['ax body'].to_numpy(),label='ax body',color='dodgerblue')

ax_acc_x.legend()

# plot lateral accleeartion
ax_acc_y.plot(df['vicon time'].to_numpy(),df['ay body'].to_numpy(),label='ay body',color='orangered')
ax_acc_y.legend()

# plot yaw rate
ax_acc_w.plot(df['vicon time'].to_numpy(),df['acc_w'].to_numpy(),label='acc_w',color='purple')
ax_acc_w.legend()










# plot long term predictions

for i in range(0,len(long_term_predictions)):
    pred = long_term_predictions[i]
    #velocities
    ax10.plot(pred[:,0],pred[:,1],color='k',alpha=0.2)
    ax11.plot(pred[:,0],pred[:,2],color='k',alpha=0.2)
    ax12.plot(pred[:,0],pred[:,3],color='k',alpha=0.2)
    # positions
    ax13.plot(pred[:,0],pred[:,6],color='k',alpha=0.2)
    ax14.plot(pred[:,0],pred[:,7],color='k',alpha=0.2)
    ax15.plot(pred[:,0],pred[:,8],color='k',alpha=0.2)
    #trajectory
    ax16.plot(pred[:,6],pred[:,7],color='k',alpha=0.2)

    #accelerations
    state_action_matrix = pred[:,[1,2,3,4,5]]
    # add Fx

    acc_x_model = np.zeros(pred.shape[0])
    acc_y_model = np.zeros(pred.shape[0])
    acc_w_model = np.zeros(pred.shape[0])

    for i in range(pred.shape[0]):
        acc = dynamic_model.forward(state_action_matrix[i,:])
        acc_x_model[i] = acc[0]
        acc_y_model[i] = acc[1]
        acc_w_model[i] = acc[2]


    # plot the model accelerations
    ax_acc_x.plot(pred[:,0],acc_x_model,color='k')
    ax_acc_y.plot(pred[:,0],acc_y_model,color='k')
    ax_acc_w.plot(pred[:,0],acc_w_model,color='k')
    ax_acc_x.legend()
    ax_acc_y.legend()
    ax_acc_w.legend()


ax16.set_aspect('equal')





plt.show()



