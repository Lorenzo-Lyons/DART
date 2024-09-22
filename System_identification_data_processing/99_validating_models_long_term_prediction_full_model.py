from functions_for_data_processing import get_data, plot_raw_data, process_raw_vicon_data,plot_vicon_data\
,dyn_model_culomb_tires,produce_long_term_predictions,full_dynamic_model,model_parameters,throttle_dynamics,steering_dynamics, produce_long_term_predictions_full_model
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
# 1 = vx
# 2 = vy
# 3 = w
# 4 = throttle
# 5 = steering
# 6 = pitch dot
# 7 = pitch
# 8 = roll dot
# 9 = roll

forward_propagate_indexes = [1,2,3,4,5,6,7,8,9]  
#forward_propagate_indexes = [1,2,3] 

# select data folder NOTE: this assumes that the current directory is DART
#folder_path = 'System_identification_data_processing/Data/90_model_validation_long_term_predictions'  # the battery was very low for this one
#folder_path = 'System_identification_data_processing/Data/91_model_validation_long_term_predictions_fast'
folder_path = 'System_identification_data_processing/Data/91_free_driving_16_sept_2024'
#folder_path = 'System_identification_data_processing/Data/91_free_driving_16_sept_2024_slow'
#folder_path = 'System_identification_data_processing/Data/free_driving_steer_rate_testing_16_sept_2024'




# load model parameters
[theta_correction, lr, l_COM, Jz, lf, m,
a_m, b_m, c_m, d_m,
a_f, b_f, c_f, d_f,
a_s, b_s, c_s, d_s, e_s,
d_t, c_t, b_t,
a_stfr, b_stfr,d_stfr,e_stfr,f_stfr,g_stfr,
max_st_dot,fixed_delay_stdn,k_stdn,
w_natural_Hz_pitch,k_f_pitch,k_r_pitch,
w_natural_Hz_roll,k_f_roll,k_r_roll] = model_parameters()



# the model gives you the derivatives of it's own states, so you can integrate them to get the states in the new time instant
dynamic_model = full_dynamic_model(lr, l_COM, Jz, lf, m,
            a_m, b_m, c_m, d_m,
            a_f, b_f, c_f, d_f,
            a_s, b_s, c_s, d_s, e_s,
            d_t, c_t, b_t,
            a_stfr, b_stfr,d_stfr,e_stfr,f_stfr,g_stfr,
            max_st_dot,fixed_delay_stdn,k_stdn,
            w_natural_Hz_pitch,k_f_pitch,k_r_pitch,
            w_natural_Hz_roll,k_f_roll,k_r_roll)









# # Starting data processing

# Starting data processing
# check if there is a processed vicon data file already
file_name = 'processed_vicon_data.csv'
# Check if the CSV file exists in the folder
file_path = os.path.join(folder_path, file_name)

# get the raw data
df_raw_data = get_data(folder_path)

# process vicon data only if not already done it
if not os.path.isfile(file_path):
    # If the file does not exist, process the raw data

    # process data
    steps_shift = 3 # decide to filter more or less the vicon data
    df = process_raw_vicon_data(df_raw_data,steps_shift)

    df.to_csv(file_path, index=False)
    print(f"File '{file_path}' saved.")
else:
    print(f"File '{file_path}' already exists, loading data.")
    df = pd.read_csv(file_path)





# Starting data processing
# # get the raw data
# df_raw_data = get_data(folder_path)

# # process raw data
# steps_shift = 3 # decide to filter more or less the vicon data
# df = process_raw_vicon_data(df_raw_data,steps_shift)

# plot raw data
#ax0,ax1,ax2 = plot_raw_data(df)

# plot vicon related data (longitudinal and lateral velocities, yaw rate related)
ax_wheels,ax_total_force_front,ax_total_force_rear,ax_lat_force,ax_long_force = plot_vicon_data(df)





# replace throttle with time integrated throttle
filtered_throttle = throttle_dynamics(df_raw_data,d_m)
df['throttle integrated'] = filtered_throttle

# add steering time integrated 
st_vec_angle_optuna, st_vec_optuna = steering_dynamics(df_raw_data,a_s,b_s,c_s,d_s,e_s,max_st_dot,fixed_delay_stdn,k_stdn)
df['steering integrated'] = st_vec_optuna

# integrate pitch and roll dynamics
pitch_vec = np.zeros(df_raw_data.shape[0])
roll_vec = np.zeros(df_raw_data.shape[0])
pitch_dot_vec = np.zeros(df_raw_data.shape[0])
roll_dot_vec = np.zeros(df_raw_data.shape[0])


for i in range(1,df.shape[0]):
    dt = df['vicon time'].iloc[i] - df['vicon time'].iloc[i-1]
    # evaluate pitch dynamics
    pitch_dot_dot = dynamic_model.critically_damped_2nd_order_dynamics(pitch_dot_vec[i-1],pitch_vec[i-1],df['ax body'].iloc[i-1],w_natural_Hz_pitch)
    
    # evaluate roll dynamics
    roll_dot_dot = dynamic_model.critically_damped_2nd_order_dynamics(roll_dot_vec[i-1],roll_vec[i-1],df['ay body'].iloc[i-1],w_natural_Hz_roll)
    
    pitch_dot_vec[i] = pitch_dot_vec[i-1] + pitch_dot_dot*dt
    roll_dot_vec[i] = roll_dot_vec[i-1] + roll_dot_dot*dt

    pitch_vec[i] = pitch_vec[i-1] + pitch_dot_vec[i-1]*dt
    roll_vec[i] = roll_vec[i-1] + roll_dot_vec[i-1]*dt

# add columns to the data
df['pitch dot'] = pitch_dot_vec
df['pitch'] = pitch_vec

df['roll dot'] = roll_dot_vec
df['roll'] = roll_vec

# time shift the steering to get the fixed time delay
# NOTE: time shifting by 1 less step due to forward Euler 
# (in an MPC this will not happen because you can update the input dynamics directly, without first having to compute the derivative and wait 1 dt for it to take effect)

st_delay = int(np.round(fixed_delay_stdn))
# Evaluate the shifted steering signal using the best fixed delay
steering_time_shifted = df_raw_data['steering'].shift(st_delay, fill_value=0).to_numpy()
df['steering time shifted'] = steering_time_shifted

# time shift the throttle to avoid the 1 time step delay due to forward Euler 
# (in an MPC this will not happen because you can update the input dynamics directly, without first having to compute the derivative and wait 1 dt for it to take effect)
#df['throttle time shifted'] = df['throttle'].shift(-1, fill_value=0).to_numpy()






# check that integrated subsystem states are correct
fig1, ((ax_throttle,ax_steering)) = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

# plot throttle vs throttle integrated
ax_throttle.plot(df['vicon time'].to_numpy(),df['throttle'].to_numpy(),color='navy',label='throttle',linewidth=1,linestyle='-')
ax_throttle.plot(df['vicon time'].to_numpy(),df['throttle integrated'].to_numpy(),color='dodgerblue',label='throttle integrated',linewidth=4,linestyle='-')
ax_throttle.set_xlabel('Time [s]')
ax_throttle.set_ylabel('Throttle')
ax_throttle.legend()
ax_throttle.set_title('Throttle vs Throttle integrated')

#plot steering vs steering integrated
ax_steering.plot(df['vicon time'].to_numpy(),df['steering'].to_numpy(),color='purple',label='steering',linewidth=1,linestyle='-')
ax_steering.plot(df['vicon time'].to_numpy(),df['steering integrated'].to_numpy(),color='orchid',label='steering integrated',linewidth=4,linestyle='-')
ax_steering.set_xlabel('Time [s]')
ax_steering.set_ylabel('Steering')
ax_steering.legend()
ax_steering.set_title('Steering vs Steering integrated')


fig1, ((ax_pitch,ax_roll)) = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

# plot pitch and pitch dot against ax body
ax_pitch.plot(df['vicon time'].to_numpy(),df['pitch'].to_numpy(),color='dodgerblue',label='pitch from acc x',linewidth=4,linestyle='-')
#ax_pitch.plot(df['vicon time'].to_numpy(),df['pitch dot'].to_numpy(),color='gray',label='pitch dot',linewidth=1,linestyle='-')
#ax_pitch.plot(df['vicon time'].to_numpy(),df['ax body'].to_numpy(),color='navy',label='ax body',linewidth=1,linestyle='-',alpha=0.2)
ax_pitch.set_xlabel('Time [s]')
ax_pitch.set_ylabel('Pitch [rad]')
ax_pitch.legend()
ax_pitch.set_title('Pitch')


# plot roll and roll dot against ay body
ax_roll.plot(df['vicon time'].to_numpy(),df['roll'].to_numpy(),color='orangered',label='roll from acc y',linewidth=4,linestyle='-')
#ax_roll.plot(df['vicon time'].to_numpy(),df['roll dot'].to_numpy(),color='gray',label='roll dot',linewidth=1,linestyle='-')
#ax_roll.plot(df['vicon time'].to_numpy(),df['ay body'].to_numpy(),color='darkred',label='ay body',linewidth=1,linestyle='-',alpha=0.2)
ax_roll.set_xlabel('Time [s]')
ax_roll.set_ylabel('Roll [rad]')
ax_roll.legend()
ax_roll.set_title('Roll')






# producing long term predictions



columns_to_extract = ['vicon time', 'vx body', 'vy body', 'w_abs_filtered', 'throttle integrated' ,'steering integrated',
                      'pitch dot','pitch','roll dot','roll','throttle','steering time shifted','vicon x','vicon y','vicon yaw']
input_data_long_term_predictions = df[columns_to_extract].to_numpy()
prediction_window = 1.5 # [s]
jumps = 25
long_term_predictions = produce_long_term_predictions_full_model(input_data_long_term_predictions, dynamic_model,prediction_window,jumps,forward_propagate_indexes)

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



ax13.plot(time_vec_data,input_data_long_term_predictions[:,12],color='dodgerblue',label='x',linewidth=4,linestyle='-')
ax13.set_xlabel('time [s]')
ax13.set_ylabel('y [m]')
ax13.legend()
ax13.set_title('trajectory in the x-y plane')

ax14.plot(time_vec_data,input_data_long_term_predictions[:,13],color='orangered',label='y',linewidth=4,linestyle='-')
ax14.set_xlabel('time [s]')
ax14.set_ylabel('y [m]')
ax14.legend()
ax14.set_title('trajectory in the x-y plane')

ax15.plot(time_vec_data,input_data_long_term_predictions[:,14],color='orchid',label='yaw',linewidth=4,linestyle='-')
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

ax16.plot(input_data_long_term_predictions[:,12],input_data_long_term_predictions[:,13],color='orange',label='trajectory',linewidth=4,linestyle='-')
ax16.set_xlabel('x [m]')
ax16.set_ylabel('y [m]')
ax16.legend()
ax16.set_title('vehicle trajectory in the x-y plane')




for i in range(0,len(long_term_predictions)):
    pred = long_term_predictions[i]
    #velocities
    ax10.plot(pred[:,0],pred[:,1],color='k',alpha=0.2)
    ax11.plot(pred[:,0],pred[:,2],color='k',alpha=0.2)
    ax12.plot(pred[:,0],pred[:,3],color='k',alpha=0.2)
    # positions
    ax13.plot(pred[:,0],pred[:,12],color='k',alpha=0.2)
    ax14.plot(pred[:,0],pred[:,13],color='k',alpha=0.2)
    ax15.plot(pred[:,0],pred[:,14],color='k',alpha=0.2)
    #trajectory
    ax16.plot(pred[:,12],pred[:,13],color='k',alpha=0.2)
    # throttle
    ax_throttle.plot(pred[:,0],pred[:,4],color='k',alpha=0.2)
    # steering
    ax_steering.plot(pred[:,0],pred[:,5],color='k',alpha=0.2)
    # pitch
    ax_pitch.plot(pred[:,0],pred[:,7],color='k',alpha=0.2)
    # roll
    ax_roll.plot(pred[:,0],pred[:,9],color='k',alpha=0.2)

    # tire forces
    
    v_y_wheel_front = np.cos(pred[:,4])*(pred[:,2] + lf*pred[:,3]) - np.sin(pred[:,4]) * pred[:,1]
    v_y_wheel_rear = pred[:,2] - pred[:,3]*lf
    # pass through the tire model
    lateral_force_front = dynamic_model.lateral_tire_forces(v_y_wheel_front)
    lateral_force_rear = dynamic_model.lateral_tire_forces(v_y_wheel_rear)

    # pass through the correction terms
    Fy_wheel_f_corrected,Fy_wheel_r_corrected = dynamic_model.correct_F_y_roll_pitch(lateral_force_front,lateral_force_rear,pred[:,5],pred[:,6],pred[:,7],pred[:,8])

    ax_wheels.scatter(v_y_wheel_front,Fy_wheel_f_corrected,color='peru',alpha=0.2)
    ax_wheels.scatter(v_y_wheel_rear,Fy_wheel_r_corrected,color='darkred',alpha=0.2)



ax16.set_aspect('equal')




plt.show()












# visualize the wheel data using the steering dynamics
df_raw_data_steering_dynamics = get_data(folder_path)
# df_raw_data_steering_dynamics = df_raw_data_steering_dynamics[df_raw_data_steering_dynamics['vicon time']>t_min]
# df_raw_data_steering_dynamics = df_raw_data_steering_dynamics[df_raw_data_steering_dynamics['vicon time']<t_max]

df_raw_data_steering_dynamics['steering angle'] = df['steering angle integrated']
# providing the steering  will make processing data use that instead of recovering it from the raw data
df_steering_dynamics = process_raw_vicon_data(df_raw_data_steering_dynamics,steps_shift)


v_y_wheel_plotting = torch.unsqueeze(torch.linspace(-1.5,1.5,100),1)
lateral_force_vec = dynamic_model.lateral_tire_forces(v_y_wheel_plotting).detach().cpu().numpy()
ax_wheels.scatter(df_steering_dynamics['V_y front wheel'],df_steering_dynamics['Fy front wheel'],color='skyblue',label='front wheel data',s=6)
ax_wheels.scatter(df_steering_dynamics['V_y rear wheel'] ,df_steering_dynamics['Fy rear wheel'],color='teal',label='rear wheel data',s=3)
#ax_wheels.plot(v_y_wheel_plotting.detach().cpu().numpy(),lateral_force_vec,color='#2c4251',label='Tire model',linewidth=4,linestyle='-')
# Create the scatter plot
sc = ax_wheels.scatter(df_steering_dynamics['V_y front wheel'],
                       df_steering_dynamics['Fy front wheel'],
                       c=df_steering_dynamics['vx body'],  # Set color based on 'vx body'
                       cmap='viridis',  # You can choose any cmap, e.g., 'viridis', 'plasma', etc.
                       label='front wheel data',
                       s=3)

# Add a color bar to indicate what the colors represent
cbar = plt.colorbar(sc, ax=ax_wheels)
cbar.set_label('Vx Body')



ax_wheels.legend()





plt.show()


























# columns_to_extract = ['vicon time', 'vx body', 'vy body', 'w_abs_filtered', 'throttle' ,'steering','vicon x','vicon y','vicon yaw']
# input_data_long_term_predictions = df[columns_to_extract].to_numpy()
# prediction_window = 1.5 # [s]
# jumps = 50
# forward_propagate_indexes = [1,2,3] # 1 =vx, 2=vy, 3=w
# long_term_predictions = produce_long_term_predictions(input_data_long_term_predictions, dynamic_model,prediction_window,jumps,forward_propagate_indexes)

# # plot long term predictions over real data
# fig, ((ax10,ax11,ax12)) = plt.subplots(3, 1, figsize=(10, 6))
# fig.subplots_adjust(top=0.995,
#                     bottom=0.11,
#                     left=0.095,
#                     right=0.995,
#                     hspace=0.345,
#                     wspace=0.2)

# time_vec_data = df['vicon time'].to_numpy()

# ax10.plot(time_vec_data,input_data_long_term_predictions[:,1],color='dodgerblue',label='vx',linewidth=4,linestyle='-')
# ax10.set_xlabel('Time [s]')
# ax10.set_ylabel('Vx body[m/s]')
# ax10.legend()
# ax10.set_title('Vx')

# ax11.plot(time_vec_data,input_data_long_term_predictions[:,2],color='orangered',label='vy',linewidth=4,linestyle='-')
# ax11.set_xlabel('Time [s]')
# ax11.set_ylabel('Vy body[m/s]')
# ax11.legend()
# ax11.set_title('Vy')


# ax12.plot(time_vec_data,input_data_long_term_predictions[:,3],color='orchid',label='w',linewidth=4,linestyle='-')
# ax12.set_xlabel('Time [s]')
# ax12.set_ylabel('W [rad/s]')
# ax12.legend()
# ax12.set_title('W')



# for i in range(0,len(long_term_predictions)):
#     pred = long_term_predictions[i]
#     ax10.plot(pred[:,0],pred[:,1],color='k',alpha=0.1)
#     ax11.plot(pred[:,0],pred[:,2],color='k',alpha=0.2)
#     ax12.plot(pred[:,0],pred[:,3],color='k',alpha=0.2)


# plt.show()

