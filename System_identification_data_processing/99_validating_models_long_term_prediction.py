from functions_for_data_processing import get_data, plot_raw_data, process_raw_vicon_data,plot_vicon_data\
,dyn_model_culomb_tires,produce_long_term_predictions,culomb_pacejka_tire_model,model_parameters
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
forward_propagate_indexes = [1,2,3] # 1 =vx, 2=vy, 3=w

# select data folder NOTE: this assumes that the current directory is DART
#folder_path = 'System_identification_data_processing/Data/90_model_validation_long_term_predictions'  # the battery was very low for this one
folder_path = 'System_identification_data_processing/Data/91_model_validation_long_term_predictions_fast'





# load model parameters
[theta_correction, lr, l_COM, Jz, lf, m,
a_m, b_m, c_m, d_m,
a_f, b_f, c_f, d_f,
a_s, b_s, c_s, d_s, e_s,
d_t, c_t, b_t,
a_stfr, b_stfr] = model_parameters()

# the model gives you the derivatives of it's own states, so you can integrate them to get the states in the new time instant
dynamic_model = dyn_model_culomb_tires(m,lr,lf,l_COM,Jz,
                 d_t,c_t,b_t,
                 a_m,b_m,c_m,
                 a_f,b_f,c_f,d_f,
                 a_stfr,b_stfr)





# Starting data processing

# get the raw data
df_raw_data = get_data(folder_path)
df_raw_data = df_raw_data[df_raw_data['vicon time']>2.8]
df_raw_data = df_raw_data[df_raw_data['vicon time']<3.4]


# process raw data
df = process_raw_vicon_data(df_raw_data)





# add filtered throttle
T = df['vicon time'].diff().mean()  # Calculate the average time step
# Filter coefficient in the new sampling frequency
d_m_100Hz = 0.01/(0.01+(0.1/d_m-0.1)) #convert to new sampling frequency

# Initialize the filtered steering angle list
filtered_throttle = [df['throttle'].iloc[0]]
# Apply the first-order filter
for i in range(1, len(df)):
    filtered_value = d_m_100Hz * df['throttle'].iloc[i] + (1 - d_m_100Hz) * filtered_throttle[-1]
    filtered_throttle.append(filtered_value)

df['throttle filtered'] = filtered_throttle









# plot raw data
ax0,ax1,ax2 = plot_raw_data(df)

# plot vicon related data (longitudinal and lateral velocities, yaw rate related)
ax_wheels = plot_vicon_data(df)


# add wheel curve on top of the vicon data
vy_range = np.linspace(-1,1,100)
Fy_wheels = dynamic_model.lateral_tire_forces(vy_range)
ax_wheels.plot(vy_range,Fy_wheels,color='k',label='wheel model curve')


#plt.show()

# plot filtered throttle signal
plt.figure()
plt.plot(df['vicon time'].to_numpy(),df['throttle'].to_numpy(),label='throttle')
plt.plot(df['vicon time'].to_numpy(),df['throttle filtered'].to_numpy(),label='throttle filtered')
plt.plot(df['vicon time'].to_numpy(),df['ax body no centrifugal'].to_numpy(),label='acc_x')
plt.xlabel('Time [s]')
plt.ylabel('Throttle')
plt.legend()







# producing long term predictions



columns_to_extract = ['vicon time', 'vx body', 'vy body', 'w_abs_filtered', 'throttle filtered' ,'steering angle','vicon x','vicon y','vicon yaw']
input_data_long_term_predictions = df[columns_to_extract].to_numpy()
prediction_window = 1.5 # [s]
jumps = 25
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

ax16.set_aspect('equal')
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

