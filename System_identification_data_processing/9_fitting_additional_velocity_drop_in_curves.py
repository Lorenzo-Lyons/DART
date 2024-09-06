from functions_for_data_processing import get_data, plot_raw_data, process_raw_vicon_data,plot_vicon_data\
,linear_tire_model,pacejka_tire_model,dyn_model_culomb_tires,produce_long_term_predictions,culomb_pacejka_tire_model
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
forward_propagate_indexes = [1] # 1 =vx, 2=vy, 3=w

# ---------------  ----------------
theta_correction = +0.5/180*np.pi 
lr = 0.135 # reference point location taken by the vicon system
COM_positon = 0.09375 #0.0925 #measuring from the rear wheel
# ------------------------------------------------------

# select data folder NOTE: this assumes that the current directory is DART
#folder_path = 'System_identification_data_processing/Data/9_model_validation_long_term_predictions'
folder_path = 'System_identification_data_processing/Data/8_circles_rubbery_floor_1_file'
#folder_path = 'System_identification_data_processing/Data/91_model_validation_long_term_predictions_fast'


# steering dynamics time constant
# Time constant in the steering dynamics
steer_time_constant = 0.065  # should be in time domain, not discrete time filtering coefficient




# car parameters
l = 0.175 # length of the car
m = 1.67 # mass
Jz_0 = 0.006513 # Moment of inertia of uniform rectangle of shape 0.18 x 0.12

# Automatically adjust following parameters according to tweaked values
l_COM = lr - COM_positon #distance of the reference point from the centre of mass)
Jz = Jz_0 + m*l_COM**2 # correcting the moment of inertia for the extra distance of the reference point from the COM
lf = l-lr


# fitted parameters
# construct a model that takes as inputs Vx,Vy,W,tau,Steer ---> Vx_dot,Vy_dot,W_dot

# motor model
# # low velocity (1.5 max)
# a_m =  28.08614730834961
# b_m =  8.511195182800293
# c_m =  -0.14750763773918152
# d_m =  0.6848964691162109  # filtering coefficient for throttle

# high velocity (4.5 max)
a_m =  25.795652389526367
b_m =  4.820503234863281
c_m =  -0.1558982878923416
d_m =  0.7068579792976379


# rolling friction model
a_f =  1.5837167501449585
b_f =  14.215554237365723
c_f =  0.5013455152511597
d_f =  -0.057962968945503235

# steering angle curve
a_s =  1.6379064321517944
b_s =  0.3301370143890381 + 0.04
c_s =  0.019644200801849365 - 0.03 # this value can be tweaked to get the tyre model curves to allign better
d_s =  0.37879398465156555 + 0.04
e_s =  1.6578725576400757

# tire model
d_t =  -6.080334186553955
c_t =  1.0502581596374512
b_t =  4.208724021911621


# filtering coefficients
steer_time_constant = 0.065 
throttle_time_constant = 0.046 # evaluated by converting alpha from 10 Hz to 100 Hz



# Starting data processing

# get the raw data
df_raw_data = get_data(folder_path)



# account for latency between vehicle and vicon system (the vehicle inputs are relayed with a certain delay)
# NOTE that the delay is actually not constant, but it is assumed to be constant for simplicity
# so there will be some little timing discrepancies between predicted stated and data

robot_vicon_time_delay_st = 5 #6 # seven periods (at 100 Hz is 0.07s)
robot_vicon_time_delay_th = 10 # seven periods (at 100 Hz is 0.07s)
df_raw_data['steering'] = df_raw_data['steering'].shift(periods=-robot_vicon_time_delay_st)
df_raw_data['throttle'] = df_raw_data['throttle'].shift(periods=-robot_vicon_time_delay_th)


# process the data
df = process_raw_vicon_data(df_raw_data,lf,lr,theta_correction,m,Jz,l_COM,a_s,b_s,c_s,d_s,e_s,steer_time_constant)

# cut the data
df = df[df['vicon time'] < 85]

# add filtered throttle
T = df['vicon time'].diff().mean()  # Calculate the average time step
# Filter coefficient
alpha_throttle = T / (T + throttle_time_constant)
# Initialize the filtered steering angle list
filtered_throttle = [df['throttle'].iloc[0]]
# Apply the first-order filter
for i in range(1, len(df)):
    filtered_value = alpha_throttle * df['throttle'].iloc[i] + (1 - alpha_throttle) * filtered_throttle[-1]
    filtered_throttle.append(filtered_value)

df['throttle filtered'] = filtered_throttle












# plot raw data
#ax0,ax1,ax2 = plot_raw_data(df)

# plot vicon related data (longitudinal and lateral velocities, yaw rate related)
#plot_vicon_data(df)

# plot filtered throttle signal
plt.figure()
plt.plot(df['vicon time'].to_numpy(),df['throttle'].to_numpy(),label='throttle')
plt.plot(df['vicon time'].to_numpy(),df['throttle filtered'].to_numpy(),label='throttle filtered')
plt.plot(df['vicon time'].to_numpy(),df['ax body no centrifugal'].to_numpy(),label='acc_x',color = 'dodgerblue')
plt.xlabel('Time [s]')
plt.ylabel('Throttle')
plt.legend()
 


# --- produce data for fitting ---

# define model NOTE: this will give you the absolute accelerations measured in the body frame
dynamic_model = dyn_model_culomb_tires(m,lr,lf,l_COM,Jz,d_t,c_t,b_t,
                 a_m,b_m,c_m,
                 a_f,b_f,c_f,d_f)



columns_to_extract = ['vx body', 'vy body', 'w_abs_filtered', 'throttle filtered' ,'steering angle time delayed']
input_data = df[columns_to_extract].to_numpy()

acc_x_model = np.zeros(input_data.shape[0])
acc_y_model = np.zeros(input_data.shape[0])
acc_w_model = np.zeros(input_data.shape[0])
acc_centrifugal_in_x = np.zeros(input_data.shape[0])


for i in range(input_data.shape[0]):
    # correct for centrifugal acceleration
    accelerations = dynamic_model.forward(input_data[i,:])
    acc_x_model[i] = accelerations[0]
    acc_y_model[i] = accelerations[1]
    acc_w_model[i] = accelerations[2]
    acc_centrifugal_in_x[i] = - input_data[i,1] * input_data[i,2]


# plot the modelled acceleration
plt.figure()
plt.plot(df['vicon time'].to_numpy(),df['ax body no centrifugal'].to_numpy(),label='acc_x',color='dodgerblue')
plt.plot(df['vicon time'].to_numpy(),acc_x_model,label='acc_x model',color='k',alpha=0.5)
plt.plot(df['vicon time'].to_numpy(),df['throttle filtered'].to_numpy(),label='throttle filtered',color='blue')
plt.plot(df['vicon time'].to_numpy(),df['steering angle time delayed'].to_numpy(),label='steering angle time delayed',color='orangered')
plt.plot(df['vicon time'].to_numpy(),df['vx body'].to_numpy(),label='vx body',color='green')
plt.plot(df['vicon time'].to_numpy(),acc_centrifugal_in_x,label='acc centrifugal in x',color='purple')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration x')
plt.legend()


# y accelerations
plt.figure()
plt.plot(df['vicon time'].to_numpy(),df['ay body no centrifugal'].to_numpy(),label='acc_y',color='orangered')
plt.plot(df['vicon time'].to_numpy(),acc_y_model,label='acc_y model',color='k',alpha=0.5)
plt.xlabel('Time [s]')
plt.ylabel('Acceleration y')
plt.legend()

# w accelerations
plt.figure()
plt.plot(df['vicon time'].to_numpy(),df['aw_abs_filtered_more'].to_numpy(),label='acc_w',color='purple')
plt.plot(df['vicon time'].to_numpy(),acc_w_model,label='acc_w model',color='k',alpha=0.5)
plt.xlabel('Time [s]')
plt.ylabel('Acceleration w')
plt.legend()



# evaluate friction curve
velocity_range = np.linspace(0,df['vx body'].max(),100)
friction_curve = dynamic_model.friction(velocity_range)

rescale = - 0.8 * np.array(acc_x_model - df['ax body no centrifugal'].to_numpy()).max() / - friction_curve.min()


plt.figure()
scatter = plt.scatter(df['vx body'].to_numpy(),acc_x_model - df['ax body no centrifugal'].to_numpy(),label='acc_x_model - measured', c=df['steering angle time delayed'].to_numpy(),
                      cmap='viridis')
plt.plot(velocity_range,friction_curve*rescale,label='friction curve rescaled',color='k',alpha=0.5)
#plt.plot(df['vicon time'].to_numpy(),df['steering angle time delayed'].to_numpy(),label='steering',color='orangered')
plt.xlabel('vx body')
plt.ylabel('Acceleration x')
# Add a colorbar to the plot
plt.colorbar(scatter, label='steering angle time delayed')
plt.legend()



plt.show()
