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



# NOTE:
# this script requires some tuning, mainly concerning the measuring system

# step 1: set the body reference frame properly:
# 1.1 set theta_correction
# i.e. make sure that the axis of the vicon is alligned perfectly with the car body frame.
# to do this look at a portion of the data where the car is going straight (look at x-y plane), then adjust theta_correction
# until Vy is exacly zero. (notice that Vy goes up for theta_correction going down)
# If you are sure there is no slip, i.e. low velocities and high friction with the ground, you can also look at the plot of
# Vy vs theta and adjust theta_correction until the curve is centered around zero.
# Note that even a single degree more or less can affect the model fitting significantly.

# 1.2 adjust lr
# i.e. make sure that the centre of the car body frame is in the centre of mass.
# Do this by lookiing at a W_dot=0 segment,i.e. a constant curvature (look at the end of a long curve if the dataset doesn't containt constant circles)
# and make sure that the rear tyre slip angle is 0.  (you need to be sure there is no slip though, so low velocity)
# notice that for lr down, a_r down too.

# 1.3 adjust COM_position
# We don't really know the exact location of the COM, but if we assume that the wheel should have the same curve,
# we can tweek this parameter until the two curves are the same (assuming no other mistakes in the front steer angle ecc)


# 1.4 
# another trick you can do is to fine tune the c parmeter in the steering curve inside "process_raw_vicon_data" cause it will
# afffect the belived steering angle, and hence the slip angles. Gives you an extra degree of freedom to tweack things
# shifts the entire front wheel data left and right


# --------------- TWEAK THESE PARAMETERS ---------------

theta_correction = +0.5/180*np.pi 
lr = 0.135 # reference point location taken by the vicon system
COM_positon = 0.09375 #measuring from the rear wheel
#steering angle curve
a =  1.6379064321517944
b =  0.3301370143890381 #+ 0.04
c =  0.019644200801849365 #- 0.04 # this value can be tweaked to get the tyre model curves to allign better
d =  0.37879398465156555 #+ 0.2 #0.04
e =  1.6578725576400757
# ------------------------------------------------------

# select data folder NOTE: this assumes that the current directory is DART
#folder_path = 'System_identification_data_processing/Data/8_circles_rubbery_floor_1_file'
#folder_path = 'System_identification_data_processing/Data/81_throttle_ramps'
#folder_path = 'System_identification_data_processing/Data/81_circles_tape_and_tiles'
folder_path = 'System_identification_data_processing/Data/81_throttle_ramps_only_steer03'






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



# Starting data processing

# get the raw data
df_raw_data = get_data(folder_path)


# process the data
df = process_raw_vicon_data(df_raw_data,lf,lr,theta_correction,m,Jz,l_COM,a,b,c,d,e,steer_time_constant)

# select time interval to fit
if folder_path == 'System_identification_data_processing/Data/8_circles_rubbery_floor_1_file':
    df = df[df['elapsed time sensors'] > 90]
    #df = df[df['elapsed time sensors'] < 235]

# df = df[df['vicon time'] > 69.87058401107788 - 10]
# df = df[df['vicon time'] < 69.87058401107788 + 10]

# plot raw data
ax0,ax1,ax2 = plot_raw_data(df)

# plot vicon related data (longitudinal and lateral velocities, yaw rate related)
plot_vicon_data(df)





# plotting the data that will be given to the models
fig, ((ax10)) = plt.subplots(1, 1, figsize=(15, 15))

# Front wheel
scatter_front = ax10.scatter(df['V_y front wheel'].to_numpy(), df['Fy front wheel'].to_numpy(), color = "#72a0c1", label='front wheel data',alpha=0.5,s=3)
# Rear wheel
scatter_rear = ax10.scatter(df['V_y rear wheel'].to_numpy(), df['Fy rear wheel'].to_numpy(),  color = "#cc92c2", label='rear wheel data',alpha=0.5,s=3)

ax10.set_xlabel('Wheel lateral velocity [m/s]')
ax10.set_ylabel('Fy wheel [N]')
ax10.legend()
ax10.set_title('Tire model')





# --------------- fitting tire model---------------
# fitting tyre models
# define first guess for parameters
initial_guess = torch.ones(3) * 0.5 # initialize parameters in the middle of their range constraint
# define number of training iterations
train_its = 1000
learning_rate = 0.003

print('')
print('Fitting pacejka-like culomb friction tire model ')

#instantiate the model
pacejka_culomb_tire_model_obj = culomb_pacejka_tire_model(initial_guess)

#define loss and optimizer objects
loss_fn = torch.nn.MSELoss() 
optimizer_object = torch.optim.Adam(pacejka_culomb_tire_model_obj.parameters(), lr=learning_rate)

# generate data in tensor form for torch
train_x = torch.unsqueeze(torch.tensor(np.concatenate((df['V_y front wheel'].to_numpy(),df['V_y rear wheel'].to_numpy()))),1).cuda()
train_y = torch.unsqueeze(torch.tensor(np.concatenate((df['Fy front wheel'].to_numpy(),df['Fy rear wheel'].to_numpy()))),1).cuda() 

# save loss values for later plot
loss_vec = np.zeros(train_its)

# train the model
for i in range(train_its):
    # clear gradient information from previous step before re-evaluating it for the current iteration
    optimizer_object.zero_grad()  
    
    # compute fitting outcome with current model parameters
    output = pacejka_culomb_tire_model_obj(train_x)

    # evaluate loss function
    loss = loss_fn(output,  train_y)
    loss_vec[i] = loss.item()

    # evaluate the gradient of the loss function with respect to the fitting parameters
    loss.backward() 

    # use the evaluated gradient to perform a gradient descent step 
    optimizer_object.step() # this updates parameters


# --- print out parameters ---
[d,c,b] = pacejka_culomb_tire_model_obj.transform_parameters_norm_2_real()
d, c, b= d.item(), c.item(), b.item()
print('Front Wheel parameters:')
print('d = ', d)
print('c = ', c)
print('b = ', b)

# # # --- plot loss function ---
plt.figure()
plt.title('Loss function')
plt.plot(loss_vec,label='tire model loss')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.legend()




# evaluate model on plotting interval
v_y_wheel_plotting = torch.unsqueeze(torch.linspace(torch.min(train_x),torch.max(train_x),100),1).cuda()
lateral_force_vec = pacejka_culomb_tire_model_obj(v_y_wheel_plotting).detach().cpu().numpy()
ax10.plot(v_y_wheel_plotting.detach().cpu().numpy(),lateral_force_vec,color='#2c4251',label='Tire model',linewidth=4,linestyle='-')
ax10.scatter(np.array([0.0]),np.array([0.0]),color='orangered',label='zero',marker='+', zorder=20) # plot zero as an x 
ax10.legend()
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

