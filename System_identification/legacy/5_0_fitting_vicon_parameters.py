from dart_dynamic_models import get_data, plot_raw_data, process_vicon_data_kinematics,plot_vicon_data,model_parameters,\
vicon_parameters_model
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


# NOTE: all this tweaking is now inside model_parameters

# # --------------- TWEAK THESE PARAMETERS ---------------
# theta_correction = +0.5/180*np.pi 
# lr = 0.135 # reference point location taken by the vicon system
# COM_positon = 0.09375 #measuring from the rear wheel
# #steering angle curve
# a_s =  1.6379064321517944
# b_s =  0.3301370143890381 #+ 0.04
# c_s =  0.019644200801849365 #- 0.04 # this value can be tweaked to get the tyre model curves to allign better
# d_s =  0.37879398465156555 #+ 0.2 #0.04
# e_s =  1.6578725576400757
# # ------------------------------------------------------
# load model parameters

# [theta_correction, lr, l_COM, Jz, lf, m, a_m, b_m, c_m, d_m,
# a_f, b_f, c_f, d_f,
# a_s, b_s, c_s, d_s, e_s,
# d_t, c_t, b_t,
# a_stfr, b_stfr,d_stfr,e_stfr,f_stfr,g_stfr,
# max_st_dot,fixed_delay_stdn,k_stdn,
# w_natural_Hz_pitch,k_f_pitch,k_r_pitch,
# w_natural_Hz_roll,k_f_roll,k_r_roll]= model_parameters()





# select data folder NOTE: this assumes that the current directory is DART
# here you need data with straight line driving and some curves
folder_path = 'System_identification_data_processing/Data/00_calibrating_vicon_parameters'






# --- Starting data processing  ------------------------------------------------
df_raw_data = get_data(folder_path)

# process the data
steps_shift = 3 # decide to filter more or less the vicon data
l_COM = 0 # set it to zero cause you don't have it yet (this will use the real vicon reference point instead of the COM)
theta_correction = 0 # set it to zero cause you don't have it yet
df = process_vicon_data_kinematics(df_raw_data,steps_shift,l_COM,theta_correction)


# chose only low velocity data because we assume there is no slip
df = df[df['vel encoder'] < 0.7] 

#robot2vicon_delay = 5 # samples delay

# # check if there is a processed vicon data file already
# file_name = 'processed_vicon_data.csv'
# # Check if the CSV file exists in the folder
# file_path = os.path.join(folder_path, file_name)

# if not os.path.isfile(file_path):
#     # If the file does not exist, process the raw data
#     # get the raw data
#     df_raw_data = get_data(folder_path)

#     # process the data
#     steps_shift = 3 # decide to filter more or less the vicon data
#     df = process_raw_vicon_data(df_raw_data,steps_shift)

#     df.to_csv(file_path, index=False)
#     print(f"File '{file_path}' saved.")
# else:
#     print(f"File '{file_path}' already exists, loading data.")
#     df = pd.read_csv(file_path)




# plot raw data
ax0,ax1,ax2 = plot_raw_data(df)

# plot vicon related data (longitudinal and lateral velocities, yaw rate related)
#ax_wheels,ax_total_force_front,ax_total_force_rear,ax_lat_force,ax_long_force = plot_vicon_data(df) 








# --------------- fitting tire model---------------
# fitting tyre models
# define first guess for parameters
initial_guess = torch.ones(2) * 0.5 # initialize parameters in the middle of their range constraint
# define number of training iterations
train_its = 1000
learning_rate = 0.001

print('')
print('Fitting vicon parameters (making the no slip assumption)')

#instantiate the model
vicon_parameters_model_obj = vicon_parameters_model(initial_guess)

#define loss and optimizer objects
loss_fn = torch.nn.MSELoss() 
optimizer_object = torch.optim.Adam(vicon_parameters_model_obj.parameters(), lr=learning_rate)

# generate data in tensor form for torch
columns_to_extract = ['vx_abs_filtered','vy_abs_filtered','unwrapped yaw']

train_x = torch.tensor(df[columns_to_extract].to_numpy()).cuda()
train_y = torch.unsqueeze(torch.tensor(np.concatenate((df['vel encoder'].to_numpy(),df['w_abs_filtered'].to_numpy()))),1).cuda() 

# save loss values for later plot
loss_vec = np.zeros(train_its)

# train the model
for i in range(train_its):
    # clear gradient information from previous step before re-evaluating it for the current iteration
    optimizer_object.zero_grad()  
    
    # compute fitting outcome with current model parameters
    Vx_body,w_output = vicon_parameters_model_obj(train_x)
    output = torch.cat((Vx_body,w_output),0)

    # evaluate loss function
    loss = loss_fn(output,  train_y)
    loss_vec[i] = loss.item()

    # evaluate the gradient of the loss function with respect to the fitting parameters
    loss.backward() 

    # use the evaluated gradient to perform a gradient descent step 
    optimizer_object.step() # this updates parameters


# --- print out parameters ---
[theta_correction,lr] = vicon_parameters_model_obj.transform_parameters_norm_2_real()
theta_correction, lr = theta_correction.item(), lr.item()
print('vicon parameters:')
print('theta_correction = ', theta_correction)
print('lr = ', lr)
print('')
print('theta correction degrees = ', theta_correction*180/np.pi)


# # # --- plot loss function ---
plt.figure()
plt.title('Loss function')
plt.plot(loss_vec,label='tire model loss')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.legend()








# correct theta rotation
theta_raw_vec = df['unwrapped yaw'].to_numpy() + theta_correction 

vy_body_vec = np.zeros(df.shape[0])

for i in range(df.shape[0]):
    rot_angle =  - theta_raw_vec[i] # from global to body you need to rotate by -theta!

    R     = np.array([[ np.cos(rot_angle), -np.sin(rot_angle)],
                        [ np.sin(rot_angle),  np.cos(rot_angle)]])
    

    vxvy = np.expand_dims(np.array(df[['vx_abs_filtered','vy_abs_filtered']].iloc[i]),1)

    vxvy_body = R @ vxvy

    vy_body_vec[i] = vxvy_body[1]


# df['vx body'] = vx_body_vec
# df['vy body'] = vy_body_vec





# plot lateral velocity at wheels
#lr = 0.1125 #0.1175# reference point location taken by the vicon system measured from the rear wheel
# car parameters
l = 0.175 # length of the car
lf = l-lr
#df = df[df['vel encoder'] < 0.5] # remove data when the car is not moving
fig1, ((ax0)) = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
ax0.set_title('Lateral velocity at wheels')
#ax0.plot(df['vicon time'].to_numpy(), df['vy body'].to_numpy()+df['w_abs_filtered'] * lf, label="V_y front wheel", color='peru')
ax0.plot(vy_body_vec-df['w_abs_filtered'].to_numpy() * lr, label="V_y rear wheel", color='darkred')
ax0.set_xlabel('time [s]')
ax0.legend()



plt.show()





