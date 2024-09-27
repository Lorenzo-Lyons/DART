from functions_for_data_processing import get_data, plot_raw_data, process_raw_vicon_data,\
plot_vicon_data,culomb_pacejka_tire_model,model_parameters,directly_measured_model_parameters,process_vicon_data_kinematics
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

[theta_correction, l_COM, l_lateral_shift_reference ,lr, lf, Jz, m,m_front_wheel,m_rear_wheel] = directly_measured_model_parameters()

[a_m, b_m, c_m, d_m,
a_f, b_f, c_f, d_f,
a_s, b_s, c_s, d_s, e_s,
d_t_f, c_t_f, b_t_f,d_t_r, c_t_r, b_t_r,
a_stfr, b_stfr,d_stfr,e_stfr,f_stfr,g_stfr,
max_st_dot,fixed_delay_stdn,k_stdn,
w_natural_Hz_pitch,k_f_pitch,k_r_pitch,
w_natural_Hz_roll,k_f_roll,k_r_roll]= model_parameters()





# select data folder NOTE: this assumes that the current directory is DART
#folder_path = 'System_identification_data_processing/Data/8_circles_rubbery_floor_1_file'

#folder_path = 'System_identification_data_processing/Data/81_circles_tape_and_tiles'

#folder_path = 'System_identification_data_processing/Data/81_throttle_ramps'
#folder_path = 'System_identification_data_processing/Data/81_throttle_ramps_only_steer03'

#folder_path = 'System_identification_data_processing/Data/91_free_driving_16_sept_2024'


#folder_path = 'System_identification_data_processing/Data/steering_identification_25_sept_2024'

#folder_path = 'System_identification_data_processing/Data/circles_27_sept_2024'

folder_path = 'System_identification_data_processing/Data/circles_27_sept_2024_fast_ramp'



# --- Starting data processing  ------------------------------------------------


# #robot2vicon_delay = 5 # samples delay
# df_raw_data = get_data(folder_path)

# # process the data
# steps_shift = 10 # decide to filter more or less the vicon data
# df = process_raw_vicon_data(df_raw_data,steps_shift)



# check if there is a processed vicon data file already
file_name = 'processed_vicon_data.csv'
# Check if the CSV file exists in the folder
file_path = os.path.join(folder_path, file_name)

if not os.path.isfile(file_path):
    steps_shift = 10 # decide to filter more or less the vicon data
    df_raw_data = get_data(folder_path)
    df_kinematics = process_vicon_data_kinematics(df_raw_data,steps_shift,theta_correction, l_COM, l_lateral_shift_reference)
    df = process_raw_vicon_data(df_kinematics,steps_shift)

    df.to_csv(file_path, index=False)
    print(f"File '{file_path}' saved.")
else:
    print(f"File '{file_path}' already exists, loading data.")
    df = pd.read_csv(file_path)











if folder_path == 'System_identification_data_processing/Data/81_throttle_ramps_only_steer03':
    # cut the data in two parts cause something is wrong in the middle (probably a temporary lag in the network)
    df1=df[df['vicon time']<110]  #  60 150
    df2=df[df['vicon time']>185.5] 
    # Concatenate vertically
    df = pd.concat([df1, df2], axis=0)
    # Reset the index if you want a clean, continuous index
    df.reset_index(drop=True, inplace=True)

elif folder_path == 'System_identification_data_processing/Data/steering_identification_25_sept_2024':
    # cut the data in two parts cause something is wrong in the middle (probably a temporary lag in the network)
    df=df[df['vicon time']<460]
#cut off time instances where the vicon missed a detection to avoid corrupted datapoints
elif  folder_path == 'System_identification_data_processing/Data/81_throttle_ramps':
    df1 = df[df['vicon time']<100]
    df1 = df1[df1['vicon time']>20]

    df2 = df[df['vicon time']>185]
    df2 = df2[df2['vicon time']<268]

    df3 = df[df['vicon time']>283]
    df3 = df3[df3['vicon time']<350]

    df4 = df[df['vicon time']>365]
    df4 = df4[df4['vicon time']<410]

    df5 = df[df['vicon time']>445]
    df5 = df5[df5['vicon time']<500]

    df6 = df[df['vicon time']>540]
    df6 = df6[df6['vicon time']<600]

    df7 = df[df['vicon time']>645]
    df7 = df7[df7['vicon time']<658]

    df8 = df[df['vicon time']>661]
    df8 = df8[df8['vicon time']<725]

    df9 = df[df['vicon time']>745]
    df9 = df9[df9['vicon time']<820]

    df = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9]) 



if folder_path == 'System_identification_data_processing/Data/circles_27_sept_2024':
    df1 = df[df['vicon time']>1]
    df1 = df1[df1['vicon time']<375]

    df2 = df[df['vicon time']>377]
    df2 = df2[df2['vicon time']<830]

    df3 = df[df['vicon time']>860]
    df3 = df3[df3['vicon time']<1000]

    df = pd.concat([df1,df2,df3])




# plot raw data
ax0,ax1,ax2 = plot_raw_data(df)


ax_wheel_f_alpha,ax_wheel_r_alpha,ax_total_force_front,\
ax_total_force_rear,ax_lat_force,ax_long_force,\
ax_acc_x_body,ax_acc_y_body,ax_acc_w = plot_vicon_data(df) 

# # plot lateral forces vs slip angles
# plt.figure()
# plt.plot(df['slip angle front'],df['Fy front wheel'],'o',label='front wheel')
# plt.plot(df['slip angle rear'],df['Fy rear wheel'],'o',label='rear wheel')
# plt.xlabel('slip angle [rad]')
# plt.ylabel('lateral force [N]')
# plt.legend()
# plt.show()







# --------------- fitting tire model---------------
# fitting tyre models
# define first guess for parameters
initial_guess = torch.ones(3) * 0.5 # initialize parameters in the middle of their range constraint
# define number of training iterations
train_its = 1000
learning_rate = 0.001 # 00#06

print('')
print('Fitting pacejka-like culomb friction tire model ')

#instantiate the model
pacejka_culomb_tire_model_obj = culomb_pacejka_tire_model(initial_guess,m_front_wheel,m_rear_wheel)

#define loss and optimizer objects
loss_fn = torch.nn.MSELoss() 
optimizer_object = torch.optim.Adam(pacejka_culomb_tire_model_obj.parameters(), lr=learning_rate)

# generate data in tensor form for torch
# data_columns = ['V_y front wheel','V_y rear wheel'] # velocities
data_columns = ['slip angle front','slip angle rear','ax body'] # velocities


# using slip angles instead of velocities
#data_columns = ['slip angle front','slip angle rear'] # slip angles

train_x = torch.tensor(df[data_columns].to_numpy()).cuda()
#train_x = torch.unsqueeze(torch.tensor(np.concatenate((df['V_y front wheel'].to_numpy(),df['V_y rear wheel'].to_numpy()))),1).cuda()
train_y = torch.unsqueeze(torch.tensor(np.concatenate((df['Fy front wheel'].to_numpy(),df['Fy rear wheel'].to_numpy()))),1).cuda() 

# save loss values for later plot
loss_vec = np.zeros(train_its)

# train the model
for i in range(train_its):
    # clear gradient information from previous step before re-evaluating it for the current iteration
    optimizer_object.zero_grad()  
    
    # compute fitting outcome with current model parameters
    F_y_f,F_y_r = pacejka_culomb_tire_model_obj(train_x)
    output = torch.cat((F_y_f,F_y_r),0) 

    # evaluate loss function
    loss = loss_fn(output,  train_y)
    loss_vec[i] = loss.item()

    # evaluate the gradient of the loss function with respect to the fitting parameters
    loss.backward() 

    # use the evaluated gradient to perform a gradient descent step 
    optimizer_object.step() # this updates parameters


# --- print out parameters ---
[d_t_f,c_t_f,b_t_f,d_t_r,c_t_r,b_t_r,k_pitch] = pacejka_culomb_tire_model_obj.transform_parameters_norm_2_real()

print('# Front wheel parameters:')
print('d_t_f = ', d_t_f.item())
print('c_t_f = ', c_t_f.item())
print('b_t_f = ', b_t_f.item())
#print('e_t_f = ', e_t_f.item())


print('# Rear wheel parameters:')
print('d_t_r = ', d_t_r.item())
print('c_t_r = ', c_t_r.item())
print('b_t_r = ', b_t_r.item())
#print('e_t_r = ', e_t_r.item())

# pitch influence
print('k_pitch = ', k_pitch.item())

# # # --- plot loss function ---
plt.figure()
plt.title('Loss function')
plt.plot(loss_vec,label='tire model loss')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.legend()



# evaluate model on plotting interval
#v_y_wheel_plotting_front = torch.unsqueeze(torch.linspace(torch.min(train_x[:,0]),torch.max(train_x[:,0]),100),1).cuda()
alpha_f_plotting_front = torch.unsqueeze(torch.linspace(torch.min(train_x[:,0]),torch.max(train_x[:,0]),100),1).cuda()
lateral_force_vec_front = pacejka_culomb_tire_model_obj.lateral_tire_force(alpha_f_plotting_front,d_t_f,c_t_f,b_t_f,m_front_wheel).detach().cpu().numpy()


# do the same for he rear wheel
alpha_r_plotting_rear = torch.unsqueeze(torch.linspace(torch.min(train_x[:,1]),torch.max(train_x[:,1]),100),1).cuda()
lateral_force_vec_rear = pacejka_culomb_tire_model_obj.lateral_tire_force(alpha_r_plotting_rear,d_t_r,c_t_r,b_t_r,m_rear_wheel).detach().cpu().numpy()


ax_wheel_f_alpha.plot(alpha_f_plotting_front.cpu(),lateral_force_vec_front,color='#2c4251',label='Tire model',linewidth=4,linestyle='-')
ax_wheel_f_alpha.legend()

ax_wheel_r_alpha.plot(alpha_r_plotting_rear.cpu(),lateral_force_vec_rear,color='#2c4251',label='Tire model',linewidth=4,linestyle='-')
ax_wheel_r_alpha.legend()

# plot model outputs
ax_wheel_f_alpha.scatter(df['slip angle front'],F_y_f.detach().cpu().numpy(),color='k',label='Tire model output (with pitch influece)',s=2)
ax_wheel_r_alpha.scatter(df['slip angle rear'],F_y_r.detach().cpu().numpy(),color='k',label='Tire model output (with pitch influece)',s=2)


plt.show()







