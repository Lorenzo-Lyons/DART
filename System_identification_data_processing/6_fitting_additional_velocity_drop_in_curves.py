from functions_for_data_processing import get_data, plot_raw_data, process_raw_vicon_data,plot_vicon_data\
,dyn_model_culomb_tires,steering_friction_model, model_parameters,directly_measured_model_parameters,process_vicon_data_kinematics
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

# select data folder NOTE: this assumes that the current directory is DART
#folder_path = 'System_identification_data_processing/Data/9_model_validation_long_term_predictions'
#folder_path = 'System_identification_data_processing/Data/8_circles_rubbery_floor_1_file'
#folder_path = 'System_identification_data_processing/Data/91_model_validation_long_term_predictions_fast'
#folder_path = 'System_identification_data_processing/Data/81_throttle_ramps'
#folder_path = 'System_identification_data_processing/Data/81_throttle_ramps_only_steer03'
#folder_path = 'System_identification_data_processing/Data/91_free_driving_16_sept_2024'
#folder_path = 'System_identification_data_processing/Data/steering_identification_25_sept_2024'
folder_path = 'System_identification_data_processing/Data/circles_27_sept_2024'





[theta_correction, l_COM, l_lateral_shift_reference ,lr, lf, Jz, m,m_front_wheel,m_rear_wheel] = directly_measured_model_parameters()

[a_m, b_m, c_m, d_m,
a_f, b_f, c_f, d_f,
a_s, b_s, c_s, d_s, e_s,
d_t_f, c_t_f, b_t_f,d_t_r, c_t_r, b_t_r,
a_stfr, b_stfr,d_stfr,e_stfr,f_stfr,g_stfr,
max_st_dot,fixed_delay_stdn,k_stdn,
w_natural_Hz_pitch,k_f_pitch,k_r_pitch,
w_natural_Hz_roll,k_f_roll,k_r_roll]= model_parameters()





# --- Starting data processing  ------------------------------------------------
# check if there is a processed vicon data file already
#file_name = 'processed_vicon_data.csv' 
file_name = 'processed_vicon_data.csv'
# Check if the CSV file exists in the folder
file_path = os.path.join(folder_path, file_name)


# steps_shift = 10 # decide to filter more or less the vicon data
#     # If the file does not exist, process the raw data
#     # get the raw data
# df_raw_data = get_data(folder_path)

steps_shift = 10


if not os.path.isfile(file_path):
    # If the file does not exist, process the raw data
    # get the raw data
    df_raw_data = get_data(folder_path)

    # process the data
    df_kinematics = process_vicon_data_kinematics(df_raw_data,steps_shift,theta_correction, l_COM, l_lateral_shift_reference)
    df = process_raw_vicon_data(df_kinematics,steps_shift)

    df.to_csv(file_path, index=False)
    print(f"File '{file_path}' saved.")
else:
    print(f"File '{file_path}' already exists, loading data.")
    df = pd.read_csv(file_path)


# # process the data
# df_raw_data = get_data(folder_path)
# df_kinematics = process_vicon_data_kinematics(df_raw_data,steps_shift,theta_correction, l_COM, l_lateral_shift_reference)





# cut off time instances where the vicon missed a detection to avoid corrupted datapoints
# df1 = df[df['vicon time']<100]

# df2 = df[df['vicon time']>165]
# df2 = df2[df2['vicon time']<409]

# df3 = df[df['vicon time']>417]
# df3 = df3[df3['vicon time']<500]

# df4 = df[df['vicon time']>506]
# df4 = df4[df4['vicon time']<600]

# df5 = df[df['vicon time']>604]
# df5 = df5[df5['vicon time']<658]

# df6 = df[df['vicon time']>661]

# df = pd.concat([df1,df2,df3,df4,df5,df6])
    
#cut off time instances where the vicon missed a detection to avoid corrupted datapoints
if  folder_path == 'System_identification_data_processing/Data/81_throttle_ramps':
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

elif folder_path == 'System_identification_data_processing/Data/steering_identification_25_sept_2024':
    df = df[df['vicon time']<460]

elif folder_path == 'System_identification_data_processing/Data/circles_27_sept_2024':

    df1 = df[df['vicon time']>1]
    df1 = df1[df1['vicon time']<375]

    df2 = df[df['vicon time']>377]
    df2 = df2[df2['vicon time']<830]

    df3 = df[df['vicon time']>860]
    df3 = df3[df3['vicon time']<1000]

    df = pd.concat([df1,df2,df3])


#df = df[df['vx body']>0.5]


# df = df[df['vx body']<2]
# plot vicon related data (longitudinal and lateral velocities, yaw rate related)
ax_wheel_f_alpha,ax_wheel_r_alpha,ax_total_force_front,\
ax_total_force_rear,ax_lat_force,ax_long_force,\
ax_acc_x_body,ax_acc_y_body,ax_acc_w = plot_vicon_data(df) 



# # plotting sensitivity of V_y_front to steering angle
# dvy_f_ddelta = np.cos(-df['steering angle'].to_numpy()) * df['vx body'].to_numpy()\
#              - np.sin(-df['steering angle'].to_numpy()) * (df['vy body'].to_numpy()+df['w'].to_numpy()*lf)
# plt.figure()
# plt.plot(df['vicon time'].to_numpy(),dvy_f_ddelta,label='dV_y_f/d delta')
# plt.xlabel('Time [s]')
# plt.ylabel('Sensitivity [m/s/rad]')
# plt.legend()

# NOTE
# Because the test have been done in a quasi static setting for the throttle it is not necessary to integrate it's dynamics




# # plot filtered throttle signal
# plt.figure()
# plt.plot(df['vicon time'].to_numpy(),df['throttle'].to_numpy(),label='throttle')
# plt.plot(df['vicon time'].to_numpy(),df['ax body no centrifugal'].to_numpy(),label='acc_x',color = 'dodgerblue')
# plt.xlabel('Time [s]')
# plt.ylabel('Throttle')
# plt.legend()
 


# --- the following is just to visualize what the model error is without the additional friction term due to the steering ---
# --- the actual model will be fitted differently cause the extra term will affect the motor force, while here we are plotting the overall
# --- missing longitudinal force. (Adding to Fx will also, slightly, affect lateral and yaw dynamics that we do not show here)

a_stfr=[]  # give empty correction terms to not use them
b_stfr=[]
d_stfr=[]
e_stfr=[]
f_stfr=[]
g_stfr=[]


# define model NOTE: this will give you the absolute accelerations measured in the body frame
dynamic_model = dyn_model_culomb_tires(m,m_front_wheel,m_rear_wheel,lr,lf,l_COM,Jz,
                 a_m,b_m,c_m,
                 a_f,b_f,c_f,d_f,
                 d_t_f, c_t_f, b_t_f,d_t_r, c_t_r, b_t_r,
                 a_stfr, b_stfr,d_stfr,e_stfr,f_stfr,g_stfr)



columns_to_extract = ['vx body', 'vy body', 'w', 'throttle' ,'steering angle','Fx wheel','Fy front wheel','Fy rear wheel','ax body']
input_data = df[columns_to_extract].to_numpy()

acc_x_model = np.zeros(input_data.shape[0])
acc_y_model = np.zeros(input_data.shape[0])
acc_w_model = np.zeros(input_data.shape[0])

# acc_centrifugal_in_x = df['vy body'].to_numpy() * df['w'].to_numpy()
# acc_centrifugal_in_y = - df['vx body'].to_numpy() * df['w'].to_numpy()

for i in range(df.shape[0]):
    # correct for centrifugal acceleration
    accelerations = dynamic_model.forward(input_data[i,:])
    acc_x_model[i] = accelerations[0]
    acc_y_model[i] = accelerations[1]
    acc_w_model[i] = accelerations[2]




# # plot the modelled acceleration
# fig, ax_accx = plt.subplots()
# ax_accx.plot(df['vicon time'].to_numpy(),df['ax body'].to_numpy(),label='acc_x body frame',color='dodgerblue')
ax_acc_x_body.plot(df['vicon time'].to_numpy(),acc_x_model,label='acc_x model',color='k',alpha=0.5)
# ax_accx.set_xlabel('Time [s]')
# ax_accx.set_ylabel('Acceleration x')

# # y accelerations
# fig, ax_accy = plt.subplots()
# ax_accy.plot(df['vicon time'].to_numpy(),df['ay body'].to_numpy(),label='acc_y in the body frame',color='orangered')
ax_acc_y_body.plot(df['vicon time'].to_numpy(),acc_y_model,label='acc_y model',color='k',alpha=0.5)
# ax_accy.set_xlabel('Time [s]')
# ax_accy.set_ylabel('Acceleration y')

# # w accelerations
# fig, ax_accw = plt.subplots()
# ax_accw.plot(df['vicon time'].to_numpy(),df['acc_w'].to_numpy(),label='acc_w',color='purple')
ax_acc_w.plot(df['vicon time'].to_numpy(),acc_w_model,label='acc_w model',color='k',alpha=0.5)
# ax_accw.set_xlabel('Time [s]')
# ax_accw.set_ylabel('Acceleration w')








# evaluate friction curve
velocity_range = np.linspace(0,df['vx body'].max(),100)
friction_curve = dynamic_model.rolling_friction(velocity_range,a_f,b_f,c_f,d_f)

# model error in [N]
missing_force = (acc_x_model - df['ax body'].to_numpy())*m



from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    df['vx body'].to_numpy(),                    # x-axis
    df['steering angle'].to_numpy(),  # y-axis
    missing_force,                    # z-axis 
    c=df['steering angle'].to_numpy(),  # color coded by 'steering angle time delayed'
    cmap='viridis'  # Colormap
)

ax.set_xlabel('vx body')
ax.set_ylabel('steering angel [rad]')
ax.set_zlabel('Force [N]')
colorbar = fig.colorbar(scatter, label='w')









# --------------- fitting extra steering friction model---------------


# fitting tyre models
# define first guess for parameters
initial_guess = torch.ones(6) * 0.5 # initialize parameters in the middle of their range constraint
# define number of training iterations
train_its = 1000
learning_rate = 0.01

print('')
print('Fitting extra steering friction model ')

#instantiate the model
steering_friction_model_obj = steering_friction_model(initial_guess,
                 m,m_front_wheel,m_rear_wheel,lr,lf,l_COM,Jz,
                 a_m,b_m,c_m,
                 a_f,b_f,c_f,d_f,
                 d_t_f, c_t_f, b_t_f,d_t_r, c_t_r, b_t_r)

#define loss and optimizer objects
loss_fn = torch.nn.MSELoss() 
optimizer_object = torch.optim.Adam(steering_friction_model_obj.parameters(), lr=learning_rate)

# generate data in tensor form for torch
# train_x = torch.tensor(input_data).cuda()
# train_y = torch.unsqueeze(torch.tensor(df['ax body'].to_numpy()),1).cuda()

# generate data in tensor form for torch
#train_x = torch.unsqueeze(torch.tensor(np.concatenate((df['V_y front wheel'].to_numpy(),df['V_y rear wheel'].to_numpy()))),1).cuda()
train_x = torch.tensor(input_data).cuda()

# -- Y lables --
train_y = torch.unsqueeze(torch.tensor(np.concatenate((df['ax body'].to_numpy(),df['ay body'].to_numpy(),df['acc_w'].to_numpy()))),1).cuda() #,df['aw_abs_filtered_more'].to_numpy()




# save loss values for later plot
loss_vec = np.zeros(train_its)

# train the model
for i in range(train_its):
    # clear gradient information from previous step before re-evaluating it for the current iteration
    optimizer_object.zero_grad()  
    
    # compute fitting outcome with current model parameters
    acc_x,acc_y,acc_w = steering_friction_model_obj(train_x)


    output = torch.cat((acc_x.double(),acc_y.double(),acc_w.double()),0)

    # evaluate loss function
    loss = loss_fn(output,  train_y)
    loss_vec[i] = loss.item()

    # evaluate the gradient of the loss function with respect to the fitting parameters
    loss.backward() 

    # use the evaluated gradient to perform a gradient descent step 
    optimizer_object.step() # this updates parameters


# --- print out parameters ---
[a_stfr,b_stfr,d_stfr,e_stfr,f_stfr,g_stfr,
a_s,b_s,c_s,d_s,e_s,
d_t_f_model,c_t_f_model,b_t_f_model,d_t_r_model,c_t_r_model,b_t_r_model,k_pitch] = steering_friction_model_obj.transform_parameters_norm_2_real()


a_stfr,b_stfr,d_stfr,e_stfr,f_stfr,g_stfr,a_s,b_s,c_s,d_s,e_s,\
d_t_f_model,c_t_f_model,b_t_f_model,d_t_r_model,c_t_r_model,b_t_r_model,k_pitch = a_stfr.item(),b_stfr.item(),d_stfr.item(),\
e_stfr.item(),f_stfr.item(),g_stfr.item(),a_s.item(),b_s.item(),c_s.item(),d_s.item(),e_s.item(),d_t_f_model.item(),\
c_t_f_model.item(),b_t_f_model.item(),d_t_r_model.item(),c_t_r_model.item(),b_t_r_model.item(),k_pitch.item()

print('Friction due to steering parameters:')
print('a_stfr = ', a_stfr)
print('b_stfr = ', b_stfr)
print('d_stfr = ', d_stfr)
print('e_stfr = ', e_stfr)
print('f_stfr = ', f_stfr)
print('g_stfr = ', g_stfr)
print('steering curve parameters:')
print('a_s = ', a_s)
print('b_s = ', b_s)
print('c_s = ', c_s)
print('d_s = ', d_s)
print('e_s = ', e_s)
# wheel parameters
print('Front wheel parameters:')
print('d_t_f = ', d_t_f_model)
print('c_t_f = ', c_t_f_model)
print('b_t_f = ', b_t_f_model)
print('Rear wheel parameters:')
print('d_t_r = ', d_t_r_model)
print('c_t_r = ', c_t_r_model)
print('b_t_r = ', b_t_r_model)

# pitch coefficient
print('pitch influece coefficient')
print('k_pitch = ',k_pitch)









# # # --- plot loss function ---
plt.figure()
plt.title('Loss function')
plt.plot(loss_vec,label='tire model loss')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.legend()



# plot fitting results on the model
ax_acc_x_body.plot(df['vicon time'].to_numpy(),acc_x.detach().cpu().numpy(),label='acc_x model with steering friction (model output)',color='k')
ax_acc_x_body.legend()

ax_acc_y_body.plot(df['vicon time'].to_numpy(),acc_y.detach().cpu().numpy(),label='acc_y model with steering friction (model output)',color='k')
ax_acc_y_body.legend()

ax_acc_w.plot(df['vicon time'].to_numpy(),acc_w.detach().cpu().numpy(),label='acc_w model with steering friction (model output)',color='k')
ax_acc_w.legend()

plt.show()

