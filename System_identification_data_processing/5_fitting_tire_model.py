from functions_for_data_processing import get_data, plot_raw_data, process_raw_vicon_data,plot_vicon_data,linear_tire_model,pacejka_tire_model
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

matplotlib.rc('font', **font)



# this assumes that the current directory is DART
folder_path = 'System_identification_data_processing/Data/5_tire_model_data' 

# get the raw data
df_raw_data = get_data(folder_path)


#these are delays between robot time and robot reaction
delay_th = 0.01 # [s]
delay_st = 0.14 # [s]
#this is the delay betwee the vicon time and the robot time
delay_vicon_to_robot = 0.1 #0.05 #[s]


l = 0.175
lr = 0.54*l # the reference point taken by the data is not exaclty in the center of the vehicle
#lr = 0.06 # reference position from rear axel
lf = l-lr
theta_correction = -0.95/180*np.pi

df = process_raw_vicon_data(df_raw_data,delay_th,delay_st,delay_vicon_to_robot,lf,lr,theta_correction)


df1 = df[df['elapsed time sensors'] > 5.0]
df1 = df1[df1['elapsed time sensors'] < 58.7]

df2 = df[df['elapsed time sensors'] > 77.7]
df2 = df2[df2['elapsed time sensors'] < 135.0] # 140.0 156

# Concatenate all DataFrames into a single DataFrame vertically
df = pd.concat((df1,df2), axis=0, ignore_index=True)


# plot raw data
ax0,ax1,ax2 = plot_raw_data(df)

# plot 
plot_vicon_data(df)


# inertial charcteristics
m = 1.67
Jz = 0.006513 # uniform rectangle of shape 0.18 x 0.12


# evaluate forces in body frame starting from the ones in the absolute frame
Fx_vec = np.zeros(df.shape[0])
Fy_r_vec = np.zeros(df.shape[0])
Fy_f_vec = np.zeros(df.shape[0])
Fy_f_wheel_vec = np.zeros(df.shape[0])

# evaluate lateral forces from lateral and yaw dynamics
for i in range(0,df.shape[0]):
    b = np.array([df['ax_abs_filtered_more'].iloc[i]*m,
                  df['ay_abs_filtered_more'].iloc[i]*m,
                  df['aw_abs_filtered_more'].iloc[i]*Jz])
    

    yaw_i = df['unwrapped yaw'].iloc[i]

    A = np.array([[+np.cos(yaw_i),-np.sin(yaw_i),-np.sin(yaw_i)],
                  [+np.sin(yaw_i),np.cos(yaw_i),np.cos(yaw_i)],
                  [0             ,lf           ,-lr]])
    
    [Fx_i, Fy_f, Fy_r] = np.linalg.solve(A, b)

    # for the front wheel the body forces need to be rotated by -steer_angle
    rot_angle = - df['steering angle'].iloc[i]
    R = np.array([[ np.cos(rot_angle), -np.sin(rot_angle)],
                  [ np.sin(rot_angle),  np.cos(rot_angle)]])
    # we assume that the Fx is equally shared across the two wheels
    [Fx_f_wheel,Fy_f_wheel] = R @ np.array([[Fx_i/2],[Fy_f]])

    Fy_f_vec[i] = Fy_f
    Fy_r_vec[i] = Fy_r
    Fx_vec[i]   = Fx_i
    Fy_f_wheel_vec[i] = Fy_f_wheel










# fit the rear tire model
    
# --------------- fitting acceleration curve--------------- 
print('')
print('Fitting linear tire model')


# define first guess for parameters
initial_guess = torch.ones(2) * 0.5 # initialize parameters in the middle of their range constraint
# NOTE that the parmeter range constraint is set in the self.transform_parameters_norm_2_real method.

#instantiate the model
linear_tire_model_obj = linear_tire_model(initial_guess)

# define number of training iterations
train_its = 100

#define loss and optimizer objects
loss_fn = torch.nn.MSELoss() # reduction = 'mean'
optimizer_object = torch.optim.Adam(linear_tire_model_obj.parameters(), lr=0.1)
        
# generate data in tensor form for torch
train_x = torch.unsqueeze(torch.tensor(df['slip angle rear'].to_numpy()),1).cuda()
train_y = torch.unsqueeze(torch.tensor(Fy_r_vec),1).cuda()

# save loss values for later plot
loss_vec = np.zeros(train_its)

# train the model
for i in range(train_its):
    # clear gradient information from previous step before re-evaluating it for the current iteration
    optimizer_object.zero_grad()  
    
    # compute fitting outcome with current model parameters
    output = linear_tire_model_obj(train_x)

    # evaluate loss function
    loss = loss_fn(output,  train_y)
    loss_vec[i] = loss.item()

    # evaluate the gradient of the loss function with respect to the fitting parameters
    loss.backward() 

    # use the evaluated gradient to perform a gradient descent step 
    optimizer_object.step() # this updates parameters


# --- print out parameters ---
[a,b] = linear_tire_model_obj.transform_parameters_norm_2_real()
a, b = a.item(), b.item()
print('a = ', a)
print('b = ', b)



# # --- plot loss function ---
# plt.figure()
# plt.title('Loss')
# plt.plot(loss_vec)
# plt.xlabel('iterations')
# plt.ylabel('loss')







# fit the rear tire model
    
# --------------- fitting acceleration curve--------------- 
print('')
print('Fitting Pacejka tire model')


# define first guess for parameters
initial_guess = torch.ones(4) * 0.5 # initialize parameters in the middle of their range constraint
# NOTE that the parmeter range constraint is set in the self.transform_parameters_norm_2_real method.

#instantiate the model
pacejka_tire_model_obj = pacejka_tire_model(initial_guess)

# define number of training iterations
train_its = 500

#define loss and optimizer objects
loss_fn = torch.nn.MSELoss() # reduction = 'mean'
optimizer_object = torch.optim.Adam(pacejka_tire_model_obj.parameters(), lr=0.01)
        
# generate data in tensor form for torch
train_x = torch.unsqueeze(torch.tensor(df['slip angle front'].to_numpy()),1).cuda()
train_y = torch.unsqueeze(torch.tensor(Fy_f_wheel_vec),1).cuda()

# save loss values for later plot
loss_vec = np.zeros(train_its)

# train the model
for i in range(train_its):
    # clear gradient information from previous step before re-evaluating it for the current iteration
    optimizer_object.zero_grad()  
    
    # compute fitting outcome with current model parameters
    output = pacejka_tire_model_obj(train_x)

    # evaluate loss function
    loss = loss_fn(output,  train_y)
    loss_vec[i] = loss.item()

    # evaluate the gradient of the loss function with respect to the fitting parameters
    loss.backward() 

    # use the evaluated gradient to perform a gradient descent step 
    optimizer_object.step() # this updates parameters


# --- print out parameters ---
[d,c,b,e] = pacejka_tire_model_obj.transform_parameters_norm_2_real()
d,c,b,e = d.item(),c.item(), b.item(),e.item()

print('d = ', d)
print('c = ', c)
print('b = ', b)
print('e = ', e)



# # --- plot loss function ---
plt.figure()
plt.title('Loss')
plt.plot(loss_vec)
plt.xlabel('iterations')
plt.ylabel('loss')












fig = plt.figure(figsize=(10, 6))
plt.plot(df['vicon time'].to_numpy(),Fx_vec,label='Fx')
plt.plot(df['vicon time'].to_numpy(),Fy_f_vec,label='Fy_f')
plt.plot(df['vicon time'].to_numpy(),Fy_r_vec,label='Fy_r')
plt.plot(df['vicon time'].to_numpy(),Fy_r_vec + Fy_f_vec,label='Fy_sum')
plt.legend()





fig, ((ax3,ax4)) = plt.subplots(2, 1, figsize=(10, 6))
fig.subplots_adjust(top=0.995,
                    bottom=0.11,
                    left=0.095,
                    right=0.995,
                    hspace=0.345,
                    wspace=0.2)





# plot the fron slip angle - Fy data
alpha_front_vec = torch.unsqueeze(torch.linspace(df['slip angle front'].min(),df['slip angle front'].max(),100),1).cuda()
lateral_force_vec_front = pacejka_tire_model_obj(alpha_front_vec).detach().cpu().numpy()




ax3.scatter(df['slip angle front'].to_numpy(),Fy_f_wheel_vec,label='data')
#plt.scatter(0,0,color='red',label='zero')
ax3.plot(alpha_front_vec.cpu().numpy(),lateral_force_vec_front,label = 'Front tire model',zorder=20,color='orangered',linewidth=4,linestyle='-')
ax3.set_xlabel('slip angle [deg]')
ax3.set_ylabel('Fy [N]')
ax3.legend()
ax3.legend(bbox_to_anchor=(1.01, -0.05),loc='lower right')
ax3.set_xlim([-10.5,+10.5])

# plot the rear slip angle - Fy data
alpha_rear_vec = torch.unsqueeze(torch.linspace(df['slip angle rear'].min(),df['slip angle rear'].max(),100),1).cuda()
lateral_force_vec = linear_tire_model_obj(alpha_rear_vec).detach().cpu().numpy()


#plt.scatter(df['slip angle front'].to_numpy(),Fy_f_vec,color='navy',label='front')
ax4.scatter(df['slip angle rear'].to_numpy(),Fy_r_vec,label='data')
#plt.scatter(0,0,color='red',label='zero')
ax4.plot(alpha_rear_vec.cpu().numpy(),lateral_force_vec,label = 'Rear tire model',zorder=20,color='orangered',linewidth=4,linestyle='-')
ax4.set_xlabel('slip angle [deg]')
ax4.set_ylabel('Fy [N]')
ax4.set_xlim([-10.5,+10.5])
ax4.legend()

ax4.legend(bbox_to_anchor=(1.01, -0.05),loc='lower right')


plt.show()

