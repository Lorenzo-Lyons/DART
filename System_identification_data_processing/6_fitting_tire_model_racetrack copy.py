from functions_for_data_processing import get_data, plot_raw_data, process_raw_vicon_data,plot_vicon_data\
,linear_tire_model,pacejka_tire_model,dyn_model_culomb_tires,produce_long_term_predictions
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

# This script is used to fit the tire model to the data collected on the racetrack with a 
# SIMPLE CULOMB friction tyre model.

# TO DO when I get back:
# MUST REDO the friction estimation on the new floor cause it does change thiings a lot
# what about the motor curve? Do I need to re estimate it? That part should not change acoording to the ground,
# but check if this is the case.

# collect new data, constant steer increasing velocity, as previous datasets, until saturation is reached.
# going straight and then sharp turn to see what happens in the transition period.






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
# and make sure that the rear tyre slip angle is 0.
# notice that for lr down, a_r down too.


# NOTE:
# another trick you can do is to fine tune the c parmeter in the steering curve inside processing raw data cause it will
# afffect the belived steering angle, and hence the slip angles. Gives you an extra degree of freedom to tweack things
# shifts the entire top figure left and right


# this assumes that the current directory is DART
#folder_path = 'System_identification_data_processing/Data/5_tire_model_data' 
#folder_path = 'System_identification_data_processing/Data/6_racetrack_lap_v_060' 
folder_path = 'System_identification_data_processing/Data/7_racetrack_lap_v_1' 

# get the raw data
df_raw_data = get_data(folder_path)


#these are delays between robot time and robot reaction
delay_th = 0.01 # [s]
delay_st = 0.14 # [s]
#this is the delay between the vicon time and the robot time
delay_vicon_to_robot = 0.1 #0.05 #[s]


l = 0.175
lr = 0.53*l #0.45 the reference point taken by the data is not exaclty in the center of the vehicle

lf = l-lr
theta_correction = 1.5/180*np.pi #0.5/180*np.pi works for front wheel and 1.5 works for back wheel

df = process_raw_vicon_data(df_raw_data,delay_th,delay_st,delay_vicon_to_robot,lf,lr,theta_correction)



# use this if using v=0.60 dataset
if folder_path == 'System_identification_data_processing/Data/6_racetrack_lap_v_060':
    time_cut = 30
    df = df[df['elapsed time sensors'] > time_cut]

elif folder_path == 'System_identification_data_processing/Data/7_racetrack_lap_v_1':
    time_cut =  18
    time_finish = 28 # 33.5 #
    #highlighting sharp corner clockwise (w<0)
    # time_cut =  27
    # time_finish = 31
    #highlighting slow corner anticlockwise (w>0)
    # time_cut =  19.5
    # time_finish = 28
    df = df[df['elapsed time sensors'] > time_cut]
    df = df[df['elapsed time sensors'] < time_finish]

    





elif folder_path == 'System_identification_data_processing/Data/5_tire_model_data':
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

# evalauting lateral velocities on wheels
V_y_f_wheel = np.zeros(df.shape[0])



# since now the steering angle is not fixed, it is a good idea to apply the first order filter to it, to recover the true steering angle
# Time step (sampling period)
df = df.copy()
T = df['elapsed time sensors'].diff().mean()  # Calculate the average time step
# Time delay
tau = 0.15
# Filter coefficient
alpha = T / (T + tau)
# Initialize the filtered steering angle list
filtered_steering_angle = [df['steering angle'].iloc[0]]
# Apply the first-order filter
for i in range(1, len(df)):
    filtered_value = alpha * df['steering angle'].iloc[i] + (1 - alpha) * filtered_steering_angle[-1]
    filtered_steering_angle.append(filtered_value)

# Add the filtered values to the DataFrame
df['steering angle time delayed'] = filtered_steering_angle

# plot the steering angle time delayed vs W
plt.figure()
plt.plot(df['vicon time'].to_numpy(),df['steering angle'].to_numpy(),label='steering angle')
plt.plot(df['vicon time'].to_numpy(),df['steering angle time delayed'].to_numpy(),label='steering angle time delayed')
plt.plot(df['vicon time'].to_numpy(),df['w_abs_filtered'].to_numpy(),label='w filtered')
plt.legend()


#plot wheel lateral velocity front vs time and steering agle



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
    #rot_angle = - df['steering angle'].iloc[i] # using time delayed version instead
    rot_angle = - df['steering angle time delayed'].iloc[i]

    R = np.array([[ np.cos(rot_angle), -np.sin(rot_angle)],
                  [ np.sin(rot_angle),  np.cos(rot_angle)]])
    # we assume that the Fx is equally shared across the two wheels
    [Fx_f_wheel,Fy_f_wheel] = R @ np.array([[Fx_i/2],[Fy_f]])

    Fy_f_vec[i] = Fy_f
    Fy_r_vec[i] = Fy_r
    Fx_vec[i]   = Fx_i
    Fy_f_wheel_vec[i] = Fy_f_wheel

    # evaluate wheel lateral velocities
    V_y_f_wheel[i] = np.cos(rot_angle)*(df['vy body'].to_numpy()[i] + lf*df['w_abs_filtered'].to_numpy()[i]) + np.sin(rot_angle) * df['vx body'].to_numpy()[i]
V_y_r_wheel = df['vy body'].to_numpy() - lr*df['w_abs_filtered'].to_numpy()


# evaluate centripetal force
F_centripetal = m * np.multiply(df['vx body'].to_numpy(), df['w_abs_filtered'].to_numpy())
Fy_f_torquing = Fy_f_vec - lf/(lf+lr) * F_centripetal
Fy_r_torquing = Fy_f_vec - lr/(lf+lr) * F_centripetal







# fitting tyre models
# define first guess for parameters
initial_guess = torch.ones(2) * 0.5 # initialize parameters in the middle of their range constraint
# parameters min-max values
a_minmax = [-35,-100]
b_minmax = [-0.02,0.02]
# define number of training iterations
train_its = 1000
learning_rate = 0.001





# --------------- fitting tire model FRONT--------------- 
print('')
print('Fitting linear culomb friction tire model FRONT')

#instantiate the model
linear_tire_model_obj_f = linear_tire_model(initial_guess,a_minmax,b_minmax)

#define loss and optimizer objects
loss_fn = torch.nn.MSELoss() # reduction = 'mean'
optimizer_object = torch.optim.Adam(linear_tire_model_obj_f.parameters(), lr=learning_rate)
        
# generate data in tensor form for torch
train_x = torch.unsqueeze(torch.tensor(V_y_f_wheel),1).cuda()
train_y = torch.unsqueeze(torch.tensor(Fy_f_wheel_vec),1).cuda() 

# save loss values for later plot
loss_vec_f = np.zeros(train_its)

# train the model
for i in range(train_its):
    # clear gradient information from previous step before re-evaluating it for the current iteration
    optimizer_object.zero_grad()  
    
    # compute fitting outcome with current model parameters
    output = linear_tire_model_obj_f(train_x)

    # evaluate loss function
    loss = loss_fn(output,  train_y)
    loss_vec_f[i] = loss.item()

    # evaluate the gradient of the loss function with respect to the fitting parameters
    loss.backward() 

    # use the evaluated gradient to perform a gradient descent step 
    optimizer_object.step() # this updates parameters


# --- print out parameters ---
[a_f,b_f] = linear_tire_model_obj_f.transform_parameters_norm_2_real()
a_f, b_f = a_f.item(), b_f.item()
print('Front Wheel parameters:')
print('a_f = ', a_f)
print('b_f = ', b_f)



    
# --------------- fitting tire model REAR--------------- 
print('')
print('Fitting linear culomb friction tire model REAR')

#instantiate the model
linear_tire_model_obj_r = linear_tire_model(initial_guess,a_minmax,b_minmax)

#define loss and optimizer objects
loss_fn = torch.nn.MSELoss() # reduction = 'mean'
optimizer_object = torch.optim.Adam(linear_tire_model_obj_r.parameters(), lr=learning_rate)
        
# generate data in tensor form for torch
train_x = torch.unsqueeze(torch.tensor(V_y_r_wheel),1).cuda()
train_y = torch.unsqueeze(torch.tensor(Fy_r_vec),1).cuda() 

# save loss values for later plot
loss_vec_r = np.zeros(train_its)

# train the model
for i in range(train_its):
    # clear gradient information from previous step before re-evaluating it for the current iteration
    optimizer_object.zero_grad()  
    
    # compute fitting outcome with current model parameters
    output = linear_tire_model_obj_r(train_x)

    # evaluate loss function
    loss = loss_fn(output,  train_y)
    loss_vec_r[i] = loss.item()

    # evaluate the gradient of the loss function with respect to the fitting parameters
    loss.backward() 

    # use the evaluated gradient to perform a gradient descent step 
    optimizer_object.step() # this updates parameters


# --- print out parameters ---
[a_r,b_r] = linear_tire_model_obj_r.transform_parameters_norm_2_real()
a_r, b_r = a_r.item(), b_r.item()
print('Rear Wheel parameters:')
print('a_r = ', a_r)
print('b_r = ', b_r)


# # --- plot loss function ---
plt.figure()
plt.title('Loss function')
plt.plot(loss_vec_r,label='front tire model loss')
plt.plot(loss_vec_f,label='rear tire model loss')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.legend()






# plotting absolute forces
fig = plt.figure(figsize=(10, 6))
plt.plot(df['vicon time'].to_numpy(),Fx_vec,label='Fx')
plt.plot(df['vicon time'].to_numpy(),Fy_f_vec,label='Fy_f')
plt.plot(df['vicon time'].to_numpy(),Fy_r_vec,label='Fy_r')
plt.plot(df['vicon time'].to_numpy(),Fy_r_vec + Fy_f_vec,label='Fy_sum')
# net moment
plt.plot(df['vicon time'].to_numpy(),-(Fy_r_vec)*lr + Fy_f_vec*lf,label='Angular Moment')
plt.legend()



# # plotting absolute forces
# fig = plt.figure(figsize=(10, 6))
# plt.plot(df['vicon time'].to_numpy(),Fx_vec,label='Fx')
# plt.plot(df['vicon time'].to_numpy(),Fy_f_vec,label='Fy_f')
# plt.plot(df['vicon time'].to_numpy(),Fy_r_vec,label='Fy_r')
# plt.plot(df['vicon time'].to_numpy(),Fy_r_vec + Fy_f_vec,label='Fy_sum')
# plt.plot(df['vicon time'].to_numpy(),-(Fy_r_vec)*lr + Fy_f_vec*lf,label='Angular Moment')
# plt.legend()


# plotting absolute forces
fig = plt.figure(figsize=(10, 6))
plt.plot(df['vicon time'].to_numpy(),Fy_f_vec,label='Fy_f')
plt.plot(df['vicon time'].to_numpy(),Fy_r_vec,label='Fy_r')
plt.plot(df['vicon time'].to_numpy(),Fy_f_wheel_vec ,label='Fy_f_wheel')

plt.title('forces on wheels vs forces on body')
plt.legend()







# fig, ((ax3,ax4)) = plt.subplots(2, 1, figsize=(10, 6))
# fig.subplots_adjust(top=0.995,
#                     bottom=0.11,
#                     left=0.095,
#                     right=0.995,
#                     hspace=0.345,
#                     wspace=0.2)





# # plot the front slip angle - Fy data
# alpha_front_vec = torch.unsqueeze(torch.linspace(df['slip angle front'].min(),df['slip angle front'].max(),100),1).cuda()
# lateral_force_vec_front = pacejka_tire_model_obj(alpha_front_vec).detach().cpu().numpy()




# ax3.scatter(df['slip angle front'].to_numpy(),Fy_f_wheel_vec,label='data')
# #plt.scatter(0,0,color='red',label='zero')
# ax3.plot(alpha_front_vec.cpu().numpy(),lateral_force_vec_front,label = 'Front tire model',zorder=20,color='orangered',linewidth=4,linestyle='-')
# ax3.set_xlabel('slip angle [deg]')
# ax3.set_ylabel('Fy [N]')
# ax3.legend()
# ax3.legend(bbox_to_anchor=(1.01, -0.05),loc='lower right')
# ax3.set_xlim([-10.5,+10.5])

# # plot the rear slip angle - Fy data
# alpha_rear_vec = torch.unsqueeze(torch.linspace(df['slip angle rear'].min(),df['slip angle rear'].max(),100),1).cuda()
# lateral_force_vec = linear_tire_model_obj(alpha_rear_vec).detach().cpu().numpy()


# #plt.scatter(df['slip angle front'].to_numpy(),Fy_f_vec,color='navy',label='front')
# #ax4.scatter(df['slip angle rear'].to_numpy(),Fy_r_vec,label='data')

# ax4.scatter(df['slip angle rear'].to_numpy(),Fy_r_torquing,label='data') 
# #plt.scatter(0,0,color='red',label='zero')
# ax4.plot(alpha_rear_vec.cpu().numpy(),lateral_force_vec,label = 'Rear tire model',zorder=20,color='orangered',linewidth=4,linestyle='-')
# ax4.set_xlabel('slip angle [deg]')
# ax4.set_ylabel('Fy [N]')
# ax4.set_xlim([-10.5,+10.5])
# ax4.legend()

# ax4.legend(bbox_to_anchor=(1.01, -0.05),loc='lower right')







# plot lateral forces against lateral velocities

# evaluate model on plotting interval
# v_y_f_plotting = torch.unsqueeze(torch.linspace(np.min(Fy_f_wheel),df['slip angle rear'].max(),100),1).cuda()
# lateral_force_vec = linear_tire_model_obj(alpha_rear_vec).detach().cpu().numpy()


# fig, ((ax5,ax6)) = plt.subplots(2, 1, figsize=(10, 6))
# fig.subplots_adjust(top=0.995,
#                     bottom=0.11,
#                     left=0.095,
#                     right=0.995,
#                     hspace=0.345,
#                     wspace=0.2)


# ax5.scatter(V_y_f_wheel,Fy_f_wheel_vec,label='data') 
# #ax5.plot(alpha_rear_vec.cpu().numpy(),lateral_force_vec,label = 'Rear tire model',zorder=20,color='orangered',linewidth=4,linestyle='-')
# ax5.set_xlabel('Wheel lateral velocity [m/s]')
# ax5.set_ylabel('Fy [N]')
# ax5.legend()
# ax5.set_title('Front wheel')


# ax6.scatter(V_y_r_wheel,Fy_r_vec,label='data') 
# ax6.set_xlabel('Wheel lateral velocity [m/s]')
# ax6.set_ylabel('Fy [N]')
# ax6.legend()
# ax6.set_title('Rear wheel')




# plot tire model fitting results, i.e. lateral force on wheel vs lateral wheel velocity

# evaluate model on plotting interval
v_y_f_wheel_plotting = torch.unsqueeze(torch.linspace(np.min(V_y_f_wheel),np.max(V_y_f_wheel),100),1).cuda()
lateral_force_vec_f = linear_tire_model_obj_f(v_y_f_wheel_plotting).detach().cpu().numpy()

# evaluate model on plotting interval
v_y_r_wheel_plotting = torch.unsqueeze(torch.linspace(np.min(V_y_r_wheel),np.max(V_y_r_wheel),100),1).cuda()
lateral_force_vec_r = linear_tire_model_obj_r(v_y_r_wheel_plotting).detach().cpu().numpy()


fig, ((ax5,ax6)) = plt.subplots(2, 1, figsize=(10, 6))
fig.subplots_adjust(top=0.995,
                    bottom=0.11,
                    left=0.095,
                    right=0.995,
                    hspace=0.345,
                    wspace=0.2)


c = df['steering angle time delayed'].to_numpy()
#c = df['vicon time'].to_numpy()

# Front wheel
scatter_front = ax5.scatter(V_y_f_wheel, Fy_f_wheel_vec, c=c, cmap='viridis', label='data')
ax5.plot(v_y_f_wheel_plotting.detach().cpu().numpy(),lateral_force_vec_f,color='darkorange',label='Front tire model',linewidth=4,linestyle='-')
ax5.scatter(np.array([0.0]),np.array([0.0]),color='orangered',label='zero',marker='+', zorder=20)
ax5.set_xlabel('Wheel lateral velocity [m/s]')
ax5.set_ylabel('Fy wheel [N]')
ax5.legend()
ax5.set_title('Front wheel')
cbar_front = plt.colorbar(scatter_front, ax=ax5)
cbar_front.set_label('Steering Value')

# Rear wheel
scatter_rear = ax6.scatter(V_y_r_wheel, Fy_r_vec, c=df['steering angle time delayed'].to_numpy(), cmap='viridis', label='data')
ax6.plot(v_y_r_wheel_plotting.detach().cpu().numpy(),lateral_force_vec_r,color='darkorange',label='Rear tire model',linewidth=4,linestyle='-')
ax6.scatter(np.array([0.0]),np.array([0.0]),color='orangered',label='zero',marker='+',zorder=20)
ax6.set_xlabel('Wheel lateral velocity [m/s]')
ax6.set_ylabel('Fy wheel[N]')
ax6.set_xlim(ax5.get_xlim())
ax6.legend()
ax6.set_title('Rear wheel')
cbar_rear = plt.colorbar(scatter_rear, ax=ax6)
cbar_rear.set_label('Steering Value')








# producing long term predictions
# construct a model that takes as inputs Vx,Vy,W,tau,Steer ---> Vx_dot,Vy_dot,W_dot
dynamic_model = dyn_model_culomb_tires(m,lr,lf,Jz,a_f,b_f,a_r,b_r)





columns_to_extract = ['vicon time', 'vx body', 'vy body', 'w_abs_filtered', 'throttle' ,'steering','vicon x','vicon y','vicon yaw']
input_data_long_term_predictions = df[columns_to_extract].to_numpy()
prediction_window = 1.5 # [s]
jumps = 25
forward_propagate_indexes = [1,2,3] # 1 =vx, 2=vy, 3=w
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

