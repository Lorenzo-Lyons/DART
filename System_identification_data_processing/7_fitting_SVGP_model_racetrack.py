from functions_for_data_processing import get_data, plot_raw_data, process_raw_vicon_data,plot_vicon_data\
,dyn_model_culomb_tires,produce_long_term_predictions,train_SVGP_model,dyn_model_SVGP
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
# multi-output SVGP model.

# TO DO: collect a dataset with no weird jumps in the vicon data, velocity range between -.6 and 1.2 m/s, and also slowing up/down
# to give the SVGP a chance at guessing the friction





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
ax_body_vec = np.zeros(df.shape[0])
ay_body_vec = np.zeros(df.shape[0])
aw_body_vec = np.zeros(df.shape[0])

# just for checking
ay_centripetal_vec = np.zeros(df.shape[0])


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




# evaluate accelerations in the body frame
for i in range(0,df.shape[0]):
    b = np.array([df['ax_abs_filtered_more'].iloc[i],
                  df['ay_abs_filtered_more'].iloc[i],
                  df['aw_abs_filtered_more'].iloc[i]])
    
    yaw_i = df['unwrapped yaw'].iloc[i]

    R = np.array([[+np.cos(-yaw_i),-np.sin(-yaw_i),0],
                  [+np.sin(-yaw_i),np.cos(-yaw_i), 0],
                  [0             ,0          ,1]])
    
    [ax_i, ay_i, aw_i] = R @ b


    ax_body_vec[i] = ax_i
    ay_body_vec[i] = ay_i- df['w_abs_filtered'].iloc[i]*df['vx body'].iloc[i]
    aw_body_vec[i] = aw_i
    ay_centripetal_vec[i] = df['w_abs_filtered'].iloc[i]*df['vx body'].iloc[i]




# train the SVGP model
num_epochs = 50
n_inducing_points = 400
learning_rate = 0.05

      
# generate data in tensor form for torch
#df['throttle'] = df['throttle'] - 0.2 # shifting to actual activation range
columns_to_extract = ['vx body', 'vy body', 'w_abs_filtered', 'throttle' ,'steering'] #  angle time delayed
SVGP_training_data = df[columns_to_extract].to_numpy()
train_x = torch.tensor(SVGP_training_data).cuda()
train_y_vx = torch.unsqueeze(torch.tensor(ax_body_vec),1).cuda() 
train_y_vy = torch.unsqueeze(torch.tensor(ay_body_vec),1).cuda() 
train_y_w  = torch.unsqueeze(torch.tensor(aw_body_vec),1).cuda() 

#cast to float to avoid issues with data types
# add some state noise to stabilize predictions in the long term
# Define different standard deviations for each column
std_devs = [0.05, 0.05, 0.05]  

# Generate noise for each column with the specified standard deviations
noise1 = torch.randn(train_x.size(0)) * std_devs[0]
noise2 = torch.randn(train_x.size(0)) * std_devs[1]
noise3 = torch.randn(train_x.size(0)) * std_devs[2]

# Add the noise to the first three columns
train_x[:, 0] += noise1.cuda()
train_x[:, 1] += noise2.cuda()
train_x[:, 2] += noise3.cuda()


# convert to float to avoid issues with data types
train_x = train_x.to(torch.float32)
train_y_vx = train_y_vx.to(torch.float32)
train_y_vy = train_y_vy.to(torch.float32)
train_y_w = train_y_w.to(torch.float32)




model_vx, model_vy, model_w, likelihood_vx, likelihood_vy, likelihood_w = train_SVGP_model(learning_rate,num_epochs, train_x, train_y_vx, train_y_vy, train_y_w, n_inducing_points)



# display fitting results

# Get into evaluation (predictive posterior) mode
model_vx.eval()
model_vy.eval()
model_w.eval()

#evaluate model on training dataset
preds_ax = model_vx(train_x)
preds_ay = model_vy(train_x)
preds_aw = model_w(train_x)

lower_x, upper_x = preds_ax.confidence_region()
lower_y, upper_y = preds_ay.confidence_region()
lower_w, upper_w = preds_aw.confidence_region()


# plotting body frame accelerations and data fitting results 
fig, ((ax1,ax2,ax3)) = plt.subplots(3, 1, figsize=(10, 6))
fig.subplots_adjust(top=0.995,
                    bottom=0.11,
                    left=0.095,
                    right=0.995,
                    hspace=0.345,
                    wspace=0.2)

# plot fitting results
ax1.set_title('ax in body frame')
ax1.plot(df['vicon time'].to_numpy(),ax_body_vec,label='ax',color='dodgerblue')
#ax1.plot(df['vicon time'].to_numpy(),train_y_vx.detach().cpu().numpy(),label='labels',color='navy')
ax1.plot(df['vicon time'].to_numpy(),preds_ax.mean.detach().cpu().numpy(),color='k',label='model prediction')
ax1.fill_between(df['vicon time'].to_numpy(), lower_x.detach().cpu().numpy(), upper_x.detach().cpu().numpy(), alpha=0.2,color='k',label='2 sigma confidence')
ax1.legend()

ax2.plot(df['vicon time'].to_numpy(),ay_body_vec,label='ay',color='orangered')
#ax2.plot(df['vicon time'].to_numpy(),train_y_vy.detach().cpu().numpy(),label='labels',color='navy')
ax2.plot(df['vicon time'].to_numpy(),preds_ay.mean.detach().cpu().numpy(),color='k',label='model prediction')
ax2.plot(df['vicon time'].to_numpy(),ay_centripetal_vec,color='orange',label='centripetal acc',linestyle='--')
ax2.fill_between(df['vicon time'].to_numpy(), lower_y.detach().cpu().numpy(), upper_y.detach().cpu().numpy(), alpha=0.2,color='k',label='2 sigma confidence')
ax2.legend()

ax3.plot(df['vicon time'].to_numpy(),aw_body_vec,label='aw',color='orchid')
#ax3.plot(df['vicon time'].to_numpy(),train_y_w.detach().cpu().numpy(),label='labels',color='navy')
ax3.plot(df['vicon time'].to_numpy(),preds_aw.mean.detach().cpu().numpy(),color='k',label='model prediction')
ax3.fill_between(df['vicon time'].to_numpy(), lower_w.detach().cpu().numpy(), upper_w.detach().cpu().numpy(), alpha=0.2,color='k',label='2 sigma confidence')
ax3.legend()






# producing long term predictions
# construct a model that takes as inputs Vx,Vy,W,tau,Steer ---> Vx_dot,Vy_dot,W_dot
dynamic_model = dyn_model_SVGP(model_vx,model_vy,model_w)




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

