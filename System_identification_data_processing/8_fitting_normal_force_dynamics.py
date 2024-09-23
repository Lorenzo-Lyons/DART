from functions_for_data_processing import get_data, plot_raw_data, process_raw_vicon_data,plot_vicon_data\
    ,model_parameters,pitch_and_roll_dynamics_model,throttle_dynamics,steering_dynamics,dyn_model_culomb_tires,generate_tensor_past_actions,full_dynamic_model
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

# This script is used to fit the steering dynamics once we are sure about what the tire model is, and the longitudinal dynamics are properly modelled.
# The additional friction due to steering must be identified already.










# ---------- NOTE ------------ #
# this is the EXCESS longitudinal and lateral acceleration effect compared to the static case (because of course the roll dpends on the whole lateral acc i.e. with centrifugal)
# but this is already accounted for in the wheel curve from the static tests





# select data folder NOTE: this assumes that the current directory is DART
#folder_path = 'System_identification_data_processing/Data/8_circles_rubbery_floor_1_file'
#folder_path = 'System_identification_data_processing/Data/81_throttle_ramps'
folder_path = 'System_identification_data_processing/Data/91_free_driving_16_sept_2024'
#folder_path = 'System_identification_data_processing/Data/91_free_driving_16_sept_2024_slow'


# define the number of past steering signals to use
n_past_acc_x = 200 
refinement_factor = 10 # refine for stable integration purpouses


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


# # Starting data processing

# Starting data processing
# check if there is a processed vicon data file already
# file_name = 'processed_vicon_data.csv'
# # Check if the CSV file exists in the folder
# file_path = os.path.join(folder_path, file_name)

# if not os.path.isfile(file_path):
#     # If the file does not exist, process the raw data

#     # get the raw data
#     df_raw_data = get_data(folder_path)

#     # replace throttle with time integrated throttle
#     filtered_throttle = throttle_dynamics(df_raw_data,d_m)
#     df_raw_data['throttle'] = filtered_throttle

#     # replace steering and steering angle with the time integrated version
#     st_vec_angle_optuna, st_vec_optuna = steering_dynamics(df_raw_data,a_s,b_s,c_s,d_s,e_s,max_st_dot,fixed_delay_stdn,k_stdn)
#     df_raw_data['steering angle'] = st_vec_angle_optuna
#     df_raw_data['steering'] = st_vec_optuna

#     # process data
#     steps_shift = 3 # decide to filter more or less the vicon data
#     df = process_raw_vicon_data(df_raw_data,steps_shift)

#     df.to_csv(file_path, index=False)
#     print(f"File '{file_path}' saved.")
# else:
#     print(f"File '{file_path}' already exists, loading data.")
#     df = pd.read_csv(file_path)









# get the raw data
df_raw_data = get_data(folder_path)

# replace throttle with time integrated throttle
filtered_throttle = throttle_dynamics(df_raw_data,d_m)
df_raw_data['throttle'] = filtered_throttle

# replace steering and steering angle with the time integrated version
# st_vec_angle_optuna, st_vec_optuna = steering_dynamics(df_raw_data,a_s,b_s,c_s,d_s,e_s,max_st_dot,fixed_delay_stdn,k_stdn)
# df_raw_data['steering angle'] = st_vec_angle_optuna
# df_raw_data['steering'] = st_vec_optuna

# process data
steps_shift = 3 # decide to filter more or less the vicon data
df = process_raw_vicon_data(df_raw_data,steps_shift)



















# plot raw data
ax0,ax1,ax2 = plot_raw_data(df)

# plot vicon related data (longitudinal and lateral velocities, yaw rate related)
ax_wheels,ax_total_force_front,ax_total_force_rear,ax_lat_force,ax_long_force = plot_vicon_data(df)






# plot total force as predicted by the model
# the model gives you the derivatives of it's own states, so you can integrate them to get the states in the new time instant
dynamic_model = dyn_model_culomb_tires(m,lr,lf,l_COM,Jz,
                 d_t,c_t,b_t,
                 a_m,b_m,c_m,
                 a_f,b_f,c_f,d_f,
                 a_stfr, b_stfr,d_stfr,e_stfr,f_stfr,g_stfr)



total_force_model_front = np.zeros(df.shape[0])
total_force_model_rear = np.zeros(df.shape[0])
Fy_front_base_model_vec = np.zeros(df.shape[0])
Fy_rear_base_model_vec = np.zeros(df.shape[0])
Fx_wheels_vec = np.zeros(df.shape[0])

for i in range(df.shape[0]):
    vx = df['vx body'].iloc[i]
    vy = df['vy body'].iloc[i]
    w = df['w_abs_filtered'].iloc[i]
    throttle = df['throttle'].iloc[i]
    steer_angle = df['steering angle'].iloc[i]

    Fx_wheel = 0.5 * (dynamic_model.motor_force(throttle,vx) + dynamic_model.friction(vx) + dynamic_model.friction_due_to_steering(vx,steer_angle))
    # wheel forces
    Vy_wheel_f = np.cos(steer_angle)*(vy + lf*w) - np.sin(steer_angle) * vx
    Vy_wheel_r = vy - lr*w
    Fy_front = dynamic_model.lateral_tire_forces(Vy_wheel_f)
    Fy_rear = dynamic_model.lateral_tire_forces(Vy_wheel_r)

    total_force_model_front[i] =  (Fx_wheel**2 + Fy_front**2)**0.5
    total_force_model_rear[i] =   (Fx_wheel**2 + Fy_rear**2)**0.5
    Fy_front_base_model_vec[i] = Fy_front
    Fy_rear_base_model_vec[i] = Fy_rear
    Fx_wheels_vec[i] = Fx_wheel

ax_total_force_rear.plot(df['vicon time'].to_numpy(),total_force_model_rear,label='total force model rear',color='k')
ax_total_force_front.plot(df['vicon time'].to_numpy(),total_force_model_front,label='total force model front',color='k')
ax_total_force_rear.legend()
ax_total_force_front.legend()

# plot components
# ax_lat_force.plot(df['vicon time'].to_numpy(),Fy_front_base_model_vec,label='Fy front model',color='k')
# ax_lat_force.plot(df['vicon time'].to_numpy(),Fy_rear_base_model_vec,label='Fy rear model',color='b')
# ax_lat_force.legend()

# ax_long_force.plot(df['vicon time'].to_numpy(),Fx_wheels_vec,label='Fx model',color='k')
# ax_long_force.legend()




# plot error between model and measured lateral forces compared to longitudinal acceleration

wheel_slippage = np.abs(df['vel encoder'].to_numpy() - df['vx body'].to_numpy())
front_lat_force_mismatch = (Fy_front_base_model_vec - df['Fy front wheel'].to_numpy()) * np.sign(df['Fy front wheel'].to_numpy())
rear_lat_force_mismatch = (Fy_rear_base_model_vec - df['Fy rear wheel'].to_numpy()) * np.sign(df['Fy rear wheel'].to_numpy())


fig1, ((ax1,ax2)) = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

# front tire
ax1.plot(df['vicon time'].to_numpy(), wheel_slippage,label='longitudinal slippage',color = 'gray')
ax1.plot(df['vicon time'].to_numpy(), df['ax body'].to_numpy(),label='longitudinal acceleration',color = 'dodgerblue')
ax1.plot(df['vicon time'].to_numpy(), df['ay body'].to_numpy(),label='lateral acceleration',color = 'orangered')
ax1.plot(df['vicon time'].to_numpy(), front_lat_force_mismatch,label='front lat force mismatch',color = 'k')
ax1.set_xlabel('Time [s]')
ax1.set_title('Front tire lat forces mismatch (following sign)')
ax1.legend()

#rear tire
ax2.plot(df['vicon time'].to_numpy(), wheel_slippage,label='longitudinal slippage',color = 'gray')
ax2.plot(df['vicon time'].to_numpy(), df['ax body'].to_numpy(),label='longitudinal acceleration',color = 'dodgerblue')
ax2.plot(df['vicon time'].to_numpy(), df['ay body'].to_numpy(),label='lateral acceleration',color = 'orangered')
ax2.plot(df['vicon time'].to_numpy(), rear_lat_force_mismatch,label='rear lat force mismatch',color = 'k')
ax2.set_xlabel('Time [s]')
ax2.set_title('Rear tire lat forces mismatch (following sign)')
ax2.legend()













# --------------- fitting model---------------

# fitting tyre models
# define first guess for parameters
initial_guess = torch.ones(6) * 0.5 # initialize parameters in the middle of their range constraint
#initial_guess[0] = 0.01 # w_natural_Hz_pitch
#initial_guess[1] = 0.99 # k_f_pitch
# initial_guess[2] = 0.001 # k_r_pitch
#initial_guess[3] = 0.9 # w_natural_Hz_roll
#initial_guess[4] = 0.001 # k_f_roll
#initial_guess[5] = 0.2 # k_r_roll


# define number of training iterations
train_its =  160 # 400 #
learning_rate = 0.005 # 0.002 #

print('')
print('Fitting roll and pitch dynamics')


# generate data in tensor form for torch
# -- X data --
Fy_front_model_tensor = torch.unsqueeze(torch.tensor(Fy_front_base_model_vec),1).cuda() 
Fy_rear_model_tensor = torch.unsqueeze(torch.tensor(Fy_rear_base_model_vec),1).cuda() 
train_x_wheel_forces_model = torch.hstack((Fy_front_model_tensor, Fy_rear_model_tensor))
# past longitudinal accelerations

n_past_actions = 50 # 50
refinement_factor = 10 # 10
train_x_past_acc_x = generate_tensor_past_actions(df, n_past_actions,refinement_factor, key_to_repeat = 'ax body')
train_x_past_acc_y = generate_tensor_past_actions(df, n_past_actions,refinement_factor, key_to_repeat = 'ay body')
train_x = torch.cat((train_x_wheel_forces_model, train_x_past_acc_x,train_x_past_acc_y), dim=1)

# -- Y lables --
# front_lat_force_mismatch_training = torch.unsqueeze(torch.tensor(Fy_front_base_model_vec - df['Fy front wheel'].to_numpy()),1).cuda() 
# rear_lat_force_mismatch_training = torch.unsqueeze(torch.tensor(Fy_rear_base_model_vec - df['Fy rear wheel'].to_numpy()),1).cuda() 
# train_y = torch.vstack((front_lat_force_mismatch_training, rear_lat_force_mismatch_training))
# measured wheel forces
columns_to_extract = ['Fy front wheel', 'Fy rear wheel']
train_y = torch.Tensor(df[columns_to_extract].to_numpy()).cuda().double()


#instantiate the model
dt_int_steering =  np.diff(df['vicon time'].to_numpy()).mean() / refinement_factor
pitch_and_roll_dynamics_model_obj = pitch_and_roll_dynamics_model(initial_guess,dt_int_steering,n_past_actions*refinement_factor,lr,lf)

#define loss and optimizer objects
loss_fn = torch.nn.MSELoss() 
optimizer_object = torch.optim.Adam(pitch_and_roll_dynamics_model_obj.parameters(), lr=learning_rate)

# save loss values for later plot
loss_vec = np.zeros(train_its)



from tqdm import tqdm
tqdm_obj = tqdm(range(0,train_its), desc="training", unit="iterations")


for i in tqdm_obj:
    # clear gradient information from previous step before re-evaluating it for the current iteration
    optimizer_object.zero_grad()  
    
    # compute fitting outcome with current model parameters
    F_y_front, F_y_rear, pitch_unscaled, roll_unscaled = pitch_and_roll_dynamics_model_obj(train_x)
    # stack vertically
    output = torch.cat([F_y_front.double(),F_y_rear.double()],1)

    loss = loss_fn(output,  train_y)
    loss_vec[i] = loss.item()

    # evaluate the gradient of the loss function with respect to the fitting parameters
    loss.backward() 

    # use the evaluated gradient to perform a gradient descent step 
    optimizer_object.step() # this updates parameters


# # # --- plot loss function ---
plt.figure()
plt.title('Loss function')
plt.plot(loss_vec,label='tire model loss')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.legend()



# --- print out parameters ---
[w_natural_Hz_pitch,k_f_pitch,k_r_pitch,w_natural_Hz_roll,k_f_roll,k_r_roll]= pitch_and_roll_dynamics_model_obj.transform_parameters_norm_2_real()
w_natural_Hz_pitch,k_f_pitch,k_r_pitch,w_natural_Hz_roll,k_f_roll,k_r_roll=  w_natural_Hz_pitch.item(),k_f_pitch.item(),k_r_pitch.item(),w_natural_Hz_roll.item(),k_f_roll.item(),k_r_roll.item()
print('')
print('')
print('# pitch dynamics parameters:')
print(f'w_natural_Hz_pitch = {w_natural_Hz_pitch}')
print(f'k_f_pitch = {k_f_pitch}')
print(f'k_r_pitch = {k_r_pitch}')

print('# roll dynamics parameters:')
print(f'w_natural_Hz_roll = {w_natural_Hz_roll}')
print(f'k_f_roll = {k_f_roll}')
print(f'k_r_roll = {k_r_roll}')



# plot results
ax_lat_force.plot(df['vicon time'].to_numpy(),F_y_front.detach().cpu().view(-1).numpy(),label='Fy front model fitted',color='teal')
ax_lat_force.plot(df['vicon time'].to_numpy(),F_y_rear.detach().cpu().view(-1).numpy(),label='Fy rear model fitted',color='green')
ax_lat_force.legend()



plt.figure()
#plot acc in body frame
plt.plot(df['vicon time'].to_numpy(),df['ax body'].to_numpy(),label='ax body',color='dodgerblue')
plt.plot(df['vicon time'].to_numpy(),pitch_unscaled.detach().cpu().view(-1).numpy(),label='pitch unscaled',color='blue')

plt.plot(df['vicon time'].to_numpy(),df['ay body'].to_numpy(),label='ax body',color='orangered')
plt.plot(df['vicon time'].to_numpy(),roll_unscaled.detach().cpu().view(-1).numpy(),label='roll unscaled',color='red')
plt.legend()




fig1, ((ax_lat_force_front,ax_lat_force_rear)) = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

ax_lat_force_front.plot(df['vicon time'].to_numpy(), df['Fy front wheel'].to_numpy(),label='Fy front measured',color = 'peru')
#ax_lat_force_front.plot(df['vicon time'].to_numpy(), df['ax body'].to_numpy(),label='longitudinal acceleration (with cent))',color = 'dodgerblue')
ax_lat_force_front.plot(df['vicon time'].to_numpy(),pitch_unscaled.detach().cpu().view(-1).numpy(),label='pitch unscaled',color='blue')
ax_lat_force_front.plot(df['vicon time'].to_numpy(),roll_unscaled.detach().cpu().view(-1).numpy(),label='roll unscaled',color='red')
ax_lat_force_front.plot(df['vicon time'].to_numpy(), wheel_slippage,label='longitudinal slippage',color = 'gray')
ax_lat_force_front.plot(df['vicon time'].to_numpy(),Fy_front_base_model_vec,label='Fy front model',color='k')
ax_lat_force_front.plot(df['vicon time'].to_numpy(),F_y_front.detach().cpu().view(-1).numpy(),label='Fy front model fitted',color='teal')


ax_lat_force_front.set_title('Front lateral forces')
ax_lat_force_front.set_xlabel('time [s]')
ax_lat_force_front.set_title('Lateral wheel forces')
ax_lat_force_front.legend()

# rear tire
ax_lat_force_rear.plot(df['vicon time'].to_numpy(), df['Fy rear wheel'].to_numpy(),label='Fy rear measured',color = 'darkred')
#ax_lat_force_rear.plot(df['vicon time'].to_numpy(), df['ay body'].to_numpy(),label='lateral acceleration (with cent)',color = 'orangered')
ax_lat_force_rear.plot(df['vicon time'].to_numpy(),roll_unscaled.detach().cpu().view(-1).numpy(),label='roll unscaled',color='red')
ax_lat_force_rear.plot(df['vicon time'].to_numpy(),pitch_unscaled.detach().cpu().view(-1).numpy(),label='pitch unscaled',color='blue')
ax_lat_force_rear.plot(df['vicon time'].to_numpy(), wheel_slippage,label='longitudinal slippage',color = 'gray')
ax_lat_force_rear.plot(df['vicon time'].to_numpy(),Fy_rear_base_model_vec,label='Fy rear model',color='k')
ax_lat_force_rear.plot(df['vicon time'].to_numpy(),F_y_rear.detach().cpu().view(-1).numpy(),label='Fy rear model fitted',color='teal')

ax_lat_force_rear.set_title('Rear lateral forces')
ax_lat_force_rear.set_xlabel('time [s]')
ax_lat_force_rear.set_title('Lateral wheel forces')
ax_lat_force_rear.legend()







# integrate the roll and pitch as you would in an MPC controller

# integrate pitch and roll dynamics
pitch_vec = np.zeros(df.shape[0])
roll_vec = np.zeros(df.shape[0])
pitch_dot_vec = np.zeros(df.shape[0])
roll_dot_vec = np.zeros(df.shape[0])

full_dynamic_model_obj = full_dynamic_model(lr, l_COM, Jz, lf, m,
            a_m, b_m, c_m, d_m,
            a_f, b_f, c_f, d_f,
            a_s, b_s, c_s, d_s, e_s,
            d_t, c_t, b_t,
            a_stfr, b_stfr,d_stfr,e_stfr,f_stfr,g_stfr,
            max_st_dot,fixed_delay_stdn,k_stdn,
            w_natural_Hz_pitch,k_f_pitch,k_r_pitch,
            w_natural_Hz_roll,k_f_roll,k_r_roll)


for i in range(1,df.shape[0]):
    dt = df['vicon time'].iloc[i] - df['vicon time'].iloc[i-1]
    # evaluate pitch dynamics
    pitch_dot_dot = full_dynamic_model_obj.critically_damped_2nd_order_dynamics(pitch_dot_vec[i-1],pitch_vec[i-1],df['ax body'].iloc[i-1],w_natural_Hz_pitch)
    
    # evaluate roll dynamics
    roll_dot_dot = full_dynamic_model_obj.critically_damped_2nd_order_dynamics(roll_dot_vec[i-1],roll_vec[i-1],df['ay body'].iloc[i-1],w_natural_Hz_roll)
    
    pitch_dot_vec[i] = pitch_dot_vec[i-1] + pitch_dot_dot*dt
    roll_dot_vec[i] = roll_dot_vec[i-1] + roll_dot_dot*dt

    pitch_vec[i] = pitch_vec[i-1] + pitch_dot_vec[i-1]*dt
    roll_vec[i] = roll_vec[i-1] + roll_dot_vec[i-1]*dt

# add columns to the data
df['pitch dot'] = pitch_dot_vec
df['pitch'] = pitch_vec

df['roll dot'] = roll_dot_vec
df['roll'] = roll_vec




fig1, ((ax_pitch,ax_roll)) = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

# plot pitch and pitch dot against ax body
ax_pitch.plot(df['vicon time'].to_numpy(),df['pitch'].to_numpy(),color='dodgerblue',label='pitch (from integrating acc x)',linewidth=4,linestyle='-')
ax_pitch.plot(df['vicon time'].to_numpy(),pitch_unscaled.detach().cpu().view(-1).numpy(),label='pitch unscaled (from model)',color='k')
ax_pitch.set_xlabel('Time [s]')
ax_pitch.set_ylabel('Pitch [rad]')
ax_pitch.legend()
ax_pitch.set_title('Pitch')


# plot roll and roll dot against ay body
ax_roll.plot(df['vicon time'].to_numpy(),df['roll'].to_numpy(),color='orangered',label='roll (from integrating acc y)',linewidth=4,linestyle='-')
ax_roll.plot(df['vicon time'].to_numpy(),roll_unscaled.detach().cpu().view(-1).numpy(),label='roll unscaled (from model)',color='k')
ax_roll.set_xlabel('Time [s]')
ax_roll.set_ylabel('Roll [rad]')
ax_roll.legend()
ax_roll.set_title('Roll')




plt.show()