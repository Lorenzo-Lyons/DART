from functions_for_data_processing import get_data, plot_raw_data, process_raw_vicon_data,plot_vicon_data,\
fullmodel_with_steering_dynamics_model,model_parameters,steering_dynamics_model,directly_measured_model_parameters
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




# select data folder NOTE: this assumes that the current directory is DART
#folder_path = 'System_identification_data_processing/Data/8_circles_rubbery_floor_1_file'
folder_path = 'System_identification_data_processing/Data/81_throttle_ramps'

#folder_path = 'System_identification_data_processing/Data/81_circles_tape_and_tiles'
#folder_path = 'System_identification_data_processing/Data/81_throttle_ramps_only_steer03'
#folder_path = 'System_identification_data_processing/Data/91_model_validation_long_term_predictions_fast'
#folder_path = 'System_identification_data_processing/Data/90_model_validation_long_term_predictions'

# Decide how many past steering signals to use. Note that this shold be enough to capture the dynamics of the system. 
# steering_time_window = 0.04  # [s] # this should be enough to capture the dynamics of the impulse response of the steering dynamics
# dt_steering = 0.001  # this small enough to limit the numerical error when solving the convolution integral in the steering dynamics model
#folder_path = 'System_identification_data_processing/Data/91_free_driving_16_sept_2024'
#folder_path = 'System_identification_data_processing/Data/91_free_driving_16_sept_2024_slow'




# define the number of past steering signals to use
n_past_steering = 50 #int(np.ceil(steering_time_window/T)) + 1
refinement_factor = 5 #int(np.ceil(T/dt_steering))




[theta_correction, l_COM, l_lateral_shift_reference ,lr, lf, Jz, m,m_front_wheel,m_rear_wheel] = directly_measured_model_parameters()

# load model parameters
[a_m, b_m, c_m, d_m,
a_f, b_f, c_f, d_f,
a_s, b_s, c_s, d_s, e_s,
d_t_f, c_t_f, b_t_f,d_t_r, c_t_r, b_t_r,
a_stfr, b_stfr,d_stfr,e_stfr,f_stfr,g_stfr,
max_st_dot,fixed_delay_stdn,k_stdn,
w_natural_Hz_pitch,k_f_pitch,k_r_pitch,
w_natural_Hz_roll,k_f_roll,k_r_roll] = model_parameters()


# decide if you want to tweak the steering curve
tweak_steering_curve = True



df_raw_data = get_data(folder_path)

# process the data
steps_shift = 3 # decide to filter more or less the vicon data
df = process_raw_vicon_data(df_raw_data,steps_shift)


# # Starting data processing
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


# define sampling time
T = df['vicon time'].diff().mean()  # Calculate the average time step




# reverse engineering the steering angle 
def fun(x,Fx,Fy,Vx,Vy,steering_angle_command, d_t ,c_t ,b_t):
    # x is the true steering angle
    R_tilde = np.array([[np.cos(steering_angle_command), -np.sin(steering_angle_command)],
                        [np.sin(steering_angle_command), +np.cos(steering_angle_command)]])
    
    F = np.array([[Fx],[Fy]])
    V = np.array([[Vx],[Vy]])
    projection_vec = np.array([-np.sin(x), np.cos(x)]).T
    vy_wheel = projection_vec @ V
    F_y_wheel = d_t * np.sin(c_t * np.arctan(b_t * vy_wheel[0][0] ))
    # measured lateral force
    measured = projection_vec @ R_tilde @ F
    return  measured[0][0] - F_y_wheel
    


true_steering_angle_vec = np.zeros(df.shape[0])
from scipy.optimize import fsolve

for i in range(0,df.shape[0]):
    # get the current data
    Vx = df['vx body'].iloc[i]
    if Vx > 0.1:
        Fx = df['Fx wheel'].iloc[i]
        Fy = df['Fy front wheel'].iloc[i]
        
        Vy = df['vy body'].iloc[i] + lf*df['w'].iloc[i]
        steering_angle_command = df['steering angle'].iloc[i]
        # solve the equation
        equation = lambda x: fun(x,Fx,Fy,Vx,Vy,steering_angle_command, d_t ,c_t ,b_t)
        
        # Solve the equation starting from an initial guess
        initial_guess = df['steering angle'].iloc[i]
        true_steering_angle = fsolve(equation, initial_guess)
        initial_guess = true_steering_angle # update initial guess
        true_steering_angle_vec[i] = true_steering_angle
    else:
        true_steering_angle_vec[i] = df['steering angle'].iloc[i] # just keep previous value



fig, ax_steering_angle = plt.subplots()
ax_steering_angle.plot(df['vicon time'],df['steering angle'],label='steering angle',color='purple')
ax_steering_angle.plot(df['vicon time'],true_steering_angle_vec,label='true steering angle',color='dodgerblue')
ax_steering_angle.set_xlabel('Time [s]')
ax_steering_angle.set_ylabel('Steering angle')






# using optuna to find a simple steering model using a fixed time delay and a maximum slope
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Define the objective function for Optuna
def objective(trial):
    # Suggest values for parameters 'a' and 'b'
    max_st_dot = trial.suggest_float('max_st_dot', 0, 10) # /s
    gain = trial.suggest_float('gain', 0, 1) # /s
    delay = trial.suggest_float('fixed_delay', 0, 20) # timesteps


    # convert to int the value of the fixed timedelay
    delay_int = int(np.round(delay))
    # evaluate shifted steering signal
    steering_time_shifted = df['steering'].shift(delay_int, fill_value=0)
    
    # Predict y based on the current values of 'a' and 'b'
    st = 0
    st_vec_angle = np.zeros(len(true_steering_angle_vec))
    # to interpolate the steering signal

    for k in range(1,len(st_vec_angle)):
        st_dot = (steering_time_shifted[k-1]-st)/T * gain # first order system
        st_dot = np.min([st_dot,max_st_dot])
        st_dot = np.max([st_dot,-max_st_dot])
        st += st_dot * T
        # convert to steerign angle
        w_s = 0.5 * (np.tanh(30*(st+c_s))+1)
        steering_angle1 = b_s * np.tanh(a_s * (st + c_s)) 
        steering_angle2 = d_s * np.tanh(e_s * (st + c_s)) 
        steering_angle = (w_s)*steering_angle1+(1-w_s)*steering_angle2

        st_vec_angle[k] = steering_angle
    
    # Calculate the mean squared error (least squares error)
    mse = mean_squared_error(true_steering_angle_vec, st_vec_angle)
    
    return mse

# Create a study and optimize
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# Get the best parameters
best_max_st_dot = study.best_trial.params['max_st_dot']
best_fixed_delay = study.best_trial.params['fixed_delay']
best_gain = study.best_trial.params['gain']


# re-run the model to get the plot of the best prediction

# Convert the best fixed_delay to an integer
best_delay_int = int(np.round(best_fixed_delay))

# Evaluate the shifted steering signal using the best fixed delay
steering_time_shifted = df['steering'].shift(best_delay_int, fill_value=0).to_numpy()

# Initialize variables for the steering prediction
st = 0
st_vec_angle_optuna = np.zeros(df.shape[0])

# Loop through the data to compute the predicted steering angles
for k in range(1, len(true_steering_angle_vec)):
    # Calculate the rate of change of steering (steering dot)
    st_dot = (steering_time_shifted[k-1] - st) / T * best_gain
    # Apply max_st_dot limits
    st_dot = np.min([st_dot, best_max_st_dot])
    st_dot = np.max([st_dot, -best_max_st_dot])
    
    # Update the steering value with the time step
    st += st_dot * T
    
    # Compute the steering angle using the two models with weights
    w_s = 0.5 * (np.tanh(30 * (st + c_s)) + 1)
    steering_angle1 = b_s * np.tanh(a_s * (st + c_s))
    steering_angle2 = d_s * np.tanh(e_s * (st + c_s))
    
    # Combine the two steering angles using the weight
    steering_angle = (w_s) * steering_angle1 + (1 - w_s) * steering_angle2
    
    # Store the predicted steering angle
    st_vec_angle_optuna[k] = steering_angle

# Now `st_vec_angle` contains the predicted steering angles with the best parameters
#print(f"Predicted steering angles: {st_vec_angle_optuna}")



ax_steering_angle.plot(df['vicon time'].to_numpy(),st_vec_angle_optuna,label='steering angle optuna',color='maroon')
ax_steering_angle.legend()


# print out the best parameters
print('max_st_dot =',best_max_st_dot)
print('fixed_delay_stdn =',best_fixed_delay)
print('k_stdn =',best_gain)


plt.show()






# --- fit steering dynamics ----



def generate_tensor(df, n_past_steering,refinement_factor,key_to_repeat):
    # due to numerical errors in evauating the convolution integral we need a finer resolution for the time step on the steering
    df_past_steering = pd.DataFrame()
    df_past_steering[key_to_repeat] = df[key_to_repeat]
    # Add delayed steering signals based on user input
    for i in range(0, n_past_steering):
        df_past_steering[key_to_repeat + f' prev{i}'] = df[key_to_repeat].shift(i, fill_value=0)
    # doing a zero order hold on the steering signal to get  a finer resolution
    df_past_steering_refined = pd.DataFrame()
    for i in range(0, (n_past_steering)):
        for k in range(refinement_factor):
            df_past_steering_refined[key_to_repeat + f' prev{i*refinement_factor+k}'] = df_past_steering[key_to_repeat + f' prev{i}']


    # Select columns for generating tensor

    selected_columns_df_steering = [key_to_repeat + f' prev{i}' for i in range(0, (n_past_steering)*refinement_factor)]
    
    # Convert the selected columns into a tensor and send to GPU (if available)
    
    train_x_df_steering = torch.tensor(df_past_steering_refined[selected_columns_df_steering].to_numpy()).cuda()

    
    return train_x_df_steering























# # NOTE this is a bit of a hack
# T = df['vicon time'].diff().mean()  # Calculate the average time step
# # forard integrate the steering signal
# st_vec = np.zeros(df.shape[0])
# st_dot = 0

# #saturate the steering rate
# max_s_dot = 18 # steering/sec
# #max_s_ddot = 60 # steering/sec^2

# # Make the steering reference into a ramp (we simulate a maximum steering rate)
# for i in range(1, df.shape[0]):

#     # just give a ramp
#     st_dot =(df['steering'].iloc[i-1]-st_vec[i-1]) / T
#     st_dot = np.min([st_dot,max_s_dot])
#     st_dot = np.max([st_dot,-max_s_dot])

#     st_vec[i] = st_vec[i-1] + T * st_dot

# df['steering'] = st_vec # substitute the original steering signal with the ramped one









# add filtered throttle

# Filter coefficient in the new sampling frequency

d_m_100Hz = 0.01/(0.01+(0.1/d_m-0.1)) #convert to new sampling frequency

# Initialize the filtered steering angle list
filtered_throttle = [df['throttle'].iloc[0]]
# Apply the first-order filter
for i in range(1, len(df)):
    filtered_value = d_m_100Hz * df['throttle'].iloc[i] + (1 - d_m_100Hz) * filtered_throttle[-1]
    filtered_throttle.append(filtered_value)

df['throttle filtered'] = filtered_throttle



n_past_steering_refined = n_past_steering * refinement_factor
dt_int_steering =  T / refinement_factor

# plot raw data
ax0,ax1,ax2 = plot_raw_data(df)

# plot vicon related data (longitudinal and lateral velocities, yaw rate related)
ax_wheels = plot_vicon_data(df)












# --------------- fitting model---------------

# fitting tyre models
# define first guess for parameters
initial_guess = torch.ones(8) * 0.5 # initialize parameters in the middle of their range constraint
initial_guess[2] = 0.1  # low delay initial guess

# define number of training iterations
train_its = 100
learning_rate = 0.01

print('')
print('Fitting steering dynamics by looking at the reconstructed steering angle')

#instantiate the model
#dt = np.diff(df['vicon time'].to_numpy()).mean()

dt_int_steering =  T / refinement_factor
steering_dynamics_model_obj = steering_dynamics_model(initial_guess,dt_int_steering,tweak_steering_curve)

#define loss and optimizer objects
loss_fn = torch.nn.MSELoss() 
optimizer_object = torch.optim.Adam(steering_dynamics_model_obj.parameters(), lr=learning_rate)


# generate data in tensor form for torch
key_to_repeat = 'steering'
train_x_df_steering = generate_tensor(df, n_past_steering,refinement_factor, key_to_repeat)

# y labels are the measured accelerations in the body frame
train_y = torch.unsqueeze(torch.tensor(true_steering_angle_vec),1).cuda() # acc_x_body_measured,


# save loss values for later plot
loss_vec = np.zeros(train_its)



from tqdm import tqdm
tqdm_obj = tqdm(range(0,train_its), desc="training", unit="iterations")


for i in tqdm_obj:
    # clear gradient information from previous step before re-evaluating it for the current iteration
    optimizer_object.zero_grad()  
    
    # compute fitting outcome with current model parameters
    output = steering_dynamics_model_obj(train_x_df_steering)


    loss = loss_fn(output,  train_y)
    loss_vec[i] = loss.item()

    # evaluate the gradient of the loss function with respect to the fitting parameters
    loss.backward() 

    # use the evaluated gradient to perform a gradient descent step 
    optimizer_object.step() # this updates parameters


# --- print out parameters ---
[damping,w_natural,fixed_delay,a_s,b_s,c_s,d_s,e_s]= steering_dynamics_model_obj.transform_parameters_norm_2_real()
damping,w_natural,fixed_delay,a_s,b_s,c_s,d_s,e_s=  damping.item(), w_natural.item() ,fixed_delay.item(),a_s.item(),b_s.item(),c_s.item(),d_s.item(),e_s.item()


print('steering dynamics parameters from direct steering angle matching:')
print('damping = ', damping)
print('w_natural_Hz = ', w_natural, '   # Hz')
print('fixed_delay = ', fixed_delay)

print('steering curve parameters')
print('a_s = ', a_s)
print('b_s = ', b_s)
print('c_s = ', c_s)
print('d_s = ', d_s)
print('e_s = ', e_s)




ax_steering_angle.plot(df['vicon time'].to_numpy(),output.detach().cpu().view(-1).numpy(),label='steering angle model',color='k')
ax_steering_angle.legend()
plt.show()






















# --- produce and plot data to train the model ---
# these are the measured acceleration in the body frame, i.e. the same ones that the model should predict

acc_centrifugal_in_x = df['vy body'].to_numpy() * df['w'].to_numpy()
acc_centrifugal_in_y = - df['vx body'].to_numpy() * df['w'].to_numpy()
# accelerations in the body frame
acc_x_body_measured = df['ax body no centrifugal'].to_numpy() + acc_centrifugal_in_x
acc_y_body_measured = df['ay body no centrifugal'].to_numpy() + acc_centrifugal_in_y


# plot the modelled acceleration
fig, ax_accx = plt.subplots()
ax_accx.plot(df['vicon time'].to_numpy(),acc_x_body_measured,label='acc_x body frame',color='dodgerblue')
ax_accx.set_xlabel('Time [s]')
ax_accx.set_ylabel('Acceleration x')

# y accelerations
fig, ax_accy = plt.subplots()
ax_accy.plot(df['vicon time'].to_numpy(),acc_y_body_measured,label='acc_y in the body frame',color='orangered')
ax_accy.set_xlabel('Time [s]')
ax_accy.set_ylabel('Acceleration y')

# w accelerations
fig, ax_accw = plt.subplots()
ax_accw.plot(df['vicon time'].to_numpy(),df['aw_more'].to_numpy(),label='acc_w',color='purple')
ax_accw.set_xlabel('Time [s]')
ax_accw.set_ylabel('Acceleration w')
















# --------------- fitting model---------------
# fitting tyre models
# define first guess for parameters
initial_guess = torch.ones(11) * 0.5 # initialize parameters in the middle of their range constraint
#initial_guess[2] = 0.1

# define number of training iterations
train_its = 100
learning_rate = 0.01

print('')
print('Fitting steering dynamics')

#instantiate the model
#dt = np.diff(df['vicon time'].to_numpy()).mean()
fullmodel_with_steering_dynamics_model_obj = fullmodel_with_steering_dynamics_model(initial_guess,(n_past_steering)*refinement_factor,dt_int_steering)

#define loss and optimizer objects
loss_fn = torch.nn.MSELoss() 
optimizer_object = torch.optim.Adam(fullmodel_with_steering_dynamics_model_obj.parameters(), lr=learning_rate)



#generate data in tensor form for torch
#train_x = torch.unsqueeze(torch.tensor(np.concatenate((df['V_y front wheel'].to_numpy(),df['V_y rear wheel'].to_numpy()))),1).cuda()
selected_columns_df = ['vx body', 'vy body', 'w','throttle filtered'] 
train_x_df = torch.tensor(df[selected_columns_df].to_numpy()).cuda()


# generate data in tensor form for torch
#train_x_df, train_x_df_steering = generate_tensor(df, n_past_steering,refinement_factor) 
train_x = torch.cat((train_x_df, train_x_df_steering), dim=1)
# y labels are the measured accelerations in the body frame
train_y = torch.unsqueeze(torch.tensor(np.concatenate((acc_y_body_measured,df['aw_more'].to_numpy()))),1).cuda() # acc_x_body_measured,
#train_y = torch.unsqueeze(torch.tensor((df['aw_more'].to_numpy())),1).cuda() # acc_x_body_measured,


# save loss values for later plot
loss_vec = np.zeros(train_its)



from tqdm import tqdm
tqdm_obj = tqdm(range(0,train_its), desc="training", unit="iterations")

max_accw = np.abs(df['aw_more'].to_numpy()).max()
max_accy = np.abs(acc_y_body_measured).max()

for i in tqdm_obj:
    # clear gradient information from previous step before re-evaluating it for the current iteration
    optimizer_object.zero_grad()  
    
    # compute fitting outcome with current model parameters
    acc_x,acc_y,acc_w, steering_angle = fullmodel_with_steering_dynamics_model_obj(train_x)
    acc_output = torch.vstack([acc_y/max_accy,acc_w/max_accw]) # acc_x
    #acc_output = acc_w
    # evaluate loss function
    loss = loss_fn(acc_output,  train_y)
    loss_vec[i] = loss.item()

    # evaluate the gradient of the loss function with respect to the fitting parameters
    loss.backward() 

    # use the evaluated gradient to perform a gradient descent step 
    optimizer_object.step() # this updates parameters


# --- print out parameters ---
[damping,w_natural,fixed_delay,a_s,b_s,c_s,d_s,e_s,d_t,c_t,b_t]= fullmodel_with_steering_dynamics_model_obj.transform_parameters_norm_2_real()
damping, w_natural,fixed_delay,a_s,b_s,c_s,d_s,e_s,d_t,c_t,b_t =  damping.item(), w_natural.item() ,fixed_delay.item(),a_s.item(),b_s.item(),c_s.item(),d_s.item(),e_s.item(),\
                                                                d_t.item(),c_t.item(),b_t.item()


print('steering dynamics parameters:')
print('damping = ', damping)
print('w_natural = ', w_natural)
print('fixed_delay = ', fixed_delay)

print('steering curve parameters')
print('a_s = ', a_s)
print('b_s = ', b_s)
print('c_s = ', c_s)
print('d_s = ', d_s)
print('e_s = ', e_s)

print('tire model parameters')
print('d_t = ', d_t)
print('c_t = ', c_t)
print('b_t = ', b_t)




# # # --- plot loss function ---
plt.figure()
plt.title('Loss function')
plt.plot(loss_vec,label='tire model loss')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.legend()



# plot fitting results
ax_accx.plot(df['vicon time'].to_numpy(),acc_x.detach().cpu().view(-1).numpy(),label='acc_x model with steering dynamics (model output)',color='k',alpha=0.5)
ax_accx.legend()

ax_accy.plot(df['vicon time'].to_numpy(),acc_y.detach().cpu().view(-1).numpy(),label='acc_y model with steering dynamics (model output)',color='k',alpha=0.5)
ax_accy.legend()

ax_accw.plot(df['vicon time'].to_numpy(),acc_w.detach().cpu().view(-1).numpy(),label='acc_w model with steering dynamics (model output)',color='k',alpha=0.5)
ax_accw.legend()






# plot the steering delay
k_vec = fullmodel_with_steering_dynamics_model_obj.produce_past_action_coefficients(torch.Tensor([damping]).cuda(),torch.Tensor([w_natural]).cuda(),torch.Tensor([fixed_delay]).cuda()).cpu().view(-1).numpy()    
t_vec = np.linspace(0,dt_int_steering*(n_past_steering_refined-1),n_past_steering_refined)

plt.figure()
plt.plot(t_vec,k_vec,label='k_vec',color='gray')
plt.plot(df['vicon time'].to_numpy(),df['steering angle'].to_numpy(),label='steering angle',color='purple')
ax_steering_angle.plot(df['vicon time'].to_numpy(),steering_angle.detach().cpu().view(-1).numpy(),label='steering angle model',color='k')
plt.xlabel('Time [s]')
plt.ylabel('Steering angle')
plt.legend()



# visualize the wheel data using the steering dynamics
df_raw_data_steering_dynamics = get_data(folder_path)
df_raw_data_steering_dynamics['steering angle'] = steering_angle.detach().cpu().view(-1).numpy()
# providing the steering  will make processing data use that instead of recovering it from the raw data
# t_min = 2.8 # 2.8
# t_max = 3.4 #3.4
# df_raw_data_steering_dynamics = df_raw_data_steering_dynamics[df_raw_data_steering_dynamics['vicon time']>t_min]
# df_raw_data_steering_dynamics = df_raw_data_steering_dynamics[df_raw_data_steering_dynamics['vicon time']<t_max]
df_steering_dynamics = process_raw_vicon_data(df_raw_data_steering_dynamics)


v_y_wheel_plotting = torch.unsqueeze(torch.linspace(-1.5,1.5,100),1).cuda()
lateral_force_vec = fullmodel_with_steering_dynamics_model_obj.F_y_wheel_model(v_y_wheel_plotting,d_t,c_t,b_t).detach().cpu().numpy()
ax_wheels.scatter(df_steering_dynamics['V_y front wheel'],df_steering_dynamics['Fy front wheel'],color='skyblue',label='front wheel data',s=6)
#ax_wheels.scatter(df_steering_dynamics['V_y rear wheel'],df_steering_dynamics['Fy rear wheel'],color='teal',label='rear wheel data',s=3)
ax_wheels.plot(v_y_wheel_plotting.detach().cpu().numpy(),lateral_force_vec,color='#2c4251',label='Tire model',linewidth=4,linestyle='-')

ax_wheels.legend()



plt.figure()
steering_input = np.linspace(-1,1,100)
# model curve
w_s = 0.5 * (np.tanh(30*(steering_input+c_s))+1)
steering_angle1_model = b_s * np.tanh(a_s * (steering_input + c_s)) 
steering_angle2_model = d_s * np.tanh(e_s * (steering_input + c_s)) 
steering_angle_model = (w_s)*steering_angle1_model+(1-w_s)*steering_angle2_model

# historical data from on board sensors (and different floor)
a_s_original =  1.6379064321517944
b_s_original =  0.3301370143890381 
c_s_original =  0.019644200801849365 
d_s_original =  0.37879398465156555 
e_s_original =  1.6578725576400757

w_s_original = 0.5 * (np.tanh(30*(steering_input+c_s_original))+1)
steering_angle1_original = b_s_original * np.tanh(a_s_original * (steering_input + c_s_original)) 
steering_angle2_original = d_s_original * np.tanh(e_s_original * (steering_input + c_s_original)) 
steering_angle_original = (w_s_original)*steering_angle1_original+(1-w_s_original)*steering_angle2_original

plt.plot(steering_input,steering_angle_model,label='steering angle model',color='k')
plt.plot(steering_input,steering_angle_original,label='steering angle original',color='orange')
plt.scatter(np.array([0.0]),np.array([0.0]),color='orangered',label='zero',marker='+', zorder=20) # plot zero as an x 
plt.legend()





plt.show()































# columns_to_extract = ['vicon time', 'vx body', 'vy body', 'w', 'throttle' ,'steering','vicon x','vicon y','vicon yaw']
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

