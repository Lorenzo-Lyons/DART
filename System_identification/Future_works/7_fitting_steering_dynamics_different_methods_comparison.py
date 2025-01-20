from dart_dynamic_models import get_data, plot_raw_data, process_raw_vicon_data,plot_vicon_data,\
model_parameters,\
process_vicon_data_kinematics,model_functions,steering_dynamics_model_NN,generate_tensor_past_actions,steering_dynamics_model_first_order
from matplotlib import pyplot as plt
import torch
import numpy as np
import pandas as pd
from scipy import interpolate
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
#folder_path = 'System_identification_data_processing/Data/81_throttle_ramps'

#folder_path = 'System_identification_data_processing/Data/81_circles_tape_and_tiles'
#folder_path = 'System_identification_data_processing/Data/81_throttle_ramps_only_steer03'
#folder_path = 'System_identification_data_processing/Data/91_model_validation_long_term_predictions_fast'
#folder_path = 'System_identification_data_processing/Data/90_model_validation_long_term_predictions'

# Decide how many past steering signals to use. Note that this shold be enough to capture the dynamics of the system. 
# steering_time_window = 0.04  # [s] # this should be enough to capture the dynamics of the impulse response of the steering dynamics
# dt_steering = 0.001  # this small enough to limit the numerical error when solving the convolution integral in the steering dynamics model
#folder_path = 'System_identification_data_processing/Data/91_free_driving_16_sept_2024'
folder_path = 'System_identification_data_processing/Data/91_free_driving_16_sept_2024_slow'
#folder_path = 'System_identification_data_processing/Data/steering_identification_25_sept_2024'



# define the number of past steering signals to use
n_past_steering = 50 #int(np.ceil(steering_time_window/T)) + 1
refinement_factor = 5 #int(np.ceil(T/dt_steering))


mf = model_functions()



# process the data
steps_shift = 5 # decide to filter more or less the vicon data
df_raw_data = get_data(folder_path)

# select low acceleration data
#df_raw_data = df_raw_data[df_raw_data['vicon time']<67]





df_kinematics = process_vicon_data_kinematics(df_raw_data,steps_shift)
df = process_raw_vicon_data(df_kinematics,steps_shift)


# manually shift the steering by 1 vehicle control loop (0.1s)
steering_time_shifted = df['steering'].shift(0, fill_value=0).to_numpy()
# overwrite the steering signal
df['steering time shifted'] = steering_time_shifted

df = df[df['vx body']>0.2] # remove low velocity points that give undefined slip angles
















# plot raw data
ax0,ax1,ax2 = plot_raw_data(df)


ax_wheel_f_alpha,ax_wheel_r_alpha,ax_total_force_front,\
ax_total_force_rear,ax_lat_force,ax_long_force,\
ax_acc_x_body,ax_acc_y_body,ax_acc_w = plot_vicon_data(df) 


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
#     df = process_raw_vicon_data(df_raw_data,steps_shift)

#     df.to_csv(file_path, index=False)
#     print(f"File '{file_path}' saved.")
# else:
#     print(f"File '{file_path}' already exists, loading data.")
#     df = pd.read_csv(file_path)





# reverse engineer the true steering angle using the assumption that the tire forces should be on previously identified
# tire model. This is done by interpolating from the value of the measured force what the steering angle should be to have that force.

# this is just needed to recreate the tire curve
model_functions_obj = model_functions()


alpha_range = np.linspace(-1,1,400)
tire_curve_F = model_functions_obj.lateral_tire_force(alpha_range,mf.d_t_f_self,mf.c_t_f_self,mf.b_t_f_self,mf.m_front_wheel_self)
inverted_tire_model = interpolate.interp1d(tire_curve_F, alpha_range,bounds_error=False, fill_value=0.0)
Reconstructed_slip_angle = inverted_tire_model(df['Fy front wheel'].to_numpy())



# plot slip angle vs reconstructed steering angle
plt.figure()
plt.plot(df['vicon time'].to_numpy(),df['slip angle front'].to_numpy(),label='slip angle',color='purple')
plt.plot(df['vicon time'].to_numpy(),Reconstructed_slip_angle,label='Reconstructed slip angle',color='k')
plt.xlabel('Time [s]')
plt.ylabel('Steering angle')
plt.title('Front slip angle, measured vs reconstructed from tire model')

plt.legend()


# from the slip angle reverse engineer the steering angle
def fun(x,Vx,Vylfw,slip_angle_reconstructed):
    # x is the true steering angle
    steer_angle = x[0]
    R_tilde = np.array([[np.cos(-steer_angle), -np.sin(-steer_angle)],
                        [np.sin(-steer_angle), +np.cos(-steer_angle)]])
    
    V = np.array([[Vx],[Vylfw]])
    V_wheel = R_tilde @ V

    slip_angle = np.arctan2(V_wheel[1][0],V_wheel[0][0])
    return slip_angle - slip_angle_reconstructed # the solver will try to get this value to 0



true_steering_angle_vec = np.zeros(df.shape[0])
from scipy.optimize import fsolve

for i in range(0,df.shape[0]):
    # get the current data
    Vx = df['vx body'].iloc[i]
    Vylfw = df['vy body'].iloc[i] + mf.lf_self*df['w'].iloc[i]
    slip_angle_reconstructed = Reconstructed_slip_angle[i]


    equation = lambda x: fun(x,Vx,Vylfw,slip_angle_reconstructed)
    
    # Solve the equation starting from an initial guess
    initial_guess = df['steering angle'].iloc[i] # initial guess is the steering angle command
    true_steering_angle = fsolve(equation, initial_guess)
    initial_guess = true_steering_angle # update initial guess
    true_steering_angle_vec[i] = true_steering_angle

fig, ax_steering_angle = plt.subplots()
ax_steering_angle.plot(df['vicon time'].to_numpy(),df['steering angle'].to_numpy(),label='steering angle',color='navy',linestyle='--')
ax_steering_angle.plot(df['vicon time'].to_numpy(),true_steering_angle_vec,label='true steering angle',color='dodgerblue',linewidth=3)
ax_steering_angle.set_xlabel('Time [s]')
ax_steering_angle.set_ylabel('Steering angle')
ax_steering_angle.legend()





# using a linear layer in an NN with the past steering signals as input to predict the steering angle
n_past_actions = 50 # 50
refinement_factor = 10
dt =  np.diff(df['vicon time'].to_numpy()).mean() / refinement_factor

train_x = generate_tensor_past_actions(df, n_past_actions,refinement_factor, key_to_repeat = 'steering angle')
# convert to float
train_x = train_x.float()
train_y = torch.unsqueeze(torch.tensor(true_steering_angle_vec),1).cuda().float()


input_size = train_x.shape[1]
output_size = 1

steering_dynamics_model_NN_obj = steering_dynamics_model_NN(input_size, output_size)
steering_dynamics_model_NN_obj.cuda()



# define number of training iterations
train_its = 1000
learning_rate = 0.001

print('')
print('Fitting steering dynamics by looking at the reconstructed steering angle NN')


#define loss and optimizer objects
loss_fn = torch.nn.MSELoss() 
optimizer_object = torch.optim.Adam(steering_dynamics_model_NN_obj.parameters(), lr=learning_rate)


# save loss values for later plot
loss_vec = np.zeros(train_its)

from tqdm import tqdm
tqdm_obj = tqdm(range(0,train_its), desc="training", unit="iterations")

for i in tqdm_obj:
    # clear gradient information from previous step before re-evaluating it for the current iteration
    optimizer_object.zero_grad()  
    
    # compute fitting outcome with current model parameters
    output_NN = steering_dynamics_model_NN_obj(train_x)

    loss = loss_fn(output_NN,  train_y)
    loss_vec[i] = loss.item()

    # evaluate the gradient of the loss function with respect to the fitting parameters
    loss.backward() 

    # use the evaluated gradient to perform a gradient descent step 
    optimizer_object.step() # this updates parameters


# # # --- plot loss function ---
plt.figure()
plt.title('Loss function NN')
plt.plot(loss_vec,label='tire model loss')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.legend()


# plot fitting results
ax_steering_angle.plot(df['vicon time'].to_numpy(),output_NN.detach().cpu().numpy(),label='model output NN',color='k',linewidth=3)
ax_steering_angle.legend()


# plotting weight distributions
weights = steering_dynamics_model_NN_obj.linear_layer.weight.detach().cpu().numpy()
plt.figure()
plt.plot(weights[0,:])
plt.xlabel('Past steering signals')
plt.ylabel('Weight value')
plt.title('Weight distribution of the linear layer in the steering dynamics model NN')






# fitting the same data with a first order dynamics satureated model
steering_dynamics_model_modulated_first_order_obj = steering_dynamics_model_first_order(n_past_actions * refinement_factor,dt)


# define number of training iterations
train_its = 300
learning_rate = 0.001

print('')
print('Fitting steering dynamics by looking at the reconstructed steering angle with first order dynamics')


#define loss and optimizer objects
loss_fn = torch.nn.MSELoss() 
optimizer_object = torch.optim.Adam(steering_dynamics_model_modulated_first_order_obj.parameters(), lr=learning_rate)


# convert to double again
#train_x = train_x.double()
train_x = generate_tensor_past_actions(df, n_past_actions,refinement_factor, key_to_repeat = 'steering').double()
train_y = train_y.double()


# save loss values for later plot
loss_vec = np.zeros(train_its)

from tqdm import tqdm
tqdm_obj = tqdm(range(0,train_its), desc="training", unit="iterations")

for i in tqdm_obj:
    # clear gradient information from previous step before re-evaluating it for the current iteration
    optimizer_object.zero_grad()  
    
    # compute fitting outcome with current model parameters
    output_1st_order = steering_dynamics_model_modulated_first_order_obj(train_x)

    loss = loss_fn(output_1st_order,  train_y)
    loss_vec[i] = loss.item()

    # evaluate the gradient of the loss function with respect to the fitting parameters
    loss.backward() 

    # use the evaluated gradient to perform a gradient descent step 
    optimizer_object.step() # this updates parameters


# print out parameters
[k] = steering_dynamics_model_modulated_first_order_obj.transform_parameters_norm_2_real()

# print parameters
print('First order integrator parameter:')
print('k = ', k.item())






# # # --- plot loss function ---
plt.figure()
plt.title('Loss function 1st order low pass')
plt.plot(loss_vec)
plt.xlabel('iterations')
plt.ylabel('loss')


# plot fitting results
ax_steering_angle.plot(df['vicon time'].to_numpy(),output_1st_order.detach().cpu().numpy(),label='model output low pass filter on st',color='orangered',linewidth=3)
ax_steering_angle.legend()
# Forwards integrate the steering angle to check that the model is working as expected
# Initialize variables for the steering prediction
st = 0
st_vec_angle_forward_integrated = np.zeros(df.shape[0])

# Loop through the data to compute the predicted steering angles
for t in range(1, len(true_steering_angle_vec)):
    # Calculate the rate of change of steering (steering dot)
    st_dot = k.item() * (df['steering'].iloc[t-1] - st) / dt

    # Update the steering value with the time step
    st += st_dot * dt
    
    # Store the predicted steering angle
    st_vec_angle_forward_integrated[t] = model_functions_obj.steering_2_steering_angle(st,mf.a_s_self,mf.b_s_self,mf.c_s_self,mf.d_s_self,mf.e_s_self)

ax_steering_angle.plot(df['vicon time'].to_numpy(),st_vec_angle_forward_integrated,label='forwards integrated 1st order',color='navy',linewidth=3,linestyle='--')
ax_steering_angle.legend()









# fitting with optuna to mimic decresing reactiveness when steering angle is close to target
#model_functions_obj = model_functions()
steering = df['steering'].to_numpy()  

# using optuna to find a simple steering model using a fixed time delay and a maximum slope
import optuna
from sklearn.metrics import mean_squared_error


def generate_input_optuna(steering_command,st,a,b,c,d,f):
    Dst = np.tanh(steering_command-st)

    # slow down convergence when steering input is small
    low_steering_coeff = d + f * np.tanh((steering_command/c)**4)

    input = a * np.tanh((Dst/b)) * low_steering_coeff
    return input






# Define the objective function for Optuna
# define sampling time
T = df['vicon time'].diff().mean()  # Calculate the average time step
def objective(trial):
    # Suggest values for parameters 'a' and 'b'
    C = trial.suggest_float('C', 0, 1) # /s
    a = trial.suggest_float('a', 0, 1) # /s
    b = trial.suggest_float('b', 0, 1) # timesteps
    c = trial.suggest_float('c', 0, 1) # timesteps
    d = trial.suggest_float('d', 0, 1) # timesteps
    f = trial.suggest_float('f', 0, 1) # timesteps


    # Predict y based on the current values of 'a' and 'b'
    st = 0
    st_vec_angle = np.zeros(len(true_steering_angle_vec))
    # to interpolate the steering signal

    for k in range(1,len(st_vec_angle)):
        # input to the model
        input = generate_input_optuna(steering[k-1],st,a,b,c,d,f)

        st_dot = input/C # first order system derivative

        st += st_dot * T

        steering_angle_optuna = model_functions_obj.steering_2_steering_angle(st,mf.a_s_self,mf.b_s_self,mf.c_s_self,mf.d_s_self,mf.e_s_self)

        st_vec_angle[k] = steering_angle_optuna
    

    # Calculate the mean squared error (least squares error)
    mse = mean_squared_error(true_steering_angle_vec, st_vec_angle)
    
    return mse

# Create a study and optimize
print('')
print('Fitting steering dynamics by looking at the reconstructed steering angle with optuna')
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200)

# Get the best parameters
best_a = study.best_trial.params['a']
best_b = study.best_trial.params['b']
best_c = study.best_trial.params['c']
best_f = study.best_trial.params['f']
best_d = study.best_trial.params['d']

best_C = study.best_trial.params['C']






# print out optuna parameters
print('Fitted optuna parameters:')
print('a =',best_a)
print('b =',best_b)
print('c =',best_c)
print('d =',best_d)
print('f =',best_f)

print('C =',best_C)




# now use the optimizzed parameters to forwar integrate the steering angle 

# Initialize variables for the steering prediction
st = 0
st_vec_angle_optuna = np.zeros(df.shape[0])

# Loop through the data to compute the predicted steering angles
for k in range(1, len(true_steering_angle_vec)):

    input = generate_input_optuna(steering[k-1],st,best_a,best_b,best_c,best_d,best_f)

    st_dot = input/best_C # first order system derivative

    st += st_dot * T

    steering_angle_optuna = model_functions_obj.steering_2_steering_angle(st,mf.a_s_self,mf.b_s_self,mf.c_s_self,mf.d_s_self,mf.e_s_self)

    st_vec_angle_optuna[k] = steering_angle_optuna



ax_steering_angle.plot(df['vicon time'].to_numpy(),st_vec_angle_optuna,label='steering angle optuna',color='peru',linewidth=3)
ax_steering_angle.legend()






print('')
print('Fitting steering dynamics using optuna with saturated speed and a fixed time delay')

# Define the objective function for Optuna
def objective(trial):
    # Suggest values for parameters 'a' and 'b'
    max_st_dot = trial.suggest_float('max_st_dot', 0, 10) # /s
    gain = trial.suggest_float('gain', 0, 1) # /s
    delay = trial.suggest_float('fixed_delay', 0, 20) # timesteps


    # convert to int the value of the fixed timedelay
    delay_int = int(np.round(delay))
    # evaluate shifted steering signal
    steering_time_shifted = df['steering'].shift(delay_int, fill_value=0).to_numpy()
    
    # Predict y based on the current values of the parameters
    st = 0
    st_vec_angle = np.zeros(len(true_steering_angle_vec))
    # to interpolate the steering signal

    for k in range(1,len(st_vec_angle)):
        st_dot = (steering_time_shifted[k-1]-st)/T * gain # first order system
        st_dot = np.min([st_dot,max_st_dot])
        st_dot = np.max([st_dot,-max_st_dot])
        st += st_dot * T

        # convert to steerign angle
        steering_angle_optuna = model_functions_obj.steering_2_steering_angle(st,mf.a_s_self,mf.b_s_self,mf.c_s_self,mf.d_s_self,mf.e_s_self)

        st_vec_angle[k] = steering_angle_optuna
    
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
    steering_angle_optuna = model_functions_obj.steering_2_steering_angle(st,mf.a_s_self,mf.b_s_self,mf.c_s_self,mf.d_s_self,mf.e_s_self)
    
    # Store the predicted steering angle
    st_vec_angle_optuna[k] = steering_angle_optuna

# Now `st_vec_angle` contains the predicted steering angles with the best parameters
#print(f"Predicted steering angles: {st_vec_angle_optuna}")


ax_steering_angle.plot(df['vicon time'].to_numpy(),st_vec_angle_optuna,label='steering angle optuna saturated speed',color='maroon')
ax_steering_angle.legend()


# print out the best parameters
print('max_st_dot =',best_max_st_dot)
print('fixed_delay_stdn =',best_fixed_delay)
print('k_stdn =',best_gain)


plt.show()



