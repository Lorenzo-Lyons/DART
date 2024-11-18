from dart_dynamic_models import get_data, plot_raw_data, process_raw_vicon_data,plot_vicon_data,\
model_parameters,directly_measured_model_parameters,\
process_vicon_data_kinematics,model_functions,generate_tensor_past_actions,steering_dynamics_model_first_order
from matplotlib import pyplot as plt
import torch
import numpy as np
from scipy import interpolate
import os


#matplotlib.rc('font', **font)

# This script is used to fit the steering dynamics once we are sure about what the tire model is, and the longitudinal dynamics are properly modelled.
# The additional friction due to steering must be identified already.




# select data folder NOTE: this assumes that the current directory is DART
#folder_path = 'System_identification/Data/8_circles_rubbery_floor_1_file'
#folder_path = 'System_identification/Data/81_throttle_ramps'

#folder_path = 'System_identification/Data/81_circles_tape_and_tiles'
#folder_path = 'System_identification/Data/81_throttle_ramps_only_steer03'
#folder_path = 'System_identification/Data/91_model_validation_long_term_predictions_fast'
#folder_path = 'System_identification/Data/90_model_validation_long_term_predictions'

# Decide how many past steering signals to use. Note that this shold be enough to capture the dynamics of the system. 
# steering_time_window = 0.04  # [s] # this should be enough to capture the dynamics of the impulse response of the steering dynamics
# dt_steering = 0.001  # this small enough to limit the numerical error when solving the convolution integral in the steering dynamics model
#folder_path = 'System_identification/Data/91_free_driving_16_sept_2024'
folder_path = 'System_identification/Data/91_free_driving_16_sept_2024_slow'
#folder_path = 'System_identification/Data/steering_identification_25_sept_2024'



# decide if you want to tweak the steering curve
tweak_steering_curve = True



# process the data
steps_shift = 5 # decide to filter more or less the vicon data
df_raw_data = get_data(folder_path)



df_kinematics = process_vicon_data_kinematics(df_raw_data,steps_shift)
df = process_raw_vicon_data(df_kinematics,steps_shift)


df = df[df['vicon time']<70] # just use first part of data where velocity is quite low to avoid dynamic effects like pitch dynamics
df = df[df['vx body']>0.2] # remove low velocity points that give undefined slip angles



# # plot raw data
# ax0,ax1,ax2 = plot_raw_data(df)


# ax_wheel_f_alpha,ax_wheel_r_alpha,ax_total_force_front,\
# ax_total_force_rear,ax_lat_force,ax_long_force,\
# ax_acc_x_body,ax_acc_y_body,ax_acc_w = plot_vicon_data(df) 





# reverse engineer the true steering angle using the assumption that the tire forces should be on previously identified
# tire model. This is done by interpolating from the value of the measured force what the steering angle should be to have that force.

# this is just needed to recreate the tire curve
mf = model_functions()


alpha_range = np.linspace(-1,1,400)
tire_curve_F = mf.lateral_tire_force(alpha_range,mf.d_t_f_self,mf.c_t_f_self,mf.b_t_f_self,mf.m_front_wheel_self)
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
refinement_factor = 1

dt =  np.diff(df['vicon time'].to_numpy()).mean() / refinement_factor

# fitting the same data with a first order dynamics satureated model
steering_dynamics_model_first_order_obj = steering_dynamics_model_first_order(n_past_actions * refinement_factor,dt)

# define number of training iterations
train_its = 300
learning_rate = 0.001

print('')
print('Fitting steering dynamics by looking at the reconstructed steering angle with first order dynamics')


#define loss and optimizer objects
loss_fn = torch.nn.MSELoss() 
optimizer_object = torch.optim.Adam(steering_dynamics_model_first_order_obj.parameters(), lr=learning_rate)


# convert to double again
train_x = generate_tensor_past_actions(df, n_past_actions,refinement_factor, key_to_repeat = 'steering')
train_y = torch.unsqueeze(torch.tensor(true_steering_angle_vec),1).cuda()


# save loss values for later plot
loss_vec = np.zeros(train_its)

from tqdm import tqdm
tqdm_obj = tqdm(range(0,train_its), desc="training", unit="iterations")

for i in tqdm_obj:
    # clear gradient information from previous step before re-evaluating it for the current iteration
    optimizer_object.zero_grad()  
    
    # compute fitting outcome with current model parameters
    output_1st_order = steering_dynamics_model_first_order_obj(train_x)

    loss = loss_fn(output_1st_order,  train_y)
    loss_vec[i] = loss.item()

    # evaluate the gradient of the loss function with respect to the fitting parameters
    loss.backward() 

    # use the evaluated gradient to perform a gradient descent step 
    optimizer_object.step() # this updates parameters


# print out parameters
[k_stdn] = steering_dynamics_model_first_order_obj.transform_parameters_norm_2_real()

# print parameters
print('First order integrator parameter:')
print('k_stdn = ', k_stdn.item())






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
ground_truth_refinement = 100 # this is used to integrate the steering angle with a higher resolution to avoid numerical errors
for t in range(1, len(true_steering_angle_vec)):
    # Calculate the rate of change of steering (steering dot)
    for k in range(ground_truth_refinement):
        st_dot = mf.continuous_time_1st_order_dynamics(st,df['steering'].iloc[t-1],k_stdn.item()) 
        # Update the steering value with the time step
        st += st_dot * dt/ground_truth_refinement
    
    # Store the predicted steering angle
    st_vec_angle_forward_integrated[t] = mf.steering_2_steering_angle(st,mf.a_s_self,mf.b_s_self,mf.c_s_self,mf.d_s_self,mf.e_s_self)



ax_steering_angle.plot(df['vicon time'].to_numpy(),st_vec_angle_forward_integrated,label='Forward Euler integration',color='navy',linewidth=3,linestyle='--')
ax_steering_angle.legend()


plt.show()



