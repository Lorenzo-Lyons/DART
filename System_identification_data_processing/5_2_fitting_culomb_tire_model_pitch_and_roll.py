from functions_for_data_processing import get_data, plot_raw_data, process_raw_vicon_data,\
plot_vicon_data,model_parameters,directly_measured_model_parameters,process_vicon_data_kinematics,\
    pacejka_tire_model_pitch_roll,throttle_dynamics,steering_dynamics,\
    generate_tensor_past_actions

from matplotlib import pyplot as plt
import torch
import numpy as np
import pandas as pd
import os


[theta_correction, l_COM, l_lateral_shift_reference ,lr, lf, Jz, m,m_front_wheel,m_rear_wheel] = directly_measured_model_parameters()

[a_m, b_m, c_m, d_m,
a_f, b_f, c_f, d_f,
a_s, b_s, c_s, d_s, e_s,
d_t_f, c_t_f, b_t_f,d_t_r, c_t_r, b_t_r,
a_stfr, b_stfr,d_stfr,e_stfr,
max_st_dot,fixed_delay_stdn,k_stdn,
w_natural_Hz_pitch,k_f_pitch,k_r_pitch,
w_natural_Hz_roll,k_f_roll,k_r_roll]= model_parameters()





# select data folder NOTE: this assumes that the current directory is DART
#folder_path = 'System_identification_data_processing/Data/8_circles_rubbery_floor_1_file'

#folder_path = 'System_identification_data_processing/Data/81_circles_tape_and_tiles'

#folder_path = 'System_identification_data_processing/Data/81_throttle_ramps'
#folder_path = 'System_identification_data_processing/Data/81_throttle_ramps_only_steer03'

folder_path = 'System_identification_data_processing/Data/91_free_driving_16_sept_2024'


#folder_path = 'System_identification_data_processing/Data/steering_identification_25_sept_2024'

#folder_path = 'System_identification_data_processing/Data/circles_27_sept_2024'

#folder_path = 'System_identification_data_processing/Data/circles_27_sept_2024_fast_ramp'





# --- Starting data processing  ------------------------------------------------


# #robot2vicon_delay = 5 # samples delay
# df_raw_data = get_data(folder_path)

# # process the data
# steps_shift = 10 # decide to filter more or less the vicon data
# df = process_raw_vicon_data(df_raw_data,steps_shift)



# check if there is a processed vicon data file already
file_name = 'processed_vicon_data_throttle_steering_dynamics.csv'
# Check if the CSV file exists in the folder
file_path = os.path.join(folder_path, file_name)

if not os.path.isfile(file_path):

    steps_shift = 5 # decide to filter more or less the vicon data
    df_raw_data = get_data(folder_path)

    # replace throttle with time integrated throttle
    filtered_throttle = throttle_dynamics(df_raw_data,d_m)
    df_raw_data['throttle'] = filtered_throttle

    # add steering time integrated 
    st_vec_angle_optuna, st_vec_optuna = steering_dynamics(df_raw_data,a_s,b_s,c_s,d_s,e_s,max_st_dot,fixed_delay_stdn,k_stdn)
    # over-write the actual data with the forward integrated data
    df_raw_data['steering angle'] = st_vec_angle_optuna
    df_raw_data['steering'] = st_vec_optuna


    df_kinematics = process_vicon_data_kinematics(df_raw_data,steps_shift,theta_correction, l_COM, l_lateral_shift_reference)
    df = process_raw_vicon_data(df_kinematics,steps_shift)

    # save the file
    df.to_csv(file_path, index=False)
    print(f"File '{file_path}' saved.")
else:
    print(f"File '{file_path}' already exists, loading data.")
    df = pd.read_csv(file_path)



# plot raw data
ax0,ax1,ax2 = plot_raw_data(df)


ax_wheel_f_alpha,ax_wheel_r_alpha,ax_total_force_front,\
ax_total_force_rear,ax_lat_force,ax_long_force,\
ax_acc_x_body,ax_acc_y_body,ax_acc_w = plot_vicon_data(df) 




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
pacejka_culomb_tire_model_obj = pacejka_tire_model_pitch_roll(initial_guess,m_front_wheel,m_rear_wheel,lf,lr,d_t_f,c_t_f,b_t_f,d_t_r,c_t_r,b_t_r)

#define loss and optimizer objects
loss_fn = torch.nn.MSELoss() 
optimizer_object = torch.optim.Adam(pacejka_culomb_tire_model_obj.parameters(), lr=learning_rate)

# generate data in tensor form for torch
data_columns = ['slip angle front','slip angle rear','ax body','ay body'] # velocities


# add past actions to fit pitch dynamics
# past longitudinal accelerations
n_past_actions = 50 # 50
refinement_factor = 10 # 10
train_x_past_acc_x = generate_tensor_past_actions(df, n_past_actions,refinement_factor, key_to_repeat = 'ax body')






train_x = torch.tensor(df[data_columns].to_numpy()).cuda()
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
[k_pitch,k_roll] = pacejka_culomb_tire_model_obj.transform_parameters_norm_2_real()

print('# pitch and roll:')
print('k_pitch = ', k_pitch.item())
print('k_roll = ', k_roll.item())


# # # --- plot loss function ---
plt.figure()
plt.title('Loss function')
plt.plot(loss_vec,label='tire model loss')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.legend()





# plot model outputs
ax_wheel_f_alpha.scatter(df['slip angle front'],F_y_f.detach().cpu().numpy(),color='k',label='Tire model output with pitch influece',s=3,alpha=0.5)
ax_wheel_r_alpha.scatter(df['slip angle rear'],F_y_r.detach().cpu().numpy(),color='k',label='Tire model output with pitch influece',s=3,alpha=0.5)


plt.show()







