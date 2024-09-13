from functions_for_data_processing import get_data, plot_raw_data, process_raw_vicon_data,plot_vicon_data,\
steering_dynamics_model,model_parameters
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
#folder_path = 'System_identification_data_processing/Data/81_throttle_ramps'

#folder_path = 'System_identification_data_processing/Data/81_circles_tape_and_tiles'
#folder_path = 'System_identification_data_processing/Data/81_throttle_ramps_only_steer03'
folder_path = 'System_identification_data_processing/Data/91_model_validation_long_term_predictions_fast'


# Decide how many past steering signals to use. Note that this shold be enough to capture the dynamics of the system. 
# steering_time_window = 0.04  # [s] # this should be enough to capture the dynamics of the impulse response of the steering dynamics
# dt_steering = 0.001  # this small enough to limit the numerical error when solving the convolution integral in the steering dynamics model


# define the number of past steering signals to use
n_past_steering = 50 #int(np.ceil(steering_time_window/T)) + 1
refinement_factor = 5 #int(np.ceil(T/dt_steering))




# load model parameters
[theta_correction, lr, l_COM, Jz, lf, m,
a_m, b_m, c_m, d_m,
a_f, b_f, c_f, d_f,
a_s, b_s, c_s, d_s, e_s,
d_t, c_t, b_t,
a_stfr, b_stfr] = model_parameters()



# Starting data processing
# check if there is a processed vicon data file already
file_name = 'processed_vicon_data.csv'
# Check if the CSV file exists in the folder
file_path = os.path.join(folder_path, file_name)

if not os.path.isfile(file_path):
    # If the file does not exist, process the raw data
    # get the raw data
    df_raw_data = get_data(folder_path)

    # process the data
    
    df = process_raw_vicon_data(df_raw_data)

    df.to_csv(file_path, index=False)
    print(f"File '{file_path}' saved.")
else:
    print(f"File '{file_path}' already exists, loading data.")
    df = pd.read_csv(file_path)





# add filtered throttle
T = df['vicon time'].diff().mean()  # Calculate the average time step
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







# --- produce and plot data to train the model ---
# these are the measured acceleration in the body frame, i.e. the same ones that the model should predict

acc_centrifugal_in_x = df['vy body'].to_numpy() * df['w_abs_filtered'].to_numpy()
acc_centrifugal_in_y = - df['vx body'].to_numpy() * df['w_abs_filtered'].to_numpy()
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
ax_accw.plot(df['vicon time'].to_numpy(),df['aw_abs_filtered_more'].to_numpy(),label='acc_w',color='purple')
ax_accw.set_xlabel('Time [s]')
ax_accw.set_ylabel('Acceleration w')

















# --------------- fitting tire model---------------
# fitting tyre models
# define first guess for parameters
initial_guess = torch.ones(3) * 0.5 # initialize parameters in the middle of their range constraint


# define number of training iterations
train_its = 200
learning_rate = 0.01

print('')
print('Fitting steering dynamics')

#instantiate the model
#dt = np.diff(df['vicon time'].to_numpy()).mean()
steering_dynamics_model_obj = steering_dynamics_model(initial_guess,(n_past_steering)*refinement_factor,dt_int_steering)

#define loss and optimizer objects
loss_fn = torch.nn.MSELoss() 
optimizer_object = torch.optim.Adam(steering_dynamics_model_obj.parameters(), lr=learning_rate)





#generate data in tensor form for torch
#train_x = torch.unsqueeze(torch.tensor(np.concatenate((df['V_y front wheel'].to_numpy(),df['V_y rear wheel'].to_numpy()))),1).cuda()

def generate_tensor(df, n_past_steering,refinement_factor):

    # due to numerical errors in evauating the convolution integral we need a finer resolution for the time step on the steering
    df_past_steering = pd.DataFrame()
    df_past_steering['steering'] = df['steering']
    # Add delayed steering signals based on user input
    for i in range(0, n_past_steering):
        df_past_steering[f'steering prev{i}'] = df['steering'].shift(i, fill_value=0)
    # doing a zero order hold on the steering signal to get  a finer resolution
    df_past_steering_refined = pd.DataFrame()
    for i in range(0, (n_past_steering)):
        for k in range(refinement_factor):
            df_past_steering_refined[f'steering prev{i*refinement_factor+k}'] = df_past_steering[f'steering prev{i}']


    # Select columns for generating tensor
    selected_columns_df = ['vx body', 'vy body', 'w_abs_filtered','throttle filtered'] 
    selected_columns_df_steering = [f'steering prev{i}' for i in range(0, (n_past_steering)*refinement_factor)]
    
    # Convert the selected columns into a tensor and send to GPU (if available)
    train_x_df = torch.tensor(df[selected_columns_df].to_numpy()).cuda()
    train_x_df_steering = torch.tensor(df_past_steering_refined[selected_columns_df_steering].to_numpy()).cuda()

    train_x = torch.cat((train_x_df, train_x_df_steering), dim=1)
    return train_x




# generate data in tensor form for torch
train_x = generate_tensor(df, n_past_steering,refinement_factor) 
# y labels are the measured accelerations in the body frame
train_y = torch.unsqueeze(torch.tensor(np.concatenate((acc_x_body_measured,acc_y_body_measured,df['aw_abs_filtered_more'].to_numpy()))),1).cuda() 




# save loss values for later plot
loss_vec = np.zeros(train_its)



from tqdm import tqdm
tqdm_obj = tqdm(range(0,train_its), desc="training", unit="iterations")

for i in tqdm_obj:
    # clear gradient information from previous step before re-evaluating it for the current iteration
    optimizer_object.zero_grad()  
    
    # compute fitting outcome with current model parameters
    acc_x,acc_y,acc_w, steering_angle = steering_dynamics_model_obj(train_x)
    acc_output = torch.vstack([acc_x,acc_y,acc_w])

    # evaluate loss function
    loss = loss_fn(acc_output,  train_y)
    loss_vec[i] = loss.item()

    # evaluate the gradient of the loss function with respect to the fitting parameters
    loss.backward() 

    # use the evaluated gradient to perform a gradient descent step 
    optimizer_object.step() # this updates parameters


# --- print out parameters ---
[damping,w_natural,fixed_delay]= steering_dynamics_model_obj.transform_parameters_norm_2_real()
damping, w_natural,fixed_delay =  damping.item(), w_natural.item() ,fixed_delay.item()


print('steering dynamics parameters:')
print('damping = ', damping)
print('w_natural = ', w_natural)
print('fixed_delay = ', fixed_delay)


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
k_vec = steering_dynamics_model_obj.produce_past_action_coefficients(torch.Tensor([damping]).cuda(),torch.Tensor([w_natural]).cuda(),torch.Tensor([fixed_delay]).cuda()).cpu().view(-1).numpy()    
t_vec = np.linspace(0,dt_int_steering*(n_past_steering_refined-1),n_past_steering_refined)

plt.figure()
plt.plot(t_vec,k_vec,label='k_vec',color='gray')
plt.plot(df['vicon time'].to_numpy(),df['steering angle'].to_numpy(),label='steering angle',color='purple')
plt.plot(df['vicon time'].to_numpy(),steering_angle.detach().cpu().view(-1).numpy(),label='steering angle model',color='k')
plt.xlabel('Time [s]')
plt.ylabel('Steering angle')
plt.legend()



# visualize the wheel data using the steering dynamics
df_raw_data_steering_dynamics = get_data(folder_path)
df_raw_data_steering_dynamics['steering angle'] = steering_angle.detach().cpu().view(-1).numpy()
# providing the steering  will make processing data use that instead of recovering it from the raw data
df_steering_dynamics = process_raw_vicon_data(df_raw_data_steering_dynamics)


v_y_wheel_plotting = torch.unsqueeze(torch.linspace(-1,1,100),1).cuda()
lateral_force_vec = steering_dynamics_model_obj.F_y_wheel_model(v_y_wheel_plotting).detach().cpu().numpy()
ax_wheels.scatter(df_steering_dynamics['V_y front wheel'],df_steering_dynamics['Fy front wheel'],color='skyblue',label='front wheel data',s=3)
ax_wheels.scatter(df_steering_dynamics['V_y rear wheel'],df_steering_dynamics['Fy rear wheel'],color='teal',label='rear wheel data',s=3)
ax_wheels.plot(v_y_wheel_plotting.detach().cpu().numpy(),lateral_force_vec,color='#2c4251',label='Tire model',linewidth=4,linestyle='-')

ax_wheels.legend()



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

