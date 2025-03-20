from dart_dynamic_models import produce_long_term_predictions,process_vicon_data_kinematics,generate_tensor_past_actions,\
plot_kinemaitcs_data, model_functions, dynamic_bicycle_actuator_delay_fitting

from matplotlib import pyplot as plt
import torch
import numpy as np
import pandas as pd
import os
import tqdm
mf = model_functions() # instantiate the model functions class


# in this we will use the dynamic bicycle model to fit the actuator dynamics using a linear layer 




# change current folder where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))
rosbag_folder = '83_7_march_2025_MPCC_rosbag'



rosbag_data_folder = os.path.join('Data',rosbag_folder,'rosbags')
csv_folder = os.path.join('Data',rosbag_folder,'csv')
rosbag_files = os.listdir(rosbag_data_folder)
csv_files = os.listdir(csv_folder)


# load the impulse response data
folder_path_act_dyn_params = os.path.join('Data',rosbag_folder,'actuator_dynamics_saved_parameters/')
raw_weights_throttle = np.load(folder_path_act_dyn_params + 'raw_weights_throttle.npy')
raw_weights_steering = np.load(folder_path_act_dyn_params + 'raw_weights_steering.npy')
n_past_actions = np.load(folder_path_act_dyn_params + 'n_past_actions.npy')
dt = np.load(folder_path_act_dyn_params + 'dt.npy')



# load the model (just needed to get the weights)
model_obj = dynamic_bicycle_actuator_delay_fitting(n_past_actions,dt)
raw_weights_throttle_tens = torch.tensor(raw_weights_throttle)
raw_weights_steering_tens = torch.tensor(raw_weights_steering)
weights_throttle_tens = model_obj.constrained_linear_layer(raw_weights_throttle_tens)[0]
weights_steering_tens = model_obj.constrained_linear_layer(raw_weights_steering_tens)[0]
weights_th= np.transpose(weights_throttle_tens.cpu().numpy())
weights_st = np.transpose(weights_steering_tens.cpu().numpy())

# load the data
df = pd.read_csv(os.path.join(csv_folder,csv_files[0])) # load first csv file
columns_to_extract = ['vx body', 'vy body', 'w'] 
train_x_states = torch.tensor(df[columns_to_extract].to_numpy())
train_x_throttle = generate_tensor_past_actions(df, n_past_actions, key_to_repeat = 'throttle')
train_x_steering = generate_tensor_past_actions(df, n_past_actions, key_to_repeat = 'steering')

train_x = torch.cat((train_x_states,train_x_throttle,train_x_steering),1) # concatenate





# # show linear layer weights
fig, (ax_weights) = plt.subplots(1, 1, figsize=(12, 4), sharex=True)


# evalaute the filtered inputs
th_filtered = np.zeros(train_x.shape[0])
st_filtered = np.zeros(train_x.shape[0])
for i in range(train_x.shape[0]):
    # produce action to pass to the model
    data_row = np.expand_dims(train_x.cpu().numpy()[i,:],0)
    th_past  = data_row[0, 3 : 3 + n_past_actions]
    st_past  = data_row[0, 3 + n_past_actions :]

    th_filtered[i] = np.expand_dims(th_past,0) @ weights_th
    st_filtered[i] = np.expand_dims(st_past,0) @ weights_st



time_axis = np.linspace(0,n_past_actions*dt,n_past_actions)
ax_weights.plot(time_axis,np.squeeze(weights_th),color='dodgerblue',label='throttle weights')
ax_weights.plot(time_axis,np.squeeze(weights_st),color='orangered',label='steering weights')
ax_weights.set_xlabel('time delay[s]')
ax_weights.set_ylabel('weight')
ax_weights.set_ylim(0,1.1)
ax_weights.legend()

# plot the filtered inputs
fig, (ax_th,ax_st) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
ax_th.plot(df['vicon time'].to_numpy(),train_x_throttle[:,0].cpu().numpy(),color='gray',label='throttle')
ax_th.plot(df['vicon time'].to_numpy(),th_filtered,color='dodgerblue',label='throttle filtered')
ax_th.set_ylabel('throttle')
ax_th.legend()
ax_th.set_xlabel('time [s]')


ax_st.plot(df['vicon time'].to_numpy(),train_x_steering[:,0].cpu().numpy(),color='gray',label='steering')
ax_st.plot(df['vicon time'].to_numpy(),st_filtered,color='orangered',label='steering filtered')
ax_st.set_ylabel('steering')
ax_st.legend()
ax_st.set_xlabel('time [s]')

plt.show()