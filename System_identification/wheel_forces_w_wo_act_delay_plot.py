from dart_dynamic_models import plot_wheel_forces, process_raw_vicon_data,\
generate_tensor_past_actions,model_functions
from matplotlib import pyplot as plt
import torch
import numpy as np
import pandas as pd
import os
font = {'family' : 'normal',
        'size'   : 22}
import matplotlib
matplotlib.rc('font', **font)

# load parameters
mf = model_functions()


# change current folder where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#rosbag_folder = '83_7_march_2025_MPCC_rosbag'
rosbag_folder = '83_9_april_2025_MPCCPP_rosbag'


# load data
csv_folder = os.path.join('Data',rosbag_folder,'csv')
csv_files = os.listdir(csv_folder)
folder_path_act_dyn_params = os.path.join('Data',rosbag_folder,'actuator_dynamics_saved_parameters/')




#check if only 1 file is in the folder
if len(csv_files) > 1:
    raise ValueError('More than 1 csv file in folder')
else:
    csv_file = os.path.join(csv_folder,csv_files[0])
    df = pd.read_csv(csv_file)


df = process_raw_vicon_data(df,2)
df_no_delay = df.copy() # make a copy of the dataframe without actuator dynamics





# ---- repeat the process with the actuator dynamics model --------------------------------

weights_th = np.load(folder_path_act_dyn_params + 'weights_throttle.npy')
weights_st = np.load(folder_path_act_dyn_params + 'weights_steering.npy')
n_past_actions = np.load(folder_path_act_dyn_params + 'n_past_actions.npy')
dt = np.load(folder_path_act_dyn_params + 'dt.npy')



# load the data
torch_output = False
past_Actions_matrx_throttle = generate_tensor_past_actions(df, n_past_actions, 'throttle', torch_output)
past_Actions_matrx_steering = generate_tensor_past_actions(df, n_past_actions, 'steering', torch_output)

# # show linear layer weights
fig, (ax_weights) = plt.subplots(1, 1, figsize=(8, 4), sharex=True)
time_axis = np.linspace(0,n_past_actions*dt,n_past_actions)
ax_weights.plot(time_axis,np.squeeze(weights_th),color='dodgerblue',label='throttle weights',linewidth=5)
ax_weights.plot(time_axis,np.squeeze(weights_st),color='orangered',label='steering weights',linewidth=5)
ax_weights.set_xlabel('time delay[s]')
ax_weights.set_ylabel('weight')
ax_weights.set_ylim(0,1.1)
ax_weights.legend()

fig.subplots_adjust(
    top=1.0,
    bottom=0.19,
    left=0.145,
    right=0.995,
    hspace=0.2,
    wspace=0.2
)



th_filtered = past_Actions_matrx_throttle @ weights_th
st_filtered = past_Actions_matrx_steering @ weights_st

df['throttle'] = th_filtered
df['steering'] = st_filtered


df = process_raw_vicon_data(df,2)



# plot the figures
fig0, ((ax_wheel_f_alpha_no_delay,ax_wheel_f_alpha)) = plt.subplots(1, 2, figsize=(16, 4), constrained_layout=True)
fig1, ((ax_wheel_r_alpha_no_delay,ax_wheel_r_alpha)) = plt.subplots(1, 2, figsize=(16, 4), constrained_layout=True)

scatter_front,color_code_label = plot_wheel_forces(df_no_delay,mf,fig0,ax_wheel_f_alpha_no_delay,ax_wheel_r_alpha_no_delay)

scatter_rear,color_code_label = plot_wheel_forces(df,mf,fig1,ax_wheel_f_alpha,ax_wheel_r_alpha)



cbar1 = fig1.colorbar(scatter_front, ax=ax_wheel_f_alpha) # right most axis
cbar1.set_label('Longitudinal accelration')  # Label the colorbar  'vel encoder-vx body'

plt.show()






