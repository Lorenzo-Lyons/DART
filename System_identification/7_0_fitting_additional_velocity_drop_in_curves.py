from dart_dynamic_models import get_data, plot_raw_data, process_raw_vicon_data,plot_vicon_data\
,dyn_model_culomb_tires,steering_friction_model,process_vicon_data_kinematics, model_functions,\
throttle_dynamics_data_processing,steering_dynamics_data_processing
from matplotlib import pyplot as plt
import torch
import numpy as np
import pandas as pd
import os


# This script is used to fit the tire model to the data collected using the vicon external tracking system with a 
# SIMPLE CULOMB friction tyre model.

# select data folder NOTE: this assumes that the current directory is DART

#folder_path = 'System_identification/Data/81_throttle_ramps'
#folder_path = 'System_identification/Data/circles_27_sept_2024'
folder_path = 'System_identification/Data/steer_friction_training_data' # this uses both the above files



mf = model_functions()


# --- Starting data processing  ------------------------------------------------
# # check if there is a processed vicon data file already
# file_name = 'processed_vicon_data.csv'
# # Check if the CSV file exists in the folder
# file_path = os.path.join(folder_path, file_name)


# steps_shift = 5

# if not os.path.isfile(file_path):
#     # If the file does not exist, process the raw data
#     # get the raw data
#     df_raw_data = get_data(folder_path)
#     df_kinematics = process_vicon_data_kinematics(df_raw_data,steps_shift)
#     df = process_raw_vicon_data(df_kinematics,steps_shift)

#     df.to_csv(file_path, index=False)
#     print(f"File '{file_path}' saved.")
# else:
#     print(f"File '{file_path}' already exists, loading data.")
#     df = pd.read_csv(file_path)


# process data

steps_shift = 5 # decide to filter more or less the vicon data

# check if there is a processed vicon data file already
file_name = 'processed_vicon_data_throttle_steering_dynamics.csv'
# Check if the CSV file exists in the folder
file_path = os.path.join(folder_path, file_name)

if not os.path.isfile(file_path):
    df_raw_data = get_data(folder_path)

    # add throttle with time integrated throttle
    filtered_throttle = throttle_dynamics_data_processing(df_raw_data)
    df_raw_data['throttle filtered'] = filtered_throttle

    # add steering angle with time integated version
    st_angle_vec_FEuler, st_vec_FEuler = steering_dynamics_data_processing(df_raw_data)
    # over-write the actual data with the forward integrated data
    df_raw_data['steering angle filtered'] = st_angle_vec_FEuler
    df_raw_data['steering filtered'] = st_vec_FEuler

    # process kinematics and dynamics
    df_kinematics = process_vicon_data_kinematics(df_raw_data,steps_shift)
    df = process_raw_vicon_data(df_kinematics,steps_shift)

    #save the processed data file
    df.to_csv(file_path, index=False)
    print(f"File '{file_path}' saved.")
else:
    print(f"File '{file_path}' already exists, loading data.")
    df = pd.read_csv(file_path)







#cut off time instances where the vicon missed a detection to avoid corrupted datapoints
if  folder_path == 'System_identification/Data/81_throttle_ramps':
    df1 = df[df['vicon time']<100]
    df1 = df1[df1['vicon time']>20]

    df2 = df[df['vicon time']>185]
    df2 = df2[df2['vicon time']<268]

    df3 = df[df['vicon time']>283]
    df3 = df3[df3['vicon time']<350]

    df4 = df[df['vicon time']>365]
    df4 = df4[df4['vicon time']<410]

    df5 = df[df['vicon time']>445]
    df5 = df5[df5['vicon time']<500]

    df6 = df[df['vicon time']>540]
    df6 = df6[df6['vicon time']<600]

    df7 = df[df['vicon time']>645]
    df7 = df7[df7['vicon time']<658]

    df8 = df[df['vicon time']>661]
    df8 = df8[df8['vicon time']<725]

    df9 = df[df['vicon time']>745]
    df9 = df9[df9['vicon time']<820]

    df = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9]) 

elif folder_path == 'System_identification/Data/steering_identification_25_sept_2024':
    df = df[df['vicon time']<460]

elif folder_path == 'System_identification/Data/circles_27_sept_2024':

    df1 = df[df['vicon time']>1]
    df1 = df1[df1['vicon time']<375]

    df2 = df[df['vicon time']>377]
    df2 = df2[df2['vicon time']<830]

    df3 = df[df['vicon time']>860]
    df3 = df3[df3['vicon time']<1000]

    df = pd.concat([df1,df2,df3])

elif folder_path == 'System_identification/Data/steer_friction_training_data':
    df1 = df[df['vicon time']<100]
    df1 = df1[df1['vicon time']>20]

    df2 = df[df['vicon time']>185]
    df2 = df2[df2['vicon time']<268]

    df3 = df[df['vicon time']>283]
    df3 = df3[df3['vicon time']<350]

    df4 = df[df['vicon time']>365]
    df4 = df4[df4['vicon time']<410]

    df5 = df[df['vicon time']>445]
    df5 = df5[df5['vicon time']<500]

    df6 = df[df['vicon time']>540]
    df6 = df6[df6['vicon time']<600]

    df7 = df[df['vicon time']>645]
    df7 = df7[df7['vicon time']<658]

    df8 = df[df['vicon time']>661]
    df8 = df8[df8['vicon time']<725]

    df9 = df[df['vicon time']>745]
    df9 = df9[df9['vicon time']<820]

    # merge time
    t_file_1 = 827.984035550927

    df10 = df[df['vicon time']>1+t_file_1]
    df10 = df10[df10['vicon time']<375+t_file_1]

    df11 = df[df['vicon time']>377+t_file_1]
    df11 = df11[df11['vicon time']<830+t_file_1]

    df12 = df[df['vicon time']>860+t_file_1]
    df12 = df12[df12['vicon time']<1000+t_file_1]

    df = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12])


# select data to use for fitting
df = df[df['vx body']>0.5]
df = df[df['ax body']<2]
df = df[df['ax body']>-1]



# --- the following is just to visualize what the model error is without the additional friction term due to the steering ---
# --- the actual model will be fitted differently cause the extra term will affect the motor force, while here we are plotting the overall
# --- missing longitudinal force. (Adding to Fx will also, slightly, affect lateral and yaw dynamics that we do not show here)


steering_friction_flag = False
pitch_dynamics_flag = False

# the model gives you the derivatives of it's own states, so you can integrate them to get the states in the new time instant
dyn_model_culomb_tires_obj = dyn_model_culomb_tires(steering_friction_flag,pitch_dynamics_flag)





columns_to_extract = ['vx body', 'vy body', 'w', 'throttle filtered' ,'steering filtered','throttle','steering']
input_data = df[columns_to_extract].to_numpy()

acc_x_model = np.zeros(input_data.shape[0])
acc_y_model = np.zeros(input_data.shape[0])
acc_w_model = np.zeros(input_data.shape[0])

for i in range(df.shape[0]):
    # correct for centrifugal acceleration
    accelerations = dyn_model_culomb_tires_obj.forward(input_data[i,:])

    acc_x_model[i] = accelerations[0]
    acc_y_model[i] = accelerations[1]
    acc_w_model[i] = accelerations[2]




# evaluate friction curve
velocity_range = np.linspace(0,df['vx body'].max(),100)
#friction_curve = dynamic_model.rolling_friction(velocity_range,a_f,b_f,c_f,d_f)

# model error in [N]
missing_force = (acc_x_model - df['ax body'].to_numpy()) * mf.m_self



from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
color_code_label = 'Force [N]'
# x axis   throttle_filtered,v,a_m,b_m,c_m
x_axis_data =  df['vx body'].to_numpy()
scatter = ax.scatter(
    x_axis_data,                    # x-axis
    df['steering angle'].to_numpy(),  # y-axis
    missing_force,                    # z-axis 
    c=missing_force,  # color coded by 'steering angle time delayed'
    cmap='viridis'  # Colormap
)

ax.set_xlabel('vx body')
ax.set_ylabel('steering angel [rad]')
ax.set_zlabel('Force [N]')
colorbar = fig.colorbar(scatter, label=color_code_label)





# --------------- fitting extra steering friction model---------------


# fitting tyre models
# define first guess for parameters
initial_guess = torch.ones(6) * 0.5 # initialize parameters in the middle of their range constraint
# define number of training iterations
train_its = 1000
learning_rate = 0.001

print('')
print('Fitting extra steering friction model ')

#instantiate the model
steering_friction_model_obj = steering_friction_model()

#define loss and optimizer objects
loss_fn = torch.nn.MSELoss() 
optimizer_object = torch.optim.Adam(steering_friction_model_obj.parameters(), lr=learning_rate)


# generate data in tensor form for torch
train_x = torch.tensor(input_data).cuda()

# -- Y lables --
train_y = torch.unsqueeze(torch.tensor(df['ax body'].to_numpy()),1).cuda() 


# save loss values for later plot
loss_vec = np.zeros(train_its)

# train the model
for i in range(train_its):
    # clear gradient information from previous step before re-evaluating it for the current iteration
    optimizer_object.zero_grad()  
    
    # compute fitting outcome with current model parameters
    acc_x,acc_y,acc_w = steering_friction_model_obj(train_x)

    # evaluate loss function
    loss = loss_fn(acc_x,  train_y)
    loss_vec[i] = loss.item()

    # evaluate the gradient of the loss function with respect to the fitting parameters
    loss.backward() 

    # use the evaluated gradient to perform a gradient descent step 
    optimizer_object.step() # this updates parameters


# --- print out parameters ---
[a_stfr_model,b_stfr_model,d_stfr_model,e_stfr_model] = steering_friction_model_obj.transform_parameters_norm_2_real()
a_stfr,b_stfr,d_stfr,e_stfr = a_stfr_model.item(),b_stfr_model.item(),d_stfr_model.item(),e_stfr_model.item()

print('# Friction due to steering parameters:')
print('    a_stfr = ', a_stfr)
print('    b_stfr = ', b_stfr)
print('    d_stfr = ', d_stfr)
print('    e_stfr = ', e_stfr)


# # # --- plot loss function ---
plt.figure()
plt.title('Loss function')
plt.plot(loss_vec,label='tire model loss')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.legend()


# plot model putputs
fig1, ((ax_acc_x,ax_acc_y,ax_acc_w)) = plt.subplots(3, 1, figsize=(10, 6), constrained_layout=True)

# plot longitudinal accleeartion
ax_acc_x.plot(df['vicon time'].to_numpy(),df['ax body'].to_numpy(),label='ax body',color='dodgerblue')
ax_acc_x.plot(df['vicon time'].to_numpy(),acc_x.detach().cpu().numpy(),color='k',label='model output')

ax_acc_x.legend()

# plot lateral accleeartion
ax_acc_y.plot(df['vicon time'].to_numpy(),df['ay body'].to_numpy(),label='ay body',color='orangered')
ax_acc_y.plot(df['vicon time'].to_numpy(),acc_y.detach().cpu().numpy(),color='k',label='model output')
ax_acc_y.legend()

# plot yaw rate
ax_acc_w.plot(df['vicon time'].to_numpy(),df['acc_w'].to_numpy(),label='acc_w',color='purple')
ax_acc_w.plot(df['vicon time'].to_numpy(),acc_w.detach().cpu().numpy(),color='k',label='model output')
ax_acc_w.legend()




# --- plot model output over model labels ---

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')

training_lables =  torch.squeeze(train_y).detach().cpu().numpy()

scatter2 = ax2.scatter(x_axis_data,df['steering angle'].to_numpy(),training_lables,c=training_lables,cmap='plasma',label='model labels')
colorbar = fig2.colorbar(scatter2, label=color_code_label)

#missing_force_model =  (torch.squeeze(acc_x).detach().cpu().numpy() - df['ax body'].to_numpy())*m
y_output_model =  torch.squeeze(acc_x).detach().cpu().numpy() #- df['Fx wheel'].to_numpy()

# add output to the scatter plot cause it's difficult to see if the fitting went well or not otherwise
scatter3 = ax2.scatter(
    x_axis_data,                    # x-axis
    df['steering angle'].to_numpy(),  # y-axis
    y_output_model,                    # z-axis 
    color='k',
    alpha=0.5,
    label='model output'
)

ax2.set_xlabel('vx body')
ax2.set_ylabel('steering angel [rad]')
ax2.set_zlabel('Force [N]')

plt.show()

