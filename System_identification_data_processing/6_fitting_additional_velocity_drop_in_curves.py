from functions_for_data_processing import get_data, plot_raw_data, process_raw_vicon_data,plot_vicon_data\
,dyn_model_culomb_tires,steering_friction_model,model_parameters
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

# This script is used to fit the tire model to the data collected using the vicon external tracking system with a 
# SIMPLE CULOMB friction tyre model.

# select data folder NOTE: this assumes that the current directory is DART
#folder_path = 'System_identification_data_processing/Data/9_model_validation_long_term_predictions'
#folder_path = 'System_identification_data_processing/Data/8_circles_rubbery_floor_1_file'
#folder_path = 'System_identification_data_processing/Data/91_model_validation_long_term_predictions_fast'
folder_path = 'System_identification_data_processing/Data/81_throttle_ramps'
#folder_path = 'System_identification_data_processing/Data/81_throttle_ramps_only_steer03'






# load model parameters
[theta_correction, lr, l_COM, Jz, lf, m,
a_m, b_m, c_m, d_m,
a_f, b_f, c_f, d_f,
a_s, b_s, c_s, d_s, e_s,
d_t, c_t, b_t,
a_stfr, b_stfr] = model_parameters()

# the model gives you the derivatives of it's own states, so you can integrate them to get the states in the new time instant
dynamic_model = dyn_model_culomb_tires(m,lr,lf,l_COM,Jz,
                 d_t,c_t,b_t,
                 a_m,b_m,c_m,
                 a_f,b_f,c_f,d_f,
                 a_stfr,b_stfr)





# --- Starting data processing  ------------------------------------------------
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




# plot vicon related data (longitudinal and lateral velocities, yaw rate related)
ax_wheels = plot_vicon_data(df)


# NOTE
# Because the test have been done in a quasi static setting for the throttle it is not necessary to integrate it's dynamics




# plot filtered throttle signal
plt.figure()
plt.plot(df['vicon time'].to_numpy(),df['throttle'].to_numpy(),label='throttle')
plt.plot(df['vicon time'].to_numpy(),df['ax body no centrifugal'].to_numpy(),label='acc_x',color = 'dodgerblue')
plt.xlabel('Time [s]')
plt.ylabel('Throttle')
plt.legend()
 


# --- the following is just to visualize what the model error is without the additional friction term due to the steering ---
# --- the actual model will be fitted differently cause the extra term will affect the motor force, while here we are plotting the overall
# --- missing longitudinal force. (Adding to Fx will also, slightly, affect lateral and yaw dynamics that we do not show here)

a_stfr= []  # give empty correction terms to not use them
b_stfr=[]
# define model NOTE: this will give you the absolute accelerations measured in the body frame
dynamic_model = dyn_model_culomb_tires(m,lr,lf,l_COM,Jz,d_t,c_t,b_t,
                 a_m,b_m,c_m,
                 a_f,b_f,c_f,d_f,
                 a_stfr,b_stfr)

columns_to_extract = ['vx body', 'vy body', 'w_abs_filtered', 'throttle' ,'steering angle']
input_data = df[columns_to_extract].to_numpy()

acc_x_model = np.zeros(input_data.shape[0])
acc_y_model = np.zeros(input_data.shape[0])
acc_w_model = np.zeros(input_data.shape[0])

acc_centrifugal_in_x = df['vy body'].to_numpy() * df['w_abs_filtered'].to_numpy()
acc_centrifugal_in_y = - df['vx body'].to_numpy() * df['w_abs_filtered'].to_numpy()

for i in range(df.shape[0]):
    # correct for centrifugal acceleration
    accelerations = dynamic_model.forward(input_data[i,:])
    acc_x_model[i] = accelerations[0]
    acc_y_model[i] = accelerations[1]
    acc_w_model[i] = accelerations[2]


# accelerations in the body frame
acc_x_body_measured = df['ax body no centrifugal'].to_numpy() + acc_centrifugal_in_x
acc_y_body_measured = df['ay body no centrifugal'].to_numpy() + acc_centrifugal_in_y

# plot the modelled acceleration
fig, ax_accx = plt.subplots()
ax_accx.plot(df['vicon time'].to_numpy(),acc_x_body_measured,label='acc_x body frame',color='dodgerblue')
ax_accx.plot(df['vicon time'].to_numpy(),acc_x_model,label='acc_x model',color='k',alpha=0.5)
ax_accx.set_xlabel('Time [s]')
ax_accx.set_ylabel('Acceleration x')

# y accelerations
fig, ax_accy = plt.subplots()
ax_accy.plot(df['vicon time'].to_numpy(),acc_y_body_measured,label='acc_y in the body frame',color='orangered')
ax_accy.plot(df['vicon time'].to_numpy(),acc_y_model,label='acc_y model',color='k',alpha=0.5)
ax_accy.set_xlabel('Time [s]')
ax_accy.set_ylabel('Acceleration y')

# w accelerations
fig, ax_accw = plt.subplots()
ax_accw.plot(df['vicon time'].to_numpy(),df['aw_abs_filtered_more'].to_numpy(),label='acc_w',color='purple')
ax_accw.plot(df['vicon time'].to_numpy(),acc_w_model,label='acc_w model',color='k',alpha=0.5)
ax_accw.set_xlabel('Time [s]')
ax_accw.set_ylabel('Acceleration w')

# evaluate friction curve
velocity_range = np.linspace(0,df['vx body'].max(),100)
friction_curve = dynamic_model.friction(velocity_range)

# model error in [N]
missing_force = (acc_x_model - acc_x_body_measured)*m

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    df['vx body'].to_numpy(),                    # x-axis
    df['steering angle'].to_numpy(),  # y-axis
    missing_force,                    # z-axis 
    c=df['steering angle'].to_numpy(),  # color coded by 'steering angle time delayed'
    cmap='viridis'  # Colormap
)

ax.set_xlabel('vx body')
ax.set_ylabel('steering angle')
ax.set_zlabel('Force [N]')
colorbar = fig.colorbar(scatter, label='steering angle time delayed')











# --------------- fitting extra steering friction model---------------


# fitting tyre models
# define first guess for parameters
initial_guess = torch.ones(2) * 0.5 # initialize parameters in the middle of their range constraint
initial_guess[2] = 0.1
# define number of training iterations
train_its = 1000
learning_rate = 0.003

print('')
print('Fitting extra steering friction model ')

#instantiate the model
steering_friction_model_obj = steering_friction_model(initial_guess,m,lr,lf,l_COM,Jz,
                 d_t,c_t,b_t,
                 a_m,b_m,c_m,
                 a_f,b_f,c_f,d_f,
                 a_stfr,b_stfr)

#define loss and optimizer objects
loss_fn = torch.nn.MSELoss() 
optimizer_object = torch.optim.Adam(steering_friction_model_obj.parameters(), lr=learning_rate)

# generate data in tensor form for torch
train_x = torch.tensor(input_data).cuda()
train_y = torch.unsqueeze(torch.tensor(acc_x_body_measured),1).cuda()


# save loss values for later plot
loss_vec = np.zeros(train_its)

# train the model
for i in range(train_its):
    # clear gradient information from previous step before re-evaluating it for the current iteration
    optimizer_object.zero_grad()  
    
    # compute fitting outcome with current model parameters
    output_x, output_y, output_w = steering_friction_model_obj(train_x)

    # evaluate loss function
    loss = loss_fn(output_x,  train_y)
    loss_vec[i] = loss.item()

    # evaluate the gradient of the loss function with respect to the fitting parameters
    loss.backward() 

    # use the evaluated gradient to perform a gradient descent step 
    optimizer_object.step() # this updates parameters


# --- print out parameters ---
[a,b] = steering_friction_model_obj.transform_parameters_norm_2_real()
a,b= a.item(), b.item()
print('Friction due to steering parameters:')
print('a_stfr = ', a)
print('b_stfr = ', b)

# # # --- plot loss function ---
plt.figure()
plt.title('Loss function')
plt.plot(loss_vec,label='tire model loss')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.legend()



# plot fitting results on the model
ax_accx.plot(df['vicon time'].to_numpy(),output_x.detach().cpu().view(-1).numpy(),label='acc_x model with steering friction (model output)',color='k')
ax_accx.legend()

ax_accy.plot(df['vicon time'].to_numpy(),output_y.detach().cpu().view(-1).numpy(),label='acc_y model with steering friction (model output)',color='k')
ax_accy.legend()

ax_accw.plot(df['vicon time'].to_numpy(),output_w.detach().cpu().view(-1).numpy(),label='acc_w model with steering friction (model output)',color='k')
ax_accw.legend()

plt.show()

