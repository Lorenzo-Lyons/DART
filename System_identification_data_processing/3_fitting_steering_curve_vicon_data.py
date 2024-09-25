from functions_for_data_processing import get_data,process_raw_data_steering, steering_curve_model,plot_raw_data,process_vicon_data_kinematics,\
directly_measured_model_parameters,plot_vicon_data
from matplotlib import pyplot as plt
import torch
import numpy as np
import pandas as pd
import os
# set font size for figures
import matplotlib
font = {'family' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

[theta_correction, l_COM, l_lateral_shift_reference ,lr, lf, Jz, m,m_front_wheel,m_rear_wheel] = directly_measured_model_parameters()

# this assumes that the current directory is DART
folder_path = 'System_identification_data_processing/Data/steering_identification_25_sept_2024'  # small sinusoidal input

# # get the raw data
# df_raw_data = get_data(folder_path)

# # process the data
# steps_shift = 5 # decide to filter more or less the vicon data





# --- Starting data processing  ------------------------------------------------

# # check if there is a processed vicon data file already
file_name = 'processed_kinematics_vicon_data.csv'
# Check if the CSV file exists in the folder
file_path = os.path.join(folder_path, file_name)

if not os.path.isfile(file_path):
    # If the file does not exist, process the raw data
    # get the raw data
    df_raw_data = get_data(folder_path)

    # process the data
    steps_shift = 5 # decide to filter more or less the vicon data
    df = process_vicon_data_kinematics(df_raw_data,steps_shift,theta_correction, l_COM, l_lateral_shift_reference)

    df.to_csv(file_path, index=False)
    print(f"File '{file_path}' saved.")
else:
    print(f"File '{file_path}' already exists, loading data.")
    df = pd.read_csv(file_path)



# select points where velocity is not too low
df = df[df['vx body'] > 0.3]



# plot raw data
ax0,ax1,ax2 = plot_raw_data(df)



# --- process the raw data ---

w_vec = df['w'].to_numpy()
vx = df['vx body'].to_numpy()
measured_steering_angle= np.arctan2(w_vec * (lf+lr) ,  vx) 









#plot the processed data
plotting_time_vec = df['elapsed time sensors'].to_numpy()
fig1, ((ax0)) = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
ax0.set_title('Steering curve fitting data')
#ax0.plot(plotting_time_vec, df['steering angle'].to_numpy(), label="steering angle [rad]", color='orchid')
ax0.plot(plotting_time_vec, df['steering'].to_numpy(), label="steering raw", color='pink')
#ax0.plot(plotting_time_vec, df['steering delayed'].to_numpy(), label="steering delayed ", color='k')
ax0.set_xlabel('time [s]')
ax0.legend()




# --------------- fitting steering curve--------------- 
print('')
print('Fitting steering curve model')

initial_guess = torch.ones(5)*0.5
#initial_guess[0] = torch.Tensor([0.95])

#instantiate class object
steering_curve_model_obj = steering_curve_model(initial_guess)

# define number of training iterations
Steer_train_its = 1000

#define loss and optimizer objects
steer_loss_fn = torch.nn.MSELoss(reduction = 'mean') 
steer_optimizer_object = torch.optim.Adam(steering_curve_model_obj.parameters(), lr=0.003)
        
# generate data in tensor form for torch
train_x_steering = torch.tensor(df['steering'].to_numpy())
train_y_steering = torch.tensor(measured_steering_angle)

# save loss values for later plot
loss_vec = np.zeros(Steer_train_its)

# train the model
for i in range(Steer_train_its):
    # clear gradient information from previous step before re-evaluating it for the current iteration
    steer_optimizer_object.zero_grad()  
    
    # compute fitting outcome with current model parameters
    steering_angle_pred = steering_curve_model_obj(train_x_steering)

    # evaluate loss function
    loss = steer_loss_fn(steering_angle_pred,  train_y_steering)
    loss_vec[i] = loss.item()

    # evaluate the gradient of the loss function with respect to the fitting parameters
    loss.backward() 

    # use the evaluated gradient to perform a gradient descent step 
    steer_optimizer_object.step() # this updates parameters automatically according to the optimizer you chose


# plot loss function
plt.figure()
plt.title('Loss')
plt.plot(loss_vec)
plt.xlabel('iterations')
plt.ylabel('loss')

# --- print out parameters ---
[a_s,b_s,c_s,d_s,e_s] = steering_curve_model_obj.transform_parameters_norm_2_real()
a_s, b_s, c_s,d_s,e_s = a_s.item(), b_s.item(), c_s.item(), d_s.item(),e_s.item()
print('a_s = ', a_s)
print('b_s = ', b_s)
print('c_s = ', c_s)
print('d_s = ', d_s)
print('e_s = ', e_s)

# plot curve over the fitting data
input_vec = np.linspace(-1,1,100)
y_fitted = steering_curve_model_obj(torch.tensor(input_vec)).detach().numpy()

fig = plt.figure(figsize=(10, 6))
fig.subplots_adjust(top=0.985,
bottom=0.11,
left=0.13,
right=0.99,
hspace=0.2,
wspace=0.2)


plt.scatter(df['steering'].to_numpy(), measured_steering_angle, label = 'data') 
plt.plot(input_vec, y_fitted ,'orangered',label = "steering curve",linewidth=4)
plt.xlabel("Steering input")
plt.ylabel("Steering angle [rad]")
#plt.title('Steering angle vs steering command scatter plot')
plt.legend()
plt.show()


