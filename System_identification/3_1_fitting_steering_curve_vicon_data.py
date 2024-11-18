from dart_dynamic_models import get_data, steering_curve_model,plot_raw_data,process_vicon_data_kinematics,\
directly_measured_model_parameters,plot_vicon_data, model_functions
from matplotlib import pyplot as plt
import torch
import numpy as np
import pandas as pd
import os

mf = model_functions()

#[theta_correction, l_COM, l_lateral_shift_reference ,lr, lf, Jz, m,m_front_wheel,m_rear_wheel] = directly_measured_model_parameters()




# this assumes that the current directory is DART
folder_path = 'System_identification/Data/steering_identification_25_sept_2024'  # small sinusoidal input
#folder_path = 'System_identification/Data/81_throttle_ramps'
#folder_path = 'System_identification/Data/circles_27_sept_2024'


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
    df = process_vicon_data_kinematics(df_raw_data,steps_shift)

    df.to_csv(file_path, index=False)
    print(f"File '{file_path}' saved.")
else:
    print(f"File '{file_path}' already exists, loading data.")
    df = pd.read_csv(file_path)




if folder_path == 'System_identification/Data/81_throttle_ramps':
    #cut off time instances where the vicon missed a detection to avoid corrupted datapoints
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









# select points where velocity is not too low
df = df[df['vx body'] > 0.3]
df = df[df['vx body'] < 0.6]

# plot raw data
ax0,ax1,ax2 = plot_raw_data(df)



# --- process the raw data ---

w_vec = df['w'].to_numpy()
vx = df['vx body'].to_numpy()

measured_steering_angle= np.arctan(w_vec * (mf.lf_self+mf.lr_self) / vx) 









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

#instantiate class object
steering_curve_model_obj = steering_curve_model()

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


# from historical data
# steering angle curve --from fitting on vicon data
# a_s =  1.4141819477081299
# b_s =  0.36395299434661865
# c_s =  -0.0004661157727241516 - 0.03 # littel adjustment to allign the tire curves
# d_s =  0.517351508140564
# e_s =  1.0095096826553345
model_functions_obj = model_functions()
curve_usual_data = model_functions_obj.steering_2_steering_angle(input_vec,model_functions_obj.a_s_self,
                                                                 model_functions_obj.b_s_self,
                                                                 model_functions_obj.c_s_self,
                                                                 model_functions_obj.d_s_self,
                                                                 model_functions_obj.e_s_self)


y_fitted = steering_curve_model_obj(torch.tensor(input_vec)).detach().numpy()

fig = plt.figure(figsize=(10, 6))
fig.subplots_adjust(top=0.985,
bottom=0.11,
left=0.13,
right=0.99,
hspace=0.2,
wspace=0.2)

fig1, ((ax)) = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
color_code_label = 'vx body'
scatter = plt.scatter(df['steering'].to_numpy(), measured_steering_angle, label = 'data',c=df[color_code_label].to_numpy(),cmap='plasma') 
ax.scatter(np.array([0.0]),np.array([0.0]),color='k',label='zero',marker='+', zorder=20)
ax.plot(input_vec, y_fitted ,'orangered',label = "steering curve",linewidth=4)
ax.plot(input_vec, curve_usual_data ,'navy',label = "steering curve from default parameters",linewidth=4)
ax.set_xlabel("Steering input")
ax.set_ylabel("Steering angle [rad]")

cbar1 = fig1.colorbar(scatter,ax=ax)
cbar1.set_label(color_code_label)  # Label the colorbar
#plt.title('Steering angle vs steering command scatter plot')
ax.legend()



plt.show()

