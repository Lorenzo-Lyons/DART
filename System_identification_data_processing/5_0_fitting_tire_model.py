from dart_dynamic_models import get_data, plot_raw_data, process_raw_vicon_data,\
plot_vicon_data,pacejka_tire_model,process_vicon_data_kinematics,model_functions
from matplotlib import pyplot as plt
import torch
import numpy as np
import pandas as pd
import os


# load parameters
mf = model_functions()



# select data folder NOTE: this assumes that the current directory is DART
#folder_path = 'System_identification_data_processing/Data/8_circles_rubbery_floor_1_file'
folder_path = 'System_identification_data_processing/Data/circles_27_sept_2024'

# --- Starting data processing  ------------------------------------------------


# #robot2vicon_delay = 5 # samples delay
# df_raw_data = get_data(folder_path)

# # process the data
# steps_shift = 10 # decide to filter more or less the vicon data
# df = process_raw_vicon_data(df_raw_data,steps_shift)



# check if there is a processed vicon data file already
file_name = 'processed_vicon_data.csv'
# Check if the CSV file exists in the folder
file_path = os.path.join(folder_path, file_name)

if not os.path.isfile(file_path):
    steps_shift = 5 # decide to filter more or less the vicon data
    df_raw_data = get_data(folder_path)

    # cut time
    #df_raw_data = df_raw_data[df_raw_data['vicon time']<235]

    df_kinematics = process_vicon_data_kinematics(df_raw_data,steps_shift)
    df = process_raw_vicon_data(df_kinematics,steps_shift)

    df.to_csv(file_path, index=False)
    print(f"File '{file_path}' saved.")
else:
    print(f"File '{file_path}' already exists, loading data.")
    df = pd.read_csv(file_path)










if folder_path == 'System_identification_data_processing/Data/81_throttle_ramps_only_steer03':
    # cut the data in two parts cause something is wrong in the middle (probably a temporary lag in the network)
    df1=df[df['vicon time']<110]  #  60 150
    df2=df[df['vicon time']>185.5] 
    # Concatenate vertically
    df = pd.concat([df1, df2], axis=0)
    # Reset the index if you want a clean, continuous index
    df.reset_index(drop=True, inplace=True)

elif folder_path == 'System_identification_data_processing/Data/steering_identification_25_sept_2024':
    # cut the data in two parts cause something is wrong in the middle (probably a temporary lag in the network)
    df=df[df['vicon time']<460]
#cut off time instances where the vicon missed a detection to avoid corrupted datapoints
elif  folder_path == 'System_identification_data_processing/Data/81_throttle_ramps':
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



if folder_path == 'System_identification_data_processing/Data/circles_27_sept_2024':
    df1 = df[df['vicon time']>1]
    df1 = df1[df1['vicon time']<375]

    df2 = df[df['vicon time']>377]
    df2 = df2[df2['vicon time']<830]

    df3 = df[df['vicon time']>860]
    df3 = df3[df3['vicon time']<1000]

    df = pd.concat([df1,df2,df3])




# plot raw data
ax0,ax1,ax2 = plot_raw_data(df)


ax_wheel_f_alpha,ax_wheel_r_alpha,ax_total_force_front,\
ax_total_force_rear,ax_lat_force,ax_long_force,\
ax_acc_x_body,ax_acc_y_body,ax_acc_w = plot_vicon_data(df) 




# --------------- fitting tire model---------------
# fitting tyre models
# define number of training iterations
train_its = 1000
learning_rate = 0.001 

print('')
print('Fitting pacejka-like culomb friction tire model ')

#instantiate the model
pacejka_tire_model_obj = pacejka_tire_model()

#define loss and optimizer objects
loss_fn = torch.nn.MSELoss() 
optimizer_object = torch.optim.Adam(pacejka_tire_model_obj.parameters(), lr=learning_rate)

# generate data in tensor form for torch
data_columns = ['slip angle front','slip angle rear'] # velocities

train_x = torch.tensor(df[data_columns].to_numpy()).cuda()
train_y = torch.unsqueeze(torch.tensor(np.concatenate((df['Fy front wheel'].to_numpy(),df['Fy rear wheel'].to_numpy()))),1).cuda() 

# save loss values for later plot
loss_vec = np.zeros(train_its)

# train the model
for i in range(train_its):
    # clear gradient information from previous step before re-evaluating it for the current iteration
    optimizer_object.zero_grad()  
    
    # compute fitting outcome with current model parameters
    F_y_f,F_y_r = pacejka_tire_model_obj(train_x)
    output = torch.cat((F_y_f,F_y_r),0) 

    # evaluate loss function
    loss = loss_fn(output,  train_y)
    loss_vec[i] = loss.item()

    # evaluate the gradient of the loss function with respect to the fitting parameters
    loss.backward() 

    # use the evaluated gradient to perform a gradient descent step 
    optimizer_object.step() # this updates parameters


# --- print out parameters ---
[d_t_f,c_t_f,b_t_f,d_t_r,c_t_r,b_t_r] = pacejka_tire_model_obj.transform_parameters_norm_2_real()

print('# Front wheel parameters:')
print('d_t_f = ', d_t_f.item())
print('c_t_f = ', c_t_f.item())
print('b_t_f = ', b_t_f.item())
#print('e_t_f = ', e_t_f.item())


print('# Rear wheel parameters:')
print('d_t_r = ', d_t_r.item())
print('c_t_r = ', c_t_r.item())
print('b_t_r = ', b_t_r.item())
#print('e_t_r = ', e_t_r.item())


# # # --- plot loss function ---
plt.figure()
plt.title('Loss function')
plt.plot(loss_vec,label='tire model loss')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.legend()



# evaluate model on plotting interval
#v_y_wheel_plotting_front = torch.unsqueeze(torch.linspace(torch.min(train_x[:,0]),torch.max(train_x[:,0]),100),1).cuda()
alpha_f_plotting_front = torch.unsqueeze(torch.linspace(torch.min(train_x[:,0]),torch.max(train_x[:,0]),100),1).cuda()
lateral_force_vec_front = pacejka_tire_model_obj.lateral_tire_force(alpha_f_plotting_front,d_t_f,c_t_f,b_t_f,mf.m_front_wheel_self).detach().cpu().numpy()


# do the same for he rear wheel
alpha_r_plotting_rear = torch.unsqueeze(torch.linspace(torch.min(train_x[:,1]),torch.max(train_x[:,1]),100),1).cuda()
lateral_force_vec_rear = pacejka_tire_model_obj.lateral_tire_force(alpha_r_plotting_rear,d_t_r,c_t_r,b_t_r,mf.m_rear_wheel_self).detach().cpu().numpy()


ax_wheel_f_alpha.plot(alpha_f_plotting_front.cpu(),lateral_force_vec_front,color='#2c4251',label='Tire model',linewidth=4,linestyle='-')
ax_wheel_f_alpha.legend()

ax_wheel_r_alpha.plot(alpha_r_plotting_rear.cpu(),lateral_force_vec_rear,color='#2c4251',label='Tire model',linewidth=4,linestyle='-')
ax_wheel_r_alpha.legend()

# plot model outputs
ax_wheel_f_alpha.scatter(df['slip angle front'],F_y_f.detach().cpu().numpy(),color='k',label='Tire model output (with pitch influece)',s=2)
ax_wheel_r_alpha.scatter(df['slip angle rear'],F_y_r.detach().cpu().numpy(),color='k',label='Tire model output (with pitch influece)',s=2)


plt.show()







