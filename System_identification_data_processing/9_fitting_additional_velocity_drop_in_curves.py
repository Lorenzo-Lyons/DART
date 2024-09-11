from functions_for_data_processing import get_data, plot_raw_data, process_raw_vicon_data,plot_vicon_data\
,dyn_model_culomb_tires,steering_friction_model
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





# ---------------  ----------------
theta_correction = +0.5/180*np.pi 
lr = 0.135 # reference point location taken by the vicon system
COM_positon = 0.09375 #0.0925 #measuring from the rear wheel
# ------------------------------------------------------

# select data folder NOTE: this assumes that the current directory is DART
#folder_path = 'System_identification_data_processing/Data/9_model_validation_long_term_predictions'
#folder_path = 'System_identification_data_processing/Data/8_circles_rubbery_floor_1_file'
#folder_path = 'System_identification_data_processing/Data/91_model_validation_long_term_predictions_fast'
folder_path = 'System_identification_data_processing/Data/81_throttle_ramps'
#folder_path = 'System_identification_data_processing/Data/81_throttle_ramps_only_steer03'


# steering dynamics time constant
# Time constant in the steering dynamics
steer_time_constant = 0.065  # in time domain,later converted to discrete time filtering coefficient




# car parameters
l = 0.175 # length of the car
m = 1.67 # mass
Jz_0 = 0.006513 # Moment of inertia of uniform rectangle of shape 0.18 x 0.12

# Automatically adjust following parameters according to tweaked values
l_COM = lr - COM_positon #distance of the reference point from the centre of mass)
Jz = Jz_0 + m*l_COM**2 # correcting the moment of inertia for the extra distance of the reference point from the COM
lf = l-lr


# fitted parameters
# construct a model that takes as inputs Vx,Vy,W,tau,Steer ---> Vx_dot,Vy_dot,W_dot

# motor model  (from fitting both friction and motor model at the same time) 
a_m =  24.492280960083008
b_m =  4.685335159301758
c_m =  -0.15649330615997314
d_m =  0.7791252732276917


# rolling friction model
a_f =  1.3823319673538208
b_f =  5.101551055908203
c_f =  0.7144842743873596
d_f =  -0.10897014290094376



# steering angle curve
a_s =  1.6379064321517944
b_s =  0.3301370143890381 #+ 0.04
c_s =  0.019644200801849365 #- 0.03 # this value can be tweaked to get the tyre model curves to allign better
d_s =  0.37879398465156555 #+ 0.04
e_s =  1.6578725576400757

# tire model
d_t =  -7.446990013122559
c_t =  0.7474039196968079
b_t =  5.093936443328857

# filtering coefficients
alpha_steer_filter = 0.60

# Starting data processing

# check if there is a processed vicon data file already
# Check if the file exists
file_name = 'processed_vicon_data.csv'
# Check if the CSV file exists in the folder
file_path = os.path.join(folder_path, file_name)

if not os.path.isfile(file_path):
    # If the file does not exist, process the raw data
    # get the raw data
    df_raw_data = get_data(folder_path)

    # account for latency between vehicle and vicon system (the vehicle inputs are relayed with a certain delay)
    # NOTE that the delay is actually not constant, but it is assumed to be constant for simplicity
    # so there will be some little timing discrepancies between predicted stated and data

    robot_vicon_time_delay_st = 5 #6 # seven periods (at 100 Hz is 0.07s)
    robot_vicon_time_delay_th = 10 # seven periods (at 100 Hz is 0.07s)
    df_raw_data['steering'] = df_raw_data['steering'].shift(periods=-robot_vicon_time_delay_st)
    df_raw_data['throttle'] = df_raw_data['throttle'].shift(periods=-robot_vicon_time_delay_th)

    # handle the last values that will be nan
    df_raw_data['steering'].iloc[-robot_vicon_time_delay_st:] = 0
    df_raw_data['throttle'].iloc[-robot_vicon_time_delay_th:] = 0


    # process the data
    df = process_raw_vicon_data(df_raw_data,lf,lr,theta_correction,m,Jz,l_COM,a_s,b_s,c_s,d_s,e_s,alpha_steer_filter)

    df.to_csv(file_path, index=False)
    print(f"File '{file_path}' saved.")
else:
    print(f"File '{file_path}' already exists, loading data.")
    df = pd.read_csv(file_path)


# select subset of datapoints to do the model fitting
#df = df[df['vx body']>0.5]
#df = df[df['vx body']<3.0]





# cut the data
#df = df[df['vicon time'] < 85]

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


# # test the filter on a sine wave to see if it works
# test_x_10_Hz = np.linspace(0,10,10)
# test_x_100_Hz = np.linspace(0,10,100)

# test_y_10_Hz = np.sin(test_x_10_Hz)
# test_y_100_Hz = np.sin(test_x_100_Hz)

# filtered_10_Hz = np.zeros(len(test_x_10_Hz))
# filtered_100_Hz = np.zeros(len(test_x_100_Hz))

# for i in range(1, len(test_x_10_Hz)):
#     filtered_10_Hz[i] = d_m * test_y_10_Hz[i] + (1 - d_m) * filtered_10_Hz[i-1]
    
# for i in range(1, len(test_x_100_Hz)):
#     filtered_100_Hz[i] = d_m_100Hz * test_y_100_Hz[i] + (1 - d_m_100Hz) * filtered_100_Hz[i-1]

# figure, ax = plt.subplots()
# ax.plot(test_x_100_Hz,test_y_100_Hz,label='original signal 100 Hz')
# ax.plot(test_x_10_Hz,filtered_10_Hz,label='filtered signal 10 Hz')
# ax.plot(test_x_100_Hz,filtered_100_Hz,label='filtered signal 100 Hz')
# ax.legend()
# plt.show()





# plot raw data
#ax0,ax1,ax2 = plot_raw_data(df)

# plot vicon related data (longitudinal and lateral velocities, yaw rate related)
#plot_vicon_data(df)

# plot filtered throttle signal
plt.figure()
plt.plot(df['vicon time'].to_numpy(),df['throttle'].to_numpy(),label='throttle')
plt.plot(df['vicon time'].to_numpy(),df['throttle filtered'].to_numpy(),label='throttle filtered')
plt.plot(df['vicon time'].to_numpy(),df['ax body no centrifugal'].to_numpy(),label='acc_x',color = 'dodgerblue')
plt.xlabel('Time [s]')
plt.ylabel('Throttle')
plt.legend()
 


# --- produce data for fitting ---
a_stfr= []  # give empty correction terms to not use them
b_stfr=[]
# define model NOTE: this will give you the absolute accelerations measured in the body frame
dynamic_model = dyn_model_culomb_tires(m,lr,lf,l_COM,Jz,d_t,c_t,b_t,
                 a_m,b_m,c_m,
                 a_f,b_f,c_f,d_f,
                 a_stfr,b_stfr)



columns_to_extract = ['vx body', 'vy body', 'w_abs_filtered', 'throttle filtered' ,'steering angle time delayed']
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

# plt.plot(df['vicon time'].to_numpy(),df['throttle filtered'].to_numpy(),label='throttle filtered',color='blue')
# plt.plot(df['vicon time'].to_numpy(),df['steering angle time delayed'].to_numpy(),label='steering angle time delayed',color='orangered')
# plt.plot(df['vicon time'].to_numpy(),df['vx body'].to_numpy(),label='vx body',color='green')
# plt.plot(df['vicon time'].to_numpy(),acc_centrifugal_in_x,label='acc centrifugal in x',color='purple')
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
    df['steering angle time delayed'].to_numpy(),  # y-axis
    missing_force,                    # z-axis 
    c=df['steering angle time delayed'].to_numpy(),  # color coded by 'steering angle time delayed'
    cmap='viridis'  # Colormap
)



#plt.plot(velocity_range,friction_curve*rescale,label='friction curve rescaled',color='k',alpha=0.5)
#plt.plot(df['vicon time'].to_numpy(),df['steering angle time delayed'].to_numpy(),label='steering',color='orangered')
ax.set_xlabel('vx body')
ax.set_ylabel('steering angle')
ax.set_zlabel('Force [N]')
# Add a colorbar to the plot
colorbar = fig.colorbar(scatter, label='steering angle time delayed')
#ax.legend()





# --------------- fitting extra steering friction model---------------


# fitting tyre models
# define first guess for parameters
initial_guess = torch.ones(2) * 0.5 # initialize parameters in the middle of their range constraint
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
print('a = ', a)
print('b = ', b)

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

