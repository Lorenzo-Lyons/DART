from functions_for_data_processing import get_data, plot_raw_data, friction_curve_model
from matplotlib import pyplot as plt
import torch
import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd
from scipy.signal import savgol_filter
# set font size for figures
import matplotlib
font = {'family' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)



# this assumes that the current directory is DART
folder_path = 'System_identification_data_processing/Data/1_step_input_data'  

# get the raw data
df_raw_data = get_data(folder_path)

# plot raw data
ax0,ax1,ax2 = plot_raw_data(df_raw_data)

# smooth velocity data
# Set the window size for the moving average
window_size = 5 
poly_order = 2

# Apply Savitzky-Golay filter
smoothed_vel_encoder = savgol_filter(df_raw_data['vel encoder'].to_numpy(), window_size, poly_order)

# add smoothed velocity data to plot
ax1.plot(df_raw_data['elapsed time sensors'].to_numpy(),smoothed_vel_encoder,label="vel encoder smoothed",color='k',linestyle='--')
plt.legend()

# setdelay
delay_th = 0.1 # [s]

# --- process the raw data ---
df = df_raw_data[['elapsed time sensors','throttle']].copy() 
df['throttle delayed'] = np.interp(df_raw_data['elapsed time sensors'].to_numpy()-delay_th, df_raw_data['elapsed time sensors'].to_numpy(), df_raw_data['throttle'].to_numpy())
df['vel encoder smoothed'] =  smoothed_vel_encoder # df_raw_data['vel encoder'] #non smoothed

spl_vel = CubicSpline(df['elapsed time sensors'].to_numpy(), df['vel encoder smoothed'].to_numpy())
m =1.67 #mass of the robot
df['force'] =  m * spl_vel(df['elapsed time sensors'].to_numpy(),1) # take the first derivative of the spline

# select only the data points where th throttle is 0
df = df[df['throttle']==0]
df = df[df['force']>-4]



# plot velocity information against force
# This is usefull to guide the choice of parameter bounds
#NOTE: pay attention that the model is fitting on the accelerations, but the parameters are designed to give a force,
# so here to get a feel for a good initial guess it's better to show the force rather than the acceleration


plt.figure()
plt.title('velocity Vs force')
plt.plot(df['elapsed time sensors'].to_numpy(),df['vel encoder smoothed'].to_numpy(),label="vel encoder smoothed",color='dodgerblue')
plt.plot(df['elapsed time sensors'].to_numpy(),df['force'].to_numpy(),label="force",color='k')
plt.step(df['elapsed time sensors'].to_numpy(),df['throttle delayed'].to_numpy(),label="throttle delayed",color='dimgray')
plt.step(df['elapsed time sensors'].to_numpy(),df['throttle'].to_numpy(),label="throttle",color='gray',linestyle="--")
plt.legend()



# --------------- fitting acceleration curve--------------- 
print('')
print('Fitting acceleration curve model')


# define first guess for parameters
initial_guess = torch.ones(3) * 0.5 # initialize parameters in the middle of their range constraint
# NOTE that the parmeter range constraint is set in the self.transform_parameters_norm_2_real method.

#instantiate the model
friction_curve_model_obj = friction_curve_model(initial_guess)

# define number of training iterations
train_its = 100

#define loss and optimizer objects
loss_fn = torch.nn.MSELoss() # reduction = 'mean'
optimizer_object = torch.optim.Adam(friction_curve_model_obj.parameters(), lr=0.1)
        
# generate data in tensor form for torch
train_x = torch.unsqueeze(torch.tensor(df['vel encoder smoothed'].to_numpy()),1).cuda()
train_y = torch.unsqueeze(torch.tensor(df['force'].to_numpy()),1).cuda()

# save loss values for later plot
loss_vec = np.zeros(train_its)

# train the model
for i in range(train_its):
    # clear gradient information from previous step before re-evaluating it for the current iteration
    optimizer_object.zero_grad()  
    
    # compute fitting outcome with current model parameters
    output = friction_curve_model_obj(train_x)

    # evaluate loss function
    loss = loss_fn(output,  train_y)
    loss_vec[i] = loss.item()

    # evaluate the gradient of the loss function with respect to the fitting parameters
    loss.backward() 

    # use the evaluated gradient to perform a gradient descent step 
    optimizer_object.step() # this updates parameters


# --- print out parameters ---
[a,b,c] = friction_curve_model_obj.transform_parameters_norm_2_real()
a, b, c = a.item(), b.item(), c.item()
print('a = ', a)
print('b = ', b)
print('c = ', c)



# --- plot loss function ---
plt.figure()
plt.title('Loss')
plt.plot(loss_vec)
plt.xlabel('iterations')
plt.ylabel('loss')


# --- plot friction curve ---
v_vec = torch.unsqueeze(torch.linspace(0,df['vel encoder smoothed'].max(),100),1).cuda()
force_vec = friction_curve_model_obj(v_vec).detach().cpu().numpy()

fig = plt.figure(figsize=(10, 6))
fig.subplots_adjust(top=0.985,
                    bottom=0.11,
                    left=0.095,
                    right=0.995,
                    hspace=0.2,
                    wspace=0.2)

plt.scatter(train_x.cpu().detach().numpy(),train_y.cpu().detach().numpy(),label='data')
plt.plot(v_vec.cpu().numpy(),force_vec,label = 'Friction curve',zorder=20,color='orangered',linewidth=4,linestyle='-')
plt.xlabel('Velocity [m\s]')
plt.ylabel('Force [N]')
#ax1.set_title('Friction curve')
plt.legend()

plt.show()


