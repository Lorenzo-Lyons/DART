from functions_for_data_processing import get_data, plot_raw_data, motor_curve_model
from matplotlib import pyplot as plt
import torch
import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
# set font size for figures
import matplotlib
font = {'family' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)



# this assumes that the current directory is DART
folder_path = 'System_identification_data_processing/Data/2_step_input_data' 





# get the raw data
df_raw_data = get_data(folder_path)

# plot raw data
ax0,ax1,ax2 = plot_raw_data(df_raw_data)
#plt.show()


# smooth velocity data
# Set the window size for the moving average
window_size = 5#5
poly_order = 2

# Apply Savitzky-Golay filter
smoothed_vel_encoder = savgol_filter(df_raw_data['vel encoder'].to_numpy(), window_size, poly_order)

# v_spline = UnivariateSpline(df_raw_data['elapsed time sensors'].to_numpy(), df_raw_data['vel encoder'].to_numpy(), s=0.01)
# smoothed_vel_encoder = v_spline(df_raw_data['elapsed time sensors'].to_numpy())

ax1.plot(df_raw_data['elapsed time sensors'].to_numpy(),smoothed_vel_encoder,label="vel encoder smoothed",color='k',linestyle='--')
plt.legend()




# identify  delay
delay_th = 0.1 # [s]

# process the raw data
m =1.67 #mass of the robot
# friction curve parameters
a_friction  =  1.6837230920791626
b_friction  =  13.49715518951416
c_friction  =  0.3352389633655548


# df = process_raw_data_acceleration(df_raw_data, delay_st)
df = df_raw_data[['elapsed time sensors','throttle']].copy() 
df['throttle delayed'] = np.interp(df_raw_data['elapsed time sensors'].to_numpy()-delay_th, df_raw_data['elapsed time sensors'].to_numpy(), df_raw_data['throttle'].to_numpy())
df['vel encoder smoothed'] =  smoothed_vel_encoder # df_raw_data['vel encoder'] #non smoothed

spl_vel = CubicSpline(df['elapsed time sensors'].to_numpy(), df['vel encoder smoothed'].to_numpy())
df['force'] =   m * spl_vel(df['elapsed time sensors'].to_numpy(),1) # take the first derivative of the spline
df['friction force'] = + a_friction * np.tanh(b_friction  * df['vel encoder smoothed'] ) + df['vel encoder smoothed'] * c_friction
df['motor force'] = df['force'] + df['friction force']


#df = df[df['vel encoder smoothed']>0.2]
#v_spline_dev = v_spline.derivative()
#df['force'] =  v_spline_dev(df['elapsed time sensors'].to_numpy()) # take the first derivative of the spline








# plot velocity information against force
# This is usefull to guide the choice of parameter bounds
#NOTE: pay attention that the model is fitting on the accelerations, but the parameters are designed to give a force,
# so here to get a feel for a good initial guess it's better to show the force rather than the acceleration



fig, ((ax3)) = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
fig.subplots_adjust(top=0.985, bottom=0.11, left=0.07, right=1.0, hspace=0.2, wspace=0.2)

ax3.set_title('velocity Vs motor force')
ax3.plot(df['elapsed time sensors'].to_numpy(),df['vel encoder smoothed'].to_numpy(),label="velocity [m/s]",color='dodgerblue',linewidth=3)
#ax3.plot(df['elapsed time sensors'].to_numpy(),df['force'].to_numpy(),label="force",color='gray')
ax3.plot(df['elapsed time sensors'].to_numpy(),df['friction force'].to_numpy(),label="estimated friction force [N]",color='dimgray',linewidth=3)
ax3.plot(df['elapsed time sensors'].to_numpy(),df['motor force'].to_numpy(),label="motor force [N]",color='k',linewidth=3)
#ax3.step(df['elapsed time sensors'].to_numpy(),df['throttle delayed'].to_numpy(),label="throttle delayed",color='dimgray')
#ax3.step(df['elapsed time sensors'].to_numpy(),df['throttle'].to_numpy(),label="throttle",color='gray',linestyle="--")
ax3.set_xlabel('time [s]')
ax3.legend()


# --------------- fitting acceleration curve--------------- 
print('')
print('Fitting acceleration curve model')

# define first guess for parameters
initial_guess = torch.ones(3) * 0.5 # initialize parameters in the middle of their range constraint
# NOTE that the parmeter range constraint is set in the self.transform_parameters_norm_2_real method.
initial_guess[0] = torch.Tensor([0.95])



# NOTE that the parmeter range constraint is set in motor_curve_model.transform_parameters_norm_2_real method.
motor_curve_model_obj = motor_curve_model(initial_guess)

# define number of training iterations
train_its = 500

#define loss and optimizer objects
loss_fn = torch.nn.MSELoss(reduction = 'mean') 
optimizer_object = torch.optim.Adam(motor_curve_model_obj.parameters(), lr=0.01)
        
# generate data in tensor form for torch
train_x = torch.tensor(df[['throttle','vel encoder smoothed']].to_numpy()).cuda()
#train_x = torch.unsqueeze(torch.tensor(df['throttle'].to_numpy()),1).cuda()
train_y = torch.unsqueeze(torch.tensor(df['motor force'].to_numpy()),1).cuda()

# save loss values for later plot
loss_vec = np.zeros(train_its)







# train the model
for i in range(train_its):
    # clear gradient information from previous step before re-evaluating it for the current iteration
    optimizer_object.zero_grad()  
    
    # compute fitting outcome with current model parameters
    output = motor_curve_model_obj(train_x)

    # evaluate loss function
    loss = loss_fn(output,  train_y)
    loss_vec[i] = loss.item()

    # evaluate the gradient of the loss function with respect to the fitting parameters
    loss.backward() 

    # use the evaluated gradient to perform a gradient descent step 
    optimizer_object.step() # this updates parameters automatically according to the optimizer you chose

# --- print out parameters ---
[a,b,c] = motor_curve_model_obj.transform_parameters_norm_2_real()
a, b, c = a.item(), b.item(), c.item()
print('a = ', a)
print('b = ', b)
print('c = ', c)


# plot loss function
plt.figure()
plt.title('Loss')
plt.plot(loss_vec)
plt.xlabel('iterations')
plt.ylabel('loss')


# --- plot motor curve ---
# throttle_vec = torch.unsqueeze(torch.linspace(0,df['throttle'].max(),100),1).cuda()
# motor_force_vec = motor_curve_model_obj(throttle_vec).detach().cpu().numpy()

# fig1, ((ax1)) = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
# ax1.scatter(train_x.cpu().detach().numpy(),train_y.cpu().detach().numpy(),label='data',s=1)
# ax1.plot(throttle_vec.cpu().numpy(),motor_force_vec,label = 'motor curve',zorder=20,color='orangered',linewidth=4,linestyle='-')
# ax1.set_xlabel('throttle')
# ax1.set_ylabel('N')
# ax1.set_title('Motor curve')
# ax1.legend()
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
x = train_x[:,0].cpu().detach().numpy()
y = train_x[:,1].cpu().detach().numpy()
z = train_y.cpu().detach().numpy()
ax.scatter(x, y, z, c='b', marker='o')



# Create a meshgrid for the surface plot

# find maximum value of velocity
v_max = a/b

throttle_range = np.linspace(0, max(x), 100)
velocity_rage = np.linspace(0, v_max, 100)
throttle_grid, velocity_grid = np.meshgrid(throttle_range, velocity_rage)
# Create input points
input_points = np.column_stack(
    (
        throttle_grid.flatten(),
        velocity_grid.flatten(),
    )
)

input_grid = torch.tensor(input_points, dtype=torch.float32).cuda()
Force_grid = motor_curve_model_obj(input_grid).detach().cpu().view(100, 100).numpy()  # Replace with your surface data

# Plot the surface
ax.plot_surface(throttle_grid, velocity_grid, Force_grid, cmap='viridis', alpha=1)
# Set labels
ax.set_xlabel('throttle')
ax.set_ylabel('velocity')
ax.set_zlabel('Motor force')

# plotting the obtained motor curve as a level plot
throttle_levels = np.linspace(-c,0.4,5).tolist()

fig = plt.figure(figsize=(10, 6))
fig.subplots_adjust(top=0.985, bottom=0.11, left=0.07, right=1.0, hspace=0.2, wspace=0.2)

#heatmap = plt.imshow(throttle_grid, extent=[velocity_grid.min(), Force_grid.max(), Force_grid.min(), throttle_grid.max()], origin='lower', cmap='plasma')
contour1 = plt.contourf(velocity_grid, Force_grid ,throttle_grid, levels=100, cmap='plasma') 
contour2 = plt.contour(velocity_grid, Force_grid ,throttle_grid, levels=throttle_levels, colors='black', linestyles='solid', linewidths=2) 
cbar = plt.colorbar(contour1, label='Throttle',ticks=[0, *throttle_levels, 1],format='%.2f')

# from matplotlib.ticker import StrMethodFormatter
# cbar.ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
# cbar.set_ticks([0, *throttle_levels, 1])

#cbar.update_ticks()
# Add labels for contour lines
plt.clabel(contour2, inline=True, fontsize=18, fmt='%1.2f')
# Set labels
plt.xlabel('Velocity [m/s]')
plt.ylabel('Force [N]')

df_data = df[df['vel encoder smoothed']>1]
df_data = df_data[df_data['vel encoder smoothed']<3]
vel_data = df_data['vel encoder smoothed'][df_data['throttle']==0.4000000059604645].to_numpy()
mot_force_data = df_data['motor force'][df_data['throttle']==0.4000000059604645].to_numpy()
plt.scatter(vel_data,mot_force_data,color='k',label='data for throttle = 0.4')
plt.legend()


#add predicted motor force to previous plot
ax3.plot(df['elapsed time sensors'].to_numpy(),output.cpu().detach().numpy(),color='orangered',linewidth=3,label="estimated motor force")


plt.show()





