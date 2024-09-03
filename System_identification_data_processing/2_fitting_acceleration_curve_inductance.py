from functions_for_data_processing import get_data, plot_raw_data, motor_curve_model_inductance
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
#folder_path = 'System_identification_data_processing/Data/2_step_input_data' 
folder_path = 'System_identification_data_processing/Data/21_step_input_data_rubbery_floor' 
#folder_path = 'System_identification_data_processing/Data/21_step_input_data_rubbery_floor_v_less15' # velocity up to 1.5 m/s
#folder_path = 'System_identification_data_processing/Data/21_step_input_data_rubbery_floor_TEST' 
#folder_path = 'System_identification_data_processing/Data/21_step_input_data_rubbery_floor_TEST_inductance' 



# get the raw data
df_raw_data = get_data(folder_path)

# plot raw data
ax0,ax1,ax2 = plot_raw_data(df_raw_data)
#plt.show()


# identify  delay
# ---------------------
# The delay we need to measure is between the throttlel and a change in velocity as measured by the data.
# indeed there seems to be no delay between the two signals
# ---------------------
#delay_th = 1 # [steps, i.e. 0.1s]



# process the raw data
m =1.67 #mass of the robot
# friction curve parameters from smooth floor
# a_friction  =  1.6837230920791626
# b_friction  =  13.49715518951416
# c_friction  =  0.3352389633655548

# friction curve parameters from rubbery floor
a_friction =  1.5837167501449585
b_friction =  14.215554237365723
c_friction =  0.5013455152511597
d_friction =  -0.057962968945503235



# clone data structure
df = df_raw_data[['elapsed time sensors','throttle','vel encoder']].copy() 


# using raw velocity data
# ---------------------
# using FORWARD DIFFERENCE to compute the velocity is the best option because, as can be seen in the data,
# when the throttle is changed, the velocity SLOPE (i.e. the acceleration) changes from the next time step.
# ---------------------
# df['vel encoder'] = df_raw_data['vel encoder']
# spl_vel = CubicSpline(df['elapsed time sensors'].to_numpy(), df_raw_data['vel encoder'].to_numpy())  #df['vel encoder smoothed']
#df['force'] =   m * spl_vel(df['elapsed time sensors'].to_numpy(),1) # take the first derivative of the spline

acc = (df_raw_data['vel encoder'].to_numpy()[1:] - df_raw_data['vel encoder'].to_numpy()[:-1])/(df_raw_data['elapsed time sensors'].to_numpy()[1:]-df_raw_data['elapsed time sensors'].to_numpy()[:-1])
acc = np.append(acc, 0)# add a zero at the end to have the same length as the original data
df['force'] = m * acc 
df['friction force'] = + ( a_friction * np.tanh(b_friction  * df['vel encoder'] ) + c_friction * df['vel encoder'] + d_friction * df['vel encoder']**2)
df['motor force'] = df['force'] + df['friction force']

# evaluate derivative of motor force
F_dot = (df['motor force'].to_numpy()[1:] - df['motor force'].to_numpy()[:-1])/(df_raw_data['elapsed time sensors'].to_numpy()[1:]-df_raw_data['elapsed time sensors'].to_numpy()[:-1])
df['motor force derivative'] = np.append(0,F_dot)# add a zero at the end to have the same length as the original data




# plot velocity information against force
fig, ((ax3)) = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
fig.subplots_adjust(top=0.985, bottom=0.11, left=0.07, right=1.0, hspace=0.2, wspace=0.2)

ax3.set_title('velocity Vs motor force')
# velocity
ax3.plot(df['elapsed time sensors'].to_numpy(),df['vel encoder'].to_numpy(),label="velocity [m/s]",color='dodgerblue',linewidth=2,marker='.',markersize=10)

# throttle
ax3.step(df['elapsed time sensors'].to_numpy(),df['throttle'].to_numpy(),where='post',color='gray',linewidth=2,label="throttle")
ax3.plot(df['elapsed time sensors'].to_numpy(),df['throttle'].to_numpy(),color='gray',linewidth=2,marker='.',markersize=10,linestyle='none')

# measured acceleration
ax3.plot(df['elapsed time sensors'].to_numpy(),acc,color='darkgreen',linewidth=2,marker='.',markersize=10,linestyle='none')
ax3.step(df['elapsed time sensors'].to_numpy(),acc,label="acceleration",where='post',color='darkgreen',linewidth=2,linestyle='-')

# estimated friction force
ax3.step(df['elapsed time sensors'].to_numpy(),df['friction force'].to_numpy(),where='post',color='maroon',linewidth=2,label="estimated friction force [N]")
ax3.plot(df['elapsed time sensors'].to_numpy(),df['friction force'].to_numpy(),color='maroon',linewidth=2,marker='.',markersize=10,linestyle='none')

# estimated motor force
ax3.step(df['elapsed time sensors'].to_numpy(),df['motor force'].to_numpy(),where='post',color='k',linewidth=2,label="motor force [N]")
ax3.plot(df['elapsed time sensors'].to_numpy(),df['motor force'].to_numpy(),color='k',linewidth=2,marker='.',markersize=10,linestyle='none')

# estimated derivative of motor force
ax3.step(df['elapsed time sensors'].to_numpy(),df['motor force derivative'].to_numpy(),where='post',color='purple',linewidth=2,label="motor force derivative[N/s]")
ax3.plot(df['elapsed time sensors'].to_numpy(),df['motor force derivative'].to_numpy(),color='purple',linewidth=2,marker='.',markersize=10,linestyle='none')



ax3.set_xlabel('time [s]')
ax3.set_title('Processed training data')





# --------------- fitting acceleration curve--------------- 
print('')
print('Fitting acceleration curve model')

# define first guess for parameters
initial_guess = torch.ones(3) * 0.5 # initialize parameters in the middle of their range constraint
# NOTE that the parmeter range constraint is set in the self.transform_parameters_norm_2_real method.
#initial_guess[0] = torch.Tensor([0.5])



# NOTE that the parmeter range constraint is set in motor_curve_model.transform_parameters_norm_2_real method.
motor_curve_model_inductance_obj = motor_curve_model_inductance(initial_guess)

# define number of training iterations
normalize_output = False
train_its = 2000

#define loss and optimizer objects
loss_fn = torch.nn.MSELoss(reduction = 'mean') 
optimizer_object = torch.optim.Adam(motor_curve_model_inductance_obj.parameters(), lr=0.003)
        
# generate data in tensor form for torch
train_x = torch.tensor(df[['throttle','vel encoder','motor force']].to_numpy()).cuda()  
train_y = torch.unsqueeze(torch.tensor(df['motor force derivative'].to_numpy()),1).cuda()

# save loss values for later plot
loss_vec = np.zeros(train_its)




# determine maximum value of training data for output normalization
if normalize_output:
    max_y = train_y.max().item()
    rescale_vector = torch.Tensor(train_y.cpu().detach().numpy() / max_y).cuda()
else:
    rescale_vector=torch.ones(train_y.shape[0]).cuda()



# train the model
for i in range(train_its):
    # clear gradient information from previous step before re-evaluating it for the current iteration
    optimizer_object.zero_grad()  
    
    # compute fitting outcome with current model parameters
    output = motor_curve_model_inductance_obj(train_x)

    # evaluate loss function
    loss = loss_fn(torch.div(output , rescale_vector),  torch.div(train_y , rescale_vector))
    loss_vec[i] = loss.item()

    # evaluate the gradient of the loss function with respect to the fitting parameters
    loss.backward() 

    # use the evaluated gradient to perform a gradient descent step 
    optimizer_object.step() # this updates parameters automatically according to the optimizer you chose

# --- print out parameters ---
[a,b,c] = motor_curve_model_inductance_obj.transform_parameters_norm_2_real()
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


#add predicted motor force to previous plot
ax3.step(df['elapsed time sensors'].to_numpy(),output.cpu().detach().numpy(),where='post',color='orangered',linewidth=2,label="estimated motor force derivative")
ax3.plot(df['elapsed time sensors'].to_numpy(),output.cpu().detach().numpy(),color='orangered',linewidth=2,marker='.',markersize=10,linestyle='none')
ax3.legend()




plt.show()

# --- plot motor curve ---
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
x = train_x[:,0].cpu().detach().numpy()
y = train_x[:,1].cpu().detach().numpy()
z = train_y.cpu().detach().numpy()
ax.scatter(x, y, z, c='b', marker='o')



# Create a meshgrid for the surface plot

# find maximum value of velocity
v_max = 5 #a/b

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
Force_grid = motor_curve_model_inductance_obj(input_grid).detach().cpu().view(100, 100).numpy()  # Replace with your surface data

# Plot the surface
ax.plot_surface(throttle_grid, velocity_grid, Force_grid, cmap='viridis', alpha=1)
# Set labels
ax.set_xlabel('throttle')
ax.set_ylabel('velocity')
ax.set_zlabel('Motor force')

# plotting the obtained motor curve as a level plot
max_throttle = df['throttle'].max()
throttle_levels = np.linspace(-c,max_throttle,5).tolist()  # 0.4 set throttle

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

df_data = df[df['vel encoder']>1]
df_data = df_data[df_data['vel encoder']<3]
vel_data = df_data['vel encoder'][df_data['throttle']==0.4000000059604645].to_numpy()
mot_force_data = df_data['motor force'][df_data['throttle']==0.4000000059604645].to_numpy()
plt.scatter(vel_data,mot_force_data,color='k',label='data for throttle = 0.4')
plt.legend()













#plot error between model and data
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
z = output.cpu().detach().numpy() - train_y.cpu().detach().numpy()
flat_x = x.flatten()
flat_y = y.flatten()    
flat_z = z.flatten()
mask = flat_x >= 0.2
flat_x = flat_x[mask]
flat_y = flat_y[mask]
flat_z = flat_z[mask]

max_val_plot = np.max([np.abs(np.min(flat_z)),np.max(flat_z)])
max_val_plot = 3
from matplotlib.colors import Normalize
norm = Normalize(vmin=-max_val_plot, vmax=max_val_plot)


scatter =  ax.scatter(flat_x, flat_y, flat_z, c=flat_z.flatten(), cmap='bwr', marker='o',norm=norm)
# Add a color bar to show the color scale
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('Z value')
#ax.set_xlim([0.2,0.4])
ax.set_xlabel('throttle')
ax.set_ylabel('velocity')
plt.show()





