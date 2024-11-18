from dart_dynamic_models import get_data, plot_raw_data, motor_and_friction_model,model_functions,generate_tensor_past_actions
from matplotlib import pyplot as plt
import torch
import numpy as np



# this assumes that the current directory is DART
folder_path = 'System_identification/Data/20_step_input_data_rubbery_floor' 
#folder_path = 'System_identification/Data/21_step_input_data_rubbery_floor_v_08-15' # velocity up to 1.5 m/s


# needed to retrieve friction coefficients
mf = model_functions()


# get the raw data
df_raw_data = get_data(folder_path)



# plot raw data
ax0,ax1,ax2 = plot_raw_data(df_raw_data)


# process the raw data
m = mf.m_self #mass of the robot

df = df_raw_data[['elapsed time sensors','throttle','vel encoder']].copy() 



#evaluate acceleration
steps = 1
acc = (df_raw_data['vel encoder'].to_numpy()[steps:] - df_raw_data['vel encoder'].to_numpy()[:-steps])/(df_raw_data['elapsed time sensors'].to_numpy()[steps:]-df_raw_data['elapsed time sensors'].to_numpy()[:-steps])
for i in range(steps):
    acc = np.append(acc , 0)# add a zero at the end to have the same length as the original data


df['force'] = m * acc



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
ax3.plot(df['elapsed time sensors'].to_numpy(),df['force'].to_numpy(),color='darkgreen',linewidth=2,marker='.',markersize=10,linestyle='none')
ax3.step(df['elapsed time sensors'].to_numpy(),df['force'].to_numpy(),label="measured force",where='post',color='darkgreen',linewidth=2,linestyle='-')


ax3.set_xlabel('time [s]')
ax3.set_title('Processed training data')
ax3.legend()




# --------------- fitting acceleration curve--------------- 
print('')
print('Fitting motor and friction model')
fit_friction_flag = True



# produce data to fit the model
n_past_throttle = 10
refinement_factor = 1 # no need to refine the time interval between data points

train_x_throttle = generate_tensor_past_actions(df, n_past_throttle,refinement_factor, key_to_repeat = 'throttle')
train_x_v = torch.unsqueeze(torch.tensor(df['vel encoder'].to_numpy()),1).cuda()
train_x = torch.cat((train_x_v,train_x_throttle),1)


train_y = torch.unsqueeze(torch.tensor(df['force'].to_numpy()),1).cuda()



# NOTE that the parmeter range constraint is set in motor_curve_model.transform_parameters_norm_2_real method.
dt = np.diff(df['elapsed time sensors'].to_numpy()).mean()
motor_and_friction_model_obj = motor_and_friction_model(n_past_throttle,dt,fit_friction_flag)

# define number of training iterations
train_its = 750

#define loss and optimizer objects
loss_fn = torch.nn.MSELoss(reduction = 'mean') 
optimizer_object = torch.optim.Adam(motor_and_friction_model_obj.parameters(), lr=0.003)







# save loss values for later plot
loss_vec = np.zeros(train_its)





# train the model
for i in range(train_its):
    # clear gradient information from previous step before re-evaluating it for the current iteration
    optimizer_object.zero_grad()  
    
    # compute fitting outcome with current model parameters
    output, filtered_throttle_model, k_vec = motor_and_friction_model_obj(train_x)

    # evaluate loss function
    loss = loss_fn(output, train_y )
    loss_vec[i] = loss.item()

    # evaluate the gradient of the loss function with respect to the fitting parameters
    loss.backward() 

    # use the evaluated gradient to perform a gradient descent step 
    optimizer_object.step() # this updates parameters automatically according to the optimizer you chose


# plot loss function
plt.figure()
plt.title('Loss')
plt.plot(loss_vec)
plt.xlabel('iterations')
plt.ylabel('loss')



# --- print out parameters ---
[a_m,b_m,c_m,time_C_m,a_f,b_f,c_f,d_f] = motor_and_friction_model_obj.transform_parameters_norm_2_real()
a_m,b_m,c_m,time_C_m,a_f,b_f,c_f,d_f = a_m.item(),b_m.item(),c_m.item(),time_C_m.item(),a_f.item(),b_f.item(),c_f.item(),d_f.item()
print('# motor parameters')
print('a_m = ',a_m)
print('b_m = ',b_m) 
print('c_m = ',c_m)
print('time_C_m = ',time_C_m)

if fit_friction_flag:
    print('# friction parameters')
    print('a_f = ',a_f)
    print('b_f = ',b_f)
    print('c_f = ',c_f)
    print('d_f = ',d_f)


# print out the coefficients of the step response to be sure that the last one is very close to 0
print('coefficients of the step response')
# print them out in a float format
k_vec_numpy = k_vec.detach().cpu().numpy()
for k in k_vec_numpy:
    # print in float format with 5 decimals
    print("{:.5f}".format(k[0]))







# Add predicted motor force to previous plot
ax3.step(df['elapsed time sensors'].to_numpy(),output.cpu().detach().numpy(),where='post',color='orangered',linewidth=2,label="total force (model)")
ax3.plot(df['elapsed time sensors'].to_numpy(),output.cpu().detach().numpy(),color='orangered',linewidth=2,marker='.',markersize=10,linestyle='none')
ax3.legend()




# forwards integrate throttle using the fitted model parameters
th = 0
filtered_throttle = np.zeros(df_raw_data.shape[0])
# Loop through the data to compute the predicted steering angles
ground_truth_refinement = 100 # this is used to integrate the steering angle with a higher resolution to avoid numerical errors
for t in range(1, len(filtered_throttle)):
    # integrate ground trough with a much higher dt to have better numerical accuracy
    for k in range(ground_truth_refinement):
        th_dot = mf.continuous_time_1st_order_dynamics(th,df_raw_data['throttle'].iloc[t],time_C_m) # this 1 step delay is due to how forward euler works
        th += dt/ground_truth_refinement * th_dot
    filtered_throttle[t] = th



# plot filtered throttle

fig, ((ax4)) = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
fig.subplots_adjust(top=0.985, bottom=0.11, left=0.07, right=1.0, hspace=0.2, wspace=0.2)


# throttle
ax4.step(df['elapsed time sensors'].to_numpy(),df['throttle'].to_numpy(),where='post',color='gray',linewidth=2,label="throttle")
ax4.plot(df['elapsed time sensors'].to_numpy(),df['throttle'].to_numpy(),color='gray',linewidth=2,marker='.',markersize=10,linestyle='none')

# filtered throttle from model
ax4.step(df['elapsed time sensors'].to_numpy(),filtered_throttle_model.cpu().detach().numpy(),where='post',color='orangered',linewidth=2,label="filtered throttle (model)")
ax4.plot(df['elapsed time sensors'].to_numpy(),filtered_throttle_model.cpu().detach().numpy(),color='orangered',linewidth=2,marker='.',markersize=10,linestyle='none')

# filtered throttle from Forward Euler integration
ax4.step(df['elapsed time sensors'].to_numpy(),filtered_throttle,where='post',color='k',linewidth=2,label="filtered throttle Forward Euler",linestyle='--')
ax4.plot(df['elapsed time sensors'].to_numpy(),filtered_throttle,color='k',linewidth=2,marker='.',markersize=10,linestyle='none')


ax4.legend()








if fit_friction_flag:
    # plot friction curve
    df_friction = df[df['throttle']==0]
    v_range = np.linspace(0,df_friction['vel encoder'].max(),100)
    friction_force = motor_and_friction_model_obj.rolling_friction(torch.unsqueeze(torch.tensor(v_range).cuda(),1),a_f,b_f,c_f,d_f).detach().cpu().numpy()
    fig, ax_friction = plt.subplots()
    ax_friction.scatter(df_friction['vel encoder'],df_friction['force'],label='data',color='dodgerblue')
    ax_friction.plot(v_range,friction_force,label='friction curve model',color='orangered')

    ax_friction.set_xlabel('velocity [m/s]')
    ax_friction.set_ylabel('friction force [N]')
    ax_friction.legend()    




# --- plot motor curve ---
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
x = filtered_throttle_model.cpu().detach().numpy() #train_x[:,0].cpu().detach().numpy()
y = train_x[:,1].cpu().detach().numpy()
z = train_y.cpu().detach().numpy()
ax.scatter(x, y, z, c='b', marker='o')


# Create a meshgrid for the surface plot

# find maximum value of velocity
v_max = 6 

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
Total_force_grid = motor_and_friction_model_obj.motor_force(input_grid[:,0],input_grid[:,1],a_m,b_m,c_m).detach().cpu().view(100, 100).numpy() +\
             motor_and_friction_model_obj.rolling_friction(input_grid[:,1],a_f,b_f,c_f,d_f).detach().cpu().view(100, 100).numpy()

Motor_force_grid = motor_and_friction_model_obj.motor_force(input_grid[:,0],input_grid[:,1],a_m,b_m,c_m).detach().cpu().view(100, 100).numpy()

# Plot the surface
ax.plot_surface(throttle_grid, velocity_grid, Total_force_grid, cmap='viridis', alpha=1)
# Set labels
ax.set_xlabel('throttle')
ax.set_ylabel('velocity')
ax.set_zlabel('Force')

# plotting the obtained motor curve as a level plot
max_throttle = df['throttle'].max()
throttle_levels = np.linspace(-c_m,max_throttle,5).tolist()  # 0.4 set throttle

fig = plt.figure(figsize=(10, 6))
fig.subplots_adjust(top=0.985, bottom=0.11, left=0.07, right=1.0, hspace=0.2, wspace=0.2)

#heatmap = plt.imshow(throttle_grid, extent=[velocity_grid.min(), Force_grid.max(), Force_grid.min(), throttle_grid.max()], origin='lower', cmap='plasma')
contour1 = plt.contourf(velocity_grid, Motor_force_grid ,throttle_grid, levels=100, cmap='plasma') 
contour2 = plt.contour(velocity_grid, Motor_force_grid ,throttle_grid, levels=throttle_levels, colors='black', linestyles='solid', linewidths=2) 
cbar = plt.colorbar(contour1, label='Throttle',ticks=[0, *throttle_levels],format='%.2f')

# from matplotlib.ticker import StrMethodFormatter
# cbar.ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
# cbar.set_ticks([0, *throttle_levels, 1])

#cbar.update_ticks()
# Add labels for contour lines
plt.clabel(contour2, inline=True, fontsize=18, fmt='%1.2f')
# Set labels
plt.xlabel('Velocity [m/s]')
plt.ylabel('Force [N]')

# df_data = df[df['vel encoder']>1]
# df_data = df_data[df_data['vel encoder']<3]
# vel_data = df_data['vel encoder'][df_data['throttle']==0.4000000059604645].to_numpy()
# mot_force_data = df_data['motor force'][df_data['throttle']==0.4000000059604645].to_numpy()
# plt.scatter(vel_data,mot_force_data,color='k',label='data for throttle = 0.4')
#plt.legend()











#plot error between model and data
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Error between model and data')
z = output.cpu().detach().numpy() - train_y.cpu().detach().numpy()
flat_x = x.flatten()
flat_y = y.flatten()    
flat_z = z.flatten()
mask = flat_x >= 0.1
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
cbar.set_label('Error between model and data')
#ax.set_xlim([0.2,0.4])
ax.set_xlabel('throttle')
ax.set_ylabel('velocity')
plt.show()





