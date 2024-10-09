from functions_for_data_processing import get_data, plot_raw_data, process_raw_vicon_data,plot_vicon_data\
,dyn_model_culomb_tires,produce_long_term_predictions,train_SVGP_model,dyn_model_SVGP,rebuild_Kxy_RBF_vehicle_dynamics,RBF_kernel_rewritten
from matplotlib import pyplot as plt
import torch
import numpy as np
import pandas as pd
import glob
import os
from scipy.interpolate import CubicSpline
import matplotlib
import tqdm
font = {'family' : 'normal',
        'size'   : 10}

#matplotlib.rc('font', **font)

# This script is used to fit the tire model to the data collected on the racetrack with a 
# multi-output SVGP model.

# TO DO: collect a dataset with no weird jumps in the vicon data, velocity range between -.6 and 1.2 m/s, and also slowing up/down
# to give the SVGP a chance at guessing the friction





# set these parameters as they will determine the running time of this script
# this will re-build the plotting results using an SVGP rebuilt analytically as would a solver
check_SVGP_analytic_rebuild = False
over_write_saved_parameters = True
epochs = 50 # epochs for training the SVGP
jumps = 200 # how many intervals between the long term predicions




# this assumes that the current directory is DART
#folder_path = 'System_identification_data_processing/Data/5_tire_model_data' 
#folder_path = 'System_identification_data_processing/Data/6_racetrack_lap_v_060' 
folder_path = 'System_identification_data_processing/Data/7_racetrack_lap_v_1' 

# get the raw data
df_raw_data = get_data(folder_path)


#these are delays between robot time and robot reaction
delay_th = 0.01 # [s]
delay_st = 0.14 # [s]
#this is the delay between the vicon time and the robot time
delay_vicon_to_robot = 0.1 #0.05 #[s]


l = 0.175
lr = 0.53*l #0.45 the reference point taken by the data is not exaclty in the center of the vehicle

lf = l-lr
theta_correction = 1.5/180*np.pi #0.5/180*np.pi works for front wheel and 1.5 works for back wheel

df = process_raw_vicon_data(df_raw_data,delay_th,delay_st,delay_vicon_to_robot,lf,lr,theta_correction)



# use this if using v=0.60 dataset
if folder_path == 'System_identification_data_processing/Data/6_racetrack_lap_v_060':
    time_cut = 30
    df = df[df['elapsed time sensors'] > time_cut]

elif folder_path == 'System_identification_data_processing/Data/7_racetrack_lap_v_1':
    time_cut =  18
    time_finish = 28 # 33.5 #
    #highlighting sharp corner clockwise (w<0)
    # time_cut =  27
    # time_finish = 31
    #highlighting slow corner anticlockwise (w>0)
    # time_cut =  19.5
    # time_finish = 28
    df = df[df['elapsed time sensors'] > time_cut]
    df = df[df['elapsed time sensors'] < time_finish]


elif folder_path == 'System_identification_data_processing/Data/5_tire_model_data':
    df1 = df[df['elapsed time sensors'] > 5.0]
    df1 = df1[df1['elapsed time sensors'] < 58.7]

    df2 = df[df['elapsed time sensors'] > 77.7]
    df2 = df2[df2['elapsed time sensors'] < 135.0] # 140.0 156

    # Concatenate all DataFrames into a single DataFrame vertically
    df = pd.concat((df1,df2), axis=0, ignore_index=True)




# plot raw data
ax0,ax1,ax2 = plot_raw_data(df)

# plot 
plot_vicon_data(df)


# inertial charcteristics
m = 1.67
Jz = 0.006513 # uniform rectangle of shape 0.18 x 0.12


# evaluate forces in body frame starting from the ones in the absolute frame
ax_body_vec = np.zeros(df.shape[0])
ay_body_vec = np.zeros(df.shape[0])
aw_body_vec = np.zeros(df.shape[0])

# just for checking
ay_centripetal_vec = np.zeros(df.shape[0])


# since now the steering angle is not fixed, it is a good idea to apply the first order filter to it, to recover the true steering angle
# Time step (sampling period)
df = df.copy()
T = df['elapsed time sensors'].diff().mean()  # Calculate the average time step
# Time delay
tau = 0.15
# Filter coefficient
alpha = T / (T + tau)
# Initialize the filtered steering angle list
filtered_steering_angle = [df['steering angle'].iloc[0]]
# Apply the first-order filter
for i in range(1, len(df)):
    filtered_value = alpha * df['steering angle'].iloc[i] + (1 - alpha) * filtered_steering_angle[-1]
    filtered_steering_angle.append(filtered_value)

# Add the filtered values to the DataFrame
df['steering angle time delayed'] = filtered_steering_angle

# plot the steering angle time delayed vs W
plt.figure()
plt.plot(df['vicon time'].to_numpy(),df['steering angle'].to_numpy(),label='steering angle')
plt.plot(df['vicon time'].to_numpy(),df['steering angle time delayed'].to_numpy(),label='steering angle time delayed')
plt.plot(df['vicon time'].to_numpy(),df['w_abs_filtered'].to_numpy(),label='w filtered')
plt.legend()




# evaluate accelerations in the body frame
for i in range(0,df.shape[0]):
    b = np.array([df['ax_abs_filtered_more'].iloc[i],
                  df['ay_abs_filtered_more'].iloc[i],
                  df['aw_abs_filtered_more'].iloc[i]])
    
    yaw_i = df['unwrapped yaw'].iloc[i]

    R = np.array([[+np.cos(-yaw_i),-np.sin(-yaw_i),0],
                  [+np.sin(-yaw_i),np.cos(-yaw_i), 0],
                  [0             ,0          ,1]])
    
    [ax_i, ay_i, aw_i] = R @ b


    ax_body_vec[i] = ax_i
    ay_body_vec[i] = ay_i- df['w_abs_filtered'].iloc[i]*df['vx body'].iloc[i]
    aw_body_vec[i] = aw_i
    ay_centripetal_vec[i] = df['w_abs_filtered'].iloc[i]*df['vx body'].iloc[i]




# train the SVGP model
n_inducing_points = 400
learning_rate = 0.05

      
# generate data in tensor form for torch
#df['throttle'] = df['throttle'] - 0.2 # shifting to actual activation range
columns_to_extract = ['vx body', 'vy body', 'w_abs_filtered', 'throttle' ,'steering'] #  angle time delayed
SVGP_training_data = df[columns_to_extract].to_numpy()
train_x = torch.tensor(SVGP_training_data).cuda()
train_y_vx = torch.unsqueeze(torch.tensor(ax_body_vec),1).cuda() 
train_y_vy = torch.unsqueeze(torch.tensor(ay_body_vec),1).cuda() 
train_y_w  = torch.unsqueeze(torch.tensor(aw_body_vec),1).cuda() 

#cast to float to avoid issues with data types
# add some state noise to stabilize predictions in the long term
# Define different standard deviations for each column
std_devs = [0.0, 0.0, 0.0]  

# Generate noise for each column with the specified standard deviations
noise1 = torch.randn(train_x.size(0)) * std_devs[0]
noise2 = torch.randn(train_x.size(0)) * std_devs[1]
noise3 = torch.randn(train_x.size(0)) * std_devs[2]

# Add the noise to the first three columns
train_x[:, 0] += noise1.cuda()
train_x[:, 1] += noise2.cuda()
train_x[:, 2] += noise3.cuda()


# convert to float to avoid issues with data types
train_x = train_x.to(torch.float32)
train_y_vx = train_y_vx.to(torch.float32)
train_y_vy = train_y_vy.to(torch.float32)
train_y_w = train_y_w.to(torch.float32)




model_vx, model_vy, model_w, likelihood_vx, likelihood_vy, likelihood_w = train_SVGP_model(learning_rate,epochs, train_x, train_y_vx, train_y_vy, train_y_w, n_inducing_points)



# display fitting results

# Get into evaluation (predictive posterior) mode
model_vx.eval()
model_vy.eval()
model_w.eval()

#evaluate model on training dataset
preds_ax = model_vx(train_x)
preds_ay = model_vy(train_x)
preds_aw = model_w(train_x)

lower_x, upper_x = preds_ax.confidence_region()
lower_y, upper_y = preds_ay.confidence_region()
lower_w, upper_w = preds_aw.confidence_region()


# plotting body frame accelerations and data fitting results 
fig, ((ax1,ax2,ax3)) = plt.subplots(3, 1, figsize=(10, 6))
fig.subplots_adjust(top=0.995,
                    bottom=0.11,
                    left=0.095,
                    right=0.995,
                    hspace=0.345,
                    wspace=0.2)

# plot fitting results
ax1.set_title('ax in body frame')
ax1.plot(df['vicon time'].to_numpy(),ax_body_vec,label='ax',color='dodgerblue')
#ax1.plot(df['vicon time'].to_numpy(),train_y_vx.detach().cpu().numpy(),label='labels',color='navy')
ax1.plot(df['vicon time'].to_numpy(),preds_ax.mean.detach().cpu().numpy(),color='k',label='model prediction')
ax1.fill_between(df['vicon time'].to_numpy(), lower_x.detach().cpu().numpy(), upper_x.detach().cpu().numpy(), alpha=0.2,color='k',label='2 sigma confidence')
ax1.legend()

ax2.plot(df['vicon time'].to_numpy(),ay_body_vec,label='ay',color='orangered')
#ax2.plot(df['vicon time'].to_numpy(),train_y_vy.detach().cpu().numpy(),label='labels',color='navy')
ax2.plot(df['vicon time'].to_numpy(),preds_ay.mean.detach().cpu().numpy(),color='k',label='model prediction')
#ax2.plot(df['vicon time'].to_numpy(),ay_centripetal_vec,color='orange',label='centripetal acc',linestyle='--')
ax2.fill_between(df['vicon time'].to_numpy(), lower_y.detach().cpu().numpy(), upper_y.detach().cpu().numpy(), alpha=0.2,color='k',label='2 sigma confidence')
ax2.legend()

ax3.plot(df['vicon time'].to_numpy(),aw_body_vec,label='aw',color='orchid')
#ax3.plot(df['vicon time'].to_numpy(),train_y_w.detach().cpu().numpy(),label='labels',color='navy')
ax3.plot(df['vicon time'].to_numpy(),preds_aw.mean.detach().cpu().numpy(),color='k',label='model prediction')
ax3.fill_between(df['vicon time'].to_numpy(), lower_w.detach().cpu().numpy(), upper_w.detach().cpu().numpy(), alpha=0.2,color='k',label='2 sigma confidence')
ax3.legend()








# analytical version of the model [necessary for solver implementation]
# rebuild SVGP using m and S 
inducing_locations_x = model_vx.variational_strategy.inducing_points.cpu().detach().numpy()
outputscale_x = model_vx.covar_module.outputscale.item()
lengthscale_x = model_vx.covar_module.base_kernel.lengthscale.cpu().detach().numpy()[0]

inducing_locations_y = model_vy.variational_strategy.inducing_points.cpu().detach().numpy()
outputscale_y = model_vy.covar_module.outputscale.item()
lengthscale_y = model_vy.covar_module.base_kernel.lengthscale.cpu().detach().numpy()[0]

inducing_locations_w = model_w.variational_strategy.inducing_points.cpu().detach().numpy()
outputscale_w = model_w.covar_module.outputscale.item()
lengthscale_w = model_w.covar_module.base_kernel.lengthscale.cpu().detach().numpy()[0]




KZZ_x = rebuild_Kxy_RBF_vehicle_dynamics(np.squeeze(inducing_locations_x),np.squeeze(inducing_locations_x),outputscale_x,lengthscale_x)
KZZ_y = rebuild_Kxy_RBF_vehicle_dynamics(np.squeeze(inducing_locations_y),np.squeeze(inducing_locations_y),outputscale_y,lengthscale_y)
KZZ_w = rebuild_Kxy_RBF_vehicle_dynamics(np.squeeze(inducing_locations_w),np.squeeze(inducing_locations_w),outputscale_w,lengthscale_w)

#kXZ = rebuild_Kxy_RBF_vehicle_dynamics(np.squeeze(train_x.cpu().numpy()),np.squeeze(inducing_locations_x),outputscale,lengthscale)
#KXX = rebuild_Kxy_RBF_vehicle_dynamics(np.squeeze(train_x.cpu().numpy()),np.squeeze(train_x.cpu().numpy()),outputscale,lengthscale)

# #plot covariance matrix eigenvalues
# # Compute the eigenvalues
# eigenvalues = np.linalg.eigvalsh(KZZ)

# # Sort the eigenvalues in decreasing order
# sorted_eigenvalues = np.sort(eigenvalues)[::-1]

# # Plot the eigenvalues
# plt.figure(figsize=(8, 5))
# plt.plot(sorted_eigenvalues, 'o-', markersize=8, color='blue', label='Eigenvalues')
# plt.title('Eigenvalues of the KXX Matrix in Decreasing Order')
# plt.xlabel('Index')
# plt.ylabel('Eigenvalue')
# plt.grid(True)
# plt.legend()

# call prediction module on inducing locations
jitter_term = 0.0001 * np.eye(n_inducing_points)  # this is very important for numerical stability




preds_zz_x = model_vx(model_vx.variational_strategy.inducing_points)
preds_zz_y = model_vy(model_vy.variational_strategy.inducing_points)
preds_zz_w = model_w( model_w.variational_strategy.inducing_points)

m_x = preds_zz_x.mean.detach().cpu().numpy() # model.variational_strategy.variational_distribution.mean.detach().cpu().numpy()  #
S_x = model_vx.variational_strategy.variational_distribution.covariance_matrix.detach().cpu().numpy()  # preds_zz.covariance_matrix.detach().cpu().numpy() # 

m_y = preds_zz_y.mean.detach().cpu().numpy() # model.variational_strategy.variational_distribution.mean.detach().cpu().numpy()  #
S_y = model_vy.variational_strategy.variational_distribution.covariance_matrix.detach().cpu().numpy()  

m_w = preds_zz_w.mean.detach().cpu().numpy() # model.variational_strategy.variational_distribution.mean.detach().cpu().numpy()  #
S_w = model_w.variational_strategy.variational_distribution.covariance_matrix.detach().cpu().numpy()  

# Compute the covariance of q(f)
# K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX

# solve the pre-post multiplication block
from scipy.linalg import solve_triangular

# Define a lower triangular matrix L and a matrix B
L_inv_x = np.linalg.inv(np.linalg.cholesky(KZZ_x + jitter_term))
#KZZ_inv_x = np.linalg.inv(KZZ_x + jitter_term)
right_vec_x = np.linalg.solve(KZZ_x + jitter_term, m_x)
middle_x = S_x - np.eye(n_inducing_points)

L_inv_y = np.linalg.inv(np.linalg.cholesky(KZZ_y + jitter_term))
#KZZ_inv_y = np.linalg.inv(KZZ_y + jitter_term)
right_vec_y = np.linalg.solve(KZZ_y + jitter_term, m_y)
middle_y = S_y - np.eye(n_inducing_points)

L_inv_w = np.linalg.inv(np.linalg.cholesky(KZZ_w + jitter_term))
#KZZ_inv_w = np.linalg.inv(KZZ_w + jitter_term)
right_vec_w = np.linalg.solve(KZZ_w + jitter_term, m_w)
middle_w = S_w - np.eye(n_inducing_points)


if over_write_saved_parameters:
    # save quantities to use them later in a solver
    folder_path = 'System_identification_data_processing/SVGP_saved_parameters/'
    np.save(folder_path+'m_x.npy', m_x)
    np.save(folder_path+'middle_x.npy', middle_x)
    np.save(folder_path+'L_inv_x.npy', L_inv_x)
    np.save(folder_path+'right_vec_x.npy', right_vec_x)
    np.save(folder_path+'inducing_locations_x.npy', inducing_locations_x)
    np.save(folder_path+'outputscale_x.npy', outputscale_x)
    np.save(folder_path+'lengthscale_x.npy', lengthscale_x)

    np.save(folder_path+'m_y.npy', m_y)
    np.save(folder_path+'middle_y.npy', middle_y)
    np.save(folder_path+'L_inv_y.npy', L_inv_y)
    np.save(folder_path+'right_vec_y.npy', right_vec_y)
    np.save(folder_path+'inducing_locations_y.npy', inducing_locations_y)
    np.save(folder_path+'outputscale_y.npy', outputscale_y)
    np.save(folder_path+'lengthscale_y.npy', lengthscale_y)

    np.save(folder_path+'m_w.npy', m_w)
    np.save(folder_path+'middle_w.npy', middle_w)
    np.save(folder_path+'L_inv_w.npy', L_inv_w)
    np.save(folder_path+'right_vec_w.npy', right_vec_w)
    np.save(folder_path+'inducing_locations_w.npy', inducing_locations_w)
    np.save(folder_path+'outputscale_w.npy', outputscale_w)
    np.save(folder_path+'lengthscale_w.npy', lengthscale_w)




if check_SVGP_analytic_rebuild:
    # storing for later plot
    two_sigma_cov_rebuilt_x = np.zeros(train_x.shape[0])
    mean_mS_x = np.zeros(train_x.shape[0])
    two_sigma_cov_rebuilt_y = np.zeros(train_x.shape[0])
    mean_mS_y = np.zeros(train_x.shape[0])
    two_sigma_cov_rebuilt_w = np.zeros(train_x.shape[0])
    mean_mS_w = np.zeros(train_x.shape[0])

    print('re-evalauting SVGP using numpy array formulation')
    analytic_predictions_iter = tqdm.tqdm(range(train_x.shape[0]), desc="analytic predictions")
    for i in analytic_predictions_iter:
        loc = np.expand_dims(train_x.cpu().numpy()[i,:],0)


        kXZ_x = rebuild_Kxy_RBF_vehicle_dynamics(loc,np.squeeze(inducing_locations_x),outputscale_x,lengthscale_x)

        #X = solve_triangular(L, kXZ.T, lower=True)
        X_x = L_inv_x @ kXZ_x.T

        # prediction
        KXX_x = RBF_kernel_rewritten(loc[0],loc[0],outputscale_x,lengthscale_x)
        cov_mS_x = KXX_x + X_x.T @ middle_x @ X_x

        # store for plotting
        two_sigma_cov_rebuilt_x[i] = np.sqrt(cov_mS_x) * 2
        mean_mS_x[i] = kXZ_x @ right_vec_x




        kXZ_y = rebuild_Kxy_RBF_vehicle_dynamics(loc,np.squeeze(inducing_locations_y),outputscale_y,lengthscale_y)

        #X = solve_triangular(L, kXZ.T, lower=True)
        X_y = L_inv_y @ kXZ_y.T

        # prediction
        KXX_y = RBF_kernel_rewritten(loc[0],loc[0],outputscale_y,lengthscale_y)
        cov_mS_y = KXX_y + X_y.T @ middle_y @ X_y

        # store for plotting
        two_sigma_cov_rebuilt_y[i] = np.sqrt(cov_mS_y) * 2
        mean_mS_y[i] = kXZ_y @ right_vec_y



        kXZ_w = rebuild_Kxy_RBF_vehicle_dynamics(loc,np.squeeze(inducing_locations_w),outputscale_w,lengthscale_w)

        #X = solve_triangular(L, kXZ.T, lower=True)
        X_w = L_inv_w @ kXZ_w.T

        # prediction
        KXX_w = RBF_kernel_rewritten(loc[0],loc[0],outputscale_w,lengthscale_w)
        cov_mS_w = KXX_w + X_w.T @ middle_w @ X_w

        # store for plotting
        two_sigma_cov_rebuilt_w[i] = np.sqrt(cov_mS_w) * 2
        mean_mS_w[i] = kXZ_w @ right_vec_w





    # ----- plot over the model training results -----

    #ax.plot(train_x.cpu().numpy(), rebuilt_mean,'red',label='pseudo points',linewidth=5)
    ax1.plot(df['vicon time'].to_numpy(), mean_mS_x, 'orange',linestyle='--',label='mean mS')

    #plot covariance rebuilt using m and S
    ax1.fill_between(df['vicon time'].to_numpy(),mean_mS_x - two_sigma_cov_rebuilt_x,
                    mean_mS_x + two_sigma_cov_rebuilt_x, alpha=0.3,label='covariance mS',color='orange')


    #ax.plot(train_x.cpu().numpy(), rebuilt_mean,'red',label='pseudo points',linewidth=5)
    ax2.plot(df['vicon time'].to_numpy(), mean_mS_y, 'orange',linestyle='--',label='mean mS')

    #plot covariance rebuilt using m and S
    ax2.fill_between(df['vicon time'].to_numpy(),mean_mS_y - two_sigma_cov_rebuilt_y,
                    mean_mS_y + two_sigma_cov_rebuilt_y, alpha=0.3,label='covariance mS',color='orange')


    #ax.plot(train_x.cpu().numpy(), rebuilt_mean,'red',label='pseudo points',linewidth=5)
    ax3.plot(df['vicon time'].to_numpy(), mean_mS_w, 'orange',linestyle='--',label='mean mS')

    #plot covariance rebuilt using m and S
    ax3.fill_between(df['vicon time'].to_numpy(),mean_mS_w - two_sigma_cov_rebuilt_w,
                    mean_mS_w + two_sigma_cov_rebuilt_w, alpha=0.3,label='covariance mS',color='orange')

























# producing long term predictions
# construct a model that takes as inputs Vx,Vy,W,tau,Steer ---> Vx_dot,Vy_dot,W_dot
dynamic_model = dyn_model_SVGP(model_vx,model_vy,model_w)




columns_to_extract = ['vicon time', 'vx body', 'vy body', 'w_abs_filtered', 'throttle' ,'steering','vicon x','vicon y','vicon yaw']
input_data_long_term_predictions = df[columns_to_extract].to_numpy()
prediction_window = 1.5 # [s]

forward_propagate_indexes = [1,2,3] # 1 =vx, 2=vy, 3=w
long_term_predictions = produce_long_term_predictions(input_data_long_term_predictions, dynamic_model,prediction_window,jumps,forward_propagate_indexes)

# plot long term predictions over real data
fig, ((ax10,ax11,ax12)) = plt.subplots(3, 1, figsize=(10, 6))
fig.subplots_adjust(top=0.995,
                    bottom=0.11,
                    left=0.095,
                    right=0.995,
                    hspace=0.345,
                    wspace=0.2)

time_vec_data = df['vicon time'].to_numpy()


# velocities
ax10.plot(time_vec_data,input_data_long_term_predictions[:,1],color='dodgerblue',label='vx',linewidth=4,linestyle='-')
ax10.set_xlabel('Time [s]')
ax10.set_ylabel('Vx body[m/s]')
ax10.legend()
ax10.set_title('Vx')


ax11.plot(time_vec_data,input_data_long_term_predictions[:,2],color='orangered',label='vy',linewidth=4,linestyle='-')
ax11.set_xlabel('Time [s]')
ax11.set_ylabel('Vy body[m/s]')
ax11.legend()
ax11.set_title('Vy')


ax12.plot(time_vec_data,input_data_long_term_predictions[:,3],color='orchid',label='w',linewidth=4,linestyle='-')
ax12.set_xlabel('Time [s]')
ax12.set_ylabel('W [rad/s]')
ax12.legend()
ax12.set_title('W')


# positions
fig, ((ax13,ax14,ax15)) = plt.subplots(3, 1, figsize=(10, 6))
fig.subplots_adjust(top=0.995,
                    bottom=0.11,
                    left=0.095,
                    right=0.995,
                    hspace=0.345,
                    wspace=0.2)



ax13.plot(time_vec_data,input_data_long_term_predictions[:,6],color='dodgerblue',label='x',linewidth=4,linestyle='-')
ax13.set_xlabel('time [s]')
ax13.set_ylabel('y [m]')
ax13.legend()
ax13.set_title('trajectory in the x-y plane')

ax14.plot(time_vec_data,input_data_long_term_predictions[:,7],color='orangered',label='y',linewidth=4,linestyle='-')
ax14.set_xlabel('time [s]')
ax14.set_ylabel('y [m]')
ax14.legend()
ax14.set_title('trajectory in the x-y plane')

ax15.plot(time_vec_data,input_data_long_term_predictions[:,8],color='orchid',label='yaw',linewidth=4,linestyle='-')
ax15.set_xlabel('time [s]')
ax15.set_ylabel('yaw [rad]')
ax15.legend()
ax15.set_title('vehicle yaw')


# trajectory
fig, ((ax16)) = plt.subplots(1, 1, figsize=(10, 6))
fig.subplots_adjust(top=0.995,
                    bottom=0.11,
                    left=0.095,
                    right=0.995,
                    hspace=0.345,
                    wspace=0.2)

ax16.plot(input_data_long_term_predictions[:,6],input_data_long_term_predictions[:,7],color='orange',label='trajectory',linewidth=4,linestyle='-')
ax16.set_xlabel('x [m]')
ax16.set_ylabel('y [m]')
ax16.legend()
ax16.set_title('vehicle trajectory in the x-y plane')




for i in range(0,len(long_term_predictions)):
    pred = long_term_predictions[i]
    #velocities
    ax10.plot(pred[:,0],pred[:,1],color='k',alpha=0.2)
    ax11.plot(pred[:,0],pred[:,2],color='k',alpha=0.2)
    ax12.plot(pred[:,0],pred[:,3],color='k',alpha=0.2)
    # positions
    ax13.plot(pred[:,0],pred[:,6],color='k',alpha=0.2)
    ax14.plot(pred[:,0],pred[:,7],color='k',alpha=0.2)
    ax15.plot(pred[:,0],pred[:,8],color='k',alpha=0.2)
    #trajectory
    ax16.plot(pred[:,6],pred[:,7],color='k',alpha=0.2)

ax16.set_aspect('equal')








plt.show()