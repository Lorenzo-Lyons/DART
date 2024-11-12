from functions_for_data_processing import get_data, plot_raw_data, process_raw_vicon_data,plot_vicon_data\
,dyn_model_culomb_tires,produce_long_term_predictions,train_SVGP_model,dyn_model_SVGP,rebuild_Kxy_RBF_vehicle_dynamics,RBF_kernel_rewritten,\
throttle_dynamics_data_processing,steering_dynamics_data_processing,process_vicon_data_kinematics,generate_tensor_past_actions,\
SVGPModel_actuator_dynamics

from matplotlib import pyplot as plt
import torch
import numpy as np
import pandas as pd
import os
import tqdm


#matplotlib.rc('font', **font)

# This script is used to fit the tire model to the data collected on the racetrack with a 
# multi-output SVGP model.




# set these parameters as they will determine the running time of this script
# this will re-build the plotting results using an SVGP rebuilt analytically as would a solver
check_SVGP_analytic_rebuild = False
over_write_saved_parameters = True
epochs = 30 # epochs for training the SVGP
learning_rate = 0.01
# generate data in tensor form for torch
# 0 = no time delay fitting
# 1 = physics-based time delay fitting (1st order)
# 2 = linear layer time delay fitting
actuator_time_delay_fitting_tag = 0

# fit likelihood noise?
fit_likelihood_noise_tag = False  # this doesn't really make much difference

# fit on subsampled dataset?
fit_on_subsampled_dataset_tag = False




# select data folder NOTE: this assumes that the current directory is DART
#folder_path = 'System_identification_data_processing/Data/91_free_driving_16_sept_2024'
folder_path = 'System_identification_data_processing/Data/91_free_driving_16_sept_2024_slow'


# process data
steps_shift = 5 # decide to filter more or less the vicon data


# check if there is a processed vicon data file already
file_name = 'processed_vicon_data_throttle_steering_dynamics.csv'
# Check if the CSV file exists in the folder
file_path = os.path.join(folder_path, file_name)

if not os.path.isfile(file_path):
    df_raw_data = get_data(folder_path)

    
    # add throttle with time integrated throttle
    filtered_throttle = throttle_dynamics_data_processing(df_raw_data)
    df_raw_data['throttle filtered'] = filtered_throttle

    # add steering angle with time integated version
    st_angle_vec_FEuler, st_vec_FEuler = steering_dynamics_data_processing(df_raw_data)
    # over-write the actual data with the forward integrated data
    df_raw_data['steering angle filtered'] = st_angle_vec_FEuler
    df_raw_data['steering filtered'] = st_vec_FEuler

    # process kinematics and dynamics
    df_kinematics = process_vicon_data_kinematics(df_raw_data,steps_shift)
    df = process_raw_vicon_data(df_kinematics,steps_shift)

    #save the processed data file
    df.to_csv(file_path, index=False)
    print(f"File '{file_path}' saved.")
else:
    print(f"File '{file_path}' already exists, loading data.")
    df = pd.read_csv(file_path)



# load likelihood noise
try:
    raw_likelihood_noises = np.load(folder_path + '/raw_noise_likelihood.npy')
except:
    print('missing likelihood noise file')

# load subset of indexes
try:
    subset_indexes = np.load(folder_path + '/subset_indexes.npy')
    subset_indexes = subset_indexes.tolist()
except:
    print('missing subset indexes file')





# plot raw data
#ax0,ax1,ax2 = plot_raw_data(df)

# plot 
#plot_vicon_data(df)

# plotting body frame accelerations and data fitting results 
fig, ((ax_x,ax_y,ax_w)) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
fig.subplots_adjust(top=0.96,
bottom=0.055,
left=0.05,
right=0.995,
hspace=0.345,
wspace=0.2)

# add accelerations
ax_x.plot(df['vicon time'].to_numpy(),df['ax body'].to_numpy(),label='ax',color='dodgerblue')
ax_y.plot(df['vicon time'].to_numpy(),df['ay body'].to_numpy(),label='ay',color='orangered')
ax_w.plot(df['vicon time'].to_numpy(),df['acc_w'].to_numpy(),label='aw',color='orchid')
# add inputs
ax_x.plot(df['vicon time'].to_numpy(),df['ax body'].max() * df['throttle'].to_numpy(),color='gray',label='throttle')
ax_y.plot(df['vicon time'].to_numpy(),df['ay body'].max() * df['steering'].to_numpy(),color='gray',label='steering')
ax_w.plot(df['vicon time'].to_numpy(),df['acc_w'].max() * df['steering'].to_numpy(),color='gray',label='steering')






# train the SVGP model
n_inducing_points = 500
n_past_actions = 0 # default value




if actuator_time_delay_fitting_tag == 0:
    columns_to_extract = ['vx body', 'vy body', 'w', 'throttle' ,'steering'] #  angle time delayed
    train_x_full_dataset = torch.tensor(df[columns_to_extract].to_numpy()).cuda()
else:
    columns_to_extract = ['vx body', 'vy body', 'w'] #  angle time delayed
    train_x_states = torch.tensor(df[columns_to_extract].to_numpy()).cuda()

    n_past_actions = 50
    refinement_factor = 1 # no need to refine the time interval between data points
    train_x_throttle = generate_tensor_past_actions(df, n_past_actions,refinement_factor, key_to_repeat = 'throttle')
    train_x_steering = generate_tensor_past_actions(df, n_past_actions,refinement_factor, key_to_repeat = 'steering')

    train_x_full_dataset = torch.cat((train_x_states,train_x_throttle,train_x_steering),1) # concatenate

# produce y lables
train_y_vx_full_dataset = torch.unsqueeze(torch.tensor(df['ax body'].to_numpy()),1).cuda()
train_y_vy_full_dataset = torch.unsqueeze(torch.tensor(df['ay body'].to_numpy()),1).cuda()
train_y_w_full_dataset  = torch.unsqueeze(torch.tensor(df['acc_w'].to_numpy()),1).cuda()

# convert to float to avoid issues with data types
train_x_full_dataset = train_x_full_dataset.to(torch.float32)
train_y_vx_full_dataset = train_y_vx_full_dataset.to(torch.float32)
train_y_vy_full_dataset = train_y_vy_full_dataset.to(torch.float32)
train_y_w_full_dataset = train_y_w_full_dataset.to(torch.float32)



# if training on a subset of the data use the subset indexes
if fit_on_subsampled_dataset_tag:
    train_x = train_x_full_dataset[subset_indexes,:]
    train_y_vx = train_y_vx_full_dataset[subset_indexes,:]
    train_y_vy = train_y_vy_full_dataset[subset_indexes,:]
    train_y_w = train_y_w_full_dataset[subset_indexes,:]
else:
    train_x = train_x_full_dataset
    train_y_vx = train_y_vx_full_dataset
    train_y_vy = train_y_vy_full_dataset
    train_y_w = train_y_w_full_dataset




# active learning
# select an initial subset of the training data
import random
# random selection of initial subset of datapoints
initial_inducing_points_indexes = random.choices(range(train_x.shape[0]), k=n_inducing_points)
inducing_points = train_x[initial_inducing_points_indexes,:].to(torch.float32)




#cast to float to avoid issues with data types
# add some state noise to stabilize predictions in the long term
# Define different standard deviations for each column
# std_devs = [0.0, 0.0, 0.0]  

# # Generate noise for each column with the specified standard deviations
# noise1 = torch.randn(train_x.size(0)) * std_devs[0]
# noise2 = torch.randn(train_x.size(0)) * std_devs[1]
# noise3 = torch.randn(train_x.size(0)) * std_devs[2]

# # Add the noise to the first three columns
# train_x[:, 0] += noise1.cuda()
# train_x[:, 1] += noise2.cuda()
# train_x[:, 2] += noise3.cuda()

dt = np.mean(np.diff(df['vicon time'].to_numpy())) # time step between data points

#initialize models
model_vx = SVGPModel_actuator_dynamics(inducing_points)
model_vy = SVGPModel_actuator_dynamics(inducing_points)
model_w  = SVGPModel_actuator_dynamics(inducing_points)

# set up time filtering
model_vx.setup_time_delay_fitting(actuator_time_delay_fitting_tag,n_past_actions,dt)
model_vy.setup_time_delay_fitting(actuator_time_delay_fitting_tag,n_past_actions,dt)
model_w.setup_time_delay_fitting(actuator_time_delay_fitting_tag,n_past_actions,dt)

# define likelyhood and optimizer objects
likelihood_vx,optimizer_vx = model_vx.return_likelyhood_optimizer_objects(learning_rate,fit_likelihood_noise_tag,raw_likelihood_noises[0])
likelihood_vy,optimizer_vy = model_vy.return_likelyhood_optimizer_objects(learning_rate,fit_likelihood_noise_tag,raw_likelihood_noises[1])
likelihood_w,optimizer_w = model_w.return_likelyhood_optimizer_objects(learning_rate,fit_likelihood_noise_tag,raw_likelihood_noises[2])





# train the SVGP model
model_vx, model_vy, model_w,\
likelihood_vx, likelihood_vy, likelihood_w = train_SVGP_model(learning_rate,epochs,
                                                            train_x,
                                                            train_y_vx,
                                                            train_y_vy,
                                                            train_y_w,
                                                            inducing_points,
                                                            actuator_time_delay_fitting_tag,
                                                            n_past_actions,dt,
                                                            model_vx,model_vy,model_w,
                                                            likelihood_vx,likelihood_vy,likelihood_w,
                                                            optimizer_vx,optimizer_vy,optimizer_w)
    

# move to cpu for plotting
model_vx = model_vx.cpu()
model_vy = model_vy.cpu()
model_w = model_w.cpu()

train_x_full_dataset = train_x_full_dataset.cpu()


preds_ax = model_vx(train_x_full_dataset)
preds_ay = model_vy(train_x_full_dataset)
preds_aw = model_w(train_x_full_dataset)


# plotting
lower_x, upper_x = preds_ax.confidence_region()
lower_y, upper_y = preds_ay.confidence_region()
lower_w, upper_w = preds_aw.confidence_region()

# plot fitting results
ax_x.cla()  # Clear the axis
ax_y.cla()  # Clear the axis
ax_w.cla()  # Clear the axis

# add accelerations
ax_x.plot(df['vicon time'].to_numpy(),df['ax body'].to_numpy(),label='ax',color='dodgerblue')
ax_y.plot(df['vicon time'].to_numpy(),df['ay body'].to_numpy(),label='ay',color='orangered')
ax_w.plot(df['vicon time'].to_numpy(),df['acc_w'].to_numpy(),label='aw',color='orchid')
# add inputs
ax_x.plot(df['vicon time'].to_numpy(),df['ax body'].max() * df['throttle'].to_numpy(),color='gray',label='throttle')
ax_y.plot(df['vicon time'].to_numpy(),df['ay body'].max() * df['steering'].to_numpy(),color='gray',label='steering')
ax_w.plot(df['vicon time'].to_numpy(),df['acc_w'].max() * df['steering'].to_numpy(),color='gray',label='steering')

# add subsampled points as orange dots
if fit_on_subsampled_dataset_tag:
    for kk in range(0,len(subset_indexes)):
        ax_x.plot(df['vicon time'].iloc[subset_indexes[kk]],df['ax body'].iloc[subset_indexes[kk]],'o',color='orange',alpha=0.5)
        ax_y.plot(df['vicon time'].iloc[subset_indexes[kk]],df['ay body'].iloc[subset_indexes[kk]],'o',color='orange',alpha=0.5)
        ax_w.plot(df['vicon time'].iloc[subset_indexes[kk]],df['acc_w'].iloc[subset_indexes[kk]],'o',color='orange',alpha=0.5)



ax_x.set_title('ax in body frame')
ax_plot_mean = ax_x.plot(df['vicon time'].to_numpy(),preds_ax.mean.detach().cpu().numpy(),color='k',label='model prediction')
aax_plot_confidence = ax_x.fill_between(df['vicon time'].to_numpy(), lower_x.detach().cpu().numpy(), upper_x.detach().cpu().numpy(), alpha=0.2,color='k',label='2 sigma confidence',zorder=20)
ax_x.legend()

ax_y.set_title('ay in body frame')
ax_y.plot(df['vicon time'].to_numpy(),preds_ay.mean.detach().cpu().numpy(),color='k',label='model prediction')
ax_y.fill_between(df['vicon time'].to_numpy(), lower_y.detach().cpu().numpy(), upper_y.detach().cpu().numpy(), alpha=0.2,color='k',label='2 sigma confidence',zorder=20)
ax_y.legend()

ax_w.set_title('aw in body frame')
ax_w.plot(df['vicon time'].to_numpy(),preds_aw.mean.detach().cpu().numpy(),color='k',label='model prediction')
ax_w.fill_between(df['vicon time'].to_numpy(), lower_w.detach().cpu().numpy(), upper_w.detach().cpu().numpy(), alpha=0.2,color='k',label='2 sigma confidence',zorder=20)
ax_w.legend()






# show linear layer weights
if actuator_time_delay_fitting_tag != 0:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    # x
    if actuator_time_delay_fitting_tag == 1:
        [time_C_throttle,time_C_steering]  = model_vx.transform_parameters_norm_2_real()
        weights_throttle_vx = model_vx.produce_past_action_coefficients_1st_oder_step_response(time_C_throttle,n_past_actions,dt)
        weights_steering_vx = model_vx.produce_past_action_coefficients_1st_oder_step_response(time_C_steering,n_past_actions,dt)
        #print out time parameter
        print('model vx')
        print(f'time constant throttle: {time_C_throttle.item()}')
        print(f'time constant steering: {time_C_steering.item()}')
    elif actuator_time_delay_fitting_tag == 2:
        weights_throttle_vx = model_vx.constrained_linear_layer(model_vx.raw_weights_throttle)[0]
        weights_steering_vx = model_vx.constrained_linear_layer(model_vx.raw_weights_steering)[0]
    ax1.plot(weights_throttle_vx.detach().cpu().numpy(),color='dodgerblue',label='throttle weights')
    ax1.plot(weights_steering_vx.detach().cpu().numpy(),color='orangered',label='steering weights')


    # y
    if actuator_time_delay_fitting_tag == 1:
        [time_C_throttle,time_C_steering]  = model_vy.transform_parameters_norm_2_real()
        weights_throttle_vy = model_vy.produce_past_action_coefficients_1st_oder_step_response(time_C_throttle,n_past_actions,dt)
        weights_steering_vy = model_vy.produce_past_action_coefficients_1st_oder_step_response(time_C_steering,n_past_actions,dt)
        # print out time parameter
        print('model vy')
        print(f'time constant throttle: {time_C_throttle.item()}')
        print(f'time constant steering: {time_C_steering.item()}')
    elif actuator_time_delay_fitting_tag == 2:
        weights_throttle_vy = model_vy.constrained_linear_layer(model_vy.raw_weights_throttle)[0]
        weights_steering_vy = model_vy.constrained_linear_layer(model_vy.raw_weights_steering)[0]
    ax2.plot(weights_throttle_vy.detach().cpu().numpy(),color='dodgerblue',label='throttle weights')
    ax2.plot(weights_steering_vy.detach().cpu().numpy(),color='orangered',label='steering weights')

    # w
    if actuator_time_delay_fitting_tag == 1:
        [time_C_throttle,time_C_steering]  = model_w.transform_parameters_norm_2_real()
        weights_throttle_w = model_w.produce_past_action_coefficients_1st_oder_step_response(time_C_throttle,n_past_actions,dt)
        weights_steering_w = model_w.produce_past_action_coefficients_1st_oder_step_response(time_C_steering,n_past_actions,dt)
        # print out time parameter
        print('model w')
        print(f'time constant throttle: {time_C_throttle.item()}')
        print(f'time constant steering: {time_C_steering.item()}')

    elif actuator_time_delay_fitting_tag == 2:
        weights_throttle_w = model_w.constrained_linear_layer(model_w.raw_weights_throttle)[0]
        weights_steering_w = model_w.constrained_linear_layer(model_w.raw_weights_steering)[0]
    ax3.plot(weights_throttle_w.detach().cpu().numpy(),color='dodgerblue',label='throttle weights')
    ax3.plot(weights_steering_w.detach().cpu().numpy(),color='orangered',label='steering weights')

    # add legends
    ax1.legend()
    ax2.legend()
    ax3.legend()





# display fitting results

# Get into evaluation (predictive posterior) mode
model_vx.eval()
model_vy.eval()
model_w.eval()






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
    folder_path_SVGP_params = folder_path + '/SVGP_saved_parameters/'
    np.save(folder_path_SVGP_params+'m_x.npy', m_x)
    np.save(folder_path_SVGP_params+'middle_x.npy', middle_x)
    np.save(folder_path_SVGP_params+'L_inv_x.npy', L_inv_x)
    np.save(folder_path_SVGP_params+'right_vec_x.npy', right_vec_x)
    np.save(folder_path_SVGP_params+'inducing_locations_x.npy', inducing_locations_x)
    np.save(folder_path_SVGP_params+'outputscale_x.npy', outputscale_x)
    np.save(folder_path_SVGP_params+'lengthscale_x.npy', lengthscale_x)

    np.save(folder_path_SVGP_params+'m_y.npy', m_y)
    np.save(folder_path_SVGP_params+'middle_y.npy', middle_y)
    np.save(folder_path_SVGP_params+'L_inv_y.npy', L_inv_y)
    np.save(folder_path_SVGP_params+'right_vec_y.npy', right_vec_y)
    np.save(folder_path_SVGP_params+'inducing_locations_y.npy', inducing_locations_y)
    np.save(folder_path_SVGP_params+'outputscale_y.npy', outputscale_y)
    np.save(folder_path_SVGP_params+'lengthscale_y.npy', lengthscale_y)

    np.save(folder_path_SVGP_params+'m_w.npy', m_w)
    np.save(folder_path_SVGP_params+'middle_w.npy', middle_w)
    np.save(folder_path_SVGP_params+'L_inv_w.npy', L_inv_w)
    np.save(folder_path_SVGP_params+'right_vec_w.npy', right_vec_w)
    np.save(folder_path_SVGP_params+'inducing_locations_w.npy', inducing_locations_w)
    np.save(folder_path_SVGP_params+'outputscale_w.npy', outputscale_w)
    np.save(folder_path_SVGP_params+'lengthscale_w.npy', lengthscale_w)

    # save SVGP models in torch format
    # save the time delay realated parameters
    time_delay_parameters = np.array([actuator_time_delay_fitting_tag,n_past_actions,dt])
    np.save(folder_path_SVGP_params + 'time_delay_parameters.npy', time_delay_parameters)
    # vx
    model_path_vx = folder_path_SVGP_params + 'svgp_model_vx.pth'
    likelihood_path_vx = folder_path_SVGP_params + 'svgp_likelihood_vx.pth'
    torch.save(model_vx.state_dict(), model_path_vx)
    torch.save(likelihood_vx.state_dict(), likelihood_path_vx)
    # vy
    model_path_vy = folder_path_SVGP_params + 'svgp_model_vy.pth'
    likelihood_path_vy = folder_path_SVGP_params + 'svgp_likelihood_vy.pth'
    torch.save(model_vy.state_dict(), model_path_vy)
    torch.save(likelihood_vy.state_dict(), likelihood_path_vy)
    # w
    model_path_w = folder_path_SVGP_params + 'svgp_model_w.pth'
    likelihood_path_w = folder_path_SVGP_params + 'svgp_likelihood_w.pth'
    torch.save(model_w.state_dict(), model_path_w)
    torch.save(likelihood_w.state_dict(), likelihood_path_w)






# # create a subset of the data to speed up future taining runs
# sigma_threshold = 0.2 # when all points in the dataset have less than this value, the subsampling stops
# indexes_to_keep = []
# updated_model = SVGPModel_actuator_dynamics(model_vx.variational_strategy.inducing_points,actuator_time_delay_fitting_tag,n_past_actions,dt).cuda()
# updated_model.covar_module.base_kernel.raw_lengthscale = model_vx.covar_module.base_kernel.raw_lengthscale
# updated_model.covar_module.raw_outputscale = model_vx.covar_module.raw_outputscale

# # for visualization purposes
# plt.ion()
# fig, ax_x = plt.subplots(1, 1, figsize=(10, 6), sharex=True)
# fig.subplots_adjust(top=0.96,
# bottom=0.055,
# left=0.05,
# right=0.995,
# hspace=0.345,
# wspace=0.2)

# # add accelerations
# ax_x.plot(df['vicon time'].to_numpy(),df['ax body'].to_numpy(),label='ax',color='dodgerblue')
# ax_x.set_title('ax in body frame')
# ax_x.plot(df['vicon time'].to_numpy(),preds_ax.mean.detach().cpu().numpy(),color='k',label='model prediction')
# ax_x.fill_between(df['vicon time'].to_numpy(), lower_x.detach().cpu().numpy(), upper_x.detach().cpu().numpy(), alpha=0.2,color='k',label='2 sigma confidence')
# ax_x.legend()

# import gpytorch
# from torch.utils.data import TensorDataset, DataLoader
# train_dataset_subsampling = TensorDataset(train_x, train_y_vx)
# train_loader_subsampling = DataLoader(train_dataset_subsampling, batch_size=250, shuffle=True)


# max_sigma = sigma_threshold + 1 # initial value must be higher than the threshold
# while max_sigma > sigma_threshold:
#     preds_ax = updated_model(train_x)
#     lower_x, upper_x = preds_ax.confidence_region()  # use variance
#     uncertainty_x = (upper_x - lower_x) / 2

#     # add the point with the most uncertainty to the inducing points
#     # Find the index of the maximum uncertainty
#     max_uncertainty_idx = torch.argmax(uncertainty_x)
#     max_sigma = torch.max(uncertainty_x)

#     # Retrieve the point in train_x with the highest uncertainty
#     most_uncertain_point = train_x[max_uncertainty_idx]
#     indexes_to_keep.append(max_uncertainty_idx.item())

#     # Append this point to the inducing points
#     new_inducing_points = torch.cat([updated_model.variational_strategy.inducing_points, most_uncertain_point.unsqueeze(0)], dim=0)

#     # update the model with the new inducing point
#     updated_model = SVGPModel_actuator_dynamics(new_inducing_points,actuator_time_delay_fitting_tag,n_past_actions,dt).cuda()
#     updated_model.covar_module.base_kernel.lengthscale = model_vx.covar_module.base_kernel.lengthscale
#     updated_model.covar_module.outputscale = model_vx.covar_module.outputscale
#     # run 1 epoch of training
#     likelihood_subsampling,optimizer_subsampling = updated_model.return_likelyhood_optimizer_objects(learning_rate)
#     likelihood_subsampling.cuda()
#     mll_subsampling = gpytorch.mlls.VariationalELBO(likelihood_subsampling, updated_model, num_data=train_y_vx.size(0))

#     minibatch_iter_subsampling = tqdm.tqdm(train_loader_subsampling, desc="Minibatch subsampling", leave=False, disable=True)
#     for x_batch_subsampling, y_batch_subsampling in minibatch_iter_subsampling:
#         optimizer_subsampling.zero_grad()
#         output_subsampling = updated_model(x_batch_subsampling)
#         loss_subsampling = -mll_subsampling(output_subsampling, y_batch_subsampling[:,0])
#         minibatch_iter_subsampling.set_postfix(loss=loss_subsampling.item())
#         loss_subsampling.backward()
#         optimizer_subsampling.step()

    
#     print(f'added point {max_uncertainty_idx} with uncertainty {max_sigma.item()}') 

#     # add visuals
#     # Allow the plot to update
#     ax_x.clear()
#     ax_x.plot(df['vicon time'].to_numpy(),df['ax body'].to_numpy(),label='ax',color='dodgerblue')
#     ax_x.plot(df['vicon time'].to_numpy(),preds_ax.mean.detach().cpu().numpy(),color='k',label='model prediction')
#     ax_x.fill_between(df['vicon time'].to_numpy(), lower_x.detach().cpu().numpy(), upper_x.detach().cpu().numpy(), alpha=0.2,color='k',label='2 sigma confidence',zorder=20)
#     for kk in range(0,len(indexes_to_keep)):
#         ax_x.axvline(df['vicon time'].iloc[indexes_to_keep[kk]], color='r', linestyle='--', label='subsampled points')
#     ax_x.legend()
#     plt.pause(0.1)  # Pause to allow the plot to refresh

# plt.ioff()  # Disable interactive mode

# # when subsampling is finished take a subset of the data and save it as a new pandas dataframe
# df_subset = df.iloc[indexes_to_keep]
# subset_file_path = os.path.join(folder_path, 'active_learning_df_subset.csv')
# df_subset.to_csv(subset_file_path, index=False)
















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
    ax_x.plot(df['vicon time'].to_numpy(), mean_mS_x, 'orange',linestyle='--',label='mean mS')

    #plot covariance rebuilt using m and S
    ax_x.fill_between(df['vicon time'].to_numpy(),mean_mS_x - two_sigma_cov_rebuilt_x,
                    mean_mS_x + two_sigma_cov_rebuilt_x, alpha=0.3,label='covariance mS',color='orange')


    #ax.plot(train_x.cpu().numpy(), rebuilt_mean,'red',label='pseudo points',linewidth=5)
    ax_y.plot(df['vicon time'].to_numpy(), mean_mS_y, 'orange',linestyle='--',label='mean mS')

    #plot covariance rebuilt using m and S
    ax_y.fill_between(df['vicon time'].to_numpy(),mean_mS_y - two_sigma_cov_rebuilt_y,
                    mean_mS_y + two_sigma_cov_rebuilt_y, alpha=0.3,label='covariance mS',color='orange')


    #ax.plot(train_x.cpu().numpy(), rebuilt_mean,'red',label='pseudo points',linewidth=5)
    ax_w.plot(df['vicon time'].to_numpy(), mean_mS_w, 'orange',linestyle='--',label='mean mS')

    #plot covariance rebuilt using m and S
    ax_w.fill_between(df['vicon time'].to_numpy(),mean_mS_w - two_sigma_cov_rebuilt_w,
                    mean_mS_w + two_sigma_cov_rebuilt_w, alpha=0.3,label='covariance mS',color='orange')







plt.show()






































# REMOVE AFTER MAKING SURE THAT THE gp modle works with the new implementation of long term predictions









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