from dart_dynamic_models import SVGP_submodel_actuator_dynamics, get_data, plot_raw_data, process_raw_vicon_data,plot_vicon_data\
,dyn_model_culomb_tires,produce_long_term_predictions,train_SVGP_model,rebuild_Kxy_RBF_vehicle_dynamics,RBF_kernel_rewritten,\
throttle_dynamics_data_processing,steering_dynamics_data_processing,process_vicon_data_kinematics,generate_tensor_past_actions,\
SVGPModel_actuator_dynamics, plot_kinemaitcs_data, load_SVGPModel_actuator_dynamics,dyn_model_SVGP_4_long_term_predictions,\
SVGP_unified_model,SVGP_submodel_actuator_dynamics

from matplotlib import pyplot as plt
import torch
import numpy as np
import pandas as pd
import os
import tqdm


if torch.cuda.is_available():
    device = torch.device("cuda")
    print('using GPU')
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
    torch.cuda.empty_cache()  # Releases unused cached memory
    torch.cuda.synchronize()  # Ensures all operations are completed






# change current folder where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))
rosbag_folder = '83_7_march_2025_MPCC_rosbag'



rosbag_data_folder = os.path.join('Data',rosbag_folder,'rosbags')
csv_folder = os.path.join('Data',rosbag_folder,'csv')
rosbag_files = os.listdir(rosbag_data_folder)
csv_files = os.listdir(csv_folder)
folder_path_SVGP_params = os.path.join('Data',rosbag_folder,'SVGP_saved_parameters/')
reprocess_data = True # set to true to reprocess the data again




# set these parameters as they will determine the running time of this script
# this will re-build the plotting results using an SVGP rebuilt analytically as would a solver
check_SVGP_analytic_rebuild = False
over_write_saved_parameters = False
epochs = 200 #  epochs for training the SVGP 100
learning_rate = 0.001
# generate data in tensor form for torch
# 0 = no time delay fitting
# 1 = physics-based time delay fitting (1st order)
# 2 = linear layer time delay fitting
# 3 = GP takes as input the time-filtered inputs obtained with the input dynamics taken from the physics-based model.

actuator_time_delay_fitting_tag = 2

# fit likelihood noise?
fit_likelihood_noise_tag = True  # this doesn't really make much difference

# fit on subsampled dataset?
fit_on_subsampled_dataset_tag = False

# use nominal model (using the dynamic bicycle model as the mean function)
use_nominal_model = False




# process data
steps_shift = 1 # decide to filter more or less the vicon data














for bag_file_name in rosbag_files:
    bag_file = os.path.join(rosbag_data_folder,bag_file_name)
    csv_file = bag_file_name.replace('.bag','.csv')
    if not os.path.isfile(csv_file) or reprocess_data == True:
        # process the rosbag data
        using_rosbag_data=True
        df = process_vicon_data_kinematics([],steps_shift,using_rosbag_data,bag_file)
        # find the time when velocity is higher than 0.2 m/s and cut time before that to 1 s before that
        idx = np.where(df['vx body'].to_numpy() > 0.2)[0][0]
        activation_time = df['vicon time'].to_numpy()[idx]
        df = df[df['vicon time'] > activation_time - 1]
        # now set time to 0 in the beginning
        df['vicon time'] = df['vicon time'] - df['vicon time'].to_numpy()[0]
    
        # save the processed data with
        df.to_csv(os.path.join(csv_folder,csv_file), index=False)






ax_vx,ax_vy, ax_w, ax_acc_x,ax_acc_y,ax_acc_w = plot_kinemaitcs_data(df)








# train the SVGP model
n_inducing_points = 500
n_past_actions = 0 # default value




if actuator_time_delay_fitting_tag == 0:
    columns_to_extract = ['vx body', 'vy body', 'w', 'throttle' ,'steering'] 
    train_x_full_dataset = torch.tensor(df[columns_to_extract].to_numpy()).cuda()

elif actuator_time_delay_fitting_tag == 3:
    columns_to_extract = ['vx body', 'vy body', 'w', 'throttle filtered' ,'steering filtered'] 
    train_x_full_dataset = torch.tensor(df[columns_to_extract].to_numpy()).cuda()

else:
    columns_to_extract = ['vx body', 'vy body', 'w'] 
    train_x_states = torch.tensor(df[columns_to_extract].to_numpy()) #.cuda()

    n_past_actions = 300 # 2 seconds of past actions
    #refinement_factor = 1 # no need to refine the time interval between data points
    train_x_throttle = generate_tensor_past_actions(df, n_past_actions, key_to_repeat = 'throttle')
    train_x_steering = generate_tensor_past_actions(df, n_past_actions, key_to_repeat = 'steering')

    train_x_full_dataset = torch.cat((train_x_states,train_x_throttle,train_x_steering),1) # concatenate





# produce y lables
train_y_vx_full_dataset = torch.unsqueeze(torch.tensor(df['ax body'].to_numpy()),1)#.cuda()
train_y_vy_full_dataset = torch.unsqueeze(torch.tensor(df['ay body'].to_numpy()),1)#.cuda()
train_y_w_full_dataset  = torch.unsqueeze(torch.tensor(df['acc_w'].to_numpy()),1)#.cuda()


if use_nominal_model: # subtract the nominal model from the y predictions
    # evaluate the dynamic bicycle model on the training data
    steering_friction_flag = True
    pitch_dynamics_flag = False # don't modify these as they need to match what is used in the DART simulator node
    nominal_model_dyn_bike = dyn_model_culomb_tires(steering_friction_flag,pitch_dynamics_flag)

    for kk in range(train_y_vx_full_dataset.shape[0]):
        # make prediction using nominal model
        state_action_k = np.array([*train_x_full_dataset[kk,:].cpu().numpy(),0.0,0.0])
        # add dummy values for the throttle and steering commands in the nominal model
        # the latter takes in state_action = [vx vy w throttle steering throttle_command steering_command]
        pred_nom = nominal_model_dyn_bike.forward(state_action_k)
        acc_x = pred_nom[0]
        acc_y = pred_nom[1]
        acc_w = pred_nom[2]
        # these last two are not used
        throttle_dot = pred_nom[3]
        steering_dot = pred_nom[4]

        # subtract the nominal model from the data (Do this in place to save memory)
        train_y_vx_full_dataset[kk] = train_y_vx_full_dataset[kk] - acc_x
        train_y_vy_full_dataset[kk] = train_y_vy_full_dataset[kk] - acc_y
        train_y_w_full_dataset[kk] = train_y_w_full_dataset[kk] - acc_w










# convert to float to avoid issues with data types
train_x_full_dataset = train_x_full_dataset.to(torch.float32)
train_y_vx_full_dataset = train_y_vx_full_dataset.to(torch.float32)
train_y_vy_full_dataset = train_y_vy_full_dataset.to(torch.float32)
train_y_w_full_dataset = train_y_w_full_dataset.to(torch.float32)


train_x = train_x_full_dataset
train_y_vx = train_y_vx_full_dataset
train_y_vy = train_y_vy_full_dataset
train_y_w = train_y_w_full_dataset












dt = np.mean(np.diff(df['vicon time'].to_numpy())) # time step between data points
print('dt: ', dt)
# #initialize models
# model_vx = SVGPModel_actuator_dynamics(inducing_points)
# model_vy = SVGPModel_actuator_dynamics(inducing_points)
# model_w  = SVGPModel_actuator_dynamics(inducing_points)

# # set up time filtering
# model_vx.setup_time_delay_fitting(actuator_time_delay_fitting_tag,n_past_actions,dt)
# model_vy.setup_time_delay_fitting(actuator_time_delay_fitting_tag,n_past_actions,dt)
# model_w.setup_time_delay_fitting(actuator_time_delay_fitting_tag,n_past_actions,dt)

# # define likelyhood and optimizer objects
# raw_likelihood_noises = [0,0,0] # default value not used
# likelihood_vx,optimizer_vx = model_vx.return_likelyhood_optimizer_objects(learning_rate,fit_likelihood_noise_tag,raw_likelihood_noises[0])
# likelihood_vy,optimizer_vy = model_vy.return_likelyhood_optimizer_objects(learning_rate,fit_likelihood_noise_tag,raw_likelihood_noises[1])
# likelihood_w,optimizer_w = model_w.return_likelyhood_optimizer_objects(learning_rate,fit_likelihood_noise_tag,raw_likelihood_noises[2])






# # generate the inducing points for the three models, since they need a 5 dimensional input

# generate the inducing points for the three models, since they need a 5 dimensional input
if actuator_time_delay_fitting_tag == 2: # linear layer
    # extract the first 3 and th
    th = torch.unsqueeze(train_x[:,2+n_past_actions],1)
    st = torch.unsqueeze(train_x[:,2+2*n_past_actions],1)
    train_x_4_inducnig_points = torch.cat((train_x[:,:3],th,st),1)
elif actuator_time_delay_fitting_tag == 1:
    train_x_4_inducnig_points = train_x

# select an initial subset of the training data
import random
# random selection of initial subset of datapoints
initial_inducing_points_indexes = random.choices(range(train_x_4_inducnig_points.shape[0]), k=n_inducing_points)
inducing_points = train_x_4_inducnig_points[initial_inducing_points_indexes,:].to(torch.float32)


        # instantiate the 3 models
submodel_vx = SVGP_submodel_actuator_dynamics(inducing_points)
submodel_vy = SVGP_submodel_actuator_dynamics(inducing_points)
submodel_vw = SVGP_submodel_actuator_dynamics(inducing_points)
SVGP_unified_model_obj = SVGP_unified_model(inducing_points,n_past_actions,dt,actuator_time_delay_fitting_tag,
                                            submodel_vx,submodel_vy,submodel_vw)




# ---  first guess ---

sdt_x = 3
sdt_y = 1
sdt_w = 5
SVGP_unified_model_obj.likelihood_vx.noise = torch.tensor([sdt_x**2], dtype=torch.float32)
SVGP_unified_model_obj.likelihood_vy.noise = torch.tensor([sdt_y**2], dtype=torch.float32)
SVGP_unified_model_obj.likelihood_w.noise = torch.tensor([sdt_w**2], dtype=torch.float32)
# first guess weights
# define as zeros 
fixed_act_delay_guess_st = 0.5 # this is the time delay in seconds as we can see it from the data
fixed_act_delay_guess_th = 0.2 # this is the time delay in seconds as we can see it from the data
smoothing_window = int(5)
delay_in_steps_st = int(fixed_act_delay_guess_st/dt)
delay_in_steps_th = int(fixed_act_delay_guess_th/dt)

first_guess_weights_throttle = torch.zeros(1,n_past_actions,requires_grad=True).cuda()
first_guess_weights_steering = torch.zeros(1,n_past_actions,requires_grad=True).cuda()
first_guess_weights_throttle[0,delay_in_steps_th-smoothing_window:delay_in_steps_th+smoothing_window] = 1
first_guess_weights_steering[0,delay_in_steps_st-smoothing_window:delay_in_steps_st+smoothing_window] = 1

#SVGP_unified_model_obj.raw_weights_throttle.data = first_guess_weights_throttle
#SVGP_unified_model_obj.raw_weights_steering.data = first_guess_weights_steering


# ---------------------

# move to cuda if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    #SVGP_unified_model_obj.raw_weights_throttle = SVGP_unified_model_obj.raw_weights_throttle.cuda()
    #SVGP_unified_model_obj.raw_weights_throttle = SVGP_unified_model_obj.raw_weights_steering.cuda()
    SVGP_unified_model_obj.model_vx.to(device)
    SVGP_unified_model_obj.model_vy.to(device)
    SVGP_unified_model_obj.model_w.to(device)
    SVGP_unified_model_obj.likelihood_vx.to(device)
    SVGP_unified_model_obj.likelihood_vy.to(device)
    SVGP_unified_model_obj.likelihood_w.to(device)
    train_x = train_x.to(device)
    train_y_vx = train_y_vx.to(device)
    train_y_vy = train_y_vy.to(device)
    train_y_w = train_y_w.to(device)
    train_x_full_dataset = train_x_full_dataset.to(device)
    train_y_vx_full_dataset = train_y_vx_full_dataset.to(device)
    train_y_vy_full_dataset = train_y_vy_full_dataset.to(device)
    train_y_w_full_dataset = train_y_w_full_dataset.to(device)



SVGP_unified_model_obj.train_model(epochs,learning_rate,train_x, train_y_vx, train_y_vy, train_y_w)

# # reset initial guess for the weights
# SVGP_unified_model_obj.raw_weights_throttle.data = first_guess_weights_throttle
# # train again
# SVGP_unified_model_obj.train_model(epochs,learning_rate,train_x, train_y_vx, train_y_vy, train_y_w)







# # Save model parameters
if over_write_saved_parameters:
    SVGP_unified_model_obj.save_model(folder_path_SVGP_params,actuator_time_delay_fitting_tag,n_past_actions,dt)







from torch.utils.data import DataLoader, TensorDataset

# Assuming train_x_full_dataset is a PyTorch Tensor
#train_x_full_dataset = train_x_full_dataset.cpu()
batch_size = 2000  # You can adjust this based on your system's capacity
dataset_to_plot = TensorDataset(train_x_full_dataset)
dataloader_to_plot = DataLoader(dataset_to_plot, batch_size=batch_size)

# Initialize empty lists to store predictions
preds_ax_mean = []
preds_ay_mean = []
preds_aw_mean = []

# upper and lower margins for plotting
lower_x = []
upper_x = []
lower_y = []
upper_y = []
lower_w = []
upper_w = []


if use_nominal_model:
    # initialize lists to store nominal model predictions
    y_nominal_vx = []
    y_nominal_vy = []
    y_nominal_w = []


# Loop over mini-batches
with torch.no_grad(): # this actually saves memory allocation cause it doesn't store the gradients
    for batch in dataloader_to_plot:
        batch_data = batch[0]

        preds_ax_batch, preds_ay_batch, preds_aw_batch, weights_throttle, weights_steering = SVGP_unified_model_obj(batch_data)
        # preds_ax_batch = SVGP_unified_model_obj.model_vx(batch_data)
        # preds_ay_batch = SVGP_unified_model_obj.model_vy(batch_data)
        # preds_aw_batch = SVGP_unified_model_obj.model_w(batch_data)

        # get the mean of the predictions
        preds_ax_batch_mean = preds_ax_batch.mean.cpu().numpy()
        preds_ay_batch_mean = preds_ay_batch.mean.cpu().numpy()
        preds_aw_batch_mean = preds_aw_batch.mean.cpu().numpy()

        # if we are using the nominal model subtract the nominal model from the predictions
        if use_nominal_model:
            #evalaute nominal model predictions on the batch data

            y_nominal_vx_batch = np.zeros(batch_data.shape[0])
            y_nominal_vy_batch = np.zeros(batch_data.shape[0])
            y_nominal_w_batch = np.zeros(batch_data.shape[0])

            for kk in range(batch_data.shape[0]):

                # make prediction using nominal model
                state_action_k = np.array([*batch_data[kk,:].cpu().numpy(),0.0,0.0])
                # add dummy values for the throttle and steering commands in the nominal model
                # the latter takes in state_action = [vx vy w throttle steering throttle_command steering_command]
                pred_nom = nominal_model_dyn_bike.forward(state_action_k)
                acc_x = pred_nom[0]
                acc_y = pred_nom[1]
                acc_w = pred_nom[2]
                # these last two are not used
                throttle_dot = pred_nom[3]
                steering_dot = pred_nom[4]

                #store the nominal model predictions
                y_nominal_vx_batch[kk] = acc_x
                y_nominal_vy_batch[kk] = acc_y
                y_nominal_w_batch[kk] = acc_w

                # subtract the nominal model from the data (Do this in place to save memory)
                preds_ax_batch_mean[kk] = preds_ax_batch_mean[kk] + acc_x
                preds_ay_batch_mean[kk] = preds_ay_batch_mean[kk] + acc_y
                preds_aw_batch_mean[kk] = preds_aw_batch_mean[kk] + acc_w

        

        # Get predictions for each model
        preds_ax_mean.extend(preds_ax_batch_mean)
        preds_ay_mean.extend(preds_ay_batch_mean) 
        preds_aw_mean.extend(preds_aw_batch_mean)

        if use_nominal_model:
            y_nominal_vx.extend(y_nominal_vx_batch)
            y_nominal_vy.extend(y_nominal_vy_batch)
            y_nominal_w.extend(y_nominal_w_batch)

        # plotting
        lower_x_batch, upper_x_batch = preds_ax_batch.confidence_region()
        lower_y_batch, upper_y_batch = preds_ay_batch.confidence_region()
        lower_w_batch, upper_w_batch = preds_aw_batch.confidence_region()

        # append confidence region bounds
        lower_x.extend(lower_x_batch.cpu().numpy())
        upper_x.extend(upper_x_batch.cpu().numpy())
        lower_y.extend(lower_y_batch.cpu().numpy())
        upper_y.extend(upper_y_batch.cpu().numpy())
        lower_w.extend(lower_w_batch.cpu().numpy())
        upper_w.extend(upper_w_batch.cpu().numpy())


# print the maximum standard deviation
max_std_dev_x = np.max(upper_x_batch.cpu().numpy()-lower_x_batch.cpu().numpy())/4
max_std_dev_y = np.max(upper_y_batch.cpu().numpy()-lower_y_batch.cpu().numpy())/4
max_std_dev_w = np.max(upper_w_batch.cpu().numpy()-lower_w_batch.cpu().numpy())/4

print('max std dev ax: ', max_std_dev_x)
print('max std dev ay: ', max_std_dev_y)
print('max std dev aw: ', max_std_dev_w)

np.save(folder_path_SVGP_params + 'max_stdev_x.npy', max_std_dev_x)
np.save(folder_path_SVGP_params + 'max_stdev_y.npy', max_std_dev_y)
np.save(folder_path_SVGP_params + 'max_stdev_w.npy', max_std_dev_w)






# plot fitting results
# ax_acc_x.cla()  # Clear the axis
# ax_acc_y.cla()  # Clear the axis
# ax_acc_w.cla()  # Clear the axis

# # add accelerations
# ax_acc_x.plot(df['vicon time'].to_numpy(),df['ax body'].to_numpy(),label='ax',color='dodgerblue')
# ax_acc_y.plot(df['vicon time'].to_numpy(),df['ay body'].to_numpy(),label='ay',color='orangered')
# ax_acc_w.plot(df['vicon time'].to_numpy(),df['acc_w'].to_numpy(),label='aw',color='orchid')




# rescale the output data
# preds_ax_mean = np.array(preds_ax_mean) * Delata_y_vx + mean_y_vx
# preds_ay_mean = np.array(preds_ay_mean) * Delata_y_vy + mean_y_vy
# preds_aw_mean = np.array(preds_aw_mean) * Delata_y_w + mean_y_w







ax_acc_x.set_title('ax in body frame')
ax_plot_mean = ax_acc_x.plot(df['vicon time'].to_numpy(),preds_ax_mean,color='k',label='model prediction')
aax_plot_confidence = ax_acc_x.fill_between(df['vicon time'].to_numpy(), lower_x, upper_x, alpha=0.2,color='k',label='2 sigma confidence',zorder=20)
ax_acc_x.legend()

ax_acc_y.set_title('ay in body frame')
ax_acc_y.plot(df['vicon time'].to_numpy(),preds_ay_mean,color='k',label='model prediction')
ax_acc_y.fill_between(df['vicon time'].to_numpy(), lower_y, upper_y, alpha=0.2,color='k',label='2 sigma confidence',zorder=20)
ax_acc_y.legend()

ax_acc_w.set_title('aw in body frame')
ax_acc_w.plot(df['vicon time'].to_numpy(),preds_aw_mean,color='k',label='model prediction')
ax_acc_w.fill_between(df['vicon time'].to_numpy(), lower_w, upper_w, alpha=0.2,color='k',label='2 sigma confidence',zorder=20)
ax_acc_w.legend()

if use_nominal_model:
    # add nominal model predictions to plot
    ax_acc_x.plot(df['vicon time'].to_numpy(),y_nominal_vx,color='gray',label='nominal model',linestyle='--')
    ax_acc_y.plot(df['vicon time'].to_numpy(),y_nominal_vy,color='gray',label='nominal model',linestyle='--')
    ax_acc_w.plot(df['vicon time'].to_numpy(),y_nominal_w,color='gray',label='nominal model',linestyle='--')






# # show linear layer weights
    
if actuator_time_delay_fitting_tag == 1 or actuator_time_delay_fitting_tag == 2:
    fig, (ax1) = plt.subplots(1, 1, figsize=(12, 4), sharex=True)
    # x
    if actuator_time_delay_fitting_tag == 1:
        [time_C_throttle,time_C_steering]  = SVGP_unified_model_obj.transform_parameters_norm_2_real()
        weights_throttle_vx = SVGP_unified_model_obj.produce_past_action_coefficients_1st_oder_step_response(time_C_throttle,n_past_actions,dt)
        weights_steering_vx = SVGP_unified_model_obj.produce_past_action_coefficients_1st_oder_step_response(time_C_steering,n_past_actions,dt)
        #print out time parameter
        print('model vx')
        print(f'time constant throttle: {time_C_throttle.item()}')
        print(f'time constant steering: {time_C_steering.item()}')
    elif actuator_time_delay_fitting_tag == 2:
        weights_throttle_vx = SVGP_unified_model_obj.constrained_linear_layer(SVGP_unified_model_obj.raw_weights_throttle)[0]
        weights_steering_vx = SVGP_unified_model_obj.constrained_linear_layer(SVGP_unified_model_obj.raw_weights_steering)[0]
    ax1.plot(weights_throttle_vx.detach().cpu().numpy(),color='dodgerblue',label='throttle weights')
    ax1.plot(weights_steering_vx.detach().cpu().numpy(),color='orangered',label='steering weights')
    





plt.show()























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
    ax_acc_x.plot(df['vicon time'].to_numpy(), mean_mS_x, 'orange',linestyle='--',label='mean mS')

    #plot covariance rebuilt using m and S
    ax_acc_x.fill_between(df['vicon time'].to_numpy(),mean_mS_x - two_sigma_cov_rebuilt_x,
                    mean_mS_x + two_sigma_cov_rebuilt_x, alpha=0.3,label='covariance mS',color='orange')


    #ax.plot(train_x.cpu().numpy(), rebuilt_mean,'red',label='pseudo points',linewidth=5)
    ax_acc_y.plot(df['vicon time'].to_numpy(), mean_mS_y, 'orange',linestyle='--',label='mean mS')

    #plot covariance rebuilt using m and S
    ax_acc_y.fill_between(df['vicon time'].to_numpy(),mean_mS_y - two_sigma_cov_rebuilt_y,
                    mean_mS_y + two_sigma_cov_rebuilt_y, alpha=0.3,label='covariance mS',color='orange')


    #ax.plot(train_x.cpu().numpy(), rebuilt_mean,'red',label='pseudo points',linewidth=5)
    ax_acc_w.plot(df['vicon time'].to_numpy(), mean_mS_w, 'orange',linestyle='--',label='mean mS')

    #plot covariance rebuilt using m and S
    ax_acc_w.fill_between(df['vicon time'].to_numpy(),mean_mS_w - two_sigma_cov_rebuilt_w,
                    mean_mS_w + two_sigma_cov_rebuilt_w, alpha=0.3,label='covariance mS',color='orange')




# add legends
ax_acc_x.legend()
ax_acc_y.legend()
ax_acc_w.legend()






print('Producing long term predictions')
# copy the thorottle and steering commands from the filtered data
if actuator_time_delay_fitting_tag == 0:
    df['throttle filtered'] = df['throttle']
    df['steering filtered'] = df['steering']
elif actuator_time_delay_fitting_tag == 2:
    # NOTE this really would need to be that there is only 1 steering and throttle filtered that all models use
    # but now using the same as the one used for the model fitting
    df['throttle filtered'] = (train_x_throttle.float().cpu() @ weights_throttle_vx).detach().numpy()
    df['steering filtered'] = (train_x_steering.float().cpu() @ weights_throttle_w).detach().numpy()






columns_to_extract = ['vicon time', 'vx body', 'vy body', 'w', 'throttle filtered' ,'steering filtered', 'throttle' ,'steering','vicon x','vicon y','vicon yaw']
input_data_long_term_predictions = df[columns_to_extract].to_numpy()
prediction_window = 1.5 # [s]
jumps = 25 #25


model_vx,model_vy,model_w = load_SVGPModel_actuator_dynamics(folder_path_SVGP_params)
dynamic_model = dyn_model_SVGP_4_long_term_predictions(model_vx,model_vy,model_w)

forward_propagate_indexes = [1,2,3] # [1,2,3,4,5] # # 1 = vx, 2=vy, 3=w, 4=throttle, 5=steering

long_term_predictions = produce_long_term_predictions(input_data_long_term_predictions, dynamic_model,prediction_window,jumps,forward_propagate_indexes)




# plot long term predictions

print('plotting long term predictions')

for pred in tqdm.tqdm(long_term_predictions, desc="Rollouts"):

    #velocities
    ax_vx.plot(pred[:,0],pred[:,1],color='k',alpha=0.2)
    ax_vy.plot(pred[:,0],pred[:,2],color='k',alpha=0.2)
    ax_w.plot(pred[:,0],pred[:,3],color='k',alpha=0.2)






plt.show()