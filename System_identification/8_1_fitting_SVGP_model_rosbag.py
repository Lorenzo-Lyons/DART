from dart_dynamic_models import SVGP_submodel_actuator_dynamics, get_data, plot_raw_data, process_raw_vicon_data,plot_vicon_data\
,dyn_model_culomb_tires,produce_long_term_predictions,train_SVGP_model,rebuild_Kxy_RBF_vehicle_dynamics,RBF_kernel_rewritten,\
throttle_dynamics_data_processing,steering_dynamics_data_processing,process_vicon_data_kinematics,generate_tensor_past_actions,\
SVGPModel_actuator_dynamics, plot_kinemaitcs_data, load_SVGPModel_actuator_dynamics,dyn_model_SVGP_4_long_term_predictions,\
SVGP_unified_model,SVGP_submodel_actuator_dynamics, model_functions

from matplotlib import pyplot as plt
import torch
import numpy as np
import pandas as pd
import os
import tqdm
mf = model_functions() # instantiate the model functions class



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
evaluate_long_term_predictions = True
epochs = 100 #100 #  epochs for training the SVGP 200
learning_rate =  0.001 #0.015 # 0.0015
# generate data in tensor form for torch
# 0 = no time delay fitting
# 1 = physics-based time delay fitting (1st order)
# 2 = linear layer time delay fitting
# legacy 3rd option
# 3 = GP takes as input the time-filtered inputs obtained with the input dynamics taken from the physics-based model.

actuator_time_delay_fitting_tag = 2

# fit likelihood noise?
fit_likelihood_noise_tag = True  # this doesn't really make much difference

# fit on subsampled dataset?
fit_on_subsampled_dataset_tag = False

# use nominal model (using the dynamic bicycle model as the mean function)
use_nominal_model = True


# process data
steps_shift = 1 # decide to filter more or less the vicon data





# plot settings
thick_line_width = 2








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






ax_vx,ax_vy, ax_w, ax_acc_x,ax_acc_y,ax_acc_w,ax_vx2,ax_w2 = plot_kinemaitcs_data(df)








# train the SVGP model
n_inducing_points = 500




if actuator_time_delay_fitting_tag == 0:
    n_past_actions = 1 

    # columns_to_extract = ['vx body', 'vy body', 'w', 'throttle' ,'steering'] 
    # train_x_full_dataset = torch.tensor(df[columns_to_extract].to_numpy()).cuda()

elif actuator_time_delay_fitting_tag == 3:
    columns_to_extract = ['vx body', 'vy body', 'w', 'throttle filtered' ,'steering filtered'] 
    train_x_full_dataset = torch.tensor(df[columns_to_extract].to_numpy()).cuda()

else:
    # columns_to_extract = ['vx body', 'vy body', 'w'] 
    # train_x_states = torch.tensor(df[columns_to_extract].to_numpy()) #.cuda()
    # load first guess from folder
    folder_path_act_dyn_params = os.path.join('Data',rosbag_folder,'actuator_dynamics_saved_parameters/')
    first_guess_weights_throttle = np.load(folder_path_act_dyn_params + 'raw_weights_throttle.npy')
    first_guess_weights_steering = np.load(folder_path_act_dyn_params + 'raw_weights_steering.npy')
    n_past_actions = np.load(folder_path_act_dyn_params + 'n_past_actions.npy')
    first_guess_weights_throttle = torch.Tensor(first_guess_weights_throttle).cuda()
    first_guess_weights_steering = torch.Tensor(first_guess_weights_steering).cuda()

    #n_past_actions =  100 # 100 Hz seconds of past actions   300
    #refinement_factor = 1 # no need to refine the time interval between data points








columns_to_extract = ['vx body', 'vy body', 'w'] 
train_x_states = torch.tensor(df[columns_to_extract].to_numpy()) #.cuda()
train_x_throttle = generate_tensor_past_actions(df, n_past_actions, key_to_repeat = 'throttle')
train_x_steering = generate_tensor_past_actions(df, n_past_actions, key_to_repeat = 'steering')

train_x_full_dataset = torch.cat((train_x_states,train_x_throttle,train_x_steering),1) # concatenate



# produce y lables
train_y_vx_full_dataset = torch.unsqueeze(torch.tensor(df['ax body'].to_numpy()),1)#.cuda()
train_y_vy_full_dataset = torch.unsqueeze(torch.tensor(df['ay body'].to_numpy()),1)#.cuda()
train_y_w_full_dataset  = torch.unsqueeze(torch.tensor(df['acc_w'].to_numpy()),1)#.cuda()


if use_nominal_model: # subtract the nominal model from the y predictions
    # evaluate the dynamic bicycle model on the training data
    #steering_friction_flag = True
    #pitch_dynamics_flag = False # don't modify these as they need to match what is used in the DART simulator node
    #nominal_model_dyn_bike = dyn_model_culomb_tires(steering_friction_flag,pitch_dynamics_flag)
    nom_pred_x = np.zeros(train_y_vx_full_dataset.shape[0])
    nom_pred_y = np.zeros(train_y_vy_full_dataset.shape[0])
    nom_pred_w = np.zeros(train_y_w_full_dataset.shape[0])

    for kk in range(train_y_vx_full_dataset.shape[0]):
        # make prediction using nominal model
        #state_action_k = np.array([*train_x_full_dataset[kk,:].cpu().numpy(),0.0,0.0])
        # add dummy values for the throttle and steering commands in the nominal model
        # the latter takes in state_action = [vx vy w throttle steering throttle_command steering_command]
        #pred_nom = nominal_model_dyn_bike.forward(state_action_k)
        #unpack state
        th = train_x_throttle[kk,0].numpy().item()
        st = train_x_steering[kk,0].numpy().item()
        vx = train_x_states[kk,0].numpy().item()
        vy = train_x_states[kk,1].numpy().item()
        w =  train_x_states[kk,2].numpy().item()

        acc_x, acc_y, acc_w = mf.dynamic_bicycle(th, st, vx, vy, w)

        # subtract the nominal model from the data (Do this in place to save memory)
        train_y_vx_full_dataset[kk] = train_y_vx_full_dataset[kk] - acc_x
        train_y_vy_full_dataset[kk] = train_y_vy_full_dataset[kk] - acc_y
        train_y_w_full_dataset[kk] = train_y_w_full_dataset[kk] - acc_w

        # store the nominal model predictions
        nom_pred_x[kk] = acc_x
        nom_pred_y[kk] = acc_y
        nom_pred_w[kk] = acc_w






# to help with the delay tuning process we can repeat the portions of the data where the throttle and steering are chaging more rapidly
# derive the time derivative of the throttle and steering

from scipy.signal import savgol_filter
df['ax body filtered'] = savgol_filter(df['ax body'], window_length=30, polyorder=2)

percentile = 80


# Define a threshold_accx (e.g., 90th percentile of acc x
threshold_accx = np.percentile(np.abs(df['ax body filtered'].to_numpy()), percentile)
# Find regions where the derivative is high
mask_high_th_dev = np.abs(np.array(df['ax body filtered'])) > threshold_accx

# plot filtered signal
ax_acc_x.plot(df['vicon time'].to_numpy(),df['ax body filtered'].to_numpy(),label='ax filtered',color='navy')
ax_vx2.fill_between(df['vicon time'].to_numpy(), ax_vx2.get_ylim()[0], ax_vx2.get_ylim()[1], where=mask_high_th_dev, color='navy', alpha=0.1, label='high acc x regions')


# find high omega acceleration regions
# filter omega acc
df['acc_w filtered'] = savgol_filter(df['acc_w'], window_length=30, polyorder=2)
# Define a threshold_accx (e.g., 90th percentile of acc x
threshold_accw = np.percentile(np.abs(df['acc_w filtered'].to_numpy()), percentile)

# Find regions where the derivative is high
mask_high_th_dev_w = np.abs(np.array(df['acc_w filtered'])) > threshold_accw

# plot filtered signal
ax_acc_w.plot(df['vicon time'].to_numpy(),df['acc_w filtered'].to_numpy(),label='aw filtered',color='navy')
ax_w2.fill_between(df['vicon time'].to_numpy(), ax_w2.get_ylim()[0], ax_w2.get_ylim()[1], where=mask_high_th_dev_w, color='navy', alpha=0.1, label='high acc w regions')


# merge the two masks
mask_high_acc = mask_high_th_dev | mask_high_th_dev_w



# re-add this point to the training data so to give more importance to these regions
# get the inxes of the high acceleration regions
indexes_high_dev = np.where(mask_high_acc)[0]
# add the indexes to the training data a certain number of times
high_acc_repetition = 4 # 20
original_data_length = train_x_full_dataset.shape[0]
for _ in range(high_acc_repetition):
    # add a very small random jitter to avoid ill conditioning in the Kxx matrix later on
    jitter_level = 1e-4
    jitter_x = torch.randn(indexes_high_dev.shape[0],train_x_full_dataset.shape[1])*jitter_level
    jitter_y_vx = torch.randn(indexes_high_dev.shape[0],train_y_vx_full_dataset.shape[1])*jitter_level
    jitter_y_vy = torch.randn(indexes_high_dev.shape[0],train_y_vy_full_dataset.shape[1])*jitter_level
    jitter_y_w = torch.randn(indexes_high_dev.shape[0],train_y_w_full_dataset.shape[1])*jitter_level

    train_x_full_dataset = torch.cat((train_x_full_dataset,train_x_full_dataset[indexes_high_dev,:]+jitter_x),0)
    train_y_vx_full_dataset = torch.cat((train_y_vx_full_dataset,train_y_vx_full_dataset[indexes_high_dev,:]+jitter_y_vx),0)
    train_y_vy_full_dataset = torch.cat((train_y_vy_full_dataset,train_y_vy_full_dataset[indexes_high_dev,:]+jitter_y_vy),0)
    train_y_w_full_dataset = torch.cat((train_y_w_full_dataset,train_y_w_full_dataset[indexes_high_dev,:]+jitter_y_w),0)











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


# # first guess weights
# # define as zeros 
# fixed_act_delay_guess_st = 0.3 # this is the time delay in seconds as we can see it from the data
# fixed_act_delay_guess_th = 0.3 # this is the time delay in seconds as we can see it from the data
# smoothing_window = int(np.round(0.25/dt)) # 0.1 seconds smoothing window
# delay_in_steps_st = int(fixed_act_delay_guess_st/dt)
# delay_in_steps_th = int(fixed_act_delay_guess_th/dt)

# # -10 , 10 is mapped to 0 and 1 in torch.sigmoid so we can use this to set the initial guess
# first_guess_weights_throttle = torch.ones(1,n_past_actions,requires_grad=True).cuda() * -10
# first_guess_weights_steering = torch.ones(1,n_past_actions,requires_grad=True).cuda() * -10
# first_guess_weights_throttle[0,delay_in_steps_th-smoothing_window:delay_in_steps_th+smoothing_window] = 10
# first_guess_weights_steering[0,delay_in_steps_st-smoothing_window:delay_in_steps_st+smoothing_window] = 10




# # # gaussian first guess
# # fixed_act_delay_guess_st = 0.1  # Time delay (seconds) 0.3
# # fixed_act_delay_guess_th = 0.2  # Time delay (seconds) 0.3
# # smoothing_window = int(np.round(0.1 / dt))  # 0.1 seconds smoothing window
# # delay_in_steps_st = int(fixed_act_delay_guess_st / dt)
# # delay_in_steps_th = int(fixed_act_delay_guess_th / dt)

# # # Define Gaussian Function
# # def gaussian_weights(n_past_actions, center, std_dev=10):
# #     """Creates a 1D Gaussian distribution centered at `center`."""
# #     x = torch.arange(n_past_actions).float().cuda()
# #     gauss = torch.exp(-((x - center) ** 2) / (2 * std_dev**2))  # Gaussian formula
# #     gauss = (gauss - gauss.min()) / (gauss.max() - gauss.min())  # Normalize to [0, 1]
# #     return torch.logit(gauss * 0.98 + 0.01)  # Convert to logit space for sigmoid

# # # Initialize Weights
# # first_guess_weights_throttle = gaussian_weights(n_past_actions, delay_in_steps_th)
# # first_guess_weights_steering = gaussian_weights(n_past_actions, delay_in_steps_st)

# # # Ensure requires_grad=True for optimization
# # first_guess_weights_throttle = first_guess_weights_throttle.unsqueeze(0).requires_grad_().cuda()
# # first_guess_weights_steering = first_guess_weights_steering.unsqueeze(0).requires_grad_().cuda()


# first guess no delay
# hardsigmoid maps to -3 to 3 --> 0 to 1
# max_val_sigmoid = 2.99999
# first_guess_weights_throttle = torch.ones(1,n_past_actions,requires_grad=True).cuda() * -max_val_sigmoid
# first_guess_weights_steering = torch.ones(1,n_past_actions,requires_grad=True).cuda() * -max_val_sigmoid
# first_guess_weights_throttle[0,8] = max_val_sigmoid
# first_guess_weights_steering[0,10] = max_val_sigmoid

# apply first guess






# apply first guess
SVGP_unified_model_obj.raw_weights_throttle.data = first_guess_weights_throttle
SVGP_unified_model_obj.raw_weights_steering.data = first_guess_weights_steering


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

# reset initial guess for the weights
#SVGP_unified_model_obj.raw_weights_throttle.data = first_guess_weights_throttle
#SVGP_unified_model_obj.raw_weights_steering.data = first_guess_weights_steering
# train again
#SVGP_unified_model_obj.train_model(epochs,learning_rate,train_x, train_y_vx, train_y_vy, train_y_w)







# # Save model parameters
if over_write_saved_parameters:
    print('')
    print('saving model parameters')
    SVGP_unified_model_obj.save_model(folder_path_SVGP_params,actuator_time_delay_fitting_tag,n_past_actions,dt)





# cut training data to the original length if needed
if high_acc_repetition > 0:
    train_x = train_x[:original_data_length,:]
    train_y_vx = train_y_vx[:original_data_length,:]
    train_y_vy = train_y_vy[:original_data_length,:]
    train_y_w = train_y_w[:original_data_length,:]
    train_x_full_dataset = train_x_full_dataset[:original_data_length,:]
    train_y_vx_full_dataset = train_y_vx_full_dataset[:original_data_length,:]
    train_y_vy_full_dataset = train_y_vy_full_dataset[:original_data_length,:]
    train_y_w_full_dataset = train_y_w_full_dataset[:original_data_length,:]






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

        # output_vx,    output_vy,      output_w,       weights_throttle, weights_steering,  non_normalized_w_th, non_normalized_w_st
        preds_ax_batch, preds_ay_batch, preds_aw_batch, weights_throttle, weights_steering , non_normalized_w_th, non_normalized_w_st = SVGP_unified_model_obj(batch_data)
        # preds_ax_batch = SVGP_unified_model_obj.model_vx(batch_data)
        # preds_ay_batch = SVGP_unified_model_obj.model_vy(batch_data)
        # preds_aw_batch = SVGP_unified_model_obj.model_w(batch_data)

        # get the mean of the predictions
        preds_ax_batch_mean = preds_ax_batch.mean.cpu().numpy()
        preds_ay_batch_mean = preds_ay_batch.mean.cpu().numpy()
        preds_aw_batch_mean = preds_aw_batch.mean.cpu().numpy()

        # if we are using the nominal model subtract the nominal model from the predictions
        # if use_nominal_model:
        #     #evalaute nominal model predictions on the batch data

        #     y_nominal_vx_batch = np.zeros(batch_data.shape[0])
        #     y_nominal_vy_batch = np.zeros(batch_data.shape[0])
        #     y_nominal_w_batch = np.zeros(batch_data.shape[0])

        #     for kk in range(batch_data.shape[0]):

        #         # make prediction using nominal model
        #         state_action_k = np.array([*batch_data[kk,:].cpu().numpy(),0.0,0.0])
        #         # add dummy values for the throttle and steering commands in the nominal model
        #         # the latter takes in state_action = [vx vy w throttle steering throttle_command steering_command]
        #         pred_nom = nominal_model_dyn_bike.forward(state_action_k)
        #         acc_x = pred_nom[0]
        #         acc_y = pred_nom[1]
        #         acc_w = pred_nom[2]
        #         # these last two are not used
        #         throttle_dot = pred_nom[3]
        #         steering_dot = pred_nom[4]

        #         #store the nominal model predictions
        #         y_nominal_vx_batch[kk] = acc_x
        #         y_nominal_vy_batch[kk] = acc_y
        #         y_nominal_w_batch[kk] = acc_w

        #         # subtract the nominal model from the data (Do this in place to save memory)
        #         preds_ax_batch_mean[kk] = preds_ax_batch_mean[kk] + acc_x
        #         preds_ay_batch_mean[kk] = preds_ay_batch_mean[kk] + acc_y
        #         preds_aw_batch_mean[kk] = preds_aw_batch_mean[kk] + acc_w

        

        # Get predictions for each model
        preds_ax_mean.extend(preds_ax_batch_mean)
        preds_ay_mean.extend(preds_ay_batch_mean) 
        preds_aw_mean.extend(preds_aw_batch_mean)

        # if use_nominal_model:
        #     y_nominal_vx.extend(y_nominal_vx_batch)
        #     y_nominal_vy.extend(y_nominal_vy_batch)
        #     y_nominal_w.extend(y_nominal_w_batch)

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
        weights_th_numpy, non_normalized_w_th = SVGP_unified_model_obj.constrained_linear_layer(SVGP_unified_model_obj.raw_weights_throttle)
        weights_st_numpy, non_normalized_w_st = SVGP_unified_model_obj.constrained_linear_layer(SVGP_unified_model_obj.raw_weights_steering)
        # make into numpy
        weights_th_numpy = np.transpose(weights_th_numpy.cpu().detach().numpy())
        weights_st_numpy = np.transpose(weights_st_numpy.cpu().detach().numpy())

        # evalaute the filtered inputs
        th_filtered = np.zeros(train_x.shape[0])
        st_filtered = np.zeros(train_x.shape[0])
        for i in range(train_x.shape[0]):
            # produce action to pass to the model
            data_row = np.expand_dims(train_x.cpu().numpy()[i,:],0)
            th_past  = data_row[0, 3 : 3 + n_past_actions]
            st_past  = data_row[0, 3 + n_past_actions :]

            th_filtered[i] = np.expand_dims(th_past,0) @ weights_th_numpy
            st_filtered[i] = np.expand_dims(st_past,0) @ weights_st_numpy



        #weights_throttle = SVGP_unified_model_obj.constrained_linear_layer(SVGP_unified_model_obj.raw_weights_throttle)[0]
        #weights_steering = SVGP_unified_model_obj.constrained_linear_layer(SVGP_unified_model_obj.raw_weights_steering)[0]

    time_axis = np.linspace(0,n_past_actions*dt,n_past_actions)
    ax1.plot(time_axis,np.squeeze(weights_th_numpy),color='dodgerblue',label='throttle weights')
    ax1.plot(time_axis,np.squeeze(weights_st_numpy),color='orangered',label='steering weights')
    ax1.set_xlabel('time delay[s]')
    ax1.set_ylabel('weight')
    # set y axis limits
    # max value
    max_val = np.max([np.max(np.abs(weights_throttle.detach().cpu().numpy())),np.max(np.abs(weights_steering.detach().cpu().numpy()))])
    ax1.set_ylim(0,max_val * 1.3)
    ax1.legend()

    # plot the filtered inputs
    if actuator_time_delay_fitting_tag == 2:
        fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax1.plot(df['vicon time'].to_numpy(),train_x_throttle[:,0].cpu().numpy(),color='gray',label='throttle')
        ax1.plot(df['vicon time'].to_numpy(),th_filtered,color='dodgerblue',label='throttle filtered')
        ax1.set_ylabel('throttle')
        ax1.legend()
        ax1.set_xlabel('time [s]')

        
        ax2.plot(df['vicon time'].to_numpy(),train_x_steering[:,0].cpu().numpy(),color='gray',label='steering')
        ax2.plot(df['vicon time'].to_numpy(),st_filtered,color='orangered',label='steering filtered')
        ax2.set_ylabel('steering')
        ax2.legend()
        ax2.set_xlabel('time [s]')













if use_nominal_model:
    # re-evaluate the nominal model predictions with the filtered inputs
    if actuator_time_delay_fitting_tag == 2:
        weights_th_numpy, non_normalized_w_th = SVGP_unified_model_obj.constrained_linear_layer(SVGP_unified_model_obj.raw_weights_throttle)
        weights_st_numpy, non_normalized_w_st = SVGP_unified_model_obj.constrained_linear_layer(SVGP_unified_model_obj.raw_weights_steering)
        # make into numpy
        weights_th_numpy = np.transpose(weights_th_numpy.cpu().detach().numpy())
        weights_st_numpy = np.transpose(weights_st_numpy.cpu().detach().numpy())

        nom_pred_x_filtered_input = np.zeros(train_x.shape[0])
        nom_pred_y_filtered_input = np.zeros(train_x.shape[0])
        nom_pred_w_filtered_input = np.zeros(train_x.shape[0])

        for i in range(train_x.shape[0]):
            # produce action to pass to the model
            data_row = np.expand_dims(train_x.cpu().numpy()[i,:],0)
            th_past  = data_row[0, 3 : 3 + n_past_actions]
            st_past  = data_row[0, 3 + n_past_actions :]

            th = np.expand_dims(th_past,0) @ weights_th_numpy
            st = np.expand_dims(st_past,0) @ weights_st_numpy

            vx = data_row[0][0]
            vy = data_row[0][1]
            w = data_row[0][2]

            acc_x, acc_y, acc_w = mf.dynamic_bicycle(th, st, vx, vy, w)
            # store for later use
            nom_pred_x_filtered_input[i] = acc_x
            nom_pred_y_filtered_input[i] = acc_y
            nom_pred_w_filtered_input[i] = acc_w

    # add the nominal model predictons
    preds_ax_mean = preds_ax_mean + nom_pred_x_filtered_input
    preds_ay_mean = preds_ay_mean + nom_pred_y_filtered_input
    preds_aw_mean = preds_aw_mean + nom_pred_w_filtered_input
    # adjust the confidence bounds
    lower_x = lower_x + nom_pred_x_filtered_input
    upper_x = upper_x + nom_pred_x_filtered_input
    lower_y = lower_y + nom_pred_y_filtered_input
    upper_y = upper_y + nom_pred_y_filtered_input
    lower_w = lower_w + nom_pred_w_filtered_input
    upper_w = upper_w + nom_pred_w_filtered_input
else:

    # add the nominal model predictons
    preds_ax_mean = preds_aw_mean + nom_pred_x
    preds_ay_mean = preds_aw_mean + nom_pred_y
    preds_aw_mean = preds_aw_mean + nom_pred_w
    # adjust the confidence bounds
    lower_x = lower_x + nom_pred_x
    upper_x = upper_x + nom_pred_x
    lower_y = lower_y + nom_pred_y
    upper_y = upper_y + nom_pred_y
    lower_w = lower_w + nom_pred_w
    upper_w = upper_w + nom_pred_w





ax_acc_x.set_title('ax in body frame')
ax_plot_mean = ax_acc_x.plot(df['vicon time'].to_numpy(),preds_ax_mean,color='k',label='model prediction',linewidth=thick_line_width)
aax_plot_confidence = ax_acc_x.fill_between(df['vicon time'].to_numpy(), lower_x, upper_x, alpha=0.2,color='k',label='2 sigma confidence',zorder=20)
ax_acc_x.legend()

ax_acc_y.set_title('ay in body frame')
ax_acc_y.plot(df['vicon time'].to_numpy(),preds_ay_mean,color='k',label='model prediction',linewidth=thick_line_width)
ax_acc_y.fill_between(df['vicon time'].to_numpy(), lower_y, upper_y, alpha=0.2,color='k',label='2 sigma confidence',zorder=20)
ax_acc_y.legend()

ax_acc_w.set_title('aw in body frame')
ax_acc_w.plot(df['vicon time'].to_numpy(),preds_aw_mean,color='k',label='model prediction',linewidth=thick_line_width)
ax_acc_w.fill_between(df['vicon time'].to_numpy(), lower_w, upper_w, alpha=0.2,color='k',label='2 sigma confidence',zorder=20)
ax_acc_w.legend()

if use_nominal_model:
    # add nominal model predictions to plot
    ax_acc_x.plot(df['vicon time'].to_numpy(),nom_pred_x,color='gray',label='nominal model',linestyle='--',linewidth=thick_line_width)
    ax_acc_y.plot(df['vicon time'].to_numpy(),nom_pred_y,color='gray',label='nominal model',linestyle='--',linewidth=thick_line_width)
    ax_acc_w.plot(df['vicon time'].to_numpy(),nom_pred_w,color='gray',label='nominal model',linestyle='--',linewidth=thick_line_width)
    if actuator_time_delay_fitting_tag == 2:
        ax_acc_x.plot(df['vicon time'].to_numpy(),nom_pred_x_filtered_input,color='lightblue',label='nominal model',linestyle='--',linewidth=thick_line_width)
        ax_acc_y.plot(df['vicon time'].to_numpy(),nom_pred_y_filtered_input,color='lightblue',label='nominal model',linestyle='--',linewidth=thick_line_width)
        ax_acc_w.plot(df['vicon time'].to_numpy(),nom_pred_w_filtered_input,color='lightblue',label='nominal model',linestyle='--',linewidth=thick_line_width)







    


plt.show()




# load the model as it could be used later in long term predictions
if check_SVGP_analytic_rebuild or evaluate_long_term_predictions: 
    from dart_dynamic_models import SVGP_unified_analytic
    # instantiate the SVGP model analytic version
    SVGP_unified_analytic_obj = SVGP_unified_analytic()
    # load the parameters
    SVGP_unified_analytic_obj.load_parameters(folder_path_SVGP_params)


if check_SVGP_analytic_rebuild:
    print('')
    print('re-evalauting SVGP using numpy array formulation')

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
        # produce action to pass to the model
        data_row = np.expand_dims(train_x.cpu().numpy()[i,:],0)
        th_past  = data_row[0, 3 : 3 + SVGP_unified_analytic_obj.n_past_actions]
        st_past  = data_row[0, 3 + SVGP_unified_analytic_obj.n_past_actions :]

        th = np.expand_dims(th_past,0) @ SVGP_unified_analytic_obj.weights_throttle
        st = np.expand_dims(st_past,0) @ SVGP_unified_analytic_obj.weights_steering


        vx = data_row[0][0]
        vy = data_row[0][1]
        w = data_row[0][2]
        x_star = np.expand_dims(np.array([vx,vy,w,th.item(),st.item()]),0)
        # prediction using the SVGP model analytic
        mean_x, mean_y, mean_w, cov_mS_x, cov_mS_y, cov_mS_w = SVGP_unified_analytic_obj.predictive_mean_cov(x_star)

        # store for plotting
        # x
        two_sigma_cov_rebuilt_x[i] = np.sqrt(cov_mS_x) * 2
        mean_mS_x[i] = mean_x
        # y
        two_sigma_cov_rebuilt_y[i] = np.sqrt(cov_mS_y) * 2
        mean_mS_y[i] = mean_y
        # w
        two_sigma_cov_rebuilt_w[i] = np.sqrt(cov_mS_w) * 2
        mean_mS_w[i] = mean_w





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




if evaluate_long_term_predictions:
    print('Producing long term predictions')
    # copy the thorottle and steering commands from the filtered data
    if actuator_time_delay_fitting_tag == 0:
        df['throttle filtered'] = df['throttle']
        df['steering filtered'] = df['steering']
    elif actuator_time_delay_fitting_tag == 2:
        # NOTE this really would need to be that there is only 1 steering and throttle filtered that all models use
        # but now using the same as the one used for the model fitting
        df['throttle filtered'] = (train_x_throttle.float().cpu() @ weights_throttle.t().float().cpu()).detach().numpy()
        df['steering filtered'] = (train_x_steering.float().cpu() @ weights_steering.t().float().cpu()).detach().numpy()




    columns_to_extract = ['vicon time', 'vx body', 'vy body', 'w', 'throttle filtered' ,'steering filtered', 'throttle' ,'steering','vicon x','vicon y','vicon yaw']
    input_data_long_term_predictions = df[columns_to_extract].to_numpy()
    prediction_window = 1.5 # [s]
    jumps = 25 #25

    forward_function = SVGP_unified_analytic_obj.forward_4_long_term_prediction

    forward_propagate_indexes = [1,2,3] # [1,2,3,4,5] # # 1 = vx, 2=vy, 3=w, 4=throttle, 5=steering

    long_term_predictions = produce_long_term_predictions(input_data_long_term_predictions, forward_function,prediction_window,jumps,forward_propagate_indexes)




    # plot long term predictions

    print('plotting long term predictions')

    for pred in tqdm.tqdm(long_term_predictions, desc="Rollouts"):

        #velocities
        ax_vx.plot(pred[:,0],pred[:,1],color='k',alpha=0.2)
        ax_vy.plot(pred[:,0],pred[:,2],color='k',alpha=0.2)
        ax_w.plot(pred[:,0],pred[:,3],color='k',alpha=0.2)






plt.show()