from dart_dynamic_models import produce_long_term_predictions,process_vicon_data_kinematics,generate_tensor_past_actions,\
plot_kinemaitcs_data, model_functions, dynamic_bicycle_actuator_delay_fitting

from matplotlib import pyplot as plt
import torch
import numpy as np
import pandas as pd
import os
import tqdm
mf = model_functions() # instantiate the model functions class


# in this we will use the dynamic bicycle model to fit the actuator dynamics using a linear layer 





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
folder_path_save_params = os.path.join('Data',rosbag_folder,'actuator_dynamics_saved_parameters/')
reprocess_data = False # set to true to reprocess the data again



# set these parameters as they will determine the running time of this script
# this will re-build the plotting results using an SVGP rebuilt analytically as would a solver
check_SVGP_analytic_rebuild = False
over_write_saved_parameters = True
evaluate_long_term_predictions = False
epochs = 500 #100 #  epochs for training the SVGP 200
training_refinement = 5
learning_rate =  0.005 #0.0003
live_plot_weights = True

# decide which model to train  (better to train one at a time)
train_th = True
train_st = False





normalize_y = False



# process data
steps_shift = 1 # decide to filter more or less the vicon data

# plot settings
thick_line_width = 2

# dataset balancing settings
percentile = 80
high_acc_repetition = 4 # 20





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





# cut only the first 3.5 sec of data
#df = df[df['vicon time'] < 5]






ax_vx,ax_vy, ax_w, ax_acc_x,ax_acc_y,ax_acc_w,ax_vx2,ax_w2 = plot_kinemaitcs_data(df)





n_past_actions =  100 # 100 Hz seconds of past actions   300
#refinement_factor = 1 # no need to refine the time interval between data points


columns_to_extract = ['vx body', 'vy body', 'w'] 
train_x_states = torch.tensor(df[columns_to_extract].to_numpy()) #.cuda()
train_x_throttle = generate_tensor_past_actions(df, n_past_actions, key_to_repeat = 'throttle')
train_x_steering = generate_tensor_past_actions(df, n_past_actions, key_to_repeat = 'steering')

train_x = torch.cat((train_x_states,train_x_throttle,train_x_steering),1) # concatenate



# produce y lables
train_y_vx = torch.unsqueeze(torch.tensor(df['ax body'].to_numpy()),1)#.cuda()
train_y_vy = torch.unsqueeze(torch.tensor(df['ay body'].to_numpy()),1)#.cuda()
train_y_w  = torch.unsqueeze(torch.tensor(df['acc_w'].to_numpy()),1)#.cuda()








# to help with the delay tuning process we can repeat the portions of the data where the throttle and steering are chaging more rapidly
# derive the time derivative of the throttle and steering

from scipy.signal import savgol_filter
df['ax body filtered'] = savgol_filter(df['ax body'], window_length=30, polyorder=2)



# Define a threshold_accx (e.g., 90th percentile of acc x
threshold_accx = np.percentile(np.abs(df['ax body filtered'].to_numpy()), percentile)
# Find regions where the derivative is high
mask_high_th_dev = np.abs(np.array(df['ax body filtered'])) > threshold_accx

# plot filtered signal
ax_acc_x.plot(df['vicon time'].to_numpy(),df['ax body filtered'].to_numpy(),label='ax filtered',color='navy',linestyle='--')
ax_vx2.fill_between(df['vicon time'].to_numpy(), ax_vx2.get_ylim()[0], ax_vx2.get_ylim()[1], where=mask_high_th_dev, color='navy', alpha=0.1, label='high acc x regions')


# find high omega acceleration regions
# filter omega acc
df['acc_w filtered'] = savgol_filter(df['acc_w'], window_length=30, polyorder=2)
# Define a threshold_accx (e.g., 90th percentile of acc x
threshold_accw = np.percentile(np.abs(df['acc_w filtered'].to_numpy()), percentile)

# Find regions where the derivative is high
mask_high_th_dev_w = np.abs(np.array(df['acc_w filtered'])) > threshold_accw

# plot filtered signal
ax_acc_w.plot(df['vicon time'].to_numpy(),df['acc_w filtered'].to_numpy(),label='aw filtered',color='navy',linestyle='--')
ax_w2.fill_between(df['vicon time'].to_numpy(), ax_w2.get_ylim()[0], ax_w2.get_ylim()[1], where=mask_high_th_dev_w, color='navy', alpha=0.1, label='high acc w regions')


# merge the two masks
mask_high_acc = mask_high_th_dev | mask_high_th_dev_w



# re-add this point to the training data so to give more importance to these regions
# get the inxes of the high acceleration regions
indexes_high_dev = np.where(mask_high_acc)[0]
# add the indexes to the training data a certain number of times

original_data_length = train_x.shape[0]
for _ in range(high_acc_repetition):
    # add a very small random jitter to avoid ill conditioning in the Kxx matrix later on
    jitter_level = 1e-4
    jitter_x = torch.randn(indexes_high_dev.shape[0],train_x.shape[1])*jitter_level
    jitter_y_vx = torch.randn(indexes_high_dev.shape[0],train_y_vx.shape[1])*jitter_level
    jitter_y_vy = torch.randn(indexes_high_dev.shape[0],train_y_vy.shape[1])*jitter_level
    jitter_y_w = torch.randn(indexes_high_dev.shape[0],train_y_w.shape[1])*jitter_level

    train_x = torch.cat((train_x,train_x[indexes_high_dev,:]+jitter_x),0)
    train_y_vx = torch.cat((train_y_vx,train_y_vx[indexes_high_dev,:]+jitter_y_vx),0)
    train_y_vy = torch.cat((train_y_vy,train_y_vy[indexes_high_dev,:]+jitter_y_vy),0)
    train_y_w = torch.cat((train_y_w,train_y_w[indexes_high_dev,:]+jitter_y_w),0)











# convert to float to avoid issues with data types
train_x = train_x.to(torch.float32)
train_y_vx = train_y_vx.to(torch.float32)
train_y_vy = train_y_vy.to(torch.float32)
train_y_w = train_y_w.to(torch.float32)










dt = np.mean(np.diff(df['vicon time'].to_numpy())) # time step between data points
print('dt: ', dt)




# # generate the inducing points for the three models, since they need a 5 dimensional input

# generate the inducing points for the three models, since they need a 5 dimensional input
model_obj = dynamic_bicycle_actuator_delay_fitting(n_past_actions,dt)




# ---  first guess ---
# set first guess for the weights
#model_obj.raw_weights_throttle.data[0][0] = torch.tensor([0.00001])
#model_obj.raw_weights_steering.data[0][0] = torch.tensor([0.0000001]) 
# ---------------------



# move to cuda if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('using GPU')
    train_x = train_x.to(device)
    train_y_vx = train_y_vx.to(device)
    train_y_vy = train_y_vy.to(device)
    train_y_w = train_y_w.to(device)


if normalize_y:
    # normalize the outputs between -1 and 1 using min and max values
    train_y_vx = (train_y_vx - train_y_vx.min()) / (train_y_vx.max() - train_y_vx.min()) * 2 - 1
    train_y_vy = (train_y_vy - train_y_vy.min()) / (train_y_vy.max() - train_y_vy.min()) * 2 - 1
    train_y_w = (train_y_w - train_y_w.min()) / (train_y_w.max() - train_y_w.min()) * 2 - 1





model_obj.train_model(epochs,learning_rate,train_x, train_y_vx, train_y_vy, train_y_w,live_plot_weights,train_th,train_st)





# # Save model parameters
if over_write_saved_parameters:
    print('')
    print('saving model parameters')
    model_obj.save_model(folder_path_save_params,n_past_actions,dt,train_th,train_st)





# cut training data to the original length if needed
if high_acc_repetition > 0:
    train_x = train_x[:original_data_length,:]
    train_y_vx = train_y_vx[:original_data_length,:]
    train_y_vy = train_y_vy[:original_data_length,:]
    train_y_w = train_y_w[:original_data_length,:]
    train_x = train_x[:original_data_length,:]
    train_y_vx = train_y_vx[:original_data_length,:]
    train_y_vy = train_y_vy[:original_data_length,:]
    train_y_w = train_y_w[:original_data_length,:]






from torch.utils.data import DataLoader, TensorDataset

# Assuming train_x is a PyTorch Tensor
#train_x = train_x.cpu()
batch_size = 2000  # You can adjust this based on your system's capacity
dataset_to_plot = TensorDataset(train_x)
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



# initialize lists to store nominal model predictions
y_nominal_vx = []
y_nominal_vy = []
y_nominal_w = []


# Loop over mini-batches
with torch.no_grad(): # this actually saves memory allocation cause it doesn't store the gradients
    for batch in dataloader_to_plot:
        batch_data = batch[0]

        # output_vx,    output_vy,      output_w,       weights_throttle, weights_steering,  non_normalized_w_th, non_normalized_w_st
        preds_ax_batch, preds_ay_batch, preds_aw_batch, weights_throttle, weights_steering , non_normalized_w_th, non_normalized_w_st = model_obj(batch_data)


        # get the mean of the predictions
        preds_ax_batch_mean = preds_ax_batch.cpu().numpy()
        preds_ay_batch_mean = preds_ay_batch.cpu().numpy()
        preds_aw_batch_mean = preds_aw_batch.cpu().numpy()


        # Get predictions for each model
        preds_ax_mean.extend(preds_ax_batch_mean)
        preds_ay_mean.extend(preds_ay_batch_mean) 
        preds_aw_mean.extend(preds_aw_batch_mean)

if normalize_y:
    # de-normalize the predictions
    preds_ax_mean = np.array(preds_ax_mean)
    preds_ay_mean = np.array(preds_ay_mean)
    preds_aw_mean = np.array(preds_aw_mean)

    min_y_vx = train_y_vx.min().cpu().numpy()
    max_y_vx = train_y_vx.max().cpu().numpy()
    min_y_vy = train_y_vy.min().cpu().numpy()
    max_y_vy = train_y_vy.max().cpu().numpy()
    min_y_w = train_y_w.min().cpu().numpy()
    max_y_w = train_y_w.max().cpu().numpy()

    preds_ax_mean = (preds_ax_mean + 1) / 2 * (max_y_vx - min_y_vx) + min_y_vx
    preds_ay_mean = (preds_ay_mean + 1) / 2 * (max_y_vy - min_y_vy) + min_y_vy
    preds_aw_mean = (preds_aw_mean + 1) / 2 * (max_y_w - min_y_w) + min_y_w








# # show linear layer weights
fig, (ax1) = plt.subplots(1, 1, figsize=(12, 4), sharex=True)


weights_th_numpy, non_normalized_w_th = model_obj.constrained_linear_layer(model_obj.raw_weights_throttle)
weights_st_numpy, non_normalized_w_st = model_obj.constrained_linear_layer(model_obj.raw_weights_steering)
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


#weights_throttle = model_obj.constrained_linear_layer(model_obj.raw_weights_throttle)[0]
#weights_steering = model_obj.constrained_linear_layer(model_obj.raw_weights_steering)[0]
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







## evalaute the original model predictions without the delay
nom_pred_x = np.zeros(train_y_vx.shape[0])
nom_pred_y = np.zeros(train_y_vy.shape[0])
nom_pred_w = np.zeros(train_y_w.shape[0])

for kk in range(train_y_vx.shape[0]):
    # make prediction using nominal model
    th = train_x_throttle[kk,0].numpy().item()
    st = train_x_steering[kk,0].numpy().item()
    vx = train_x_states[kk,0].numpy().item()
    vy = train_x_states[kk,1].numpy().item()
    w =  train_x_states[kk,2].numpy().item()

    acc_x, acc_y, acc_w = mf.dynamic_bicycle(th, st, vx, vy, w)

    # store the nominal model predictions
    nom_pred_x[kk] = acc_x
    nom_pred_y[kk] = acc_y
    nom_pred_w[kk] = acc_w






ax_acc_x.set_title('ax in body frame')
ax_plot_mean = ax_acc_x.plot(df['vicon time'].to_numpy(),preds_ax_mean,color='k',label='model prediction',linewidth=thick_line_width)


ax_acc_y.set_title('ay in body frame')
ax_acc_y.plot(df['vicon time'].to_numpy(),preds_ay_mean,color='k',label='model prediction',linewidth=thick_line_width)


ax_acc_w.set_title('aw in body frame')
ax_acc_w.plot(df['vicon time'].to_numpy(),preds_aw_mean,color='k',label='model prediction',linewidth=thick_line_width)



# add nominal model predictions to plot
ax_acc_x.plot(df['vicon time'].to_numpy(),nom_pred_x,color='gray',label='nominal model',linestyle='-',linewidth=thick_line_width)
ax_acc_y.plot(df['vicon time'].to_numpy(),nom_pred_y,color='gray',label='nominal model',linestyle='-',linewidth=thick_line_width)
ax_acc_w.plot(df['vicon time'].to_numpy(),nom_pred_w,color='gray',label='nominal model',linestyle='-',linewidth=thick_line_width)
# add legend
ax_acc_x.legend()
ax_acc_y.legend()
ax_acc_w.legend()






    


# load the model as it could be used later in long term predictions
if check_SVGP_analytic_rebuild or evaluate_long_term_predictions: 
    from dart_dynamic_models import SVGP_unified_analytic
    # instantiate the SVGP model analytic version
    SVGP_unified_analytic_obj = SVGP_unified_analytic()
    # load the parameters
    SVGP_unified_analytic_obj.load_parameters(folder_path_save_params)


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