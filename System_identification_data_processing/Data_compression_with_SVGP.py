import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from functions_for_data_processing import get_data, throttle_dynamics_data_processing, steering_dynamics_data_processing, process_vicon_data_kinematics, process_raw_vicon_data


# define the folder path to load data
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


# cut time up to 66.5 s
# df = df[df['vicon time']>7.5] 
# df = df[df['vicon time']<66.5]  

# df = df[df['vicon time']>34] 
# df = df[df['vicon time']<36.5] 


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
# add title and labels
ax_x.set_title('Body frame accelerations')
ax_x.set_ylabel('ax [m/s^2]')
ax_y.set_ylabel('ay [m/s^2]')
ax_w.set_ylabel('aw [m/s^2]')
ax_w.set_xlabel('Time [s]')




# produce SVGP model
# SVGP 
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

class SVGPModel(ApproximateGP):
    def __init__(self,inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(SVGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    




def plot_GP(ax,x,y,subset_indexes,model,likelihood,resolution,df):
    resolution = resolution * 0.99
    # move to cpu
    x = x.cpu()
    y = y.cpu()
    model = model.cpu()
    likelihood = likelihood.cpu()

    # plot subset with an orange circle
    ax.plot(df['vicon time'].to_numpy()[subset_indexes], y[subset_indexes].cpu().numpy(), 'o', color='orange',alpha=0.5,markersize=3)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred_likelyhood = likelihood(model(x))
        lower_likelyhood, upper_likelyhood = observed_pred_likelyhood.confidence_region()
        ax.plot(df['vicon time'].to_numpy(), observed_pred_likelyhood.mean.cpu().numpy(), 'k')
        ax.fill_between(df['vicon time'].to_numpy(), lower_likelyhood.cpu().numpy(), upper_likelyhood.cpu().numpy(), alpha=0.3)

        # obseved pred epistemic uncertainty
        observed_pred_model = model(x)
        lower_model, upper_model = observed_pred_model.confidence_region()
        ax.fill_between(df['vicon time'].to_numpy(), lower_model.cpu().numpy(), upper_model.cpu().numpy(), alpha=0.3, color='k')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Training Data')

    #move back to gpu
    x = x.cuda()
    y = y.cuda()
    model = model.cuda()
    likelihood = likelihood.cuda()






# produce fitting data
columns_to_extract = ['vx body', 'vy body', 'w', 'throttle' ,'steering'] #  angle time delayed
x = torch.tensor(df[columns_to_extract].to_numpy()).cuda()
y = torch.tensor(df['ax body'].to_numpy()).cuda()

# convert to float32
x = x.to(torch.float32)
y = y.to(torch.float32)

# initialize inducing points by selecting them randomly from the input data
n_inducing = 500
resolution = df['vicon time'].to_numpy()[1] - df['vicon time'].to_numpy()[0]

# define data subset as the inducing points

import random
subset_indexes_initnial_fit = random.sample(range(x.size(0)), n_inducing) # to get a good initial fit you need random samples
inducing_points = x[subset_indexes_initnial_fit,:] # first n datapoints
# here you can test the algorithm by giving the first data samples as inducing points
#subset_indexes = random.sample(range(x.size(0)), n_inducing)
subset_indexes = list(range(n_inducing))
x_subset = x.clone()[subset_indexes]
y_subset = y.clone()[subset_indexes]


training_iterations = 60
training_iterations_retraining = 60
# plot training data and inducing points


# intialize SVGP model
model = SVGPModel(inducing_points)
model.train()
model = model.cuda()

# initialize likelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood()
likelihood.train()
likelihood = likelihood.cuda()
# assign the likelihood noise that was fitted previously on a static input-output point (i.e. we have measured the noise in the y data)
raw_likely_hood_noise_from_noise_measurement = -4.736683
likelihood.noise_covar.raw_noise.data = torch.tensor([raw_likely_hood_noise_from_noise_measurement]).cuda()


optimizer = torch.optim.Adam([
{'params': model.parameters()},
], lr=0.03)
# {'params': likelihood.parameters()} $ removed likelihood parameters from the optimizer cause we previously fitted the noise

# Use the adam optimizer
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y.size(0))

# define data loader to hadle large data in the GPU
from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(x, y)
train_loader = DataLoader(train_dataset, batch_size=2500, shuffle=True)

# add a progress bar
print('training SVGP model on the full dataset') 
import tqdm
epochs_iter = tqdm.tqdm(range(training_iterations), desc="Epoch")
epochs_iter_retraining = tqdm.tqdm(range(training_iterations_retraining), desc="Epoch")
minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False, disable=True)
# Initialize a list to store loss values
loss_values = []
for i in epochs_iter:
    for x_batch, y_batch in minibatch_iter:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = -mll(output, y_batch)
        loss.backward()
        optimizer.step()
        # Store the loss value for plot
    loss_values.append(loss.item())


# print out likelyhood noise
print(f"RAW Likelyhood noise = {likelihood.noise_covar.raw_noise.item():.6f}")



# Plot the loss function
fig, ax_loss = plt.subplots(1, 1, figsize=(10, 6))
ax_loss.plot(loss_values, label='Training Loss')
ax_loss.set_xlabel('Iteration')
ax_loss.set_ylabel('Loss')
ax_loss.set_title('Loss Function During Training')
ax_loss.legend()
ax_loss.grid(True)




# now remove the lengthscale and the outputscale from the training process
model.covar_module.base_kernel.raw_lengthscale.requires_grad = False
model.covar_module.raw_outputscale.requires_grad = False



# plotting body frame accelerations and data fitting results
# now data compression

plot_GP(ax_x,x,y,subset_indexes,model,likelihood,resolution,df)




# evalaute max uncertainty rateo
prior_stdd = np.sqrt(model.covar_module.outputscale.item())
preds = model(x) # evaluate predictive posterior on full dataset
max_stdd = torch.max(preds.variance).item()**0.5 # evaluate max uncertainty

# evalaute max rateo
stdd_rateo_threshold = max_stdd/prior_stdd 
acceptable_rateo_increase = 1.0 #we allow 10% increase in the rateo



#initially assign the x subset to the inducing points
# x_subset = x_subset.detach()  # Detach to avoid any gradient tracking
# model.variational_strategy.inducing_points.data = x_subset


# start training loop
max_stdd_rateo = 1 + stdd_rateo_threshold # set to high value to start the loop

# set inducing points to the subset
inducing_points = x.clone()[subset_indexes,:]
# assign the variational mean to the model
#model.variational_strategy._variational_distribution.variational_mean.data = y_subset.clone() # update the first guess variational mean
# update inducing points first guess
model.variational_strategy.inducing_points.data = inducing_points

optimizer_retrain = torch.optim.Adam([
    {'params': [param for name, param in model.named_parameters()
                if 'raw_lengthscale' not in name and 'outputscale' not in name]}
], lr=0.03)


plt.ion()
while max_stdd_rateo > stdd_rateo_threshold * acceptable_rateo_increase:
    # now just optimize the inducing points locations
    # Only optimize the inducing points


    # Use the adam optimizer
    #mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_subset.size(0))

    # update the training data loader
    train_dataset_retrain = TensorDataset(x_subset, y_subset)
    train_loader_retrain = DataLoader(train_dataset_retrain, batch_size=2500, shuffle=True)
    minibatch_iter_retrain = tqdm.tqdm(train_loader_retrain, desc="Minibatch retrain", leave=False, disable=True)
    loss_values = []
    for i in epochs_iter:
        for x_batch_retrain, y_batch_retrain in minibatch_iter_retrain:
            optimizer_retrain.zero_grad()
            output = model(x_batch_retrain)
            loss = -mll(output, y_batch_retrain)
            loss.backward()
            optimizer_retrain.step()
            loss_values.append(loss.item())

    # plot loss function
    ax_loss.clear()
    ax_loss.plot(loss_values, label='Training Loss')
    ax_loss.set_xlabel('Iteration')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Loss Function During Training')
    ax_loss.legend()
    ax_loss.grid(True)


    # plotting results
    ax_x.clear()
    ax_x.plot(df['vicon time'].to_numpy(),df['ax body'].to_numpy(),label='ax',color='dodgerblue') # plot orginal data
    plot_GP(ax_x,x,y,subset_indexes,model,likelihood,resolution,df)
    # --- finished plotting ---




    # update points in dataset and inducing points

    # evaluate maximum uncertainty
    prior_stdd = np.sqrt(model.covar_module.outputscale.item())

    # move to cpu
    x = x.cpu()
    model = model.cpu()

    preds = model(x)

    #move back to gpu
    x = x.cuda()
    model = model.cuda()

    stdd = preds.variance.sqrt()  # Get standard deviations

    # Sort uncertainties in descending order and get the indices
    sorted_indices = torch.argsort(stdd, descending=True)

    # Find the first index with the highest uncertainty that is not in subset_indexes
    for idx in sorted_indices:
        if idx.item() not in subset_indexes:
            max_uncertainty_idx = idx.item()
            max_stdd = stdd[max_uncertainty_idx].item()
            max_stdd_rateo = max_stdd/prior_stdd
            subset_indexes.append(max_uncertainty_idx)
            break
    else:
        print("All high-uncertainty points are already in the subset.")





    print(f"max_stdd_rateo = {max_stdd_rateo:.4f}, stdd_rateo_threshold = {stdd_rateo_threshold:.4f}")


    #update the subset of data
    x_subset = x.clone()[subset_indexes] # select chosen points
    y_subset = y.clone()[subset_indexes]


    # !! also update the inducing points !!
    # identify the least informative inducing point
    # evaluate kernel matrix
    KZZ = model.covar_module(model.variational_strategy.inducing_points)

    # select row with the highest sum
    sum_KZZ = torch.sum(KZZ,1)

    # find the index of the least informative inducing point
    least_informative_idx = torch.argmax(sum_KZZ).item()


    # add line to plot the next data point to be added
    ax_x.axvline(df['vicon time'].to_numpy()[max_uncertainty_idx].item(), color='g',alpha=0.3)
    # add dashed line on the inducing point that will be eliminated
    #ax_x.axvline(model.variational_strategy.inducing_points[least_informative_idx].item(), color='k',linestyle='--',alpha=0.3)
    plt.pause(0.1)

    # replace the least informative inducing point with the new data point
    with torch.no_grad():
        #inducing_points[least_informative_idx] = x.clone()[max_uncertainty_idx]
        model.variational_strategy._variational_distribution.variational_mean.data[least_informative_idx] = y.clone()[max_uncertainty_idx] # update the first guess variational mean
        model.variational_strategy.inducing_points.data[least_informative_idx,:] = x[max_uncertainty_idx,:]
        
    a = 1 # just to put a debugger stop point here


plt.ioff()
plt.show()
    


