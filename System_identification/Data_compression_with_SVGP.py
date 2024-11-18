import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from dart_dynamic_models import get_data, throttle_dynamics_data_processing, steering_dynamics_data_processing, process_vicon_data_kinematics, process_raw_vicon_data


# define the folder path to load data
folder_path = 'System_identification/Data/91_free_driving_16_sept_2024_slow'

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
subset_indexes = subset_indexes_initnial_fit.copy() #list(range(n_inducing))
x_subset = x.clone()[subset_indexes]
y_subset = y.clone()[subset_indexes]


training_iterations = 100
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
raw_likely_hood_noise_from_noise_measurement_x = 5.675664 #-4.736683
likelihood.noise_covar.raw_noise.data = torch.tensor([raw_likely_hood_noise_from_noise_measurement_x]).cuda()


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
acceptable_rateo_increase = 1 #we allow 10% increase in the rateo  # TESTING!!!! JUST TO TO SET UP Y DATA PART TOO


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
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_subset.size(0))

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

    print(f"max_stdd_rateo = {max_stdd_rateo:.4f}, stdd_rateo_threshold = {stdd_rateo_threshold*acceptable_rateo_increase:.4f}")


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
        









# ---  now repeat for the y data  ---

# ok now we repeat the process with the vy and W data, addin gnew data points to the subset
# produce y fitting data
y_y = torch.tensor(df['ay body'].to_numpy()).cuda()

# convert to float32
y_y = y_y.to(torch.float32)

# pick a rondom subset of the subsetdata
inducing_points_initial_values_y_indexes = random.sample(range(len(subset_indexes)), n_inducing)
inducing_points_y = y_y.clone()[inducing_points_initial_values_y_indexes]

# intialize SVGP model
model_y = SVGPModel(inducing_points_y)
model_y.train()
model_y = model.cuda()

# initialize likelihood
likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
likelihood_y.train()
likelihood_y = likelihood_y.cuda()

# assign the likelihood noise that was fitted previously on a static input-output point (i.e. we have measured the noise in the y data)
raw_likely_hood_noise_from_noise_measurement_y = 0.035466 #-4.736683
likelihood_y.noise_covar.raw_noise.data = torch.tensor([raw_likely_hood_noise_from_noise_measurement_y]).cuda()


optimizer_y = torch.optim.Adam([
{'params': model_y.parameters()},
], lr=0.03)

# {'params': likelihood_y.parameters()} $ removed likelihood parameters from the optimizer cause we previously fitted the noise

# Use the adam optimizer
mll_y = gpytorch.mlls.VariationalELBO(likelihood_y, model_y, num_data=y_y.size(0))

# define data loader to hadle large data in the GPU
from torch.utils.data import TensorDataset, DataLoader
train_dataset_y = TensorDataset(x, y_y)
train_loader_y = DataLoader(train_dataset_y, batch_size=2500, shuffle=True)

# add a progress bar
print('training SVGP model on the full dataset') 
minibatch_iter_y = tqdm.tqdm(train_loader_y, desc="Minibatch", leave=False, disable=True)
# Initialize a list to store loss values
loss_values = []
epochs_iter_y = tqdm.tqdm(range(training_iterations), desc="Epoch")
for i in epochs_iter_y:
    for x_batch, y_batch in minibatch_iter_y:
        optimizer_y.zero_grad()
        output_y = model(x_batch)
        loss_y = -mll_y(output_y, y_batch)
        loss_y.backward()
        optimizer_y.step()
        # Store the loss value for plot
    loss_values.append(loss_y.item())


# plot loss function
ax_loss.clear()
ax_loss.plot(loss_values, label='Training Loss')
ax_loss.set_xlabel('Iteration')
ax_loss.set_ylabel('Loss')
ax_loss.set_title('Loss Function During Training Y')
ax_loss.legend()
ax_loss.grid(True)


plot_GP(ax_y,x,y_y,subset_indexes,model_y,likelihood_y,resolution,df)


# evalaute max uncertainty rateo
prior_stdd_y = np.sqrt(model_y.covar_module.outputscale.item())
preds_y = model_y(x) # evaluate predictive posterior on full dataset
max_stdd_y = torch.max(preds_y.variance).item()**0.5 # evaluate max uncertainty

# evalaute max rateo
stdd_rateo_threshold_y = max_stdd_y/prior_stdd_y

# start training loop
max_stdd_rateo_y = 1 + stdd_rateo_threshold_y # set to high value to start the loop



optimizer_retrain_y = torch.optim.Adam([
    {'params': [param for name, param in model_y.named_parameters()
                if 'raw_lengthscale' not in name and 'outputscale' not in name]}
], lr=0.03)




plt.ion()
while max_stdd_rateo_y > stdd_rateo_threshold_y * acceptable_rateo_increase:
    # now just optimize the inducing points locations
    # Only optimize the inducing points

    # Use the adam optimizer
    mll_y = gpytorch.mlls.VariationalELBO(likelihood_y, model_y, num_data=y_subset.size(0))

    # update the training data loader
    train_dataset_retrain_y = TensorDataset(x_subset, y_subset)
    train_loader_retrain = DataLoader(train_dataset_retrain_y, batch_size=2500, shuffle=True)
    minibatch_iter_retrain_y = tqdm.tqdm(train_loader_retrain, desc="Minibatch retrain", leave=False, disable=True)
    loss_values = []
    for i in epochs_iter:
        for x_batch_retrain_y, y_batch_retrain_y in minibatch_iter_retrain_y:
            optimizer_retrain_y.zero_grad()
            output_y = model_y(x_batch_retrain_y)
            loss_y = -mll(output_y, y_batch_retrain_y)
            loss_y.backward()
            optimizer_retrain_y.step()
            loss_values.append(loss_y.item())

    # plot loss function
    ax_loss.clear()
    ax_loss.plot(loss_values, label='Training Loss')
    ax_loss.set_xlabel('Iteration')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Loss Function During Training')
    ax_loss.legend()
    ax_loss.grid(True)


    # plotting results
    ax_y.clear()
    ax_y.plot(df['vicon time'].to_numpy(),df['ay body'].to_numpy(),label='ay',color='orangered') # plot orginal data
    plot_GP(ax_y,x,y_y,subset_indexes,model_y,likelihood_y,resolution,df)
    # --- finished plotting ---

    # update points in dataset and inducing points

    # evaluate maximum uncertainty
    prior_stdd_y = np.sqrt(model_y.covar_module.outputscale.item())

    # move to cpu
    x = x.cpu()
    model_y = model_y.cpu()

    preds_y = model_y(x)

    #move back to gpu
    x = x.cuda()
    model_y = model_y.cuda()

    stdd_y = preds_y.variance.sqrt()  # Get standard deviations

    # Sort uncertainties in descending order and get the indices
    sorted_indices_y = torch.argsort(stdd_y, descending=True)

    # Find the first index with the highest uncertainty that is not in subset_indexes
    for idx in sorted_indices_y:
        if idx.item() not in subset_indexes:
            max_uncertainty_idx = idx.item()
            max_stdd_y = stdd_y[max_uncertainty_idx].item()
            max_stdd_rateo_y = max_stdd_y/prior_stdd_y
            subset_indexes.append(max_uncertainty_idx)
            break
    else:
        print("All high-uncertainty points are already in the subset.")

    print(f"max_stdd_rateo y = {max_stdd_rateo_y:.4f}, stdd_rateo_threshold y = {stdd_rateo_threshold_y*acceptable_rateo_increase:.4f}")


    #update the subset of data
    x_subset = x.clone()[subset_indexes] # select chosen points
    y_subset = y.clone()[subset_indexes]


    # !! also update the inducing points !!
    # identify the least informative inducing point
    # evaluate kernel matrix
    KZZ = model_y.covar_module(model_y.variational_strategy.inducing_points)

    # select row with the highest sum
    sum_KZZ = torch.sum(KZZ,1)

    # find the index of the least informative inducing point
    least_informative_idx = torch.argmax(sum_KZZ).item()


    # add line to plot the next data point to be added
    ax_y.axvline(df['vicon time'].to_numpy()[max_uncertainty_idx].item(), color='g',alpha=0.3)
    # add dashed line on the inducing point that will be eliminated
    #ax_x.axvline(model.variational_strategy.inducing_points[least_informative_idx].item(), color='k',linestyle='--',alpha=0.3)
    plt.pause(0.1)

    # replace the least informative inducing point with the new data point
    with torch.no_grad():
        #inducing_points[least_informative_idx] = x.clone()[max_uncertainty_idx]
        model_y.variational_strategy._variational_distribution.variational_mean.data[least_informative_idx] = y_y.clone()[max_uncertainty_idx] # update the first guess variational mean
        model_y.variational_strategy.inducing_points.data[least_informative_idx,:] = x[max_uncertainty_idx,:]
        




# now repeat for the w data
# produce w fitting data
y_w = torch.tensor(df['acc_w'].to_numpy()).cuda()

# convert to float32
y_w = y_w.to(torch.float32)

# pick a rondom subset of the subsetdata
inducing_points_initial_values_w_indexes = random.sample(range(len(subset_indexes)), n_inducing)
inducing_points_w = y_w.clone()[inducing_points_initial_values_w_indexes]

# intialize SVGP model
model_w = SVGPModel(inducing_points_w)
model_w.train()
model_w = model.cuda()

# initialize likelihood
likelihood_w = gpytorch.likelihoods.GaussianLikelihood()
likelihood_w.train()
likelihood_w = likelihood_w.cuda()

# assign the likelihood noise that was fitted previously on a static input-output point (i.e. we have measured the noise in the y data)
raw_likely_hood_noise_from_noise_measurement_w = 32.327175 
likelihood_w.noise_covar.raw_noise.data = torch.tensor([raw_likely_hood_noise_from_noise_measurement_w]).cuda()


optimizer_w = torch.optim.Adam([
{'params': model_w.parameters()},
{'params': likelihood.parameters()}
], lr=0.1)
# {'params': likelihood.parameters()} $ removed likelihood parameters from the optimizer cause we previously


# Use the adam optimizer
mll_w = gpytorch.mlls.VariationalELBO(likelihood_w, model_w, num_data=y_w.size(0))

# define data loader to hadle large data in the GPU
train_dataset_w = TensorDataset(x, y_w)
train_loader_w = DataLoader(train_dataset_w, batch_size=2500, shuffle=True)

# add a progress bar

minibatch_iter_w = tqdm.tqdm(train_loader_w, desc="Minibatch", leave=False, disable=True)
# Initialize a list to store loss values
loss_values = []
training_iterations = 200
epochs_iter_w = tqdm.tqdm(range(training_iterations), desc="Epoch")
for i in epochs_iter_w:
    for x_batch, y_batch in minibatch_iter_w:
        optimizer_w.zero_grad()
        output_w = model_w(x_batch)
        loss_w = -mll_w(output_w, y_batch)
        loss_w.backward()
        optimizer_w.step()
        # Store the loss value for plot
    loss_values.append(loss_w.item())


# plot loss function
ax_loss.clear()
ax_loss.plot(loss_values, label='Training Loss')
ax_loss.set_xlabel('Iteration')
ax_loss.set_ylabel('Loss')
ax_loss.set_title('Loss Function During Training W')
ax_loss.legend()
ax_loss.grid(True)


plot_GP(ax_w,x,y_w,subset_indexes,model_w,likelihood_w,resolution,df)


# evalaute max uncertainty rateo
prior_stdd_w = np.sqrt(model_w.covar_module.outputscale.item())
preds_w = model_w(x) # evaluate predictive posterior on full dataset
max_stdd_w = torch.max(preds_w.variance).item()**0.5 # evaluate
# evalaute max rateo
stdd_rateo_threshold_w = max_stdd_w/prior_stdd_w

# start training loop
max_stdd_rateo_w = 1 + stdd_rateo_threshold_w # set to high value to start the loop

optimizer_retrain_w = torch.optim.Adam([
    {'params': [param for name, param in model_w.named_parameters()
                if 'raw_lengthscale' not in name and 'outputscale' not in name]}
], lr=0.03)


#acceptable_rateo_increase = 1 # TEMPOARY TESTING!!!!

while max_stdd_rateo_w > stdd_rateo_threshold_w * acceptable_rateo_increase:
    # now just optimize the inducing points locations
    # Only optimize the inducing points

    # Use the adam optimizer
    mll_w = gpytorch.mlls.VariationalELBO(likelihood_w, model_w, num_data=y_subset.size(0))

    # update the training data loader
    train_dataset_retrain_w = TensorDataset(x_subset, y_subset)
    train_loader_retrain = DataLoader(train_dataset_retrain_w, batch_size=2500, shuffle=True)
    minibatch_iter_retrain_w = tqdm.tqdm(train_loader_retrain, desc="Minibatch retrain", leave=False, disable=True)
    loss_values = []
    for i in epochs_iter:
        for x_batch_retrain_w, y_batch_retrain_w in minibatch_iter_retrain_w:
            optimizer_retrain_w.zero_grad()
            output_w = model_w(x_batch_retrain_w)
            loss_w = -mll(output_w, y_batch_retrain_w)
            loss_w.backward()
            optimizer_retrain_w.step()
            loss_values.append(loss_w.item())

    # plot loss function
    ax_loss.clear()
    ax_loss.plot(loss_values, label='Training Loss')
    ax_loss.set_xlabel('Iteration')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Loss Function During Training')
    ax_loss.legend()
    ax_loss.grid(True)


    # plotting results
    ax_w.clear()
    ax_w.plot(df['vicon time'].to_numpy(),df['acc_w'].to_numpy(),label='aw',color='orchid') # plot orginal data
    plot_GP(ax_w,x,y_w,subset_indexes,model_w,likelihood_w,resolution,df)
    # --- finished plotting ---

    # update points in dataset and inducing points

    # evaluate maximum uncertainty
    prior_stdd_w = np.sqrt(model_w.covar_module.outputscale.item())

    # move to cpu
    x = x.cpu()
    model_w = model_w.cpu()

    preds_w = model_w(x)

    #move back to gpu
    x = x.cuda()
    model_w = model_w.cuda()

    stdd_w = preds_w.variance.sqrt()  # Get standard deviations

    # Sort uncertainties in descending order and get the indices
    sorted_indices_w = torch.argsort(stdd_w,
    descending=True)

    # Find the first index with the highest uncertainty that is not in subset_indexes
    for idx in sorted_indices_w:
        if idx.item() not in subset_indexes:
            max_uncertainty_idx = idx.item()
            max_stdd_w = stdd_w[max_uncertainty_idx].item()
            max_stdd_rateo_w = max_stdd_w/prior_stdd_w
            subset_indexes.append(max_uncertainty_idx)
            break
    else:
        print("All high-uncertainty points are already in the subset.")

    print(f"max_stdd_rateo w = {max_stdd_rateo_w:.4f}, stdd_rateo_threshold w = {stdd_rateo_threshold_w * acceptable_rateo_increase:.4f}")

    #update the subset of data
    x_subset = x.clone()[subset_indexes] # select chosen points
    y_subset = y.clone()[subset_indexes]


    # !! also update the inducing points !!
    # identify the least informative inducing point
    # evaluate kernel matrix
    KZZ = model_w.covar_module(model_w.variational_strategy.inducing_points)
    
    # select row with the highest sum
    sum_KZZ = torch.sum(KZZ,1)
    
    # find the index of the least informative inducing point


    least_informative_idx = torch.argmax(sum_KZZ).item()

    # add line to plot the next data point to be added
    ax_w.axvline(df['vicon time'].to_numpy()[max_uncertainty_idx].item(), color='g',alpha=0.3)
    # add dashed line on the inducing point that will be eliminated
    #ax_x.axvline(model.variational_strategy.inducing_points[least_informative_idx].item(), color='k',linestyle='--',alpha=0.3)
    plt.pause(0.1)

    # replace the least informative inducing point with the new data point

    with torch.no_grad():
        
        #inducing_points[least_informative_idx] = x.clone()[max_uncertainty_idx]
        model_w.variational_strategy._variational_distribution.variational_mean.data[least_informative_idx] = y_w.clone()[max_uncertainty_idx]
        # update the first guess variational mean
        model_w.variational_strategy.inducing_points.data[least_informative_idx,:] = x[max_uncertainty_idx,:]




# save the subset of indexes as a numpy array
name = folder_path + '/subset_indexes.npy'
np.save(name,np.array(subset_indexes))


plt.ioff()
plt.show()
    


