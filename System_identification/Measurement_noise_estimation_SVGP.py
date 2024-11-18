import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from dart_dynamic_models import get_data, throttle_dynamics_data_processing,\
steering_dynamics_data_processing, process_vicon_data_kinematics, process_raw_vicon_data,\
plot_GP, SVGPModel





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





# driving straight (for likelyhood noise estimation)
df = df[df['vicon time']>34] 
df = df[df['vicon time']<36.5] 




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





# produce fitting data
columns_to_extract = ['vx body', 'vy body', 'w', 'throttle' ,'steering'] #  angle time delayed
x = torch.tensor(df[columns_to_extract].to_numpy()).cuda()
y = torch.tensor(df['ax body'].to_numpy()).cuda()

# convert to float32
x = x.to(torch.float32)
y = y.to(torch.float32)

# initialize inducing points by selecting them randomly from the input data
n_inducing = 50
resolution = df['vicon time'].to_numpy()[1] - df['vicon time'].to_numpy()[0]


# define data subset as the inducing points
import random
subset_indexes_initnial_inducing = random.sample(range(x.size(0)), n_inducing) # to get a good initial fit you need random samples
inducing_points = x[subset_indexes_initnial_inducing,:] # first n datapoints
x_subset = x.clone()[subset_indexes_initnial_inducing]
y_subset = y.clone()[subset_indexes_initnial_inducing]

training_iterations = 100


# intialize SVGP model
model = SVGPModel(inducing_points)
model.train()
model = model.cuda()

# initialize likelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood()
likelihood.train()
likelihood = likelihood.cuda()

# set initial noise to a high value
likelihood.noise_covar.noise = 0.093056 ** 2 #1.8**2


optimizer = torch.optim.Adam([
{'params': model.parameters()},
{'params': likelihood.parameters()}
], lr=0.1)

# Use the adam optimizer
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y.size(0))

# define data loader to hadle large data in the GPU
from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(x, y)
train_loader = DataLoader(train_dataset, batch_size=2500, shuffle=True)

# add a progress bar
import tqdm
epochs_iter = tqdm.tqdm(range(training_iterations), desc="Epoch", disable=True)
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
print(f"RAW Likelyhood noise X = {likelihood.noise_covar.raw_noise.item():.6f}")
print(f"Likelyhood noise STDEV X = {np.sqrt(likelihood.noise_covar.noise.item()):.6f}")


# Plot the loss function
fig, (ax_loss_x, ax_loss_y, ax_loss_w) = plt.subplots(3, 1, figsize=(10, 18))


ax_loss_x.plot(loss_values, label='Training Loss')
ax_loss_x.set_xlabel('Iteration')
ax_loss_x.set_ylabel('Loss')
ax_loss_x.set_title('Loss Function During Training')
ax_loss_x.legend()
ax_loss_x.grid(True)


# plot the fitting results
plot_GP(ax_x,x,y,subset_indexes_initnial_inducing,model,likelihood,resolution,df)







# ok now we repeat the process with the vy and W data, addin gnew data points to the subset
# produce y fitting data
y_y = torch.tensor(df['ay body'].to_numpy()).cuda()

# convert to float32
y_y = y_y.to(torch.float32)

# intialize SVGP model
model_y = SVGPModel(inducing_points)
model_y.train()
model_y = model.cuda()

# initialize likelihood
likelihood_y = gpytorch.likelihoods.GaussianLikelihood()
likelihood_y.train()
likelihood_y = likelihood_y.cuda()

# set initial noise to a high value
likelihood_y.noise_covar.noise = 0.050152 ** 2  #1.8**2

optimizer_y = torch.optim.Adam([
{'params': model_y.parameters()},
{'params': likelihood_y.parameters()}
], lr=0.1)
# {'params': likelihood.parameters()} $ removed likelihood parameters from the optimizer cause we previously fitted the noise

# Use the adam optimizer
mll_y = gpytorch.mlls.VariationalELBO(likelihood_y, model_y, num_data=y_y.size(0))

# define data loader to hadle large data in the GPU
from torch.utils.data import TensorDataset, DataLoader
train_dataset_y = TensorDataset(x, y_y)
train_loader_y = DataLoader(train_dataset_y, batch_size=2500, shuffle=True)

# add a progress bar
minibatch_iter_y = tqdm.tqdm(train_loader_y, desc="Minibatch", leave=False, disable=True)
# Initialize a list to store loss values
loss_values = []
for i in epochs_iter:
    for x_batch, y_batch in minibatch_iter_y:
        optimizer_y.zero_grad()
        output_y = model(x_batch)
        loss_y = -mll_y(output_y, y_batch)
        loss_y.backward()
        optimizer_y.step()
        # Store the loss value for plot
    loss_values.append(loss_y.item())

print(f"RAW Likelyhood noise Y = {likelihood_y.noise_covar.raw_noise.item():.6f}")
print(f"Likelyhood noise STDEV Y = {np.sqrt(likelihood_y.noise_covar.noise.item()):.6f}")

# plot loss function
ax_loss_y.clear()
ax_loss_y.plot(loss_values, label='Training Loss')
ax_loss_y.set_xlabel('Iteration')
ax_loss_y.set_ylabel('Loss')
ax_loss_y.set_title('Loss Function During Training Y')
ax_loss_y.legend()
ax_loss_y.grid(True)


plot_GP(ax_y,x,y_y,subset_indexes_initnial_inducing,model_y,likelihood_y,resolution,df)



# repeat for w data
# produce y fitting data
y_w = torch.tensor(df['acc_w'].to_numpy()).cuda()

# convert to float32
y_w = y_w.to(torch.float32)

# intialize SVGP model
model_w = SVGPModel(inducing_points)
model_w.train()
model_w = model.cuda()

# initialize likelihood
likelihood_w = gpytorch.likelihoods.GaussianLikelihood()
likelihood_w.train()
likelihood_w = likelihood_w.cuda()


# set initial noise to a high value
likelihood_w.noise_covar.noise = 0.401365 ** 2 #5**2


optimizer_w = torch.optim.Adam([
{'params': model_w.parameters()},
{'params': likelihood_w.parameters()}
], lr=0.1)
# {'params': likelihood.parameters()} $ removed likelihood parameters from the optimizer cause we previously fitted the noise

# Use the adam optimizer
mll_w = gpytorch.mlls.VariationalELBO(likelihood_w, model_w, num_data=y_w.size(0))

# define data loader to hadle large data in the GPU
from torch.utils.data import TensorDataset, DataLoader
train_dataset_w = TensorDataset(x, y_w)
train_loader_w = DataLoader(train_dataset_w, batch_size=2500, shuffle=True)


minibatch_iter_w = tqdm.tqdm(train_loader_w, desc="Minibatch", leave=False, disable=True)
# Initialize a list to store loss values
loss_values = []
for i in epochs_iter:
    for x_batch, y_batch in minibatch_iter_w:
        optimizer_w.zero_grad()
        output_w = model(x_batch)
        loss_w = -mll_w(output_w, y_batch)
        loss_w.backward()
        optimizer_w.step()
        # Store the loss value for plot
    loss_values.append(loss_w.item())

print(f"RAW Likelyhood noise W = {likelihood_w.noise_covar.raw_noise.item():.6f}")
print(f"Likelyhood noise STDEV W = {np.sqrt(likelihood_w.noise_covar.noise.item()):.6f}")

# plot loss function
ax_loss_w.clear()
ax_loss_w.plot(loss_values, label='Training Loss')
ax_loss_w.set_xlabel('Iteration')
ax_loss_w.set_ylabel('Loss')
ax_loss_w.set_title('Loss Function During Training W')
ax_loss_w.legend()
ax_loss_w.grid(True)


plot_GP(ax_w,x,y_w,subset_indexes_initnial_inducing,model_w,likelihood_w,resolution,df)



# now filtering the signal using a gaussian filter


from scipy.ndimage import gaussian_filter

# repeat for x
sigma_x = likelihood.noise_covar.noise.item()**0.5
filtered_data_x = gaussian_filter(df['ax body'].to_numpy(), sigma=sigma_x)
ax_x.plot(df['vicon time'].to_numpy(),filtered_data_x,color='r',label='filtered data')
ax_x.legend()

# just using the measured variance of the signal
filtered_data_x2 = gaussian_filter(df['ax body'].to_numpy(), sigma=np.std(df['ax body'].to_numpy()))
ax_x.plot(df['vicon time'].to_numpy(),filtered_data_x,color='g',label='filtered data simple sigma')
ax_x.legend()
# print out value
print(f"STDEV X = {np.std(df['ax body'].to_numpy()):.6f}")


# repeat for y
sigma_y = likelihood_y.noise_covar.noise.item()**0.5
filtered_data_y = gaussian_filter(df['ay body'].to_numpy(), sigma=sigma_y)
ax_y.plot(df['vicon time'].to_numpy(),filtered_data_y,color='r',label='filtered data')
ax_y.legend()

# just using the measured variance of the signal
filtered_data_y2 = gaussian_filter(df['ay body'].to_numpy(), sigma=np.std(df['ay body'].to_numpy()))
ax_y.plot(df['vicon time'].to_numpy(),filtered_data_y2,color='g',label='filtered data simple sigma')
ax_y.legend()
# print out value
print(f"STDEV Y = {np.std(df['ay body'].to_numpy()):.6f}")





# repeat for w
sigma_w = likelihood_w.noise_covar.noise.item()**0.5
filtered_data_w = gaussian_filter(df['acc_w'].to_numpy(), sigma=sigma_w)
ax_w.plot(df['vicon time'].to_numpy(),filtered_data_w,color='r',label='filtered data')
ax_w.legend()

# just using the measured variance of the signal
filtered_data_w2 = gaussian_filter(df['acc_w'].to_numpy(), sigma=np.std(df['acc_w'].to_numpy()))
ax_w.plot(df['vicon time'].to_numpy(),filtered_data_w2,color='g',label='filtered data simple sigma')
ax_w.legend()
# print out value
print(f"STDEV W = {np.std(df['acc_w'].to_numpy()):.6f}")




# save the subset of indexes as a numpy array
name = folder_path + '/raw_noise_likelihood.npy'
np.save(name,np.array([likelihood.noise_covar.noise.item(),likelihood_y.noise_covar.noise.item(),likelihood_w.noise_covar.noise.item()]))


plt.show()
    


