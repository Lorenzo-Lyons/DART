from functions_for_data_processing import get_data,process_raw_data_steering, steering_curve_model,plot_raw_data
from matplotlib import pyplot as plt
import torch
import numpy as np
# set font size for figures
import matplotlib
font = {'family' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)



# this assumes that the current directory is DART
folder_path = 'System_identification_data_processing/Data/3_step_steering_data'  # small sinusoidal input

# get the raw data
df_raw_data = get_data(folder_path)

# plot raw data
ax0,ax1,ax2 = plot_raw_data(df_raw_data)

# --- process the raw data ---

# identify steering delay
# we assume that signal 2 is a time delayed version of signal 1
# signal1 = df_raw_data['steering'].to_numpy()
# signal2 = df_raw_data['W (IMU)'].to_numpy()
# delay_indexes = evaluate_delay(signal1, signal2)

# # convert delay in seconds
# dt = np.mean(np.diff(df_raw_data['elapsed time sensors'].to_numpy()))
# delay_st = delay_indexes * dt
# print('Steering delay = ', delay_st)
delay_st = 0
# process the rest of the data
df = process_raw_data_steering(df_raw_data)



#plot the processed data
plotting_time_vec = df['elapsed time sensors'].to_numpy()
fig1, ((ax0)) = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
ax0.set_title('Steering curve fitting data')
ax0.plot(plotting_time_vec, df['steering angle'].to_numpy(), label="steering angle [rad]", color='orchid')
ax0.plot(plotting_time_vec, df['steering'].to_numpy(), label="steering raw ", color='pink')
#ax0.plot(plotting_time_vec, df['steering delayed'].to_numpy(), label="steering delayed ", color='k')
ax0.set_xlabel('time [s]')
ax0.legend()




# --------------- fitting steering curve--------------- 
print('')
print('Fitting steering curve model')

initial_guess = torch.ones(5)*0.5
#initial_guess[0] = torch.Tensor([0.95])

#instantiate class object
steering_curve_model_obj = steering_curve_model(initial_guess)

# define number of training iterations
Steer_train_its = 300

#define loss and optimizer objects
steer_loss_fn = torch.nn.MSELoss(reduction = 'mean') 
steer_optimizer_object = torch.optim.Adam(steering_curve_model_obj.parameters(), lr=0.1)
        
# generate data in tensor form for torch
train_x_steering = torch.tensor(df['steering'].to_numpy())
train_y_steering = torch.tensor(df['steering angle'].to_numpy())

# save loss values for later plot
loss_vec = np.zeros(Steer_train_its)

# train the model
for i in range(Steer_train_its):
    # clear gradient information from previous step before re-evaluating it for the current iteration
    steer_optimizer_object.zero_grad()  
    
    # compute fitting outcome with current model parameters
    steering_angle_pred = steering_curve_model_obj(train_x_steering)

    # evaluate loss function
    loss = steer_loss_fn(steering_angle_pred,  train_y_steering)
    loss_vec[i] = loss.item()

    # evaluate the gradient of the loss function with respect to the fitting parameters
    loss.backward() 

    # use the evaluated gradient to perform a gradient descent step 
    steer_optimizer_object.step() # this updates parameters automatically according to the optimizer you chose


# plot loss function
plt.figure()
plt.title('Loss')
plt.plot(loss_vec)
plt.xlabel('iterations')
plt.ylabel('loss')

# --- print out parameters ---
[a,b,c,d,e] = steering_curve_model_obj.transform_parameters_norm_2_real()
a, b, c,d,e = a.item(), b.item(), c.item(), d.item(),e.item()
print('a = ', a)
print('b = ', b)
print('c = ', c)
print('d = ', d)
print('e = ', e)

# plot curve over the fitting data
input_vec = np.linspace(-1,1,100)
y_fitted = steering_curve_model_obj(torch.tensor(input_vec)).detach().numpy()

fig = plt.figure(figsize=(10, 6))
fig.subplots_adjust(top=0.985,
bottom=0.11,
left=0.13,
right=0.99,
hspace=0.2,
wspace=0.2)

plt.scatter(df['steering'].to_numpy(), df['steering angle'].to_numpy(), label = 'data') 
plt.plot(input_vec, y_fitted ,'orangered',label = "steering curve",linewidth=4)
plt.xlabel("Steering input")
plt.ylabel("Steering angle [rad]")
#plt.title('Steering angle vs steering command scatter plot')
plt.legend()
plt.show()


