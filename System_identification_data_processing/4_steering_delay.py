from functions_for_data_processing import get_data,process_raw_data_steering,plot_raw_data,evaluate_delay
from matplotlib import pyplot as plt
import torch
import numpy as np
from scipy.interpolate import CubicSpline


# this assumes that the current directory is DART
folder_path = 'System_identification_data_processing/Data/4_sinusoidal_steering_data'   

# get the raw data
df_raw_data = get_data(folder_path)

# plot raw data
ax0,ax1,ax2 = plot_raw_data(df_raw_data)


# define steering command to steering angle static mapping
a =  1.6379064321517944
b =  0.3301370143890381
c =  0.019644200801849365
d =  0.37879398465156555
e =  1.6578725576400757

# process the raw data
df = process_raw_data_steering(df_raw_data)

# evalaute steering angle reference as described through the static steering mapping
#self.b * torch.tanh(self.a * (steering_command + self.c)) 

w = 0.5 * (np.tanh(30*(df['steering']+c))+1)
steering_angle1 = b * np.tanh(a * (df['steering'] + c)) 
steering_angle2 = d * np.tanh(e * (df['steering'] + c)) 

df['steering angle reference'] = (w)*steering_angle1+(1-w)*steering_angle2

# evaluate steering derivative
spl_steering_angle = CubicSpline(df['elapsed time sensors'].to_numpy(), df['steering angle'].to_numpy())
df['steering angle dev'] = spl_steering_angle(df['elapsed time sensors'].to_numpy(),1)


# identify steering delay
# we assume that signal 2 is a time delayed version of signal 1
signal1 = df['steering angle reference'].to_numpy()
signal2 = df['steering angle'].to_numpy()
delay_indexes = evaluate_delay(signal1, signal2)

# convert delay in seconds
dt = np.mean(np.diff(df['elapsed time sensors'].to_numpy()))
delay_st = delay_indexes * dt
print('Steering delay = ', delay_st)


df['steering angle reference delayed'] = np.interp(df['elapsed time sensors'].to_numpy()-delay_st, df['elapsed time sensors'].to_numpy(), df['steering angle reference'].to_numpy())



#plot the processed data
plotting_time_vec = df['elapsed time sensors'].to_numpy()
fig1, ((ax0)) = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)

# plot steering angle vs steering input 
# NOTE: we invert the sign of the steering input to make it more clear
ax0.set_title('Steering angle reference Vs steering angle')
ax0.plot(plotting_time_vec, df['steering angle'].to_numpy(), label="steering angle", color='orchid')
ax0.plot(plotting_time_vec, df['steering angle reference'].to_numpy(), label="steering angle reference", color='k',linestyle='--')
ax0.plot(plotting_time_vec, df['steering angle reference delayed'].to_numpy(), label="steering angle reference delayed", color='k')
ax0.legend()


plt.show()
