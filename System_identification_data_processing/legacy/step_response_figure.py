from functions_for_data_processing import get_data, plot_raw_data, evaluate_delay, motor_curve_model, plot_motor_friction_curves
from matplotlib import pyplot as plt
import torch
import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
# set font size for figures
import matplotlib
font = {'family' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)



# this assumes that the current directory is Platooning_code
folder_path = 'platooning_ws/src/platooning_utilities/Data/Data_throttle_curve_car_1_24_JAN_new_encoder' 
#folder_path = 'platooning_ws/src/platooning_utilities/Data/Data_throttle_curve_car_1_22_JAN_delay_estimation'  




# get the raw data
df_raw_data = get_data(folder_path)

# plot raw data
ax0,ax1,ax2 = plot_raw_data(df_raw_data)
#plt.show()


# smooth velocity data
# Set the window size for the moving average
window_size = 5#5
poly_order = 2

# Apply Savitzky-Golay filter
smoothed_vel_encoder = savgol_filter(df_raw_data['vel encoder'].to_numpy(), window_size, poly_order)

# v_spline = UnivariateSpline(df_raw_data['elapsed time sensors'].to_numpy(), df_raw_data['vel encoder'].to_numpy(), s=0.01)
# smoothed_vel_encoder = v_spline(df_raw_data['elapsed time sensors'].to_numpy())

ax1.plot(df_raw_data['elapsed time sensors'].to_numpy(),smoothed_vel_encoder,label="vel encoder smoothed",color='k',linestyle='--')
plt.legend()




# identify  delay
delay_th = 0.1 # [s]

# process the raw data
m =1.67 #mass of the robot
# friction curve parameters
a_friction  =  1.6837230920791626
b_friction  =  13.49715518951416
c_friction  =  0.3352389633655548


# df = process_raw_data_acceleration(df_raw_data, delay_st)
df = df_raw_data[['elapsed time sensors','throttle']].copy() 
df['throttle delayed'] = np.interp(df_raw_data['elapsed time sensors'].to_numpy()-delay_th, df_raw_data['elapsed time sensors'].to_numpy(), df_raw_data['throttle'].to_numpy())
df['vel encoder smoothed'] =  smoothed_vel_encoder # df_raw_data['vel encoder'] #non smoothed

spl_vel = CubicSpline(df['elapsed time sensors'].to_numpy(), df['vel encoder smoothed'].to_numpy())
df['force'] =   m * spl_vel(df['elapsed time sensors'].to_numpy(),1) # take the first derivative of the spline
df['friction force'] = + a_friction * np.tanh(b_friction  * df['vel encoder smoothed'] ) + df['vel encoder smoothed'] * c_friction
df['motor force'] = df['force'] + df['friction force']

# select data between 58 and 70 seconds
t_init = 135
df = df[df['elapsed time sensors']>t_init]
df = df[df['elapsed time sensors']<153]
#v_spline_dev = v_spline.derivative()
#df['force'] =  v_spline_dev(df['elapsed time sensors'].to_numpy()) # take the first derivative of the spline








# plot velocity information against force
# This is usefull to guide the choice of parameter bounds
#NOTE: pay attention that the model is fitting on the accelerations, but the parameters are designed to give a force,
# so here to get a feel for a good initial guess it's better to show the force rather than the acceleration



fig, ((ax3,ax4)) = plt.subplots(2, 1, figsize=(10, 6))
fig.subplots_adjust(top=0.995,
bottom=0.11,
left=0.125,
right=0.995,
hspace=0.4,
wspace=0.2)

t_vec = df['elapsed time sensors'].to_numpy()-t_init
#ax3.set_title('velocity Vs motor force')
ax3.plot(t_vec,df['vel encoder smoothed'].to_numpy(),label="velocity",color='dodgerblue',linewidth=4)
mask = np.array(df['throttle']) >= 0.15
ax3.fill_between(t_vec, ax3.get_ylim()[0], ax3.get_ylim()[1], where=mask, color='skyblue', alpha=0.2, label='throttle>0')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Velocity [m/s]')
from matplotlib.ticker import FormatStrFormatter
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax3.legend()

ax4.plot(t_vec,df['force'].to_numpy(),label="resulting force",color='navy',linewidth=4)
ax4.set_ylim([-4,4])
ax4.fill_between(t_vec, ax4.get_ylim()[0], ax4.get_ylim()[1], where=mask, color='skyblue', alpha=0.2, label='throttle>0')
ax4.set_ylabel('Force [N]')
ax4.set_xlabel('Time [s]')
ax4.legend(bbox_to_anchor=(0.5, -0.075),loc='lower center')
plt.show()

