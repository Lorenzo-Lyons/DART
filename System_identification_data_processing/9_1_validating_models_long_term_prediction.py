from functions_for_data_processing import get_data, plot_raw_data, process_raw_vicon_data,plot_vicon_data\
,dyn_model_culomb_tires,produce_long_term_predictions,throttle_dynamics_data_processing,\
    process_vicon_data_kinematics,steering_dynamics_data_processing,\
    load_SVGPModel_actuator_dynamics,dyn_model_SVGP_4_long_term_predictions,\
    load_SVGPModel_actuator_dynamics_analytic,dyn_model_SVGP_4_long_term_predictions_analytical
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
from tqdm import tqdm


#matplotlib.rc('font', **font)

# This script is used to fit the tire model to the data collected using the vicon external tracking system with a 
# SIMPLE CULOMB friction tyre model.

# chose what stated to forward propagate (the others will be taken from the data, this can highlight individual parts of the model)
forward_propagate_indexes = [1,2,3] # [1,2,3,4,5] # # 1 = vx, 2=vy, 3=w, 4=throttle, 5=steering

# select data folder to test long term predictions on NOTE: this assumes that the current directory is DART
#folder_path = 'System_identification_data_processing/Data/90_model_validation_long_term_predictions'  # the battery was very low for this one
#folder_path = 'System_identification_data_processing/Data/91_model_validation_long_term_predictions_fast'
#folder_path = 'System_identification_data_processing/Data/91_free_driving_16_sept_2024'
folder_path = 'System_identification_data_processing/Data/91_free_driving_16_sept_2024_slow'
#folder_path = 'System_identification_data_processing/Data/free_driving_steer_rate_testing_16_sept_2024'

#folder_path = 'System_identification_data_processing/Data/81_throttle_ramps_only_steer03'
#folder_path = 'System_identification_data_processing/Data/circles_27_sept_2024'





# --- folder path from where to load gp parameters ---
folder_path_GP = 'System_identification_data_processing/Data/82_huge_datest_for_gp_fitting/SVGP_saved_parameters'





model_tag = 1 # 0 for physics-based model, 1 for SVGP model, 2 for SVGP rewritten in analytic form



if model_tag == 0: # pysic-based model
    steering_friction_flag = True
    pitch_dynamics_flag = False
    # the model gives you the derivatives of it's own states, so you can integrate them to get the states in the new time instant
    dynamic_model = dyn_model_culomb_tires(steering_friction_flag,pitch_dynamics_flag)

elif model_tag == 1: # SVGP model NOTE that it can take both raw and filtered inputs according to the actuator_time_delay_fitting_tag
    model_vx,model_vy,model_w = load_SVGPModel_actuator_dynamics(folder_path_GP)
    dynamic_model = dyn_model_SVGP_4_long_term_predictions(model_vx,model_vy,model_w)

elif model_tag == 2: # SVGP model analytical form NOTE that it can take both raw and filtered inputs according to the actuator_time_delay_fitting_tag
    evalaute_cov_tag = False
    model_vx,model_vy,model_w = load_SVGPModel_actuator_dynamics_analytic(folder_path_GP,evalaute_cov_tag)
    dynamic_model = dyn_model_SVGP_4_long_term_predictions_analytical(model_vx,model_vy,model_w)
    

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




# # plot raw data
# ax0,ax1,ax2 = plot_raw_data(df)

# plot vicon related data (longitudinal and lateral velocities, yaw rate related)
ax_wheel_f_alpha,ax_wheel_r_alpha,ax_total_force_front,\
ax_total_force_rear,ax_lat_force,ax_long_force,\
ax_acc_x_body,ax_acc_y_body,ax_acc_w = plot_vicon_data(df) 





# producing long term predictions
n_states = 5
n_inputs = 2


columns_to_extract = ['vicon time', 'vx body', 'vy body', 'w', 'throttle filtered' ,'steering filtered', 'throttle' ,'steering','vicon x','vicon y','vicon yaw']
input_data_long_term_predictions = df[columns_to_extract].to_numpy()
prediction_window = 1.5 # [s]
jumps = 25 #25
long_term_predictions = produce_long_term_predictions(input_data_long_term_predictions, dynamic_model,prediction_window,jumps,forward_propagate_indexes)









# plot long term predictions over real data
fig, ((ax10,ax11,ax12)) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
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

x_index = n_states + n_inputs + 1
y_index = n_states + n_inputs + 2
yaw_index = n_states + n_inputs + 3

ax13.plot(time_vec_data,input_data_long_term_predictions[:,x_index],color='dodgerblue',label='x',linewidth=4,linestyle='-')
ax13.set_xlabel('time [s]')
ax13.set_ylabel('y [m]')
ax13.legend()
ax13.set_title('trajectory in the x-y plane')

ax14.plot(time_vec_data,input_data_long_term_predictions[:,y_index],color='orangered',label='y',linewidth=4,linestyle='-')
ax14.set_xlabel('time [s]')
ax14.set_ylabel('y [m]')
ax14.legend()
ax14.set_title('trajectory in the x-y plane')

ax15.plot(time_vec_data,input_data_long_term_predictions[:,yaw_index],color='orchid',label='yaw',linewidth=4,linestyle='-')
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

ax16.plot(input_data_long_term_predictions[:,x_index],input_data_long_term_predictions[:,y_index],color='orange',label='trajectory',linewidth=4,linestyle='-')
ax16.set_xlabel('x [m]')
ax16.set_ylabel('y [m]')
ax16.legend()
ax16.set_title('vehicle trajectory in the x-y plane')




fig1, ((ax_acc_x,ax_acc_y,ax_acc_w)) = plt.subplots(3, 1, figsize=(10, 6), constrained_layout=True)

# plot longitudinal accleeartion
ax_acc_x.plot(df['vicon time'].to_numpy(),df['ax body'].to_numpy(),label='ax body',color='dodgerblue')

ax_acc_x.legend()

# plot lateral accleeartion
ax_acc_y.plot(df['vicon time'].to_numpy(),df['ay body'].to_numpy(),label='ay body',color='orangered')
ax_acc_y.legend()

# plot yaw rate
ax_acc_w.plot(df['vicon time'].to_numpy(),df['acc_w'].to_numpy(),label='acc_w',color='purple')
ax_acc_w.legend()








# plot long term predictions
input_data_acc_prediction = input_data_long_term_predictions[:,1:n_states+n_inputs+1] # [1,2,3,4,5]
acc_x_model_from_data = np.zeros(df.shape[0])
acc_y_model_from_data = np.zeros(df.shape[0])
acc_w_model_from_data = np.zeros(df.shape[0])

# print('predicting accelerations from data')
# for i in tqdm(range(0,df.shape[0]), desc="i"):
#     acc = dynamic_model.forward(input_data_acc_prediction[i,:])
#     acc_x_model_from_data[i] = acc[0]
#     acc_y_model_from_data[i] = acc[1]
#     acc_w_model_from_data[i] = acc[2]



acc_x_model_from_data = np.zeros(df.shape[0])
acc_y_model_from_data = np.zeros(df.shape[0])
acc_w_model_from_data = np.zeros(df.shape[0])

print('predicting accelerations from data')
if model_tag == 0:
    for i in range(0,df.shape[0]):
        acc = dynamic_model.forward(input_data_acc_prediction[i,:])
        acc_x_model_from_data[i] = acc[0]
        acc_y_model_from_data[i] = acc[1]
        acc_w_model_from_data[i] = acc[2]

elif model_tag == 1:
    import torch
    if model_vx.actuator_time_delay_fitting_tag == 0:
        #['vx body', 'vy body', 'w', 'throttle filtered' ,'steering filtered','throttle','steering']
        state_action_base_model = np.column_stack((input_data_acc_prediction[:, :3], input_data_acc_prediction[:, 5], input_data_acc_prediction[:, 6]))

    elif model_vx.actuator_time_delay_fitting_tag == 3:
        state_action_base_model = input_data_acc_prediction[:, :5]

    test_x = torch.Tensor(state_action_base_model)
    acc_x_model_from_data = model_vx(test_x).mean.detach().numpy()
    acc_y_model_from_data = model_vy(test_x).mean.detach().numpy()
    acc_w_model_from_data = model_w(test_x).mean.detach().numpy()

elif model_tag == 2:
    # add tqdm bar
    from tqdm import tqdm
    for i in tqdm(range(0,df.shape[0])):
        acc = dynamic_model.forward(input_data_acc_prediction[i,:])
        acc_x_model_from_data[i] = acc[0]
        acc_y_model_from_data[i] = acc[1]
        acc_w_model_from_data[i] = acc[2]









# plot longitudinal acclearation
ax_acc_x.plot(df['vicon time'].to_numpy(),acc_x_model_from_data,color='maroon',label='model output from data')
ax_acc_y.plot(df['vicon time'].to_numpy(),acc_y_model_from_data,color='maroon',label='model output from data')
ax_acc_w.plot(df['vicon time'].to_numpy(),acc_w_model_from_data,color='maroon',label='model output from data')



# plot the input dynamics
fig, ((ax_th,ax_st)) = plt.subplots(2, 1, figsize=(10, 6))
ax_th.plot(df['vicon time'].to_numpy(),df['throttle'].to_numpy(),label='throttle',color='gray',linestyle='--')
ax_th.plot(df['vicon time'].to_numpy(),df['throttle filtered'].to_numpy(),label='throttle filtered',color='dodgerblue')
ax_th.legend()

ax_st.plot(df['vicon time'].to_numpy(),df['steering'].to_numpy(),label='steering',color='gray',linestyle='--')
ax_st.plot(df['vicon time'].to_numpy(),df['steering filtered'].to_numpy(),label='steering filtered',color='orangered')
ax_st.legend()



# plot long term predictions

print('plotting long term predictions')

for pred in tqdm(long_term_predictions, desc="Rollouts"):

    #velocities
    ax10.plot(pred[:,0],pred[:,1],color='k',alpha=0.2)
    ax11.plot(pred[:,0],pred[:,2],color='k',alpha=0.2)
    ax12.plot(pred[:,0],pred[:,3],color='k',alpha=0.2)
    # positions
    ax13.plot(pred[:,0],pred[:,x_index],color='k',alpha=0.2)
    ax14.plot(pred[:,0],pred[:,y_index],color='k',alpha=0.2)
    ax15.plot(pred[:,0],pred[:,yaw_index],color='k',alpha=0.2)
    #trajectory
    ax16.plot(pred[:,x_index],pred[:,y_index],color='k',alpha=0.2)
    # input dynamics
    ax_th.plot(pred[:,0],pred[:,4],color='k',alpha=0.2)
    ax_st.plot(pred[:,0],pred[:,5],color='k',alpha=0.2)

    #accelerations
    state_action_matrix = pred[:,1:n_states+n_inputs+1] 
    # add Fx

    acc_x_model = np.zeros(pred.shape[0])
    acc_y_model = np.zeros(pred.shape[0])
    acc_w_model = np.zeros(pred.shape[0])

    for i in range(pred.shape[0]):
        acc = dynamic_model.forward(state_action_matrix[i,:])
        acc_x_model[i] = acc[0]
        acc_y_model[i] = acc[1]
        acc_w_model[i] = acc[2]


    # plot the model accelerations
    ax_acc_x.plot(pred[:,0],acc_x_model,color='k')
    ax_acc_y.plot(pred[:,0],acc_y_model,color='k')
    ax_acc_w.plot(pred[:,0],acc_w_model,color='k')
    ax_acc_x.legend()
    ax_acc_y.legend()
    ax_acc_w.legend()


ax16.set_aspect('equal')





plt.show()



