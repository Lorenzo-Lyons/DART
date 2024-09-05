import os
import glob
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline, CubicSpline
from matplotlib import pyplot as plt
import torch
from scipy.signal import savgol_filter

def get_data(folder_path):
    import csv
    import os

    # This function gets (or produces) the merged data files from the specified folder
    print('Getting data')
    print('Looking for file " merged_files.csv "  in folder "', folder_path,'"')

    file_name = 'merged_files.csv'
    # Check if the CSV file exists in the folder
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path) and os.path.isfile(file_path):
        print('The CSV file exists in the specified folder.')

    else:
        print('The CSV file does not already exist in the specified folder. Proceding with file generation.')
        merge_data_files_from_a_folder(folder_path)

    #recording_name_train = file_name
    df = pd.read_csv(file_path)
    print('Data succesfully loaded.')
    return df



def merge_data_files_from_a_folder(folder_path):
    #this method creates a single file from all .csv files in the specified folder

    # Output file name and path
    file_name = 'merged_files.csv'
    output_file_path = folder_path + '/' + file_name

    # Get all CSV file paths in the folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    csv_files.sort(key=lambda x: os.path.basename(x))

    dataframes = []
    timing_offset = 0
    # Read each CSV file and store it in the dataframes list
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        #sometimes the files have some initial lines where all values are zero, so just remove them
        df = df[df['elapsed time sensors'] != 0.0]



        # set throttle to 0 when safety is off
        df['throttle'][df['safety_value'] == 0.0] = 0.0

        # reset time in each file to start from zero
        df['elapsed time sensors'] -= df['elapsed time sensors'].iloc[0]
        df['elapsed time sensors'] += timing_offset

        if 'vicon time' in df.keys():
            df['vicon time'] -= df['vicon time'].iloc[0]
            df['vicon time'] += timing_offset
            dt = np.average(df['vicon time'].diff().to_numpy()[1:]) # evaluate dt
            timing_offset = df['vicon time'].iloc[-1] + dt 
        else:
            #update timing offset
            #extract safety off data and fix issues with timing
            dt = np.average(df['elapsed time sensors'].diff().to_numpy()[1:]) # evaluate dt
            timing_offset = df['elapsed time sensors'].iloc[-1] + dt # each file will have a dt timegap between it and the next file
        


        
        dataframes.append(df)

    # Concatenate all DataFrames into a single DataFrame vertically
    merged_df = pd.concat(dataframes, axis=0, ignore_index=True)

    #write merged csv file
    merged_df.to_csv(output_file_path, index=False)

    print('Merging complete. Merged file saved as:', output_file_path)
    return output_file_path #, num_lines



def evaluate_delay(signal1,signal2):
    # outputs delay expressed in vector index jumps
    # we assume that the signals are arrays of the same length
    if len(signal1) == len(signal2):
    
        # Use numpy's correlate function to find cross-correlation
        cross_corr = np.correlate(signal1, signal2, mode='full')
        #the length of the cross_corr vector will be N + N - 1
        # in position N you find the cross correlation for 0 delay
        # signal 1 is kept still and signal 2 is moved across. 
        # So if signal 2 is a delayed version of signal 1, the maximum
        # value of the cross correlation will accur before position N. (N means no delay)

        # Find the index of the maximum correlation
        delay_indexes = (len(signal1)) - (np.argmax(cross_corr)+1)  # plus one is needed cause np.argmax gives you the index of where that is

        return delay_indexes
    else:
        print('signals not of the same length! Stopping delay evaluation')


def process_raw_data_steering(df):
    
    # evaluate measured steering angle by doing inverse of kinematic bicycle model (only when velocity is higher than 0.8 m/s)
    # Note that dataset should not contain high velocities since the kinematic bicycle model will fail, and measured steering angle would be wrong
    L = 0.175 # distance between front and rear axels
    elapsed_time_vec = df['elapsed time sensors'][df['vel encoder'] > 0.8].to_numpy()
    #steering_delayed = df['steering delayed'][df['vel encoder'] > 0.8].to_numpy()
    steering = df['steering'][df['vel encoder'] > 0.8].to_numpy()

    vel_encoder = df['vel encoder'][df['vel encoder'] > 0.8].to_numpy()
    w_vec = df['W (IMU)'][df['vel encoder'] > 0.8].to_numpy()
    steering_angle= np.arctan2(w_vec * L ,  vel_encoder) 

    d = {'elapsed time sensors': elapsed_time_vec,
        #'W (IMU)': w_vec,
        'steering angle': steering_angle,
        #'steering delayed' : steering_delayed,
        'steering' : steering}

    df_steering_angle = pd.DataFrame(data=d)

    return df_steering_angle




def plot_raw_data(df):
    plotting_time_vec = df['elapsed time sensors'].to_numpy()

    fig1, ((ax0, ax1, ax2)) = plt.subplots(3, 1, figsize=(10, 6), constrained_layout=True)
    ax0.set_title('dt check')
    ax0.plot(np.diff(df['elapsed time sensors']), label="dt", color='gray')
    ax0.set_ylabel('dt [s]')
    ax0.set_xlabel('data point')
    ax0.legend()

    # plot raw data velocity vs throttle
    ax1.set_title('Raw Velocity vs throttle')
    ax1.plot(plotting_time_vec, df['vel encoder'].to_numpy(), label="V encoder [m/s]", color='dodgerblue')
    ax1.plot(plotting_time_vec, df['throttle'].to_numpy(), label="throttle raw []", color='gray')
    # Create a background where the safety is disingaged
    mask = np.array(df['safety_value']) == 1
    ax1.fill_between(plotting_time_vec, ax1.get_ylim()[0], ax1.get_ylim()[1], where=mask, color='gray', alpha=0.1, label='safety value disingaged')
    ax1.set_xlabel('time [s]')
    ax1.legend()

    # plot raw data w vs steering
    ax2.set_title('Raw Omega')
    ax2.plot(plotting_time_vec, df['W (IMU)'].to_numpy(),label="omega IMU raw data [rad/s]", color='orchid')
    ax2.plot(plotting_time_vec, df['steering'].to_numpy(),label="steering raw []", color='pink') 
    ax2.fill_between(plotting_time_vec, ax2.get_ylim()[0], ax2.get_ylim()[1], where=mask, color='gray', alpha=0.1, label='safety value disingaged')
    ax2.set_xlabel('time [s]')
    ax2.legend()
    return ax0,ax1,ax2







def process_raw_vicon_data(df,delay_th,delay_st,delay_vicon_to_robot,lf,lr,theta_correction):
    #reset vicon time to start from 0
    #df['vicon time'] = df['vicon time'] - df['vicon time'][0]



    # adjust robot inputs to match up with the system's response
    #NOTE that this step is really really important
    time_vec = df['elapsed time sensors'].to_numpy()
    df['throttle'] = np.interp(time_vec-delay_th, time_vec, df['throttle'].to_numpy())
    df['steering'] = np.interp(time_vec-delay_st, time_vec, df['steering'].to_numpy())


    # add new columns with derived vx vy omega from optitrack data
    df['unwrapped yaw'] = unwrap_hm(df['vicon yaw'].to_numpy()) + theta_correction


    #adjust the vicon time to match up with the robot time
    #df['vicon time'] =  df['vicon time'].to_numpy() - df['vicon time'].to_numpy()[0]
    time_vec_vicon = df['vicon time'].to_numpy() 


    df['vicon x'] = np.interp(time_vec_vicon-delay_vicon_to_robot, time_vec_vicon, df['vicon x'].to_numpy())
    df['vicon y'] = np.interp(time_vec_vicon-delay_vicon_to_robot, time_vec_vicon, df['vicon y'].to_numpy())
    df['unwrapped yaw'] = np.interp(time_vec_vicon-delay_vicon_to_robot, time_vec_vicon, df['unwrapped yaw'].to_numpy())


    # --- evaluate first time derivative ---

    # simple finite differences method for comparison
    df['vx_abs_raw'] = [*np.divide(np.diff(df['vicon x'].to_numpy()),  np.diff(df['vicon time'].to_numpy())) ,0] # add a zero for dimensionality issues
    df['vy_abs_raw'] = [*np.divide(np.diff(df['vicon y'].to_numpy()),  np.diff(df['vicon time'].to_numpy())) ,0]
    df['w_abs_raw'] =  [*np.divide(np.diff(df['unwrapped yaw'].to_numpy()),np.diff(df['vicon time'].to_numpy())) ,0]
    # Calculate the derivatives using values 5 indices apart


    steps_shift = 10
    shifted_time0 = df['vicon time'].shift(+steps_shift)
    shifted_x0 = df['vicon x'].shift(+steps_shift)
    shifted_y0 = df['vicon y'].shift(+steps_shift)
    shifted_yaw0 = df['unwrapped yaw'].shift(+steps_shift)

    shifted_time2 = df['vicon time'].shift(-steps_shift)
    shifted_x2 = df['vicon x'].shift(-steps_shift)
    shifted_y2 = df['vicon y'].shift(-steps_shift)
    shifted_yaw2 = df['unwrapped yaw'].shift(-steps_shift)


    # Finite differences
    df['vx_abs_filtered'] = (shifted_x2 - shifted_x0) / (shifted_time2 - shifted_time0)
    df['vy_abs_filtered'] = (shifted_y2 - shifted_y0) / (shifted_time2 - shifted_time0)
    df['w_abs_filtered']  = (shifted_yaw2 - shifted_yaw0) / (shifted_time2 - shifted_time0)

    # Handle the last 5 elements (they will be NaN due to the shift)
    df['vx_abs_filtered'].iloc[-steps_shift:] = 0
    df['vy_abs_filtered'].iloc[-steps_shift:] = 0
    df['w_abs_filtered'].iloc[-steps_shift:] = 0

    df['vx_abs_filtered'].iloc[:steps_shift] = 0
    df['vy_abs_filtered'].iloc[:steps_shift] = 0
    df['w_abs_filtered'].iloc[:steps_shift] = 0


    # filtered first time derivative
    # spl_x = CubicSpline(time_vec_vicon, df['vicon x'].to_numpy())
    # spl_y = CubicSpline(time_vec_vicon, df['vicon y'].to_numpy())
    # spl_theta = CubicSpline(time_vec_vicon, df['unwrapped yaw'].to_numpy())

    window_size = 10
    poly_order = 1
    # Apply Savitzky-Golay filter
    #df['vx_abs_raw'] = savgol_filter(spl_x(time_vec_vicon,1), window_size, poly_order)
    #df['vy_abs_raw'] = savgol_filter(spl_y(time_vec_vicon,1), window_size, poly_order)
    #df['w_abs_raw'] = savgol_filter(spl_theta(time_vec_vicon,1), window_size, poly_order)



    # --- evalaute second time derivative ---

    # simple finite differences method for comparison
    df['ax_abs_raw'] = [*np.divide(np.diff(df['vx_abs_raw'].to_numpy()),  np.diff(df['vicon time'].to_numpy())) ,0] # add a zero for dimensionality issues
    df['ay_abs_raw'] = [*np.divide(np.diff(df['vy_abs_raw'].to_numpy()),  np.diff(df['vicon time'].to_numpy())) ,0]
    df['aw_abs_raw'] =  [*np.divide(np.diff(df['w_abs_raw'].to_numpy()),  np.diff(df['vicon time'].to_numpy())) ,0]
    
    # Shifted values for steps_shift indices ahead
    shifted_vx0 = df['vx_abs_filtered'].shift(+steps_shift)
    shifted_vy0 = df['vy_abs_filtered'].shift(+steps_shift)
    shifted_w0 = df['w_abs_filtered'].shift(+steps_shift)

    shifted_vx2 = df['vx_abs_filtered'].shift(-steps_shift)
    shifted_vy2 = df['vy_abs_filtered'].shift(-steps_shift)
    shifted_w2 = df['w_abs_filtered'].shift(-steps_shift)

    # Calculate the finite differences for acceleration
    df['ax_abs_filtered_more'] = (shifted_vx2 - shifted_vx0) / (shifted_time2 - shifted_time0)
    df['ay_abs_filtered_more'] = (shifted_vy2 - shifted_vy0) / (shifted_time2 - shifted_time0)
    df['aw_abs_filtered_more'] = (shifted_w2 - shifted_w0) / (shifted_time2 - shifted_time0)

    # Handle the last 5 elements (they will be NaN due to the shift)
    df['ax_abs_filtered_more'].iloc[-steps_shift:] = 0
    df['ay_abs_filtered_more'].iloc[-steps_shift:] = 0
    df['aw_abs_filtered_more'].iloc[-steps_shift:] = 0

    df['ax_abs_filtered_more'].iloc[:steps_shift] = 0
    df['ay_abs_filtered_more'].iloc[:steps_shift] = 0
    df['aw_abs_filtered_more'].iloc[:steps_shift] = 0

    # filtering from second time derivative of position
    # window_size = 10
    # poly_order = 1
    # Apply Savitzky-Golay filter
    # df['ax_abs_raw'] = savgol_filter(spl_x(time_vec_vicon,2), window_size, poly_order)
    # df['ay_abs_raw'] = savgol_filter(spl_y(time_vec_vicon,2), window_size, poly_order)
    # df['aw_abs_raw'] = savgol_filter(spl_theta(time_vec_vicon,2), window_size, poly_order)


    # handle the first 5 and last 5 elements


    # filtering from filtered velocity
    spl_vx = CubicSpline(time_vec_vicon, df['vx_abs_filtered'].to_numpy())
    spl_vy = CubicSpline(time_vec_vicon, df['vy_abs_filtered'].to_numpy())
    spl_w = CubicSpline(time_vec_vicon, df['w_abs_filtered'].to_numpy())

    df['ax_abs_filtered'] =savgol_filter(spl_vx(time_vec_vicon,1), window_size, poly_order)
    df['ay_abs_filtered'] =savgol_filter(spl_vy(time_vec_vicon,1), window_size, poly_order)
    df['aw_abs_filtered'] =savgol_filter(spl_w(time_vec_vicon,1), window_size, poly_order)


    # --- convert velocity and acceleration into body frame ---
    vx_body_vec = np.zeros(df.shape[0])
    vy_body_vec = np.zeros(df.shape[0])
    ax_body_vec = np.zeros(df.shape[0])
    ay_body_vec = np.zeros(df.shape[0])
    ax_body_vec_nocent = np.zeros(df.shape[0])
    ay_body_vec_nocent = np.zeros(df.shape[0])

    for i in range(df.shape[0]):
        rot_angle =  - df['unwrapped yaw'].iloc[i] # from global to body you need to rotate by -theta!
        #thi = spl_theta(time_vec_vicon[i]) #evaluate theta from spline
        w_i = df['w_abs_filtered'].iloc[i]

        R     = np.array([[ np.cos(rot_angle), -np.sin(rot_angle)],
                          [ np.sin(rot_angle),  np.cos(rot_angle)]])
        
        R_dev = np.array([[-np.sin(rot_angle), -np.cos(rot_angle)],
                          [ np.cos(rot_angle), -np.sin(rot_angle)]])

        vxvy = np.expand_dims(np.array(df[['vx_abs_filtered','vy_abs_filtered']].iloc[i]),1)
        axay = np.expand_dims(np.array(df[['ax_abs_filtered_more','ay_abs_filtered_more']].iloc[i]),1)

        vxvy_body = R @ vxvy
        axay_body = R @ axay + R_dev * w_i @ vxvy
        axay_nocent = R @ axay

        vx_body_vec[i],vy_body_vec[i] = vxvy_body[0], vxvy_body[1]
        ax_body_vec[i],ay_body_vec[i] = axay_body[0], axay_body[1]
        ax_body_vec_nocent[i],ay_body_vec_nocent[i] = axay_nocent[0], axay_nocent[1]

    df['vx body'] = vx_body_vec
    df['vy body'] = vy_body_vec
    df['ax body'] = ax_body_vec
    df['ay body'] = ay_body_vec

    df['ax body no centrifugal'] = ax_body_vec_nocent
    df['ay body no centrifugal'] = ay_body_vec_nocent


    #add slip angles as they will be evaluated in the tire model
    # --- tire forces ---
    #rear slip angle

    #evaluate steering angle (needed for front slip angle)
    a =  1.6379064321517944
    b =  0.3301370143890381 + 0.04
    c =  0.019644200801849365 #- 0.04 # this value can be tweaked to get the tyre model curves to allign better
    d =  0.37879398465156555 + 0.04
    e =  1.6578725576400757
    w = 0.5 * (np.tanh(30*(df['steering']+c))+1)
    steering_angle1 = b * np.tanh(a * (df['steering'] + c)) 
    steering_angle2 = d * np.tanh(e * (df['steering'] + c)) 
    steering_angle = (w)*steering_angle1+(1-w)*steering_angle2 

  


    Vy_wheel_r = df['vy body'].to_numpy() - lr * df['w_abs_filtered'].to_numpy() # lateral velocity of the rear wheel
    Vx_wheel_r = df['vx body'].to_numpy() 
    Vx_correction_term_r = 0.1*np.exp(-100*Vx_wheel_r**2) # this avoids the vx term being 0 and having undefined angles for small velocities
    # note that it will be negligible outside the vx = [-0.2,0.2] m/s interval.
    Vx_wheel_r = Vx_wheel_r + Vx_correction_term_r
    
    a_slip_r = - np.arctan(Vy_wheel_r/ Vx_wheel_r) / np.pi * 180  #converting alpha into degrees
                

    # front slip angle
    Vy_wheel_f = (df['vy body'].to_numpy() + df['w_abs_filtered'].to_numpy() * lf) #* np.cos(steering_angle) - df['vx body'].to_numpy() * np.sin(steering_angle)
    Vx_wheel_f = df['vx body'].to_numpy() #(df['vy body'].to_numpy() + df['w_abs_filtered'].to_numpy() * lf) * np.sin(steering_angle) + df['vx body'].to_numpy() * np.cos(steering_angle)
    Vx_correction_term_f = 0.1*np.exp(-100*Vx_wheel_f**2)
    Vx_wheel_f = Vx_wheel_f + Vx_correction_term_f
    #a_slip_f_star = np.arctan2(Vy_wheel_f, np.exp(-30*np.abs(df['vx body'])) + np.abs(df['vx body']))
    #the steering angle changes sign if you go in reverse!
    a_slip_f = -( -steering_angle + np.arctan2(Vy_wheel_f, Vx_wheel_f)) / np.pi * 180  #converting alpha into degrees


    # add new columns
    df['steering angle'] = steering_angle
    df['slip angle front'] = a_slip_f
    df['slip angle rear'] = a_slip_r

    return df


def unwrap_hm(x):  # this function is used to unwrap the angles
    if isinstance(x, (int, float)):
        return np.unwrap([x])[0]
    elif isinstance(x, np.ndarray):
        return np.unwrap(x)
    else:
        raise ValueError("Invalid input type. Expected 'float', 'int', or 'numpy.ndarray'.")

def plot_vicon_data(df):

    # plot vicon data filtering process
    plotting_time_vec = df['vicon time'].to_numpy()

    fig1, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    ax1.set_title('velocity x')
    ax1.plot(plotting_time_vec, df['vx_abs_raw'].to_numpy(), label="vicon abs vx raw", color='k')
    ax1.plot(plotting_time_vec, df['vx_abs_filtered'].to_numpy(), label="vicon abs vx filtered", color='dodgerblue')
    ax1.legend()

    ax4.set_title('acceleration x')
    #ax4.plot(plotting_time_vec, df['ax_abs_raw'].to_numpy(), label="vicon abs ax raw", color='k')
    ax4.plot(plotting_time_vec, df['ax_abs_filtered'].to_numpy(), label="vicon abs ax filtered", color='k')
    ax4.plot(plotting_time_vec, df['ax_abs_filtered_more'].to_numpy(), label="vicon abs ax filtered more", color='dodgerblue')
    ax4.legend()


    ax2.set_title('velocity y')
    ax2.plot(plotting_time_vec, df['vy_abs_raw'].to_numpy(), label="vicon abs vy raw", color='k')
    ax2.plot(plotting_time_vec, df['vy_abs_filtered'].to_numpy(), label="vicon abs vy filtered", color='orangered')
    ax2.legend()

    ax5.set_title('acceleration y')
    #ax5.plot(plotting_time_vec, df['ay_abs_raw'].to_numpy(), label="vicon abs ay raw", color='k')
    ax5.plot(plotting_time_vec, df['ay_abs_filtered'].to_numpy(), label="vicon abs ay filtered", color='k')
    ax5.plot(plotting_time_vec, df['ay_abs_filtered_more'].to_numpy(), label="vicon abs ay filtered more", color='orangered')
    ax5.legend()


    ax3.set_title('velocity yaw')
    ax3.plot(plotting_time_vec, df['w_abs_raw'].to_numpy(), label="vicon w raw", color='k')
    ax3.plot(plotting_time_vec, df['w_abs_filtered'].to_numpy(), label="vicon w filtered", color='slateblue')
    ax3.legend()

    ax6.set_title('acceleration yaw')
    #ax6.plot(plotting_time_vec, df['aw_abs_raw'].to_numpy(), label="vicon aw raw", color='k')
    ax6.plot(plotting_time_vec, df['aw_abs_filtered'].to_numpy(), label="vicon aw filtered", color='k')
    ax6.plot(plotting_time_vec, df['aw_abs_filtered_more'].to_numpy(), label="vicon aw filtered more", color='slateblue')
    ax6.legend()





    # plot raw opti data
    fig1, ((ax1, ax2, ax3 , ax4)) = plt.subplots(4, 1, figsize=(10, 6), constrained_layout=True)
    ax1.set_title('Velocity data')
    #ax1.plot(plotting_time_vec, df['vx_abs'].to_numpy(), label="Vx abs data", color='lightblue')
    #ax1.plot(plotting_time_vec, df['vy_abs'].to_numpy(), label="Vy abs data", color='rosybrown')
    ax1.plot(plotting_time_vec, df['vx body'].to_numpy(), label="Vx body", color='dodgerblue')
    ax1.plot(plotting_time_vec, df['vy body'].to_numpy(), label="Vy body", color='orangered')
    ax1.legend()

    # plot body frame data time history
    ax2.set_title('Vy data raw vicon')
    ax2.plot(plotting_time_vec, df['throttle'].to_numpy(), label="Throttle",color='gray', alpha=1)
    ax2.plot(plotting_time_vec, df['vel encoder'].to_numpy(),label="Velocity Encoder raw", color='indigo')
    ax2.plot(plotting_time_vec, df['vx body'].to_numpy(), label="Vx body frame",color='dodgerblue')
    #ax2.plot(plotting_time_vec, df['vy body'].to_numpy(), label="Vy body frame",color='orangered')
    
    ax2.legend()
    # plot omega data time history
    ax3.set_title('Omega data time history')
    ax3.plot(plotting_time_vec, df['steering'].to_numpy(),label="steering input raw data", color='pink') #  -17 / 180 * np.pi * 
    ax3.plot(plotting_time_vec, df['W (IMU)'].to_numpy(),label="omega IMU raw data", color='orchid')
    #ax3.plot(plotting_time_vec, df['w_abs'].to_numpy(), label="omega opti", color='lightblue')
    ax3.plot(plotting_time_vec, df['w_abs_filtered'].to_numpy(), label="omega opti filtered",color='slateblue')
    ax3.legend()

    ax4.set_title('x - y - theta time history')
    ax4.plot(plotting_time_vec, df['vicon x'].to_numpy(), label="x opti",color='slateblue')
    ax4.plot(plotting_time_vec, df['vicon y'].to_numpy(), label="y opti",color='orangered')
    ax4.plot(plotting_time_vec, df['unwrapped yaw'].to_numpy(), label="unwrapped theta",color='yellowgreen')
    ax4.plot(plotting_time_vec, df['vicon yaw'].to_numpy(), label="theta raw data", color='darkgreen')
    ax4.legend()



    # plot slip angles
    fig2, ((ax1, ax2, ax3)) = plt.subplots(3, 1, figsize=(10, 6), constrained_layout=True)
    ax1.set_title('slip angle front')
    ax1.plot(plotting_time_vec, df['slip angle front'].to_numpy(), label="slip angle front", color='peru')
    ax1.plot(plotting_time_vec, df['slip angle rear'].to_numpy(), label="slip angle rear", color='darkred')
    ax1.plot(plotting_time_vec, df['aw_abs_filtered_more'].to_numpy(), label="acc w", color='slateblue')
    ax1.plot(plotting_time_vec, df['vy body'].to_numpy(), label="Vy body", color='orangered')
    ax1.legend()

    ax2.set_title('slip angle rear')
    ax2.plot(plotting_time_vec, df['slip angle rear'].to_numpy(), label="slip angle rear", color='darkred')
    ax2.legend()


    ax3.set_title('throttle and steering commands')
    ax3.plot(plotting_time_vec, df['throttle'].to_numpy(), label="throttle", color='dodgerblue')
    ax3.plot(plotting_time_vec, df['steering'].to_numpy(), label="steering", color='slateblue')
    ax3.legend()

    # plot data points
    fig1, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)
    ax1.set_title('control input map')
    ax1.scatter(df['steering'].to_numpy(), df['throttle'].to_numpy(),color='skyblue')
    ax1.set_xlabel('steering')
    ax1.set_ylabel('throttle')
    ax1.set_xlim([-1,1])

    ax2.set_title('Vy-Vx map')
    ax2.scatter(df['vy body'].to_numpy(), df['vx body'].to_numpy(),color='k')
    ax2.set_xlabel('Vy')
    ax2.set_ylabel('Vx')

    fig1, ((ax1)) = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    ax1.plot(df['vicon time'].to_numpy(),df['vicon time'].diff().to_numpy())
    ax1.set_title('time steps')

    # plot acceleration data
    fig1, ((ax1,ax2,ax3)) = plt.subplots(1, 3, figsize=(10, 6), constrained_layout=True)
    ax1.plot(df['vicon time'].to_numpy(), df['ax body no centrifugal'].to_numpy(),label='acc x absolute measured in body frame',color = 'dodgerblue')
    ax1.set_xlabel('time [s]')
    ax1.set_title('X_ddot @ R(yaw)')
    ax1.legend()

    ax2.plot(df['vicon time'].to_numpy(), df['ay body no centrifugal'].to_numpy(),label='acc y absolute measured in body frame',color = 'orangered')
    ax2.set_xlabel('time [s]')
    ax2.set_title('Y_ddot @ R(yaw)')
    ax2.legend()

    ax3.plot(df['vicon time'].to_numpy(), df['aw_abs_filtered_more'].to_numpy(),label='dt',color = 'slateblue')
    ax3.set_xlabel('time [s]')
    ax3.set_title('Acc w')
    ax3.legend()

    # plot x-y trajectory
    plt.figure()
    plt.plot(df['vicon x'].to_numpy(),df['vicon y'].to_numpy())


class steering_curve_model(torch.nn.Sequential):
    def __init__(self,initial_guess):
        super(steering_curve_model, self).__init__()
        self.register_parameter(name='a', param=torch.nn.Parameter(torch.Tensor(initial_guess[0])))
        self.register_parameter(name='b', param=torch.nn.Parameter(torch.Tensor(initial_guess[1])))
        self.register_parameter(name='c', param=torch.nn.Parameter(torch.Tensor(initial_guess[2])))
        self.register_parameter(name='d', param=torch.nn.Parameter(torch.Tensor(initial_guess[3])))
        self.register_parameter(name='e', param=torch.nn.Parameter(torch.Tensor(initial_guess[4])))

    def transform_parameters_norm_2_real(self):
        # Normalizing the fitting parameters is necessary to handle parameters that have different orders of magnitude.
        # This method converts normalized values to real values. I.e. maps from [0,1] --> [min_val, max_val]
        # so every parameter is effectively constrained to be within a certain range.
        # where min_val max_val are set here in this method as the first arguments of minmax_scale_hm
        constraint_weights = torch.nn.Hardtanh(0, 1) # this constraint will make sure that the parmeter is between 0 and 1

        a = self.minmax_scale_hm(0.1,5,constraint_weights(self.a))
        b = self.minmax_scale_hm(0.2,0.6,constraint_weights(self.b))
        c = self.minmax_scale_hm(-0.1,0.1,constraint_weights(self.c))

        d = self.minmax_scale_hm(0.2,0.6,constraint_weights(self.d))
        e = self.minmax_scale_hm(0.1,5,constraint_weights(self.e))
        return [a,b,c,d,e]
        
    def minmax_scale_hm(self,min,max,normalized_value):
    # normalized value should be between 0 and 1
        return min + normalized_value * (max-min)

    def forward(self, steering_command):
        [a,b,c,d,e] = self.transform_parameters_norm_2_real()

        # this is the model that will be fitted
        # we assume that the steering curve is sigmoidal. This can be seen from the observed data.
        w = 0.5 * (torch.tanh(30*(steering_command+c))+1)
        steering_angle1 = b * torch.tanh(a * (steering_command + c)) 
        steering_angle2 = d * torch.tanh(e * (steering_command + c))
        steering_angle = (w)*steering_angle1+(1-w)*steering_angle2 

        return steering_angle
    



class steering_actuator_model(torch.nn.Sequential):
    def __init__(self):
        super(steering_actuator_model, self).__init__()
        self.register_parameter(name='k', param=torch.nn.Parameter(torch.Tensor([10.0])))

    
    def forward(self, train_x):  # this is the model that will be fitted
        # extract data
        steering_angle_reference = train_x[:,0]
        steering_angle = train_x[:,1]
        # evalaute output
        steer_angle_dot = self.k * (steering_angle_reference - steering_angle)

        return steer_angle_dot
    







class friction_curve_model(torch.nn.Sequential):
    def __init__(self,param_vals):
        super(friction_curve_model, self).__init__()
        # initialize parameters NOTE that the initial values should be [0,1], i.e. they should be the normalized value.
        self.register_parameter(name='a', param=torch.nn.Parameter(torch.Tensor([param_vals[0]]).cuda()))
        self.register_parameter(name='b', param=torch.nn.Parameter(torch.Tensor([param_vals[1]]).cuda()))
        self.register_parameter(name='c', param=torch.nn.Parameter(torch.Tensor([param_vals[2]]).cuda()))
        self.register_parameter(name='d', param=torch.nn.Parameter(torch.Tensor([param_vals[3]]).cuda()))


    def transform_parameters_norm_2_real(self):
        # Normalizing the fitting parameters is necessary to handle parameters that have different orders of magnitude.
        # This method converts normalized values to real values. I.e. maps from [0,1] --> [min_val, max_val]
        # so every parameter is effectively constrained to be within a certain range.
        # where min_val max_val are set here in this method as the first arguments of minmax_scale_hm

        constraint_weights = torch.nn.Hardtanh(0, 1) # this constraint will make sure that the parmeter is between 0 and 1

        #friction curve = -  a * tanh(b  * v) - v * c
        a = self.minmax_scale_hm(0.1,3.0,constraint_weights(self.a))
        b = self.minmax_scale_hm(1,100,constraint_weights(self.b))
        c = self.minmax_scale_hm(0.01,2,constraint_weights(self.c))
        d = self.minmax_scale_hm(-0.2,0.2,constraint_weights(self.d))
        return [a,b,c,d]
  
    def minmax_scale_hm(self,min,max,normalized_value):
    # normalized value should be between 0 and 1
        return min + normalized_value * (max-min)
    
    def forward(self, train_x):  # this is the model that will be fitted
        # --- friction evalaution
        [a,b,c,d] = self.transform_parameters_norm_2_real()
        Fx = - ( a * torch.tanh(b  * train_x) + c * train_x + d * train_x**2 )

        return Fx



class motor_curve_model(torch.nn.Sequential):
    def __init__(self,param_vals,n_previous_throttle):
        super(motor_curve_model, self).__init__()
        # define number of past throttle actions to keep use
        self.n_previous_throttle = n_previous_throttle


        # initialize parameters NOTE that the initial values should be [0,1], i.e. they should be the normalized value.
        self.register_parameter(name='a', param=torch.nn.Parameter(torch.Tensor([param_vals[0]]).cuda()))
        self.register_parameter(name='b', param=torch.nn.Parameter(torch.Tensor([param_vals[1]]).cuda()))
        self.register_parameter(name='c', param=torch.nn.Parameter(torch.Tensor([param_vals[2]]).cuda()))
        self.register_parameter(name='d', param=torch.nn.Parameter(torch.Tensor([param_vals[3]]).cuda()))
        self.register_parameter(name='e', param=torch.nn.Parameter(torch.Tensor([param_vals[4]]).cuda()))
        self.register_parameter(name='f', param=torch.nn.Parameter(torch.Tensor([param_vals[5]]).cuda()))


    def transform_parameters_norm_2_real(self):
        # Normalizing the fitting parameters is necessary to handle parameters that have different orders of magnitude.
        # This method converts normalized values to real values. I.e. maps from [0,1] --> [min_val, max_val]
        # so every parameter is effectively constrained to be within a certain range.
        # where min_val max_val are set here in this method as the first arguments of minmax_scale_hm
        constraint_weights = torch.nn.Hardtanh(0, 1) # this constraint will make sure that the parmeter is between 0 and 1

        # motor curve F= (a - v * b) * w * (throttle+c) : w = 0.5 * (torch.tanh(100*(throttle+c))+1)
        a = self.minmax_scale_hm(0,45,constraint_weights(self.a))
        b = self.minmax_scale_hm(0,15,constraint_weights(self.b))
        c = self.minmax_scale_hm(-0.3,0,constraint_weights(self.c))
        # d = self.minmax_scale_hm(-0.3,0.3,constraint_weights(self.d))
        # e = self.minmax_scale_hm(2,10,constraint_weights(self.e))
        # f = self.minmax_scale_hm(-2,4,constraint_weights(self.f))

        #b = self.minmax_scale_hm(0,1,constraint_weights(self.b))
        d = self.minmax_scale_hm(0,1,constraint_weights(self.d))
        e = self.minmax_scale_hm(0,1,constraint_weights(self.e))
        f = self.minmax_scale_hm(0,1,constraint_weights(self.f))


        return [a,b,c,d,e,f]
        
    def minmax_scale_hm(self,min,max,normalized_value):
    # normalized value should be between 0 and 1
        return min + normalized_value * (max-min)
    
    def forward(self, train_x):  # this is the model that will be fitted
        throttle = torch.unsqueeze(train_x[:,0],1)
        v = torch.unsqueeze(train_x[:,1],1)
        #throttle_prev = torch.unsqueeze(train_x[:,2],1)
        #throttle_prev_prev = torch.unsqueeze(train_x[:,3],1)
        throttle_prev = train_x[:,2:2+self.n_previous_throttle]

        # evaluate motor force as a function of the throttle
        [a,b,c,d,e,f] = self.transform_parameters_norm_2_real()
        # evaluate coefficients for the throttle filter
        k0 = d
        k1 = d * (1-d)
        k2 = d * (1-d)**2
        k3 = d * (1-d)**3 
        k4 = d * (1-d)**4 
        k5 = d * (1-d)**5 
        sum = (k0+k1+k2+k3+k4+k5)

        k_vec = torch.unsqueeze(torch.cat([k1,k2,k3,k4,k5],0)[:self.n_previous_throttle],1).double()

        throttle_filtered = (k0 * throttle + throttle_prev @ k_vec)/sum
        
        Fx = self.motor_equation(throttle_filtered,v)

        return Fx
    
    def motor_equation(self,throttle_filtered,v):
        [a,b,c,d,e,f] = self.transform_parameters_norm_2_real()
        w = 0.5 * (torch.tanh(100*(throttle_filtered+c))+1)
        Fx =  (a - b * v) * w * (throttle_filtered+c)
        return Fx




class motor_curve_model_inductance(torch.nn.Sequential):
    def __init__(self,param_vals):
        super(motor_curve_model_inductance, self).__init__()
        # define mass of the robot


        # initialize parameters NOTE that the initial values should be [0,1], i.e. they should be the normalized value.
        self.register_parameter(name='a', param=torch.nn.Parameter(torch.Tensor([param_vals[0]]).cuda()))
        self.register_parameter(name='b', param=torch.nn.Parameter(torch.Tensor([param_vals[1]]).cuda()))
        self.register_parameter(name='c', param=torch.nn.Parameter(torch.Tensor([param_vals[2]]).cuda()))


    def transform_parameters_norm_2_real(self):
        # Normalizing the fitting parameters is necessary to handle parameters that have different orders of magnitude.
        # This method converts normalized values to real values. I.e. maps from [0,1] --> [min_val, max_val]
        # so every parameter is effectively constrained to be within a certain range.
        # where min_val max_val are set here in this method as the first arguments of minmax_scale_hm
        constraint_weights = torch.nn.Hardtanh(0, 1) # this constraint will make sure that the parmeter is between 0 and 1

        a = self.minmax_scale_hm(10,50,constraint_weights(self.a))
        b = self.minmax_scale_hm(10,-10,constraint_weights(self.b))
        c = self.minmax_scale_hm(10,-10,constraint_weights(self.c))

        return [a,b,c]
        
    def minmax_scale_hm(self,min,max,normalized_value):
    # normalized value should be between 0 and 1
        return min + normalized_value * (max-min)
    
    def forward(self, train_x):  # this is the model that will be fitted
        throttle = torch.unsqueeze(train_x[:,0],1)
        v = torch.unsqueeze(train_x[:,1],1)
        F = torch.unsqueeze(train_x[:,2],1)

        # evaluate motor force derivative as a function of the throttle,velocity and current motor force
        [a,b,c] = self.transform_parameters_norm_2_real()
        Fx_dot = a * throttle + b * v + c * F

        return Fx_dot


class force_model(torch.nn.Sequential):
    def __init__(self,param_vals):
        super(force_model, self).__init__()
        # define mass of the robot


        # initialize parameters NOTE that the initial values should be [0,1], i.e. they should be the normalized value.
        self.register_parameter(name='a_f', param=torch.nn.Parameter(torch.Tensor([param_vals[0]]).cuda()))
        self.register_parameter(name='b_f', param=torch.nn.Parameter(torch.Tensor([param_vals[1]]).cuda()))
        self.register_parameter(name='c_f', param=torch.nn.Parameter(torch.Tensor([param_vals[2]]).cuda()))
        self.register_parameter(name='a_m', param=torch.nn.Parameter(torch.Tensor([param_vals[3]]).cuda()))
        self.register_parameter(name='b_m', param=torch.nn.Parameter(torch.Tensor([param_vals[4]]).cuda()))
        self.register_parameter(name='c_m', param=torch.nn.Parameter(torch.Tensor([param_vals[5]]).cuda()))


    def transform_parameters_norm_2_real(self):
        # Normalizing the fitting parameters is necessary to handle parameters that have different orders of magnitude.
        # This method converts normalized values to real values. I.e. maps from [0,1] --> [min_val, max_val]
        # so every parameter is effectively constrained to be within a certain range.
        # where min_val max_val are set here in this method as the first arguments of minmax_scale_hm

        constraint_weights = torch.nn.Hardtanh(0, 1) # this constraint will make sure that the parmeter is between 0 and 1

        #friction curve F= -  a * tanh(b  * v) - v * c
        a_f = self.minmax_scale_hm(1.5,2.0,constraint_weights(self.a_f))
        b_f = self.minmax_scale_hm(10,15,constraint_weights(self.b_f))
        c_f = self.minmax_scale_hm(0.2,0.4,constraint_weights(self.c_f))
        # motor curve F= (a - v * b) * w * (throttle+c) : w = 0.5 * (torch.tanh(100*(throttle+c))+1)
        a_m = self.minmax_scale_hm(5,40,constraint_weights(self.a_m))
        b_m = self.minmax_scale_hm(0.1,10,constraint_weights(self.b_m))
        c_m = self.minmax_scale_hm(-0.3,0.3,constraint_weights(self.c_m))
        return [a_f,b_f,c_f,a_m,b_m,c_m]
        
    def minmax_scale_hm(self,min,max,normalized_value):
    # normalized value should be between 0 and 1
        return min + normalized_value * (max-min)
    
    def friction_force(self,v):
        [a_f,b_f,c_f,a_m,b_m,c_m] = self.transform_parameters_norm_2_real()
        return - a_f * torch.tanh(b_f  * v) - v * c_f
    
    def motor_force(self,train_x):
        throttle = torch.unsqueeze(train_x[:,0],1)
        v = torch.unsqueeze(train_x[:,1],1)
        [a_f,b_f,c_f,a_m,b_m,c_m] = self.transform_parameters_norm_2_real()
        w = 0.5 * (torch.tanh(100*(throttle+c_m))+1)
        F_motor =  (a_m - v * b_m) * w * (throttle+c_m)
        return F_motor
    
    def forward(self, train_x):  # this is the model that will be fitted
        v = torch.unsqueeze(train_x[:,1],1)

        # evalaute friction force
        F_friction = self.friction_force(v)

        # evaluate motor force
        F_motor = self.motor_force(train_x)

        return F_friction+F_motor




class linear_tire_model(torch.nn.Sequential):
    def __init__(self,param_vals,a_minmax,b_minmax):
        super(linear_tire_model, self).__init__()
        # define mass of the robot

        # initialize parameters NOTE that the initial values should be [0,1], i.e. they should be the normalized value.
        self.register_parameter(name='a', param=torch.nn.Parameter(torch.Tensor([param_vals[0]]).cuda()))
        self.register_parameter(name='b', param=torch.nn.Parameter(torch.Tensor([param_vals[0]]).cuda()))
        self.a_minmax = a_minmax
        self.b_minmax = b_minmax

    def transform_parameters_norm_2_real(self):
        # Normalizing the fitting parameters is necessary to handle parameters that have different orders of magnitude.
        # This method converts normalized values to real values. I.e. maps from [0,1] --> [min_val, max_val]
        # so every parameter is effectively constrained to be within a certain range.
        # where min_val max_val are set here in this method as the first arguments of minmax_scale_hm

        constraint_weights = torch.nn.Hardtanh(0, 1) # this constraint will make sure that the parmeter is between 0 and 1

        #friction curve F= -  a * tanh(b  * v) - v * c
        a = self.minmax_scale_hm(self.a_minmax[0],self.a_minmax[1],constraint_weights(self.a))
        b = self.minmax_scale_hm(self.b_minmax[0],self.b_minmax[1],constraint_weights(self.b))
        return [a,b]
        
    def minmax_scale_hm(self,min,max,normalized_value):
    # normalized value should be between 0 and 1
        return min + normalized_value * (max-min)
    
    def forward(self, train_x):  # this is the model that will be fitted
        [a,b] = self.transform_parameters_norm_2_real()
        # evalaute lateral tire force
        F_y = a * (train_x + b) # + b
        return F_y
    

class culomb_pacejka_tire_model(torch.nn.Sequential):
    def __init__(self,param_vals):
        super(culomb_pacejka_tire_model, self).__init__()
        # define mass of the robot

        # initialize parameters NOTE that the initial values should be [0,1], i.e. they should be the normalized value.
        self.register_parameter(name='d', param=torch.nn.Parameter(torch.Tensor([param_vals[0]]).cuda()))
        self.register_parameter(name='c', param=torch.nn.Parameter(torch.Tensor([param_vals[0]]).cuda()))
        self.register_parameter(name='b', param=torch.nn.Parameter(torch.Tensor([param_vals[0]]).cuda()))
        #self.register_parameter(name='e', param=torch.nn.Parameter(torch.Tensor([param_vals[0]]).cuda()))


    def transform_parameters_norm_2_real(self):
        # Normalizing the fitting parameters is necessary to handle parameters that have different orders of magnitude.
        # This method converts normalized values to real values. I.e. maps from [0,1] --> [min_val, max_val]
        # so every parameter is effectively constrained to be within a certain range.
        # where min_val max_val are set here in this method as the first arguments of minmax_scale_hm

        constraint_weights = torch.nn.Hardtanh(0, 1) # this constraint will make sure that the parmeter is between 0 and 1

        #friction curve F= -  a * tanh(b  * v) - v * c
        #friction curve F= -  a * tanh(b  * v) - v * c
        d = self.minmax_scale_hm(-1,-10,constraint_weights(self.d))
        c = self.minmax_scale_hm(0.5,1.5,constraint_weights(self.c))
        b = self.minmax_scale_hm(0.01,10,constraint_weights(self.b))
        #e = self.minmax_scale_hm(-0.01,0,constraint_weights(self.e))
        return [d,c,b]
        
    def minmax_scale_hm(self,min,max,normalized_value):
    # normalized value should be between 0 and 1
        return min + normalized_value * (max-min)
    
    def forward(self, train_x):  # this is the model that will be fitted
        [d,c,b] = self.transform_parameters_norm_2_real()
        # evalaute lateral tire force
        F_y = d * torch.sin(c * torch.arctan(b * train_x )) # - e * (b * train_x -torch.arctan(b * train_x))
        return F_y





class pacejka_tire_model(torch.nn.Sequential):
    def __init__(self,param_vals):
        super(pacejka_tire_model, self).__init__()
        # define mass of the robot

        # initialize parameters NOTE that the initial values should be [0,1], i.e. they should be the normalized value.
        self.register_parameter(name='d', param=torch.nn.Parameter(torch.Tensor([param_vals[0]]).cuda()))
        self.register_parameter(name='c', param=torch.nn.Parameter(torch.Tensor([param_vals[0]]).cuda()))
        self.register_parameter(name='b', param=torch.nn.Parameter(torch.Tensor([param_vals[0]]).cuda()))
        self.register_parameter(name='e', param=torch.nn.Parameter(torch.Tensor([param_vals[0]]).cuda()))


    def transform_parameters_norm_2_real(self):
        # Normalizing the fitting parameters is necessary to handle parameters that have different orders of magnitude.
        # This method converts normalized values to real values. I.e. maps from [0,1] --> [min_val, max_val]
        # so every parameter is effectively constrained to be within a certain range.
        # where min_val max_val are set here in this method as the first arguments of minmax_scale_hm

        constraint_weights = torch.nn.Hardtanh(0, 1) # this constraint will make sure that the parmeter is between 0 and 1

        #friction curve F= -  a * tanh(b  * v) - v * c
        d = self.minmax_scale_hm(1,10,constraint_weights(self.d))
        c = self.minmax_scale_hm(0.5,1.5,constraint_weights(self.c))
        b = self.minmax_scale_hm(0.01,5,constraint_weights(self.b))
        e = self.minmax_scale_hm(-100,0,constraint_weights(self.e))
        return [d,c,b,e]
        
    def minmax_scale_hm(self,min,max,normalized_value):
    # normalized value should be between 0 and 1
        return min + normalized_value * (max-min)
    
    def forward(self, train_x):  # this is the model that will be fitted
        [d,c,b,e] = self.transform_parameters_norm_2_real()
        # evalaute lateral tire force
        #F_y = d * torch.sin(c * torch.arctan(b * train_x ))
        F_y = d * torch.sin(c * torch.arctan(b * train_x - e * (b * train_x -torch.arctan(b * train_x))))
        return F_y


















def plot_motor_friction_curves(df,acceleration_curve_model_obj,fitting_friction):

    #plot motor characteristic curve
    tau_vec = torch.unsqueeze(torch.linspace(-1,1,100),1).cuda()
    v_vec = torch.unsqueeze(torch.linspace(0,df['vel encoder smoothed'].max(),100),1).cuda()
    data_vec = torch.cat((tau_vec, v_vec), 1)

    
    #plot friction curve
    friction_vec = acceleration_curve_model_obj.friction_curve(v_vec).detach().cpu().numpy()

    fig1, ((ax1)) = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    ax1.plot(v_vec.cpu().numpy(),friction_vec,label = 'Friction curve',zorder=20,color='orangered',linewidth=5)
    ax1.set_xlabel('velocity [m\s]')
    ax1.set_ylabel('[N]')
    ax1.set_title('Friction curve')
    #ax1.grid()

    if fitting_friction:
        return ax1
    
    else:
        # also plot motor curve
        Fx_vec = acceleration_curve_model_obj.motor_curve(data_vec).detach().cpu().numpy()

        fig1, ((ax2)) = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
        ax2.plot(tau_vec.cpu().numpy(),Fx_vec,label = 'Th curve')
        ax2.set_title('Motor curve curve')
        ax2.set_xlabel('Throttle')
        ax2.set_ylabel('[N]')
        ax2.grid()

        return (ax1,ax2)
    




def produce_long_term_predictions(input_data, model,prediction_window,jumps,forward_propagate_indexes):
    # plotting long term predictions on data
    # each prediction window starts from a data point and then the quantities are propagated according to the provided model,
    # so they are not tied to the Vx Vy W data in any way. Though the throttle and steering inputs are taken from the data of course.

    # --- plot fitting results ---
    # input_data = ['vicon time', 'vx body', 'vy body', 'w_abs_filtered', 'throttle' ,'steering','vicon x','vicon y','vicon yaw']

    #prepare tuple containing the long term predictions
    long_term_preds = ()
    


    # iterate through each prediction window
    print('------------------------------')
    print('producing long term predictions')
    from tqdm import tqdm
    tqdm_obj = tqdm(range(0,input_data.shape[0],jumps), desc="long term preds", unit="pred")

    for i in tqdm_obj:
        

        #reset couner
        k = 0
        elpsed_time_long_term_pred = 0

        # set up initial positions
        long_term_pred = np.expand_dims(input_data[i, :],0)


        # iterate through time indexes of each prediction window
        while elpsed_time_long_term_pred < prediction_window and k + i + 1 < len(input_data):
            #store time values
            #long_term_pred[k+1,0] = input_data[k+i, 0] 
            dt = input_data[i + k + 1, 0] - input_data[i + k, 0]
            elpsed_time_long_term_pred = elpsed_time_long_term_pred + dt

            #produce propaagated state
            state_action_k = long_term_pred[k,[1,2,3,4,5]]
            
            # run it through the model
            accelrations = model.forward(state_action_k) # absolute accelerations in the current vehicle frame of reference
            # correct for centripetal acceleration
            #accelrations[1] = accelrations[1] - state_action_k[0] * state_action_k[2]
            new_state_new_frame = long_term_pred[k,[1,2,3]] + accelrations * dt

            # convert new state in new vehicle frame of reference
            # rot_angle = long_term_pred[k,3] * dt # adjust by the delta rotation
            # R = np.array([
            #     [np.cos(rot_angle), -np.sin(rot_angle), 0],
            #     [np.sin(rot_angle), np.cos(rot_angle), 0],
            #     [0, 0, 1]
            # ])
            # new_state_new_frame = R @ new_state



            # chose quantities to forward propagate
            if 1 in forward_propagate_indexes:
                new_vx = new_state_new_frame[0]
            else:
                new_vx = input_data[i+k+1, 1]

            if 2 in forward_propagate_indexes:
                new_vy = new_state_new_frame[1]
            else:
                new_vy = input_data[i+k+1, 2]

            if 3 in forward_propagate_indexes:
                new_w = new_state_new_frame[2]
            else:
                new_w = input_data[i+k+1, 3] 

            new_state_new_frame = np.array([new_vx,new_vy,new_w])

            # forward propagate x y yaw state
            rot_angle = long_term_pred[k,8]
            R = np.array([
                [np.cos(rot_angle), -np.sin(rot_angle), 0],
                [np.sin(rot_angle), np.cos(rot_angle), 0],
                [0, 0, 1]
            ])

            # absolute velocities
            abs_vxvyw = R @ np.array([long_term_pred[k,1],long_term_pred[k,2],long_term_pred[k,3]])


            # propagate x y yaw according to the previous state
            new_xyyaw = np.array([long_term_pred[k,6],long_term_pred[k,7],long_term_pred[k,8]]) + abs_vxvyw * dt

            # put everything together
            new_row = np.array([input_data[i + k + 1, 0],*new_state_new_frame,input_data[k+i,4],input_data[k+i,5],*new_xyyaw])
            long_term_pred = np.vstack([long_term_pred, new_row])

            # update k
            k = k + 1

        long_term_preds += (long_term_pred,)  

    return long_term_preds




class dyn_model_culomb_tires():
    def __init__(self,m,lr,lf,Jz,d_f,c_f,b_f,e_f,d_r,c_r,b_r,e_r):

        self.m = m
        self.lr = lr
        self.lf = lf
        self.Jz = Jz

        self.d_f = d_f
        self.c_f = c_f
        self.b_f = b_f
        self.e_f = e_f

        self.d_r = d_r
        self.c_r = c_r
        self.b_r = b_r
        self.e_r = e_r

    def steer_angle(self,steer):
        #evaluate steering angle (needed for front slip angle)
        a =  1.6379064321517944
        b =  0.3301370143890381+ 0.04
        c =  0.019644200801849365 #- 0.04 # this value can be tweaked to get the tyre model curves to allign better
        d =  0.37879398465156555+ 0.04
        e =  1.6578725576400757
        w = 0.5 * (np.tanh(30*(steer+c))+1)
        steering_angle1 = b * np.tanh(a * (steer + c)) 
        steering_angle2 = d * np.tanh(e * (steer + c)) 
        steering_angle = (w)*steering_angle1+(1-w)*steering_angle2 
        return steering_angle

    def motor_force(self,th,v):
        a =  28.887779235839844
        b =  5.986172199249268
        c =  -0.15045104920864105
        w = 0.5 * (np.tanh(100*(th+c))+1)
        Fm =  (a - v * b) * w * (th+c)
        return Fm

    def friction(self,v):
        a =  1.7194761037826538
        b =  13.312559127807617
        c =  0.289848655462265
        Ff = - a * np.tanh(b  * v) - v * c
        return Ff
    
    def lateral_tire_forces(self,vy_wheel_f,vy_wheel_r):
        return [self.a_f * (vy_wheel_f + self.b_f), self.a_r * (vy_wheel_r + self.b_r)]
    
    def forward(self, state_action):
        #returns vx_dot,vy_dot,w_dot in the vehicle body frame
        #state_action = [vx,vy,w,throttle,steer]
        vx = state_action[0]
        vy = state_action[1]
        w = state_action[2]
        throttle = state_action[3]
        steering = state_action[4]
        

        #evaluate forward force
        Fx = self.motor_force(throttle,vx) + self.friction(vx)

        # evalaute steering angle
        steer_angle = self.steer_angle(steering)

        # evaluate lateral tire forces
        Vy_wheel_f = np.cos(-steer_angle)*(vy + self.lf*w) + np.sin(-steer_angle) * vx
        Vy_wheel_r = vy - self.lr*w

        lateral_forces_wheel = self.lateral_tire_forces(Vy_wheel_f,Vy_wheel_r)
        b = np.array(  [Fx/2,
                        lateral_forces_wheel[0],
                        lateral_forces_wheel[1]])
        
        A = np.array([  [1+np.cos(steer_angle),-np.sin(steer_angle),0],
                        [+np.sin(steer_angle),+np.cos(steer_angle),1],
                        [+np.sin(steer_angle)*self.lf  ,np.cos(steer_angle)*self.lf  ,-self.lr]])
        
        body_forces = A @ b

        return np.array([body_forces[0]/self.m,body_forces[1]/self.m,body_forces[2]/self.Jz])
    



import torch
import gpytorch
import tqdm
import random
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy



# SVGP 
class SVGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(SVGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.size(dim=1)))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)




def train_SVGP_model(learning_rate,num_epochs, train_x, train_y_vx, train_y_vy, train_y_w, n_inducing_points):
    
    # start fitting
    # make contiguous (not sure why)
    train_x = train_x.contiguous()
    train_y_vx = train_y_vx.contiguous()
    train_y_vy = train_y_vy.contiguous()
    train_y_w = train_y_w.contiguous()

    # define batches for training (each bach will be used to perform a gradient descent step in each iteration. So toal parameters updates are Epochs*n_batches)
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset_vx = TensorDataset(train_x, train_y_vx)
    train_dataset_vy = TensorDataset(train_x, train_y_vy)
    train_dataset_w = TensorDataset(train_x, train_y_w)

    # define data loaders
    train_loader_vx = DataLoader(train_dataset_vx, batch_size=250, shuffle=True)
    train_loader_vy = DataLoader(train_dataset_vy, batch_size=250, shuffle=True)
    train_loader_w = DataLoader(train_dataset_w, batch_size=250, shuffle=True)

    #choosing initial guess inducing points as a random subset of the training data
    random.seed(10) # set the seed so to have same points for every run
    # random selection of inducing points
    random_indexes = random.choices(range(train_x.shape[0]), k=n_inducing_points)
    inducing_points = train_x[random_indexes, :]

    inducing_points = inducing_points.to(torch.float32)

    #initialize models
    model_vx = SVGPModel(inducing_points=inducing_points)
    model_vy = SVGPModel(inducing_points=inducing_points)
    model_w = SVGPModel(inducing_points=inducing_points)


    # assign first guess lengthscales
    #                                                            vx, vy ,w, throttle,steer
    model_vx.covar_module.base_kernel.lengthscale = torch.tensor([0.1,1,10,  0.1,    1])
    model_vy.covar_module.base_kernel.lengthscale = torch.tensor([1,1,10,10,1])
    model_w.covar_module.base_kernel.lengthscale =  torch.tensor([2,2,2,2,2])


    # Assign training data to models just to have it all together for later plotting
    model_vx.train_x = train_x 
    model_vx.train_y_vx = train_y_vx

    model_vy.train_x = train_x 
    model_vy.train_y_vy = train_y_vy

    model_w.train_x = train_x 
    model_w.train_y_w = train_y_w


    #define likelyhood objects
    likelihood_vx = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_vy = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_w = gpytorch.likelihoods.GaussianLikelihood()
 


    #move to GPU for faster fitting
    if torch.cuda.is_available():
        model_vx = model_vx.cuda()
        model_vy = model_vy.cuda()
        model_w = model_w.cuda()
        likelihood_vx = likelihood_vx.cuda()
        likelihood_vy = likelihood_vy.cuda()
        likelihood_w = likelihood_w.cuda()

    #set to training mode
    model_vx.train()
    model_vy.train()
    model_w.train()
    likelihood_vx.train()
    likelihood_vy.train()
    likelihood_w.train()

    #set up optimizer and its options
    optimizer_vx = torch.optim.AdamW([{'params': model_vx.parameters()}, {'params': likelihood_vx.parameters()},], lr=learning_rate)
    optimizer_vy = torch.optim.AdamW([{'params': model_vy.parameters()}, {'params': likelihood_vy.parameters()},], lr=learning_rate)
    optimizer_w = torch.optim.AdamW([{'params': model_w.parameters()}, {'params': likelihood_w.parameters()},], lr=learning_rate)


    # Set up loss object. We're using the VariationalELBO
    mll_vx = gpytorch.mlls.VariationalELBO(likelihood_vx, model_vx, num_data=train_y_vx.size(0))#, beta=1)
    mll_vy = gpytorch.mlls.VariationalELBO(likelihood_vy, model_vy, num_data=train_y_vy.size(0))#, beta=1)
    mll_w = gpytorch.mlls.VariationalELBO(likelihood_w, model_w, num_data=train_y_w.size(0))#, beta=1)

    # start training (tqdm is just to show the loading bar)
    epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")



    loss_2_print_vx_vec = []
    loss_2_print_vy_vec = []
    loss_2_print_w_vec = []

    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        minibatch_iter_vx = tqdm.tqdm(train_loader_vx, desc="Minibatch vx", leave=False, disable=True)
        minibatch_iter_vy = tqdm.tqdm(train_loader_vy, desc="Minibatch vy", leave=False, disable=True)
        minibatch_iter_w  = tqdm.tqdm(train_loader_w,  desc="Minibatch w",  leave=False, disable=True)

        for x_batch_vx, y_batch_vx in minibatch_iter_vx:
            optimizer_vx.zero_grad()
            output_vx = model_vx(x_batch_vx)
            loss_vx = -mll_vx(output_vx, y_batch_vx[:,0])
            minibatch_iter_vx.set_postfix(loss=loss_vx.item())
            loss_vx.backward()
            optimizer_vx.step()

        loss_2_print_vx_vec = [*loss_2_print_vx_vec, loss_vx.item()]

        for x_batch_vy, y_batch_vy in minibatch_iter_vy:
            optimizer_vy.zero_grad()
            output_vy = model_vy(x_batch_vy)
            loss_vy = -mll_vy(output_vy, y_batch_vy[:,0])
            minibatch_iter_vy.set_postfix(loss=loss_vy.item())
            loss_vy.backward()
            optimizer_vy.step()

        loss_2_print_vy_vec = [*loss_2_print_vy_vec, loss_vy.item()]

        for x_batch_w, y_batch_w in minibatch_iter_w:
            optimizer_w.zero_grad()
            output_w = model_w(x_batch_w)
            loss_w = -mll_w(output_w, y_batch_w[:,0])
            minibatch_iter_w.set_postfix(loss=loss_w.item())
            loss_w.backward()
            optimizer_w.step()

        loss_2_print_w_vec = [*loss_2_print_w_vec, loss_w.item()]
           
    #plot loss functions
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(loss_2_print_vx_vec,label='loss vx',color='dodgerblue') 
    ax.plot(loss_2_print_vy_vec,label='loss vy',color='orangered')
    ax.plot(loss_2_print_w_vec,label='loss w',color='orchid')
    ax.legend()


    #move to gpu for later evaluation
    model_vx = model_vx.cuda()
    model_vy = model_vy.cuda()
    model_w = model_w.cuda()

    return model_vx, model_vy, model_w, likelihood_vx, likelihood_vy, likelihood_w





#define orthogonally decoupled SVGP model
# Orthogonally decoupled SVGP
def make_orthogonal_vs(model,mean_inducing_points,covar_inducing_points):
    # mean_inducing_points = torch.randn(1000, train_x.size(-1), dtype=train_x.dtype, device=train_x.device)
    # covar_inducing_points = torch.randn(100, train_x.size(-1), dtype=train_x.dtype, device=train_x.device)

    covar_variational_strategy = gpytorch.variational.VariationalStrategy(
        model, covar_inducing_points,
        gpytorch.variational.CholeskyVariationalDistribution(covar_inducing_points.size(-2)),
        learn_inducing_locations=True
    )

    variational_strategy = gpytorch.variational.OrthogonallyDecoupledVariationalStrategy(
        covar_variational_strategy, mean_inducing_points,
        gpytorch.variational.DeltaVariationalDistribution(mean_inducing_points.size(-2)),
    )
    return variational_strategy

class OrthDecoupledApproximateGP(ApproximateGP):
    def __init__(self,mean_inducing_points,covar_inducing_points):
        #variational_distribution = gpytorch.variational.DeltaVariationalDistribution(inducing_points.size(-2))
        variational_strategy = make_orthogonal_vs(self,mean_inducing_points,covar_inducing_points)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module =  gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=mean_inducing_points.size(dim=1)))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_decoupled_SVGP_model(learning_rate,num_epochs, train_x, train_y_vx, train_y_vy, train_y_w, n_inducing_points_mean,n_inducing_points_cov):
    
    # start fitting
    # make contiguous (not sure why)
    train_x = train_x.contiguous()
    train_y_vx = train_y_vx.contiguous()
    train_y_vy = train_y_vy.contiguous()
    train_y_w = train_y_w.contiguous()

    # define batches for training (each bach will be used to perform a gradient descent step in each iteration. So toal parameters updates are Epochs*n_batches)
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset_vx = TensorDataset(train_x, train_y_vx)
    train_dataset_vy = TensorDataset(train_x, train_y_vy)
    train_dataset_w = TensorDataset(train_x, train_y_w)

    # define data loaders
    train_loader_vx = DataLoader(train_dataset_vx, batch_size=250, shuffle=True)
    train_loader_vy = DataLoader(train_dataset_vy, batch_size=250, shuffle=True)
    train_loader_w = DataLoader(train_dataset_w, batch_size=250, shuffle=True)

    #choosing initial guess inducing points as a random subset of the training data
    random.seed(10) # set the seed so to have same points for every run
    # random selection of inducing points
    random_indexes_mean = random.choices(range(train_x.shape[0]), k=n_inducing_points_mean)
    inducing_points_mean = train_x[random_indexes_mean, :]
    inducing_points_mean = inducing_points_mean.to(torch.float32)

    random_indexes_cov = random.choices(range(inducing_points_mean.shape[0]), k=n_inducing_points_cov)
    inducing_points_cov = train_x[random_indexes_cov, :]


    #initialize models
    model_vx = OrthDecoupledApproximateGP(inducing_points_mean,inducing_points_cov)
    model_vy = OrthDecoupledApproximateGP(inducing_points_mean,inducing_points_cov)
    model_w = OrthDecoupledApproximateGP(inducing_points_mean,inducing_points_cov)


    # assign first guess lengthscales
    #                                                            vx, vy ,w, throttle,steer
    # model_vx.covar_module.base_kernel.lengthscale = torch.tensor([0.1,1,10,  0.1,    1])
    # model_vy.covar_module.base_kernel.lengthscale = torch.tensor([1,1,10,10,1])
    # model_w.covar_module.base_kernel.lengthscale =  torch.tensor([2,2,2,2,2])


    # Assign training data to models just to have it all together for later plotting
    model_vx.train_x = train_x 
    model_vx.train_y_vx = train_y_vx

    model_vy.train_x = train_x 
    model_vy.train_y_vy = train_y_vy

    model_w.train_x = train_x 
    model_w.train_y_w = train_y_w


    #define likelyhood objects
    likelihood_vx = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_vy = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_w = gpytorch.likelihoods.GaussianLikelihood()
 


    #move to GPU for faster fitting
    if torch.cuda.is_available():
        model_vx = model_vx.cuda()
        model_vy = model_vy.cuda()
        model_w = model_w.cuda()
        likelihood_vx = likelihood_vx.cuda()
        likelihood_vy = likelihood_vy.cuda()
        likelihood_w = likelihood_w.cuda()

    #set to training mode
    model_vx.train()
    model_vy.train()
    model_w.train()
    likelihood_vx.train()
    likelihood_vy.train()
    likelihood_w.train()

    #set up optimizer and its options
    optimizer_vx = torch.optim.AdamW([{'params': model_vx.parameters()}, {'params': likelihood_vx.parameters()},], lr=learning_rate)
    optimizer_vy = torch.optim.AdamW([{'params': model_vy.parameters()}, {'params': likelihood_vy.parameters()},], lr=learning_rate)
    optimizer_w = torch.optim.AdamW([{'params': model_w.parameters()}, {'params': likelihood_w.parameters()},], lr=learning_rate)


    # Set up loss object. We're using the VariationalELBO
    mll_vx = gpytorch.mlls.VariationalELBO(likelihood_vx, model_vx, num_data=train_y_vx.size(0))#, beta=1)
    mll_vy = gpytorch.mlls.VariationalELBO(likelihood_vy, model_vy, num_data=train_y_vy.size(0))#, beta=1)
    mll_w = gpytorch.mlls.VariationalELBO(likelihood_w, model_w, num_data=train_y_w.size(0))#, beta=1)

    # start training (tqdm is just to show the loading bar)
    epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")



    loss_2_print_vx_vec = []
    loss_2_print_vy_vec = []
    loss_2_print_w_vec = []

    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        minibatch_iter_vx = tqdm.tqdm(train_loader_vx, desc="Minibatch vx", leave=False, disable=True)
        minibatch_iter_vy = tqdm.tqdm(train_loader_vy, desc="Minibatch vy", leave=False, disable=True)
        minibatch_iter_w  = tqdm.tqdm(train_loader_w,  desc="Minibatch w",  leave=False, disable=True)

        for x_batch_vx, y_batch_vx in minibatch_iter_vx:
            optimizer_vx.zero_grad()
            output_vx = model_vx(x_batch_vx)
            loss_vx = -mll_vx(output_vx, y_batch_vx[:,0])
            minibatch_iter_vx.set_postfix(loss=loss_vx.item())
            loss_vx.backward()
            optimizer_vx.step()

        loss_2_print_vx_vec = [*loss_2_print_vx_vec, loss_vx.item()]

        for x_batch_vy, y_batch_vy in minibatch_iter_vy:
            optimizer_vy.zero_grad()
            output_vy = model_vy(x_batch_vy)
            loss_vy = -mll_vy(output_vy, y_batch_vy[:,0])
            minibatch_iter_vy.set_postfix(loss=loss_vy.item())
            loss_vy.backward()
            optimizer_vy.step()

        loss_2_print_vy_vec = [*loss_2_print_vy_vec, loss_vy.item()]

        for x_batch_w, y_batch_w in minibatch_iter_w:
            optimizer_w.zero_grad()
            output_w = model_w(x_batch_w)
            loss_w = -mll_w(output_w, y_batch_w[:,0])
            minibatch_iter_w.set_postfix(loss=loss_w.item())
            loss_w.backward()
            optimizer_w.step()

        loss_2_print_w_vec = [*loss_2_print_w_vec, loss_w.item()]
           
    #plot loss functions
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(loss_2_print_vx_vec,label='loss vx',color='dodgerblue') 
    ax.plot(loss_2_print_vy_vec,label='loss vy',color='orangered')
    ax.plot(loss_2_print_w_vec,label='loss w',color='orchid')
    ax.legend()


    #move to gpu for later evaluation
    model_vx = model_vx.cuda()
    model_vy = model_vy.cuda()
    model_w = model_w.cuda()

    return model_vx, model_vy, model_w, likelihood_vx, likelihood_vy, likelihood_w







class dyn_model_SVGP():
    def __init__(self,model_vx,model_vy,model_w):

        self.model_vx = model_vx
        self.model_vy = model_vy
        self.model_w = model_w

    def forward(self, state_action):
        input = torch.unsqueeze(torch.Tensor(state_action),0).cuda()
        ax= self.model_vx(input).mean.detach().cpu().numpy()[0]
        ay= self.model_vy(input).mean.detach().cpu().numpy()[0]
        aw= self.model_w(input).mean.detach().cpu().numpy()[0]

        return np.array([ax,ay,aw])
    
def rebuild_Kxy_RBF_vehicle_dynamics(X,Y,outputscale,lengthscale):
    n = X.shape[0]
    m = Y.shape[0]
    KXY = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            KXY[i,j] = RBF_kernel_rewritten(X[i,:],Y[j,:],outputscale,lengthscale)
    return KXY

def RBF_kernel_rewritten(x,y,outputscale,lengthscale):
    exp_arg = np.zeros(len(lengthscale))
    for i in range(len(lengthscale)):
        exp_arg[i] = (x[i]-y[i])**2/lengthscale[i]**2
    return outputscale * np.exp(-0.5*np.sum(exp_arg))
