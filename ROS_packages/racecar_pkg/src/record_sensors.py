#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32, Float32MultiArray, Header
import csv
import datetime
import rospkg
import numpy as np



class record_input_and_sensor_data:
    def __init__(self, car_number):

        self.folder_name = '/src/Data/NEW_FOLDER/'


        self.car_number = car_number
        self.file_name = 'car_' + str(car_number) + '_Data'

        #initiate this node
        rospy.init_node('data_recording' + str(car_number), anonymous=True)

        #initialize variables
        self.sensor_input_data =  [0.0, 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0 ,0.0,0.0,0.0]

        # create new file
        # to later find path to data folder
        self.rospack = rospkg.RosPack()
        date_time = datetime.datetime.now()
        date_time_str = date_time.strftime("%m_%d_%Y_%H_%M_%S")
        file_name = self.rospack.get_path('racecar_pkg') + self.folder_name + self.file_name +'recording_'+ date_time_str + '.csv'
        file = open(file_name, 'w+') 
        print(file_name)

        #write header line
        self.writer = csv.writer(file)
        #                  [elapsed_time, current, voltage, acc_x, acc_y, omega_rad, vel, safety_value, throttle, steering]
        self.writer.writerow(['elapsed time sensors', 'current', 'voltage' ,'acc x (IMU)','acc y (IMU)','W (IMU)', 'vel encoder','safety_value', 'throttle','steering'])

        #subscribe to inputs and sensor in formation topics
        #on-board sensors
        rospy.Subscriber('sensors_and_input_' + str(car_number), Float32MultiArray, self.callback_sensors_and_input)



        rospy.spin()

    #Current callback function
    def callback_sensors_and_input(self, sensors_and_input_data):
        # sensors_and_input = [elapsed_time, current, voltage, acc_x,acc_y, omega_rad, vel]
        self.sensors_and_input_data =  np.array(sensors_and_input_data.data)
        data_line = self.sensors_and_input_data
        #print('data line =', data_line)
        self.writer.writerow(data_line)





if __name__ == '__main__':
    try:
        car_number = 1
        recording = record_input_and_sensor_data(car_number)

    except rospy.ROSInterruptException:
        print('Something went wrong with setting up topics')

