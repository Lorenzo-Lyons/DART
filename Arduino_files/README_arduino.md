# Arduino file usage
Before using the arduino file contained in Arduino_BNO055_IMU_encoder_ROS the following steps must be completed:
- Open the arduino file and set the car number variable to the appropriate number in the line:
```
ros::Publisher imu_data_pub("arduino_data_1", &arduino_data_msg);
```
- Install arduino library "rosserial_arduino" version 0.7.9 from arduino library manager.
- Install ros client on DART from terminal 
```
sudo apt-get install ros-melodic-rosserial-python
```
- Upload the arduino file to the arduino board
- make sure the arduino device is assigned to the static name "arduino". [link](https://jh-byun.github.io/study/ubuntu-USB-static-name/)