# DART: Delft's Autonomous-driving Robotic Testbed

<p align="center">
  <img src="Images_for_readme/4_jetracers_in_a_line_cropped.png" width="700" title="DART ">
</p>


## What's DART?
DART is a small-scale car-like robot that is intended for autonomous driving research. It's based on the commercially available [JetracerPro AI kit](https://www.waveshare.com/wiki/JetRacer_Pro_AI_Kit)  available from Waveshare and features  additional sensors and a few other upgrades. Instructions on how to build your own will be available soon. 



## What's in this repo?
This repository contains the code to set up and start driving with DART! In particular you can find:
- **ROS packages** that can be deployed on the platform to quicky get you started on your experiments. They feature basic functionalities, low level, high level controllers and a lidar-based SLAM pipeline.
- **Simulator** to test and debug before heading out to the lab. It replicates the sensor readings you would get from the vehicle, so no time wasted between simulating and testing.
- **System Identification** code needed to build a kinematic and a dynamic bicycle model, as well as the relative data set. 


## Installation
Clone this repo.
```
git clone https://github.com/Lorenzo-Lyons/DART.git
```
The Data_processing folder contains the code and data required for system identification and can be run as simple python scripts with your favourite code editor like [Visual Studio Code](https://code.visualstudio.com/). To use the simulator and other ROS packages you will need a working ROS intallation, we used [ROS noetic](http://wiki.ros.org/noetic/Installation/Ubuntu) but other ROS versions should work too. You will then need to place the packages in a [catkin workspace](http://wiki.ros.org/catkin/Tutorials/create_a_workspace). 

## System identification


## Simulator
To run the simulator you can use the provided launch file:

```
roslaunch dart_simulator_pkg dart_simulator.launch
```


## Available low level controllers

The package *lane_following_controller_pkg* cointains low level controllers that use the previously described kinematic bicycle model to control the robot. 

**Velocity tracking controller.** This controller tracks a reference velocity. To do so simpy publish a reference velocity value to the topic *v_ref_<car_number>*.

The controller works by means of a feedforward-feedback controller defined as:

```math
\begin{align*}
\tau = - K(v-v_{ref}) + \tau^{ff}
\end{align*}
```
Where $K$ is a gain and $\tau^{ff}$ is defined as:
```math
\begin{align*}
\tau^{ff} =\tau\text{  s.t.  } f(\tau,v_{ref})=0
\end{align*}
```

The current vehicle velocity $v$ is provided by an encoder. To start the sensor publishing:

```
rosrun cytron_jetracer publish_sensors_and_inputs_universal.py
```
To start the longitudinal velocity tracking controller run:
```
rosrun lane_following_controller_pkg longitudinal_controller.py
```
**Steering controller.** To send a desired steering angle to the robot simply use the function $\delta(\sigma)$ as described in the previous section. The publish to the topic *steering_<car_number>*

## Available high level controllers
**Gamepad control with velocity reference.**
<p align="center">
  <img src="images_for_readme/gamepad_connected.jpeg" width="350" title="properly connected gamepad">
</p>
This controller allows the user to set a reference velocity and simultaneously steer the vehicle. The controller uses the gamepad provided by waveshare in the jetracer pro AI kit. First of all ensure that the gamepad is properly connected by plugging in the usb dongle and pressing the "home" button on the gamepad. A single red light should be visible on the gamepad. Then launch the controller by typing:

```
roslaunch cytron_jetracer v_ref_gamepad_controller.launch
```
To set the velocity reference press the "Y" and "A" keys.

**Lane following controller.**


<p align="center">
  <img src="https://github.com/Lorenzo-Lyons/Jetracer_WS_github/assets/94372990/97e69d59-ac2f-4e4e-8bcd-0306837a3d66" title="Lane following controller">
</p>

The lane following controller allows the vehicle to autonomously track a user-defined reference track (pink line in the video) at a certain reference velocity. It requires a previously built map of the environment.

To build a map of the environment first launch the file:

```
roslaunch localization_and_mapping_jetracer_pkg slam_jetracer_universal.launch
```
Note that the vehicle needs to navigate the environment in order to map it. A convenient way of doing so is to run the velocity tracking controller described in the previous section.



To save the map type:
```
rosrun map_server map_saver -f map_name
```

To use the map in the lane following controller first make sure it is in the folder *saved_maps_from_slam* located in the localization_and_mapping_pkg package. Then edit the *map_file* parameter in *the map_server.launch* file to match the newly created map. Then launch the map server file.

```
roslaunch localization_and_mapping_jetracer_pkg map_server.launch
```
Now launch the lane following controller.

```
roslaunch lane_following_controller_pkg lane_following_controller.launch
```
To modify the reference path edit the function *produce_track* located in the file *functions_for_controllers.py*.


>>>>>>> 4aba142b89ce729ad169342a0384498463d0ee64
