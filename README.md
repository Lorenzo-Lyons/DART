# DART: Delft's Autonomous-driving Robotic Testbed

<p align="center">
  <img src="Images_for_readme/4_jetracers_in_a_line_cropped.png" width="700" title="DART ">
</p>


## What's DART?
DART is a small-scale car-like robot that is intended for autonomous driving research. It's based on the commercially available [JetracerPro AI kit](https://www.waveshare.com/wiki/JetRacer_Pro_AI_Kit)  available from Waveshare and features  additional sensors and a few other upgrades. Instructions on how to build your own will be available soon. 



## What's in this repo?
This repository contains the code to set up and start driving with DART! In particular you can find:
- **Build instructions** to replicate the DART.
- **ROS packages** that can be deployed on the platform to quicky get you started on your experiments. They feature basic functionalities, low level, high level controllers and a lidar-based SLAM pipeline.
- **Simulator** to test and debug before heading out to the lab. It replicates the sensor readings you would get from the vehicle, so no time wasted between simulating and testing.
- **System Identification** code needed to build physics-based models: kinematic and dynamic bicycle model, and a data-driven Stochastic Variational Gaussian Process model. 

## Build instructions

To see the full build instructions go to [Build instruction section](build_instructions/).


## Installation
Clone this repo.
```
git clone https://github.com/Lorenzo-Lyons/DART.git
```
Install DART_dynamic_models python package. Navigate to the pacakge root folder *DART_dynamic_models* and install:

```
pip install dist/DART_dynamic_models-0.1.0-py3-none-any.whl
```

The Data_processing folder contains the code and data required for system identification steps and can be run as simple python scripts with your favourite code editor like [Visual Studio Code](https://code.visualstudio.com/). To use the simulator and other ROS packages you will need a working ROS intallation, we used [ROS noetic](http://wiki.ros.org/noetic/Installation/Ubuntu) but other ROS versions should work too. You will then need to place the packages in a [catkin workspace](http://wiki.ros.org/catkin/Tutorials/create_a_workspace). 

### System identification
To start using DART it's thus necessary to understand what happens when we provide the system with a certain input, i.e. we need to identify the system's model. The folder *Data_processing* cointains the code and the data to build the pysics-based kinematic and dynamic bicycle model, as well as the data-driven SVGP model.

The kinematic bicycle model is suitable for most kind of experiments that don't require to reach high speeds. Since it is simpler and computationally lighter we suggest trying it first and switching to the dynamic kinematic bicycle or data-driven SVGP model only if actually needed. Also note that the data necessary to fit the kinematic bicycle model can be collected with the on-board sensors, while the dynamic bicycle model and GP model require an external motion capture system.

To perform the system identification, run through the scripts following the numbering order, i.e. start from *1_0_fitting_friction_on_board_sensors*.
> [!NOTE]
> The outputs of the each script are simply printed out, so to update the model parameters, the user must copy-paste them in the function *dart_dynamic_models.model_functions*. When running the SVGP identification script the SVGP parameters will be automatically saved in the same folder the fitting data was loaded from.


**Fitting Results**
As a preview we show the physics-based identified sub-models.


|![](Images_for_readme/friction_curve.png)<br>Friction Curve|![](Images_for_readme/motor_curve.png)<br>Motor Curve|
|:-:|:-:
|![](Images_for_readme/steering_mapping.png)<br>**Steering input to steering angle map**|![](Images_for_readme/tire_models.png)<br>**Tire models**|





## Simulator
<p align="center">
  <img src="Images_for_readme/simulator_screenshot.png" width="700" title="simulator ">
</p>
The simulator matches the same ROS node and topic structure as in the real vehicle.

To run the simulator you can use the provided launch file:

```
roslaunch dart_simulator_pkg dart_simulator.launch
```
As a quick test, you can then control the simulated vehicle using the keyboard:
```
rosrun racecar_pkg teleop_keyboard.py
```


## Available low level controllers

The package *racecar_pkg* cointains scripts to run base functionalities, like send commands to the motors, that have been obtained by modifying the software in the github repo [cytron_jetracer](https://github.com/CytronTechnologies/cytron_jetracer). It also features some low level controllers. 

**Reference velocity and steering angle controller** This allows to  conveniently control the vehicle by sending longitudinal velocity reference and steering angle inputs. This controller relies on the kinematic bicycle, so it is suitable for low-medium speeds. To use this controller run the launch file:

```
roslaunch racecar_pkg racecar_steer_angle_v_ref.launch
```
Then send commands to the topics *v_ref_<car_number>* and *steering_angle_<car_number>*. This can be done for example with the gamepad provided in the JetracerPro AI kit. Run the following script:

```
roslaunch racecar_pkg gamepad_steer_angle_v_ref.py
```



## Lane following controller

<p align="center">
  <img src="Images_for_readme/lidar_based_lane_following-ezgif.com-cut.gif" title="Lane following controller">
</p>

The lane following controller allows the vehicle to autonomously track a user-defined reference track (pink line in the video) at a certain reference velocity. It requires a previously built map of the environment.

To build a map of the environment first launch the file:

```
roslaunch localization_and_mapping_pkg gmapping_universal.launch
```
Note that the vehicle needs to navigate the environment in order to map it. A convenient way of doing so is to run the velocity tracking controller described in the previous section. We also assume that the lidar has been properly set up as detailed in the building instructions.


To save the map type:
```
rosrun map_server map_saver -f map_name
```

To use the map in the lane following controller first make sure it is in the folder *saved_maps_from_slam* located in the localization_and_mapping_pkg package. Then edit the *map_file* parameter in *the map_server.launch* file to match the newly created map. Then launch the map server file.

```
roslaunch localization_and_mapping_pkg map_server.launch
```
Now launch the lane following controller.

```
roslaunch lane_following_controller_pkg lane_following_controller.launch
```
To modify the reference path edit the function *produce_track* located in the file *functions_for_controllers.py*.


