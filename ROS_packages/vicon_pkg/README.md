# Motion Planning Hackathon

This repository contains a rough skeleton of a ROS package that you may use as an initial guess during the hackathon.

---

>[!Important]
> This setup assumes that you have ROS 1 (Noetic) installed on your laptop. You can also use a dockerized version if you prefer, e.g.: <https://github.com/nachovizzo/ros_in_docker>

## Installing Dependencies

Clone this repo into your ROS workspace as `hackathon`: `git clone https://github.com/lassepe/motion-planning-hackathon hackathon`.

This package has two runtime dependencies:

- <https://github.com/tud-amr/vicon_bridge>
    - Install this by cloning it into your ROS workspace
- <https://github.com/cra-ros-pkg/robot_localization>
    - This is part of most ROS1 installations by default.

Finally, don't forget to `catkin build` in your workspace have it recognize the new packages.


## Quick Start

Each robot has a small onboard computer which runs a ROS master to aid communication between
low-level nodes running on the robot (motor controllers, sensors, diagnostics etc.).

>[!Important]
> In order to keep the robot in a usable state for *everyone*, **we do not modify the software on the robot!**
> 
> Instead, please adopt the following workflow:
> 
> - Treat the robot as a black box that runs the ROS master.
> - Run any additional nodes (high-level control, vicon bridge etc) on your own laptop


### Connecting Your Laptop to Robot's ROS Master

The only requirement for this workflow to work is that the nodes on your laptop are hooked up with the ROS master on the robot.


Follow these steps to enable this:

1. Connect both the robot and your laptop to the mocap network. All robots are already configured to connect to that network upon boot. For your machine, use:
    - `SSID: TP-Link_5F80_5G`
    - `Password: <ask lasse/thijs>`

2. [Consult this document to find the IP of your robot in the mobile robotics wifi as well as it's hostname](https://github.com/tud-amr/wiki/tree/main/infrastructure/network/mobile_robotics_hub))

3. Add the robot's hostname to your `/etc/hosts` file. 
    - `<robot_ip> <robot_hostname>`

3. Verify that you can ping the robot on it's ip and its hostname from your laptop.

4. In order for nodes launched on your laptop to hook up to the robot's ROS master, export the following environment variables:
    - `export ROS_MASTER_URI=http://<robot_hostname>:11311` (**note**: the exact value here may vary from platform to platform. Clearpath robots start the master on the hostname. Other plaforms (JetRacer) may launch the ROS master differently)
    - `export ROS_IP=<your_ip>` (you get this from `ip a`)

5. Finally, to verify that ROS communication works run
    - `rostopic list`; if this doesn't work it means your `ROS_MASTER_URI` is not set correctly
    - `rostopic echo <some_topic>`; if this doesn't work it means your `ROS_IP` is not set correctly or the ros master is launched on an IP alias that is not correctly resolved from your laptop


### Launching the Demo Setup

The main entry point for the demo setup is in `launch/demo.launch`.

By default, this launch file will:

1. Run an instance of the vicon bridge to translate vicon messages into ROS messages
2. Run an EKF for the robot to augment the vicon readings (pose only) with velocity information (and filter noise).
3. Launch Rviz to visualize the origin of the vicon frame and the EKF estimate of the robot state.


You can launch the demo node as: `roslaunch hackathon demo.launch robot_name:=<robot_name>` (where `<robot_name>` is the name of your robot in the Vicon tracker software).

>![!Note]
> You may see messages such as `Transform from odom to base_link was unavailable for the time requested. Using latest instead.` from time to time. This is due to wifi network delays and is not avoidable (especially if there are many WiFi devices around). So long as the update rate of the pose in RViz looks reasonable, you can ignore this message.

### Hack Away

For your own demo, add your own code in

- If you are working in C++: `src/`
- If you are working in Python: `scripts/`

Then, modify the launch file to also launch your controller / motion-planner.

