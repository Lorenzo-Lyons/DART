# DART Lab Setup 
In this folder we show the instructions needed to use the DART platform in the [`Cognitive Robotics Mobile Robot Lab`](https://github.com/cor-mobile-robotics/lab-wiki).
Please read the wiki in the link. Most of the lab-related instructions are shown there in better detail. please also be aware that to book the lab you must first be added to the relevant outlook calendar, as detailed in the wiki.

## 1. Lab Setup
Read the  [`Cognitive Robotics Mobile Robot Lab`](https://github.com/cor-mobile-robotics/lab-wiki) wiki.


### Install ROS Packages to use the vicon system
To stream the DART state data from the vicon system onto a ROS network you will need to launch contents from the `vicon_pkg` package. This relies on the more general purpose 
[`vicon_bridge`](https://github.com/cor-mobile-robotics/vicon_bridge). For the user's convenience we have also put the `vicon_bridge` package in this repo, yet this will not be updated so use caution. To use these packages add them to a catkin workspace and build them. I.e.:

- Add the `vicon_bridge` package to a catkin workspace, taking it either from this repo or from [`vicon_bridge`](https://github.com/cor-mobile-robotics/vicon_bridge).
- Add the`vicon_pkg` from this repo to the same catkin workspace.
- build and re-source the workspace

## 2. Vehicle Setup (Operation Workflow)

When everything is set up correctly, the vehicle can be operated as follows:

1. Set up and connect the motor battery.  
2. Charge the motor battery (before use).  
3. Power on the Jetson.  
4. Turn on the ESC and set the reference voltage to a **0-throttle** value.  
5. Connect the car to a screen and configure Wi-Fi (should already be working in the lab).  
   - **User:** `jetson`  
   - **Password:** `jetson`  

---

## 3. Remote Access to the Vehicle

You can access the vehicle remotely via SSH or VS Code Remote:

- Ensure you are on the same network.  
  - LAN cable preferred.  
  - Wi-Fi: `mrl-wifi-5g` (password under the modem).  
- Set up **VS Code Remote SSH** with keys for convenience.  
- **TODO:** check if the PAT is only required for `jetracer_ws`.  

### Running the Vehicle
- Start `roscore` (if this is **car 1**).  
- Run `racecar_universal.py` on your laptop (it will complain about safety, but this is expected).  

---

## 4. Laptop Setup

Update your `.bashrc` with the ROS master and your IP:  

```bash
export ROS_IP=192.168.1.84
export ROS_MASTER_URI=http://192.168.0.131:11311/  # vehicle 1 as master

