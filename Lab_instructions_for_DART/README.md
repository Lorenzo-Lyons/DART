# DART Lab Setup & Vehicle Operation Guide

## 1. Lab Setup

### Install ROS Packages
- Install [`vicon_pkg`](https://github.com/ethz-asl/vicon_bridge)  
  **TODO:** clean up references, mention Lasse, and update instructions.  
- Install `vicon_bridge`  
  **TODO:** clarify lab setup details and mention Lasse.  

---

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

