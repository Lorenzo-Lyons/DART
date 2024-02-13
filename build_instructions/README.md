## The complete build instructions for the DART

First we have to disassemble The Jetracer. Take of the camera and NVidia Jetson Nano. You are now seeing the top of the power distribution board. Then there are two options:
    - You don't need extra velocity sensors and are fine with an IMU alone, go to [shockabsorber](#shockabsorbers) section.
    - You want the extra sensors, which can take some time, go to the next section.


## Sensor placement for Velocity readings

Note: this is an extra feature that takes a bit of time. An alternative is using only an IMU.

Remove the power distribution board. Now unscrew the top of the main driveshaft and take out the main driveshaft by removing the rear differential gear housing.

To receive a velocity reading from the car, we are going to place sensors near the main gear for RPM readings. This can be done by adding an IR sensor or a magnet sensor. Use both if you want to increase the accuracy of the RPM estimate. Both sensors will be shown in this building tutorial.  

Take out the main gear and glue the two small laser cutted parts, magnet_inlay.DXF and white_ring_irsensors.DXF, to the main gear. There is the change that you have to remove some plastic from the main gear to place te inlay. Now place 4 small magnets in the magnet inlay, so they are correctly spaced. Make sure that the magnets have the correct magnetic field facing from the main gear, so the Magnet sensor can read the magnets when moving in front of the sensor.

<p align="center">
  <img src="images/main_gear.jpeg" width="700" title="Main gear">
</p>

Now use the IR sensor and the 3d printed sensor_gate.STL to create the IR sensor gate attached to the floor board of the DART.

<p align="center">
  <img src="images/irsensor.jpg" width="700" title="IR sensor gate">
</p>

The magnet sensor is placed on the rear differential gear housing using a 3D printed inlay (magnetsensor_holder.STL). 

<p align="center">
  <img src="images/magnetsensor.jpg" width="700" title="IR sensor gate">
</p>

Now we can rebuild the DART until the power idstribution board has been placed back. First insert the main gear with the driveshaft and the rear differential gear housing. Place the driveshaft cover back and then attach the power distribution board.

## Shockabsorbers

Due to the increased weight of the platform and the soft springs in the schockabsorbers, we have to increase the stiffness from the springs. This can be done by inserting a small plastic or brass ring.

<p align="center">
  <img src="images/shockabsorbers.jpg" width="700" title="Insert shockabsorbers">
</p>

## Full Design

For the top part of the DART we start by attaching the XD4 Lidar to the laser cutted baseboard.DXF And the spacers for the Jetson Nano. 

<p align="center">
  <img src="images/lidarplacement.jpeg" width="700" title="XD4 Lidar placement">
</p>


Attach the baseboard.DXF to the power distribution board. Now attach the Jetson Nano to the spacers upside down, as seen in the figure below. Make sure that the ports of the Jetson nano face backwards. 

To assemble the top plate, the camera is first inserted into the 3D printed camerafix.STL. This camerafix.STL with the camera is then mounted on the laser cutted upperboard.DXF. Then the board can be attached on top of the Jetson Nano using small spacers.

<p align="center">
  <img src="images/jetracerleft.jpeg" width="700" title="XD4 Lidar placement">
</p>

To attach the laser cutted back.DXF board, we use the small top holders, 3D printed smallholder.STL, and the bigger bottom holders, 3D printed holder.STL.


## Lipo battery placement

In the lower compartment that is not occupied by the servomotor and motor, can be used to insert the extra LiPo battery.

<p align="center">
  <img src="images/lipo.jpeg" width="700" title="LiPo placement ">
</p>