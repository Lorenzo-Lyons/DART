/*!
LTC2992: Dual Wide Range Power Monitor

@verbatim

Setting the Thresholds:
    1. Select the Threshold Configuration from the main menu.
    2. Select the desired Threshold to be changed. Then enter the minimum and maximum
       values.
    3. If any reading exceeds the set threshold, a fault will be generated that can be viewed in
     the Read/Clear Faults Menu


Reading and Clearing a Fault:
    1. Select the Read/Clear Fault option from the main menu.
    2. To read all the faults, select Read Faults option. This will display all the faults that have occured.
  3. To clear all faults, go back to the Read/Clear Faults menu and select Clear Faults option.

NOTE: Due to memory limitations of the Atmega328 processor this sketch shows limited functionality of the LTC2992. Please
      check the datasheet for a picture of the full functionality of the LTC2992.

NOTES
 Setup:
 Set the terminal baud rate to 115200 and select the newline terminator.
 Requires a power supply.
 Refer to demo manual DC2561A.

USER INPUT DATA FORMAT:
 decimal : 1024
 hex     : 0x400
 octal   : 02000  (leGPIOg 0 "zero")
 binary  : B10000000000
 float   : 1024.0

@endverbatim
http://www.linear.com/product/LTC2992

http://www.linear.com/product/LTC2992#demoboards

Copyright 2018(c) Analog Devices, Inc.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in
   the documentation and/or other materials provided with the
   distribution.
 - Neither the name of Analog Devices, Inc. nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.
 - The use of this software may or may not infringe the patent rights
   of one or more patent holders.  This license does not release you
   from the requirement that you obtain separate licenses from these
   patent holders to use this software.
 - Use of the software either in source or binary form, must be run
   on or directly connected to an Analog Devices Inc. component.

THIS SOFTWARE IS PROVIDED BY ANALOG DEVICES "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT,
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL ANALOG DEVICES BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, INTELLECTUAL PROPERTY RIGHTS, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*! @file
    @ingroup LTC2992
*/

// define frequency of data output
double dt = 0.1; //10Hz
unsigned long time_now = 0; // updated later inside loop








#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

/* Set the delay between fresh samples */
uint16_t BNO055_SAMPLERATE_DELAY_MS = dt * 1000;

// Check I2C device address and correct line below (by default address is 0x29 or 0x28)
//                                   id, address
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28, &Wire);









int16_t ax, ay, az;
int16_t gx, gy, gz;
float ax_float, ay_float, gz_float;
//float IMU_acc_conversion = 9.81 / 16384 ; // comes out in meters per second
//float IMU_gyro_conversion = 0.007633; //taken from datasheet(1 / 131) comes out in derees per second














#include <Arduino.h>
#include <stdint.h>
#include "Linduino.h"
#include "LT_I2C.h"
#include "LT_SPI.h"
#include "UserInterface.h"
#include "QuikEval_EEPROM.h"
#include "LTC2992.h"
#include <Wire.h>
#include <SPI.h>



const float resistor = .02;         //!< resistor value on demo board

// Error string
const char ack_error[] = "Error: No Acknowledge. Check I2C Address."; //!< Error message

// Global variables
static int8_t demo_board_connected;        //!< Set to 1 if the board is connected




//velocity reading related
// Changing the constant below determines the time between velocity measurements

// Constant below determines how much % of new velocity is used to update the filtered old velocity
// High value (100) means the new velocity measurement is the filtered velocity, thus no filtering is done
// Low value (1) means it takes a long time for new measurements to change the filtered velocity
// unsigned long filter_weight = 50;

// Creating global variables for the detection pins, these need to be volatile
//volatile unsigned long i_IR = 0; 
volatile int i_IR = 0;
//volatile unsigned long IR_LastTimeWeMeasured = micros();
//volatile unsigned long IR_SumPeriods = 0;

// Variables for storing the incremented detection values, so they are not changed during calculations
//unsigned long IR_store = 0;
//unsigned long IR_SumPeriods_store = 0;

// Variables for timing the loop period
unsigned long period = dt * pow(10,6);       // conversion from seconds to microseconds


// Variables for converting detections to velocities
//double conversion = 0.041291804;   // this is one full main gear rotation converted to meters
//unsigned long IR_conversion_period = pow(10,9) * conversion / 20;   // For integer division storage multiplied by 10^9
//double conversion_n_to_m = 0.002063;   //[For IR sensor] Conversion from gear and from wheel rotation to wheel motion -> 2pi/(ndetections_per_revolution(20)) * Tau_differential(11/39)*R_wheel(0.0233)
double conversion_n_to_m = 0.0022939891; //0.005158;   //[For Magnet sensor] 2pi/(ndetections_per_revolution(8)) * Tau_differential(11/39)*R_wheel(0.0233)

// Variables for storing the velocities
double IR_vel = 0;
//unsigned long period_vel = 0;
//unsigned long filter_vel = 0;

//double detections_2_velocity_factor = 2 * 3.141593 / 20 * dt * conversion; // this is 2pi/n_partitions on wheel * gear ratio * Radius of wheel [m] so it converts detections to m/s
double detections_2_velocity_factor = conversion_n_to_m / dt;

// string publishing related global variables
String current_str = "0.00";
String voltage_str = "0.00";
String ax_str = "0.00";
String ay_str = "0.00";
String gz_str = "0.00";
String vel_str = "0.00";




// variables need to set a fixed frequency
 



//! Initialize sensors
void setup()
{
  // setting the pins----------
  // IMU power supply
  pinMode(4, OUTPUT);
  digitalWrite(4, HIGH);
  // IR sensor
  pinMode(2, INPUT_PULLUP);
  pinMode(7, OUTPUT);     //for power
  digitalWrite(7, HIGH);
  attachInterrupt(digitalPinToInterrupt(2), Pulse_Event_IR, CHANGE); // setting interrupt pin to perform operations really quick on trigger event ####CHANGE
  //----------------------------


  Serial.begin(115200);

  while (!Serial) delay(10);  // wait for serial port to open!

  Serial.println("Orientation Sensor Test"); Serial.println("");

  /* Initialise the sensor */
  if (!bno.begin())
  {
    /* There was a problem detecting the BNO055 ... check your connections */
    Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
    while (1);
  }

  delay(1000);


}








//! Repeats Linduino loop
void loop()
{ 
  //start timer
  time_now = micros();
  
  //IMU sensor-----------------------------------------------------------

    //could add VECTOR_ACCELEROMETER, VECTOR_MAGNETOMETER,VECTOR_GRAVITY...
  sensors_event_t orientationData , angVelocityData , linearAccelData, magnetometerData, accelerometerData, gravityData;
  //bno.getEvent(&orientationData, Adafruit_BNO055::VECTOR_EULER);
  bno.getEvent(&angVelocityData, Adafruit_BNO055::VECTOR_GYROSCOPE);
  //bno.getEvent(&linearAccelData, Adafruit_BNO055::VECTOR_LINEARACCEL);
  //bno.getEvent(&magnetometerData, Adafruit_BNO055::VECTOR_MAGNETOMETER);
  bno.getEvent(&accelerometerData, Adafruit_BNO055::VECTOR_ACCELEROMETER);
  //bno.getEvent(&gravityData, Adafruit_BNO055::VECTOR_GRAVITY);

  //printEvent(&orientationData);
  printEvent(&angVelocityData);
  //printEvent(&linearAccelData);
  //printEvent(&magnetometerData);
  printEvent(&accelerometerData);
  //printEvent(&gravityData);

  //int8_t boardTemp = bno.getTemp();
  //Serial.println();
  //Serial.print(F("temperature: "));
  //Serial.println(boardTemp);

  //uint8_t system, gyro, accel, mag = 0;
  //bno.getCalibration(&system, &gyro, &accel, &mag);
  //Serial.println();
  //Serial.print("Calibration: Sys=");
  //Serial.print(system);
  //Serial.print(" Gyro=");
  //Serial.print(gyro);
  //Serial.print(" Accel=");
  //Serial.print(accel);
  //Serial.print(" Mag=");
  //Serial.println(mag);

  //Serial.println("--");

  // ------------------------------------------------------------------------
  
  
  // IR velocity measure ---------------------------------------------------

  // Convert incremented detections to velocities
  IR_vel = detections_2_velocity_factor * float(i_IR);
  
  // Conver to string in order to send the velocities over serial connection
  vel_str = String(IR_vel);
  
  // reset counter to 0 (it is incremented by the interrupt pin)
  //Serial.println(i_IR);
  i_IR = 0;
 
  //-------------------------------------------------------------------------
  
  

  //send out string through serial ------------------------------------------
  Serial.println("Cur" + current_str + "Vol" + voltage_str + "Acc_x" + ax_str + "Acc_y"+ ay_str + "Gyr_z" + gz_str + "Vel" + vel_str);
  //-------------------------------------------------------------------------

  // Run while loop for remaining time
  while( micros() < time_now + period)
  {
    // Wait until the sampling period 
  }
}


 
// Interrupt pins
// Run this function when pin 2 (IR sensor) is interrupted
void Pulse_Event_IR()
{  
  i_IR++; //so just increment the number of detections
  //Serial.println(String(i_IR)+"   voltage" + String(digitalRead(2)));
}




// read sensor data from IMU, these functions are triggered when you call getEvent
void printEvent(sensors_event_t* event) {
  double x = -1000000, y = -1000000 , z = -1000000; //dumb values, easy to spot problem
  if (event->type == SENSOR_TYPE_ACCELEROMETER) {
    //Serial.print("Accl:");
    ax_float = event->acceleration.x;
    ay_float = event->acceleration.y;
    z = event->acceleration.z;
    ax_str = String(ax_float);
    ay_str = String(ay_float);
  }
  else if (event->type == SENSOR_TYPE_ORIENTATION) {
    Serial.print("Orient:");
    x = event->orientation.x;
    y = event->orientation.y;
    z = event->orientation.z;
  }
  else if (event->type == SENSOR_TYPE_MAGNETIC_FIELD) {
    Serial.print("Mag:");
    x = event->magnetic.x;
    y = event->magnetic.y;
    z = event->magnetic.z;
  }
  else if (event->type == SENSOR_TYPE_GYROSCOPE) {
    //Serial.print("Gyro:");
    x = event->gyro.x;
    y = event->gyro.y;
    gz_float = event->gyro.z;
    gz_str = String(gz_float);
  }
  else if (event->type == SENSOR_TYPE_ROTATION_VECTOR) {
    Serial.print("Rot:");
    x = event->gyro.x;
    y = event->gyro.y;
    z = event->gyro.z;

  }
  else if (event->type == SENSOR_TYPE_LINEAR_ACCELERATION) {
    Serial.print("Linear:");
    x = event->acceleration.x;
    y = event->acceleration.y;
    z = event->acceleration.z;
  }
  else if (event->type == SENSOR_TYPE_GRAVITY) {
    Serial.print("Gravity:");
    x = event->acceleration.x;
    y = event->acceleration.y;
    z = event->acceleration.z;
  }
  else {
    Serial.print("Unk:");
  }
  
  
  //Serial.print("\tx= ");
  //Serial.print(x);
  //Serial.print(" |\ty= ");
  //Serial.print(y);
  //Serial.print(" |\tz= ");
  //Serial.println(z);
}
