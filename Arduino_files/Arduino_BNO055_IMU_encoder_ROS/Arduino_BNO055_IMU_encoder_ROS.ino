
// this arduino script sets up a ROS node that sends out sensor data a float array
// to read it from the ros nework you need to install
// from arduino library manager rosserial_arduino
// then in terminal
// sudo apt-get install ros-melodic-rosserial-python
// then run 
// rosrun rosserial_python serial_node.py     
// you can also specify the port by adding      _port:=/dev/ttyUSB0




// define frequency of data output
double dt = 0.1; //10Hz
unsigned long time_now = 0; // updated later inside loop



// inlcude libraries for ROS integration
#include <ros.h>
#include <std_msgs/Float32MultiArray.h>

ros::NodeHandle nh;

std_msgs::Float32MultiArray arduino_data_msg;
ros::Publisher imu_data_pub("arduino_data_1", &arduino_data_msg);

// set up varible for array publishing
// acc_x, acc_y, omega_rad, vel
float list[4] = {0.0,0.0,0.0,0.0};




//--- set up IMU
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28, &Wire);



//--- velocity reading related

// Creating global variables for the detection pins, these need to be volatile
volatile int i_IR = 0;

// Variables for timing the loop period
unsigned long period = dt * pow(10,6);       // conversion from seconds to microseconds

//[For IR sensor] Conversion from gear and from wheel rotation to wheel motion -> 2pi/(ndetections_per_revolution(20)) * Tau_differential(11/39)*R_wheel(0.0233)
double conversion_n_to_m = 0.00412708; //0.005158;   //[For Magnet sensor] 2pi/(ndetections_per_revolution(10)) * Tau_differential(11/39)*R_wheel(0.0233)

// Variables for storing the velocities
double IR_vel = 0;
double detections_2_velocity_factor = conversion_n_to_m / dt;






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

  // set up arduino related things
  nh.initNode();
  nh.advertise(imu_data_pub);
  
  /* Initialise the sensor */
  if (!bno.begin())
  {
    /* There was a problem detecting the BNO055 ... check your connections */
    //Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
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
  imu::Vector<3> gyro = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
  imu::Vector<3> accel = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);

 
  // IR velocity measure ---------------------------------------------------

  // Convert incremented detections to velocities
  IR_vel = detections_2_velocity_factor * float(i_IR);

  // reset counter to 0 (it is incremented by the interrupt pin)
  //Serial.println(i_IR);
  i_IR = 0;
 
  //-------------------------------------------------------------------------
  
  // produce ROS message
  // acc_x, acc_y, omega_rad, vel
  list[0] = accel.x();
  list[1] = accel.y();
  list[2] = gyro.z() / 180 * 3.141592;
  list[3] = IR_vel;

  // publish message
  arduino_data_msg.data = list;
  arduino_data_msg.data_length = 4;
  imu_data_pub.publish( &arduino_data_msg);

  //Allow ROS to process incoming messages
  nh.spinOnce();
  
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
