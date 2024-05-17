#!/usr/bin/env python3

import Jetson.GPIO as GPIO
import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import PointStamped
import math

GPIO.setmode(GPIO.BOARD)
my_pwm = GPIO.PWM(33,10000)
my_pwm.ChangeFrequency(10000)

# signal between 50 and 100%, with 50 at 6Hz and 100 at 12 Hz
# start at slowest
my_pwm.start(50)


def callback(point):
	dist = distance(point.point.x, point.point.y)
	
	pwm_now = min(100, dist*25 + 50)
	my_pwm.start(pwm_now);
	#print(" pwm" , pwm_now)
	
def distance(x,y):
	return math.sqrt(x*x + y*y)

rospy.init_node('lidar_pwm', anonymous=False)
rospy.Subscriber('tag_point', PointStamped, callback, queue_size=2)




rospy.spin()

GPIO.cleanup()
