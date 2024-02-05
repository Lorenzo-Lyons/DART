#!/usr/bin/env python3

import rospy
from getkey import getkey
from std_msgs.msg import Float32
from pynput.keyboard import Key, Listener



class safety_publisher:
    def __init__(self):
        self.safety_value = 0
        rospy.init_node('safety', anonymous=True)
        self.pub_safety = rospy.Publisher('safety_value', Float32, queue_size=1)

    def on_press(self,key):
        global safety_value
        if key == Key.backspace:
            self.safety_value = 1
            self.pub_safety.publish(self.safety_value)
            # print('{0} pressed'.format(
            # key))
            print('safety disingaged')
        else:
            print('press backspace to disinge safety')


    def on_release(self,key):
        global safety_value
        self.safety_value = 0
        self.pub_safety.publish(self.safety_value)
        # print('{0} release'.format(key))
        print('safety engaged')
        if key == Key.esc:
            # Stop listener
            return False



    def teleop(self):
        #Setup topics publishing and nodes

        rate = rospy.Rate(10) # 10hz

        #Print hints
        print("Press backspace key to disingage safety")

        while not rospy.is_shutdown():
            #Get key press
            #key = getkey()

            # Collect events until released
            with Listener(
                    on_press=self.on_press,
                    on_release=self.on_release) as listener:
                listener.join()
            self.pub_safety.publish(self.safety_value)

            rate.sleep()



if __name__ == '__main__':
    try:
        safety_publisher_obj = safety_publisher()
        safety_publisher_obj.teleop()
    except rospy.ROSInterruptException:
        pass
