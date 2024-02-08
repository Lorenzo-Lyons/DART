#!/usr/bin/env python3

import rospy
import pygame
from std_msgs.msg import Float32



# Initialize pygame
pygame.init()
width, height = 800, 200
gameDisplay = pygame.display.set_mode((width, height))
pygame.display.set_caption("Teleop controller")

# Set up Pygame font
font = pygame.font.Font(None, 36)  # Use the default font with size 36

# Set font color (RGB tuple)
font_color = (96, 96, 96)  # White



def teleop_keyboard(car_number):
    # Publishing safety value
    pub_safety_value = rospy.Publisher('safety_value', Float32, queue_size=8)
    pub_throttle = rospy.Publisher('throttle_'+ str(car_number), Float32, queue_size=8)
    pub_steering = rospy.Publisher('steering_'+ str(car_number), Float32, queue_size=8)

    rospy.init_node('teleop_keyboard', anonymous=True)
    rate = rospy.Rate(10)  # 10hz

    while not rospy.is_shutdown():
        pygame.event.pump()

        # Safety value publishing
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            pub_safety_value.publish(1)
            background_color = (204, 255, 153)  # Change to red
            text = "Go time!"
        else:
            pub_safety_value.publish(0)
            background_color = (192, 192, 192)  # Reset to black
            text = "WASD keys to drive and space to disingage safety"

        #publish control inputs
        if keys[pygame.K_a]:
            pub_steering.publish(1)
        elif keys[pygame.K_d]:
            pub_steering.publish(-1)
        else:
            pub_steering.publish(0)

        if keys[pygame.K_w]:
            pub_throttle.publish(0.3)
        elif keys[pygame.K_s]:
            pub_throttle.publish(-1)
        else:
            pub_throttle.publish(0)



        # Clear the screen with the current background color
        gameDisplay.fill(background_color)

        text_surface = font.render(text, True, font_color)

        # Blit the text surface onto the gameDisplay
        gameDisplay.blit(text_surface, (width // 2 - text_surface.get_width() // 2, height // 2 - text_surface.get_height() // 2))

        # Update the display
        pygame.display.flip()



        rate.sleep()

if __name__ == '__main__':
    try:
        car_number = 1
        teleop_keyboard(car_number)
    except rospy.ROSInterruptException:
        pass
