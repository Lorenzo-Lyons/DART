import pygame

def detect_gamepad_inputs():
    # Initialize pygame for gamepad usage
    pygame.init()
    pygame.joystick.init()

    # Check if any joysticks are connected
    if pygame.joystick.get_count() == 0:
        print("No joystick detected. Please connect a gamepad and try again.")
        return

    # Select the first joystick (assuming one is connected)
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Detected gamepad: {joystick.get_name()}")

    print("\nPress buttons or move axes on the gamepad to see their values (Press CTRL+C to exit).\n")

    try:
        while True:
            # Process events
            pygame.event.pump()

            # Check all buttons
            for button in range(joystick.get_numbuttons()):
                if joystick.get_button(button):
                    print(f"Button {button} pressed")

            # Check all axes
            for axis in range(joystick.get_numaxes()):
                axis_value = joystick.get_axis(axis)
                if abs(axis_value) > 0.1:  # Only print if the axis value is significant (to avoid noise)
                    print(f"Axis {axis} moved: {axis_value:.2f}")

            # Check all hats (D-pad)
            for hat in range(joystick.get_numhats()):
                hat_value = joystick.get_hat(hat)
                if hat_value != (0, 0):  # Only print if there's input
                    print(f"Hat {hat} pressed: {hat_value}")

            pygame.time.wait(100)  # Delay to make the output more readable

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Clean up pygame resources
        pygame.quit()

if __name__ == "__main__":
    detect_gamepad_inputs()
