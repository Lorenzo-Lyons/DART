import numpy as np
# NOTE this should be a method added to a controller class.
# it should be the subscriber call back for the vicon pose topic

def vicon_subscriber_callback(self,msg):

    # here we need to evaluate the velocities
    # for now a very simple derivation
    # extract current orientation
    q_x = msg.pose.pose.orientation.x
    q_y = msg.pose.pose.orientation.y
    q_z = msg.pose.pose.orientation.z
    q_w = msg.pose.pose.orientation.w
    # convert to Euler
    quaternion = [q_x, q_y, q_z, q_w]
    roll, pitch, yaw = euler_from_quaternion(quaternion)

    #update past states
    # shift them back by 1 step and update the last value
    self.past_x_vicon[:-1] = self.past_x_vicon[1:]
    self.past_y_vicon[:-1] = self.past_y_vicon[1:]
    self.past_yaw_vicon[:-1] = self.past_yaw_vicon[1:]
    self.past_time_vicon[:-1] = self.past_time_vicon[1:]

    # add last entry
    self.past_x_vicon[-1] = msg.pose.pose.position.x
    self.past_y_vicon[-1] = msg.pose.pose.position.y
    self.past_yaw_vicon[-1] = yaw
    self.past_time_vicon[-1] = msg.header.stamp.to_sec()

    #evalaute velocities using finite differences on last values

    vx_abs = (self.past_x_vicon[-1] - self.past_x_vicon[0]) / (self.past_time_vicon[-1] - self.past_time_vicon[0])
    vy_abs = (self.past_y_vicon[-1] - self.past_y_vicon[0]) / (self.past_time_vicon[-1] - self.past_time_vicon[0])

    #convert to body frame
    self.vx = +vx_abs * np.cos(yaw) + vy_abs * np.sin(yaw)
    self.vy = -vx_abs * np.sin(yaw) + vy_abs * np.cos(yaw)


    # unwrap past angles to avoid jumps when flipping from - pi to + pi
    delta_yaw = self.past_yaw_vicon[-1] - self.past_yaw_vicon[0]
    if delta_yaw > np.pi:
        delta_yaw -= 2 * np.pi
    elif delta_yaw < -np.pi:
        delta_yaw += 2 * np.pi
    self.omega = (delta_yaw) / (self.past_time_vicon[-1] - self.past_time_vicon[0])

    # update the pose
    # if delay compensation is used, forward propagate the current state into the future
    if self.delay_compensation:
        #print('delay=',self.delay)
        #determine absolute velocities
        self.x_y_yaw_state = [msg.pose.pose.position.x + vx_abs * self.delay,
        msg.pose.pose.position.y+ vy_abs * self.delay,
        yaw + self.omega * self.delay]
    else:
        self.x_y_yaw_state = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]

    self.pose_msg_time = msg.header.stamp

    # publish velocity states for rviz
    self.vx_publisher.publish(Float32(self.vx))
    self.vy_publisher.publish(Float32(self.vy))
    self.w_publisher.publish(Float32(self.omega))