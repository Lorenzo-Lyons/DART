#ifndef VICON_BRIDGE_SEGMENT_PUBLISHER_POSESTAMPED_H_
#define VICON_BRIDGE_SEGMENT_PUBLISHER_POSESTAMPED_H_

#include <vicon_bridge/segment_publisher.h>

#include <geometry_msgs/PoseStamped.h>
#include <ros/node_handle.h>

class SegmentPublisherPoseStamped: public SegmentPublisher
{
public:

  SegmentPublisherPoseStamped(ros::NodeHandle nh, std::string frame_id, std::string publish_topic, int frequency_divider, double z_axis_offset):
    SegmentPublisher(nh.advertise<geometry_msgs::PoseStamped>(publish_topic, 1), frame_id, frequency_divider, z_axis_offset)
  {};

  void publishMsg(const ros::Time frame_time, const double position[3], const double rotation[4]) override
  {
    counter++;

    if (counter < publish_on_count)
    {
      return;
    }
    counter = 0;

    geometry_msgs::PoseStamped pose_stamped;

    pose_stamped.header.stamp = frame_time;
    pose_stamped.header.frame_id = frame_id_;
    pose_stamped.pose.position.x = position[0];
    pose_stamped.pose.position.y = position[1];
    pose_stamped.pose.position.z = position[2]-z_axis_offset_;
    pose_stamped.pose.orientation.x = rotation[0];
    pose_stamped.pose.orientation.y = rotation[1];
    pose_stamped.pose.orientation.z = rotation[2];
    pose_stamped.pose.orientation.w = rotation[3];

    pub_.publish(pose_stamped);
  }

};

#endif /* VICON_BRIDGE_SEGMENT_PUBLISHER_POSESTAMPED_H_ */