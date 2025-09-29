#ifndef VICON_BRIDGE_SEGMENT_PUBLISHER_TRANSFORMSTAMPED_H_
#define VICON_BRIDGE_SEGMENT_PUBLISHER_TRANSFORMSTAMPED_H_

#include <vicon_bridge/segment_publisher.h>

#include <geometry_msgs/TransformStamped.h>
#include <ros/node_handle.h>


class SegmentPublisherTransformStamped: public SegmentPublisher
{
public:

  SegmentPublisherTransformStamped(ros::NodeHandle nh, std::string frame_id, std::string publish_topic, int frequency_divider, double z_axis_offset):
    SegmentPublisher(nh.advertise<geometry_msgs::TransformStamped>(publish_topic, 1), frame_id, frequency_divider, z_axis_offset)
  {};

  void publishMsg(const ros::Time frame_time, const double position[3], const double rotation[4]) override
  {
    counter++;

    if (counter < publish_on_count)
    {
      return;
    }
    counter = 0;

    geometry_msgs::TransformStamped transform_stamped;

    transform_stamped.header.stamp = frame_time;
    transform_stamped.header.frame_id = frame_id_;
    transform_stamped.transform.translation.x = position[0];
    transform_stamped.transform.translation.y = position[1];
    transform_stamped.transform.translation.z = position[2]-z_axis_offset_;
    transform_stamped.transform.rotation.x = rotation[0];
    transform_stamped.transform.rotation.y = rotation[1];
    transform_stamped.transform.rotation.z = rotation[2];
    transform_stamped.transform.rotation.w = rotation[3];

    pub_.publish(transform_stamped);
  }

};

#endif /* VICON_BRIDGE_SEGMENT_PUBLISHER_TRANSFORMSTAMPED_H_ */