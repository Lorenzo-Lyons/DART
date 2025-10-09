#ifndef VICON_BRIDGE_SEGMENT_PUBLISHER_POSEWITHCOVARIANCESTAMPED_H_
#define VICON_BRIDGE_SEGMENT_PUBLISHER_POSEWITHCOVARIANCESTAMPED_H_

#include <vicon_bridge/segment_publisher.h>

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <ros/node_handle.h>

class SegmentPublisherPosewithcovarianceStamped: public SegmentPublisher
{
public:

  SegmentPublisherPosewithcovarianceStamped(ros::NodeHandle nh, std::string frame_id, std::string publish_topic, int frequency_divider, double z_axis_offset):
    SegmentPublisher(nh.advertise<geometry_msgs::PoseWithCovarianceStamped>(publish_topic, 1), frame_id, frequency_divider, z_axis_offset)
  {};

  void publishMsg(const ros::Time frame_time, const double position[3], const double rotation[4]) override
  {
    counter++;

    if (counter < publish_on_count)
    {
      return;
    }
    counter = 0;

    geometry_msgs::PoseWithCovarianceStamped pose_with_covariance_stamped;

    pose_with_covariance_stamped.header.stamp = frame_time;
    pose_with_covariance_stamped.header.frame_id = frame_id_;
    pose_with_covariance_stamped.pose.pose.position.x = position[0];
    pose_with_covariance_stamped.pose.pose.position.y = position[1];
    pose_with_covariance_stamped.pose.pose.position.z = position[2]-z_axis_offset_;
    pose_with_covariance_stamped.pose.pose.orientation.x = rotation[0];
    pose_with_covariance_stamped.pose.pose.orientation.y = rotation[1];
    pose_with_covariance_stamped.pose.pose.orientation.z = rotation[2];
    pose_with_covariance_stamped.pose.pose.orientation.w = rotation[3];

    pub_.publish(pose_with_covariance_stamped);
    
  }

};

#endif /* VICON_BRIDGE_SEGMENT_PUBLISHER_POSEWITHCOVARIANCESTAMPED_H_ */
