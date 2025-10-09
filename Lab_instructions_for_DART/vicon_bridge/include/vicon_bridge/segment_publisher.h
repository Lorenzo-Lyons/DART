#ifndef VICON_BRIDGE_SEGMENT_PUBLISHER_H_
#define VICON_BRIDGE_SEGMENT_PUBLISHER_H_

#include <string>

#include <ros/publisher.h>
#include <ros/node_handle.h>

class SegmentPublisher
{
public:

  ros::Publisher pub_;
  std::string frame_id_ = "world";

  // Variables for publishing at a lower rate
  int counter = 0;
  int publish_on_count = 0;

  // If we want to reset the z-axis
  double z_axis_offset_ = 0.0;


  SegmentPublisher(ros::Publisher pub, std::string frame_id, int frequency_divider, double z_axis_offset):
    pub_(pub),
    frame_id_(frame_id),
    publish_on_count(frequency_divider),
    z_axis_offset_(z_axis_offset)
  {};

  //virtual void setMsg(ros::NodeHandle nh, std::string publish_topic)=0;

  virtual void publishMsg(const ros::Time frame_time, const double position[3], const double rotation[4])=0;

};

#endif /* VICON_BRIDGE_SEGMENT_PUBLISHER_H_ */