/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2010, UC Regents
 *  Copyright (c) 2011, Markus Achtelik, ETH Zurich, Autonomous Systems Lab (modifications)
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the University of California nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

#include <vicon_bridge/vicon_bridge.h>


#include <iostream>
#include <map>
#include <unordered_map>


#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>

#include <vicon_bridge/Markers.h>
#include <vicon_bridge/Marker.h>

#include <vicon_bridge/segment_publisher_posestamped.h>
#include <vicon_bridge/segment_publisher_posewithcovariancestamped.h>
#include <vicon_bridge/segment_publisher_transformstamped.h>

#include <yaml-cpp/yaml.h>


struct CalibrationData {
  tf::Vector3 translation;
  tf::Quaternion rotation;
};

ViconReceiver::ViconReceiver() :
    nh_priv("~")
{
  // Setting up the vicon client connection
  std::string host_name = "";
  std::string stream_mode = "ClientPull";
  bool setup_grabpose = false;


  nh_priv.param("stream_mode", stream_mode, stream_mode);
  nh_priv.param("datastream_hostport", host_name, host_name);

  if (init_vicon(host_name, stream_mode) == false)
  {
    ROS_ERROR("Error while connecting to Vicon. Exiting now.");
    return;
  }

  nh_priv.param("publish_transform", publish_tf_, publish_tf_);
  nh_priv.param("publish_markers", publish_markers_, publish_markers_);
  nh_priv.param("publish_segments", publish_segments_, publish_segments_);
  nh_priv.param("setup_grabpose", setup_grabpose, setup_grabpose);

  // Parameters for the tracked objects
  nh_priv.param("msg_type", msg_type_all_, msg_type_all_);
  nh_priv.param("frame_id", frame_id_all_, frame_id_all_);
  nh_priv.param("frequency_divider", frequency_divider_all_, frequency_divider_all_);
  nh_priv.param("reset_z_axis", reset_z_axis_, reset_z_axis_);
  nh_priv.param("calibration_folder_path", calibration_folder_path_, calibration_folder_path_);

  vicon_client_.GetFrame();
  double client_framerate = vicon_client_.GetFrameRate().FrameRateHz;
  ROS_INFO("Vicon client framerate: %f", client_framerate);


  //check if msg type is correct
  if (!(msg_type_all_ == "geometry_msgs/PoseStamped" || msg_type_all_ == "geometry_msgs/PoseWithCovarianceStamped" || msg_type_all_ == "geometry_msgs/TransformStamped"))
  {
    ROS_ERROR("msg_type %s is not supported. Please use geometry_msgs/PoseStamped, geometry_msgs/PoseWithCovarianceStamped or geometry_msgs/TransformStamped", msg_type_all_.c_str());
    return;
  }

  // Parameters for tracking specific objects
  nh_priv.param("only_use_object_specific", object_specific_only_, object_specific_only_);

  std::vector<std::string> object_names;
  std::vector<std::string> object_msg_types;
  std::vector<std::string> object_frame_ids;
  std::vector<std::string> object_publish_topics;
  std::vector<int> object_frequency_divider;
  nh_priv.param("object_specific/object_names", object_names, object_names);
  nh_priv.param("object_specific/object_msg_types", object_msg_types, object_msg_types);
  nh_priv.param("object_specific/object_frame_ids", object_frame_ids, object_frame_ids);
  nh_priv.param("object_specific/object_publish_topics", object_publish_topics, object_publish_topics);
  nh_priv.param("object_specific/object_publish_topics", object_publish_topics, object_publish_topics);
  nh_priv.param("object_specific/object_frequency_divider", object_frequency_divider, object_frequency_divider);

  // Load calibration data
  nh_priv.param("load_calibration_data", load_calibration_data_, load_calibration_data_);
  if (load_calibration_data_)  calibration_data_ = loadCalibrationData(object_names);

  // Check if the sizes of the vectors are equal
  if (!(object_names.size() == object_msg_types.size() && object_msg_types.size() == object_frame_ids.size() && object_frame_ids.size() == object_publish_topics.size() && object_publish_topics.size() == object_frequency_divider.size()))
  {
    ROS_ERROR("The sizes of the object_specific vectors are not equal. Please check the sizes of the vectors");
    return;
  }

  if (!object_names.size() == 0)
  {
    ROS_INFO("Found %d objects to determine specific settings", (int)object_names.size());

    for (int i = 0; i < object_names.size(); i++)
    {
      // Check if msg type is correct
      if (!(object_msg_types[i] == "geometry_msgs/PoseStamped" || object_msg_types[i] == "geometry_msgs/PoseWithCovarianceStamped" || object_msg_types[i] == "geometry_msgs/TransformStamped"))
      {
        ROS_ERROR("msg_type %s is not supported. Please use geometry_msgs/PoseStamped, geometry_msgs/PoseWithCovarianceStamped or geometry_msgs/TransformStamped", object_msg_types[i].c_str());
        return;
      }

      std::array<std::string, 4> object_details = {object_msg_types[i], object_frame_ids[i], object_publish_topics[i], std::to_string(object_frequency_divider[i])};

      object_specific_details_.insert(std::pair<std::string ,std::array<std::string, 4>>(object_names[i], object_details));

      int frequency = client_framerate / object_frequency_divider[i];

      ROS_INFO("Object %s: \n"
                "\t\t\t\t\t msg type: %s \n"
                "\t\t\t\t\t frame id: %s \n"
                "\t\t\t\t\t topic name: %s \n"
                "\t\t\t\t\t frequency divider: %i \n"
                "\t\t\t\t\t actual frequency: %i", object_names[i].c_str(), object_msg_types[i].c_str(), object_frame_ids[i].c_str(), object_publish_topics[i].c_str(), object_frequency_divider[i], frequency);
    }
  }

  

  
  // Service Server
  if (setup_grabpose)
  {
    ROS_INFO("setting up grab_vicon_pose service server ... ");
    m_grab_vicon_pose_service_server = nh_priv.advertiseService("grab_vicon_pose", &ViconReceiver::grabPoseCallback,
                                                                this);
  }

  // ROS_INFO("setting up segment calibration service server ... ");
  // calibrate_segment_server_ = nh_priv.advertiseService("calibrate_segment", &ViconReceiver::calibrateSegmentCallback,
  //                                                       this);

  // Publishers
  if(publish_markers_)
  {
    marker_pub_ = nh.advertise<vicon_bridge::Markers>(tracked_frame_suffix_ + "/markers", 10);
  }
  
  ros::Duration d(1);

  while (ros::ok())
  {
    // Ask for a new frame or wait untill a new frame is available
    if (vicon_client_.GetFrame().Result == Result::Success)
    {
      process_frame();
    }
    else
    {
      // Vicon client needs to be disconnected since connection is lost
      vicon_client_.Disconnect();
      ROS_WARN("Vicon client connection lost. Waiting for connection to grab a new frame ...");
      while (!vicon_client_.IsConnected().Connected)
      {
        vicon_client_.Connect(host_name);
        ROS_INFO(".");
        d.sleep();
      }
      ROS_INFO_STREAM("... connection re-established!");
    }   
  }
}

ViconReceiver::~ViconReceiver()
{
  for (size_t i = 0; i < time_log_.size(); i++)
  {
    std::cout << time_log_[i] << std::endl;
  }

  ROS_INFO_STREAM("Disconnecting from Vicon DataStream SDK");

  if (vicon_client_.Disconnect().Result == Result::Success){
    ROS_INFO("Vicon connection shut down.");
  } else
  {
    ROS_ERROR("Error while shutting down Vicon.");
  }
  ROS_ASSERT(!vicon_client_.IsConnected().Connected);
}


bool ViconReceiver::init_vicon(std::string host_name, std::string stream_mode)
{
  ROS_INFO_STREAM("Connecting to Vicon DataStream SDK at " << host_name << " ...");

  ros::Duration d(1);
  Result::Enum result(Result::Unknown);

  while (!vicon_client_.IsConnected().Connected)
  {
    vicon_client_.Connect(host_name);
    ROS_INFO(".");
    d.sleep();
    ros::spinOnce();
    if (!ros::ok())
      return false;
  }
  ROS_ASSERT(vicon_client_.IsConnected().Connected);
  ROS_INFO_STREAM("... connected!");

  // ClientPullPrefetch doesn't make much sense here, since we're only forwarding the data
  if (stream_mode == "ServerPush")
  {
    result = vicon_client_.SetStreamMode(StreamMode::ServerPush).Result;
  }
  else if (stream_mode == "ClientPull")
  {
    result = vicon_client_.SetStreamMode(StreamMode::ClientPull).Result;
  }
  else
  {
    ROS_FATAL("Unknown stream mode -- options are ServerPush, ClientPull");
    ros::shutdown();
  }

  ROS_INFO_STREAM("Setting Stream Mode to " << stream_mode<< ": "<< Adapt(result));

  vicon_client_.SetAxisMapping(Direction::Forward, Direction::Left, Direction::Up); // 'Z-up'
  Output_GetAxisMapping _Output_GetAxisMapping = vicon_client_.GetAxisMapping();

  ROS_INFO_STREAM("Axis Mapping: X-" << Adapt(_Output_GetAxisMapping.XAxis) << " Y-"
      << Adapt(_Output_GetAxisMapping.YAxis) << " Z-" << Adapt(_Output_GetAxisMapping.ZAxis));

  vicon_client_.EnableSegmentData();
  ROS_ASSERT(vicon_client_.IsSegmentDataEnabled().Enabled);

  Output_GetVersion _Output_GetVersion = vicon_client_.GetVersion();
  ROS_INFO_STREAM("Version: " << _Output_GetVersion.Major << "." << _Output_GetVersion.Minor << "."
      << _Output_GetVersion.Point);
  return true;
}


void ViconReceiver::process_frame()
{
  Output_GetFrameNumber OutputFrameNum = vicon_client_.GetFrameNumber();

  int frameDiff = OutputFrameNum.FrameNumber - lastFrameNumber_;
  lastFrameNumber_ = OutputFrameNum.FrameNumber;
  if (frameDiff == 0)
  {
    ROS_WARN_STREAM("Frame number unchanged (" << OutputFrameNum.FrameNumber << "). Skipping frame.");
    return;
  }

  ros::Duration vicon_latency(vicon_client_.GetLatencyTotal().Total);
  if(publish_segments_)
  {
    process_subjects(ros::Time::now() - vicon_latency);
  }

  // TODO: function below is not yet checked
  if(publish_markers_)
  {
    process_markers(ros::Time::now() - vicon_latency, lastFrameNumber_);
  }

  frameCount_ += frameDiff;
  if ((frameDiff) > 1)
  {
    droppedFrameCount_ += frameDiff - 1;
    double droppedFramePct = (double)droppedFrameCount_ / frameCount_ * 100;
    ROS_DEBUG_STREAM(frameDiff << " more (total " << droppedFrameCount_ << "/" << frameCount_ << ", "
        << droppedFramePct << "%) frame(s) dropped. Consider adjusting rates.");
  }
}


void ViconReceiver::process_subjects(const ros::Time& frame_time)
{
  std::vector<tf::StampedTransform, std::allocator<tf::StampedTransform> > transforms;
  static unsigned int cnt = 0;

  unsigned int n_subjects = vicon_client_.GetSubjectCount().SubjectCount;

  for (unsigned int i_subjects = 0; i_subjects < n_subjects; i_subjects++)
  {

    std::string subject_name = vicon_client_.GetSubjectName(i_subjects).SubjectName;

    unsigned int n_segments = vicon_client_.GetSegmentCount(subject_name).SegmentCount;
    for (unsigned int i_segments = 0; i_segments < n_segments; i_segments++)
    {
      std::string segment_name = vicon_client_.GetSegmentName(subject_name, i_segments).SegmentName;

      std::string name;
      if (n_segments == 1)
      {
        name = subject_name;
      }
      else
      {
        name = subject_name + "/" + segment_name;
      }

      // Check if we have a publisher for this segment
      std::map<std::string, std::unique_ptr<SegmentPublisher>>::iterator pub_it = segment_publishers_.find(name);

      // If it is not in the map, create a new one
      // Only if we want to make a new one
      if (pub_it == segment_publishers_.end())
      {
        

        // Check if the name of the object exists in the specific details
        std::map<std::string, std::array<std::string, 4>>::iterator object_specific_details_it = object_specific_details_.find(name);

        if (!object_specific_only_ && object_specific_details_it == object_specific_details_.end())
        {
          double z_axis_offset = 0.0;
          if (reset_z_axis_)
          {
            Output_GetSegmentGlobalTranslation trans = vicon_client_.GetSegmentGlobalTranslation(subject_name, segment_name);
            if (trans.Occluded)
            {
              ROS_WARN_STREAM(name <<" occluded during initialisation, not resetting the z-axis! " );
            } else
            {
              z_axis_offset = trans.Translation[2] / 1000;
            }
          }

          if (msg_type_all_ == "geometry_msgs/PoseStamped")
          {
            segment_publishers_.insert(std::make_pair(name, new SegmentPublisherPoseStamped(nh, frame_id_all_, name, frequency_divider_all_, z_axis_offset)));
          } else if (msg_type_all_ == "geometry_msgs/PoseWithCovarianceStamped")
          {
            segment_publishers_.insert(std::make_pair(name, new SegmentPublisherPosewithcovarianceStamped(nh, frame_id_all_, name, frequency_divider_all_, z_axis_offset)));
          } else if (msg_type_all_ == "geometry_msgs/TransformStamped")
          {
            segment_publishers_.insert(std::make_pair(name, new SegmentPublisherTransformStamped(nh, frame_id_all_, name, frequency_divider_all_, z_axis_offset)));
          }

        } else if (!(object_specific_details_it == object_specific_details_.end()))
        {
          double z_axis_offset = 0.0;
          if (reset_z_axis_)
          {
            Output_GetSegmentGlobalTranslation trans = vicon_client_.GetSegmentGlobalTranslation(subject_name, segment_name);
            if (trans.Occluded)
            {
              ROS_WARN_STREAM(name <<" occluded during initialisation, not resetting the z-axis! " );
            } else
            {
              z_axis_offset = trans.Translation[2] / 1000;
            }
          }

          std::string msg_type = object_specific_details_it->second[0];
          std::string frame_id = object_specific_details_it->second[1];
          std::string publish_topic = object_specific_details_it->second[2];
          int frequency_divider = std::stoi(object_specific_details_it->second[3]);

          // Use specific option
          if (msg_type == "geometry_msgs/PoseStamped")
          {
            segment_publishers_.insert(std::make_pair(name, new SegmentPublisherPoseStamped(nh, frame_id, publish_topic, frequency_divider, z_axis_offset)));
          } else if (msg_type == "geometry_msgs/PoseWithCovarianceStamped")
          {
            segment_publishers_.insert(std::make_pair(name, new SegmentPublisherPosewithcovarianceStamped(nh, frame_id, publish_topic, frequency_divider, z_axis_offset)));
          } else if (msg_type == "geometry_msgs/TransformStamped")
          {
            segment_publishers_.insert(std::make_pair(name, new SegmentPublisherTransformStamped(nh, frame_id, publish_topic, frequency_divider, z_axis_offset)));
          }

          object_specific_details_.erase(object_specific_details_it);

        } else
        {
          continue;
        }
        

        ROS_INFO("creating new object %s/%s ...",subject_name.c_str(), segment_name.c_str() );
        ROS_INFO("... done, advertised as  %s", name.c_str());

        continue;
      }

      Output_GetSegmentGlobalTranslation trans = vicon_client_.GetSegmentGlobalTranslation(subject_name, segment_name);
      Output_GetSegmentGlobalRotationQuaternion quat = vicon_client_.GetSegmentGlobalRotationQuaternion(subject_name,
                                                                                                      segment_name);

      if (!trans.Result == Result::Success || !quat.Result == Result::Success)
      {
        ROS_WARN("GetSegmentGlobalTranslation/Rotation failed (result = %s, %s), not publishing ...",
            Adapt(trans.Result).c_str(), Adapt(quat.Result).c_str());
        continue;
      }

      if (trans.Occluded || quat.Occluded)
      {
        if (cnt % 100 == 0)
            ROS_WARN_STREAM("" << name <<" occluded, not publishing... " );
        continue;
      }

      // Apply calibration if available
      auto calib_it = calibration_data_.find(subject_name);
      if (calib_it != calibration_data_.end()) {
          // Calibration data available, apply it
          tf::Transform calib_transform;
          calib_transform.setOrigin(calib_it->second.translation);
          calib_transform.setRotation(calib_it->second.rotation);

          tf::Transform current_transform;
          current_transform.setOrigin(tf::Vector3(trans.Translation[0] / 1000, trans.Translation[1] / 1000, trans.Translation[2] / 1000));
          current_transform.setRotation(tf::Quaternion(quat.Rotation[0], quat.Rotation[1], quat.Rotation[2], quat.Rotation[3]));

          tf::Quaternion corrected_rotation = calib_transform.getRotation() * current_transform.getRotation();
          tf::Vector3 corrected_translation = calib_transform.getOrigin() + current_transform.getOrigin();

          double translation[3] = {corrected_translation.x(), corrected_translation.y(), corrected_translation.z()};
          double rotation[4] = {corrected_rotation.x(), corrected_rotation.y(), corrected_rotation.z(), corrected_rotation.w()};

          pub_it->second->publishMsg(frame_time, translation, rotation);

          if (publish_tf_) {
              tf::Transform transform;
              transform.setOrigin(corrected_translation);
              transform.setRotation(corrected_rotation);
              transforms.push_back(tf::StampedTransform(transform, frame_time, frame_id_all_, name));
          }
          
      } else {
          // No calibration data available, publish original pose
          double translation[3] = {trans.Translation[0] / 1000, trans.Translation[1] / 1000,
                                                  trans.Translation[2] / 1000};
          double rotation[4] = {quat.Rotation[0], quat.Rotation[1], quat.Rotation[2],
                                                            quat.Rotation[3]};
          
          pub_it->second->publishMsg(frame_time, translation, rotation);

          if (publish_tf_)
          {
            tf::Transform transform;
            transform.setOrigin(tf::Vector3(trans.Translation[0] / 1000, trans.Translation[1] / 1000,
                                                  trans.Translation[2] / 1000));
            transform.setRotation(tf::Quaternion(quat.Rotation[0], quat.Rotation[1], quat.Rotation[2],
                                                            quat.Rotation[3]));
            transforms.push_back(tf::StampedTransform(transform, frame_time, frame_id_all_, name));
          }
      }
    }
  }

  if(publish_tf_)
  {
    tf_broadcaster_.sendTransform(transforms);
  }
  cnt++;
}

void ViconReceiver::process_markers(const ros::Time& frame_time, unsigned int vicon_frame_num)
{
  if (marker_pub_.getNumSubscribers() > 0)
  {
    if (!marker_data_enabled)
    {
      vicon_client_.EnableMarkerData();
      ROS_ASSERT(vicon_client_.IsMarkerDataEnabled().Enabled);
      marker_data_enabled = true;
    }
    if (!unlabeled_marker_data_enabled)
    {
      vicon_client_.EnableUnlabeledMarkerData();
      ROS_ASSERT(vicon_client_.IsUnlabeledMarkerDataEnabled().Enabled);
      unlabeled_marker_data_enabled = true;
    }
    n_markers_ = 0;
    vicon_bridge::Markers markers_msg;
    markers_msg.header.stamp = frame_time;
    markers_msg.frame_number = vicon_frame_num;
    // Count the number of subjects
    unsigned int SubjectCount = vicon_client_.GetSubjectCount().SubjectCount;
    // Get labeled markers
    for (unsigned int SubjectIndex = 0; SubjectIndex < SubjectCount; ++SubjectIndex)
    {
      std::string this_subject_name = vicon_client_.GetSubjectName(SubjectIndex).SubjectName;
      // Count the number of markers
      unsigned int num_subject_markers = vicon_client_.GetMarkerCount(this_subject_name).MarkerCount;
      n_markers_ += num_subject_markers;
      //std::cout << "    Markers (" << MarkerCount << "):" << std::endl;
      for (unsigned int MarkerIndex = 0; MarkerIndex < num_subject_markers; ++MarkerIndex)
      {
        vicon_bridge::Marker this_marker;
        this_marker.marker_name = vicon_client_.GetMarkerName(this_subject_name, MarkerIndex).MarkerName;
        this_marker.subject_name = this_subject_name;
        this_marker.segment_name
            = vicon_client_.GetMarkerParentName(this_subject_name, this_marker.marker_name).SegmentName;

        // Get the global marker translation
        Output_GetMarkerGlobalTranslation _Output_GetMarkerGlobalTranslation =
            vicon_client_.GetMarkerGlobalTranslation(this_subject_name, this_marker.marker_name);

        this_marker.translation.x = _Output_GetMarkerGlobalTranslation.Translation[0];
        this_marker.translation.y = _Output_GetMarkerGlobalTranslation.Translation[1];
        this_marker.translation.z = _Output_GetMarkerGlobalTranslation.Translation[2];
        this_marker.occluded = _Output_GetMarkerGlobalTranslation.Occluded;

        markers_msg.markers.push_back(this_marker);
      }
    }
    // get unlabeled markers
    unsigned int UnlabeledMarkerCount = vicon_client_.GetUnlabeledMarkerCount().MarkerCount;
    //ROS_INFO("# unlabeled markers: %d", UnlabeledMarkerCount);
    n_markers_ += UnlabeledMarkerCount;
    n_unlabeled_markers_ = UnlabeledMarkerCount;
    for (unsigned int UnlabeledMarkerIndex = 0; UnlabeledMarkerIndex < UnlabeledMarkerCount; ++UnlabeledMarkerIndex)
    {
      // Get the global marker translation
      Output_GetUnlabeledMarkerGlobalTranslation _Output_GetUnlabeledMarkerGlobalTranslation =
          vicon_client_.GetUnlabeledMarkerGlobalTranslation(UnlabeledMarkerIndex);

      if (_Output_GetUnlabeledMarkerGlobalTranslation.Result == Result::Success)
      {
        vicon_bridge::Marker this_marker;
        this_marker.translation.x = _Output_GetUnlabeledMarkerGlobalTranslation.Translation[0];
        this_marker.translation.y = _Output_GetUnlabeledMarkerGlobalTranslation.Translation[1];
        this_marker.translation.z = _Output_GetUnlabeledMarkerGlobalTranslation.Translation[2];
        this_marker.occluded = false; // unlabeled markers can't be occluded
        markers_msg.markers.push_back(this_marker);
      }
      else
      {
        ROS_WARN("GetUnlabeledMarkerGlobalTranslation failed (result = %s)",
            Adapt(_Output_GetUnlabeledMarkerGlobalTranslation.Result).c_str());
      }
    }
    marker_pub_.publish(markers_msg);
  }
}

bool ViconReceiver::grabPoseCallback(vicon_bridge::viconGrabPose::Request& req, vicon_bridge::viconGrabPose::Response& resp)
{
  ROS_INFO("Got request for a VICON pose");
  tf::TransformListener tf_listener;
  tf::StampedTransform transform;
  tf::Quaternion orientation(0, 0, 0, 0);
  tf::Vector3 position(0, 0, 0);

  std::string tracked_segment = tracked_frame_suffix_ + "/" + req.subject_name + "/" + req.segment_name;

  // Gather data:
  int N = req.n_measurements;
  int n_success = 0;
  ros::Duration timeout(0.1);
  ros::Duration poll_period(1.0 / 240.0);

  for (int k = 0; k < N; k++)
  {
    try
    {
      if (tf_listener.waitForTransform(frame_id_all_, tracked_segment, ros::Time::now(), timeout, poll_period))
      {
        tf_listener.lookupTransform(frame_id_all_, tracked_segment, ros::Time(0), transform);
        orientation += transform.getRotation();
        position += transform.getOrigin();
        n_success++;
      }
    }
    catch (tf::TransformException ex)
    {
      ROS_ERROR("%s", ex.what());
      //    		resp.success = false;
      //    		return false; // TODO: should we really bail here, or just try again?
    }
  }

  // Average the data
  orientation /= n_success;
  orientation.normalize();
  position /= n_success;

  // copy what we used to service call response:
  resp.success = true;
  resp.pose.header.stamp = ros::Time::now();
  resp.pose.header.frame_id = frame_id_all_;
  resp.pose.pose.position.x = position.x();
  resp.pose.pose.position.y = position.y();
  resp.pose.pose.position.z = position.z();
  resp.pose.pose.orientation.w = orientation.w();
  resp.pose.pose.orientation.x = orientation.x();
  resp.pose.pose.orientation.y = orientation.y();
  resp.pose.pose.orientation.z = orientation.z();

  return true;
}

std::unordered_map<std::string, ViconReceiver::CalibrationData> ViconReceiver::loadCalibrationData(const std::vector<std::string>& object_names) {
    std::unordered_map<std::string, CalibrationData> calibration_data;
    for (const auto& object_name : object_names) {
        std::string yaml_file = calibration_folder_path_ + "/" + object_name + ".yaml";
        
        try {
            YAML::Node config = YAML::LoadFile(yaml_file);
            CalibrationData data;
            data.translation.setX(config["position"]["x"].as<double>());
            data.translation.setY(config["position"]["y"].as<double>());
            data.translation.setZ(config["position"]["z"].as<double>());
            data.rotation.setX(config["orientation"]["x"].as<double>());
            data.rotation.setY(config["orientation"]["y"].as<double>());
            data.rotation.setZ(config["orientation"]["z"].as<double>());
            data.rotation.setW(config["orientation"]["w"].as<double>());
            calibration_data[object_name] = data;
            ROS_INFO("Load calibration data for object '%s'", object_name.c_str());
        } catch (const YAML::Exception& e) {
            ROS_WARN("Could not load calibration data for object '%s': %s", object_name.c_str(), e.what());
        }
    }
    return calibration_data;
}
// bool ViconReceiver::calibrateSegmentCallback(vicon_bridge::viconCalibrateSegment::Request& req,
//                               vicon_bridge::viconCalibrateSegment::Response& resp)
// {

//   std::string full_name = req.subject_name + "/" + req.segment_name;
//   ROS_INFO("trying to calibrate %s", full_name.c_str());

//   SegmentMap::iterator seg_it = segment_publishers_.find(full_name);

//   if (seg_it == segment_publishers_.end())
//   {
//     ROS_WARN("frame %s not found --> not calibrating", full_name.c_str());
//     resp.success = false;
//     resp.status = "segment " + full_name + " not found";
//     return false;
//   }

//   SegmentPublisher & seg = seg_it->second;

//   if (seg.calibrated)
//   {
//     ROS_INFO("%s already calibrated, deleting old calibration", full_name.c_str());
//     seg.calibration_pose.setIdentity();
//   }

//   vicon_bridge::viconGrabPose::Request grab_req;
//   vicon_bridge::viconGrabPose::Response grab_resp;

//   grab_req.n_measurements = req.n_measurements;
//   grab_req.subject_name = req.subject_name;
//   grab_req.segment_name = req.segment_name;

//   bool ret = grabPoseCallback(grab_req, grab_resp);

//   if (!ret)
//   {
//     resp.success = false;
//     resp.status = "error while grabbing pose from Vicon";
//     return false;
//   }

//   tf::Transform t;
//   t.setOrigin(tf::Vector3(grab_resp.pose.pose.position.x, grab_resp.pose.pose.position.y,
//                           grab_resp.pose.pose.position.z - req.z_offset));
//   t.setRotation(tf::Quaternion(grab_resp.pose.pose.orientation.x, grab_resp.pose.pose.orientation.y,
//                                 grab_resp.pose.pose.orientation.z, grab_resp.pose.pose.orientation.w));

//   seg.calibration_pose = t.inverse();

//   // write zero_pose to parameter server
//   string param_suffix(full_name + "/zero_pose/");
//   nh_priv.setParam(param_suffix + "orientation/w", t.getRotation().w());
//   nh_priv.setParam(param_suffix + "orientation/x", t.getRotation().x());
//   nh_priv.setParam(param_suffix + "orientation/y", t.getRotation().y());
//   nh_priv.setParam(param_suffix + "orientation/z", t.getRotation().z());

//   nh_priv.setParam(param_suffix + "position/x", t.getOrigin().x());
//   nh_priv.setParam(param_suffix + "position/y", t.getOrigin().y());
//   nh_priv.setParam(param_suffix + "position/z", t.getOrigin().z());

//   ROS_INFO_STREAM("calibration completed");
//   resp.pose = grab_resp.pose;
//   resp.success = true;
//   resp.status = "calibration successful";
//   seg.calibrated = true;

//   return true;
// }

int main(int argc, char** argv)
{
  ros::init(argc, argv, "vicon");

  ros::AsyncSpinner aspin(1);
  aspin.start();
  ViconReceiver vr;
  aspin.stop();
  return 0;
}
