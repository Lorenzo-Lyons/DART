#!/usr/bin/env python

import rospy	

from dynamic_reconfigure_pkg.cfg import server_dart

def callback(config, level):
	rospy.loginfo("""Reconfigure Request: {int_param}, {double_param},\ 
	{str_param}, {bool_param}, {size}""".format(**config))
	return config
 
if __name__ == "__main__":
	rospy.init_node("dart_simulator_gui_node", anonymous = False)

	srv = Server(server_dart, callback)
	rospy.spin()

