cmake_minimum_required(VERSION 3.0.2)
project(remote_pc)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  ros_numpy
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
)

#   std_msgs

catkin_package(
#  CATKIN_DEPENDS roscpp rospy std_msgs

    CATKIN_DEPENDS geometry_msgs ros_numpy rospy sensor_msgs std_msgs tf roscpp
)
install(FILES launch/TrainData2csv.launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

catkin_install_python(PROGRAMS scripts/TrainData2csv.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
