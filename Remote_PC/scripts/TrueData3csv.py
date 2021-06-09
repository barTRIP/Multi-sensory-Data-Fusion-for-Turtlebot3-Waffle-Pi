#!/usr/bin/env python3

# Import Numpy for easy array manipulation
import numpy as np
from numpy import asarray, savetxt, inf
# Import OpenCV for easy image rendering
import time
#Import csv to document stuff.
import csv
import os
#Import transformations stuff
#Check use of tf2 instead.
import tf
import tf.transformations
from geometry_msgs.msg import PointStamped
#Import math stuff
import math
# import ROS stuff
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField, LaserScan, Range
import sensor_msgs.point_cloud2 as pc2
import ros_numpy

#Various flags.
XYZ = None
j=0
k=0
a=0
b=0
prev_angle = 0
#Array management variables.
x_coord = []
y_coord = []
z_coord = []
distance_np = []
Angle = []
def_angle = []
camera=[1]
Sonar_Data = []
Def_Camera = []
DefLidarData = [''] * 360
LidarData = None
Sonar_L = None
Sonar_R = None
#Execution flags
flag = 1
Flag = 1
Final_flag = 1

#Callbacks to all the subscribers.
def callback_Lidar(msg):
	global LidarData
	LidarData = np.array(msg.ranges)

def callback_Sonar_L(msg):
	global Sonar_L
	Sonar_L = msg.range

def callback_Sonar_R(msg):
	global Sonar_R
	Sonar_R = msg.range

def cloud_callback(cloud):
	global XYZ
	XYZ = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(cloud)

#Subscription to sensor topics and node initialization.
rospy.init_node('PointCloud')
subLidar = rospy.Subscriber('/scan', LaserScan, callback_Lidar)
subSonarL = rospy.Subscriber('/L_sonar_dist', Range, callback_Sonar_L)
subSonarR = rospy.Subscriber('/R_sonar_dist', Range, callback_Sonar_R)
pc_cloud = rospy.Subscriber('/camera/depth/color/points',PointCloud2,cloud_callback)

#Waiting to receive data from topics.
rospy.wait_for_message('/L_sonar_dist', Range)
rospy.wait_for_message('/R_sonar_dist', Range)
rospy.wait_for_message('/scan', LaserScan)
rospy.wait_for_message('/camera/depth/color/points',PointCloud2)

#Continous execution while not Ctrl-c
while not rospy.is_shutdown():
	#Delay each execution to a frequency of 10 Hz
	start=time.process_time()
	ros_rate = rospy.Rate(10)
	ros_rate.sleep ()

	#Transforming the coordinates from all sensors w.r.t. lidar r.f.
	listenerSonarL=tf.TransformListener()
	listenerSonarL.waitForTransform('/sonarL_link', '/base_scan', rospy.Time(0),rospy.Duration(60.0))
	sonarL_point=PointStamped()
	sonarL_point.header.frame_id = 'sonarL_link'
	sonarL_point.header.stamp =rospy.Time()
	sonarL_point.point.x= Sonar_L
	sonarL_p = listenerSonarL.transformPoint('base_scan',sonarL_point)
	UltSonarL = sonarL_p.point.x

	listenerSonarR=tf.TransformListener()
	listenerSonarR.waitForTransform('/sonarR_link', '/base_scan', rospy.Time(0),rospy.Duration(60.0))
	sonarR_point=PointStamped()
	sonarR_point.header.frame_id = 'sonarR_link'
	sonarR_point.header.stamp =rospy.Time()
	sonarR_point.point.x= Sonar_R
	sonarR_p = listenerSonarR.transformPoint('base_scan',sonarR_point)
	UltSonarR = sonarR_p.point.x

	listenerCamera=tf.TransformListener()
	listenerCamera.waitForTransform('/camera_depth_optical_frame', '/base_scan', rospy.Time(0),rospy.Duration(60.0))

	#Camera data obtention
	#Substract the relevant points from the image and save it in Cube
	for i, data in enumerate(XYZ):
		camera_point=PointStamped()
		camera_point.header.frame_id = 'camera_depth_optical_frame'
		camera_point.header.stamp =rospy.Time()
		camera_point.point.x = data[0]
		camera_point.point.y = data[1]
		camera_point.point.z = data[2]
		camera_p = listenerCamera.transformPoint('base_scan',camera_point)
		UltCameraX = camera_p.point.x
		UltCameraY = camera_p.point.y
		UltCameraZ = camera_p.point.z
		UltCameraX = float('%.3f'%(UltCameraX))
		UltCameraY = float('%.3f'%(UltCameraY))
		if UltCameraZ < -0.12:
			break
		if UltCameraZ < 0.03:
			x_coord.append([UltCameraX])
			y_coord.append([UltCameraY])
			#z_coord.append([UltCameraZ])
	x_coord=np.array(x_coord)
	y_coord=np.array(y_coord)
	cube=np.append(x_coord, y_coord, 1)
	x_coord=cube[:,1]
	y_coord=cube[:,0]
	distance_numpy=(x_coord*x_coord+y_coord*y_coord)**0.5
	distance_numpy=distance_numpy.round(decimals=3)
	distance_numpy=np.where(distance_numpy>3.5, 10.0, distance_numpy)
	x_coord=np.where(x_coord==0, 0.001, x_coord)
	angle_numpy=np.arctan(y_coord/x_coord)
	angle_numpy=np.where(angle_numpy<0,(-angle_numpy)*(180/math.pi),(math.pi-angle_numpy)*(180/math.pi))
	angle_numpy=angle_numpy.round(decimals=0)
	distance_numpy=np.expand_dims(distance_numpy, axis=1)
	angle_numpy=np.expand_dims(angle_numpy, axis=1)
	cube=np.append(distance_numpy, angle_numpy, 1)
	#Change angle and distance position.
	cube[:,[0,1]]=cube[:,[1,0]]
	#Sort by first column
	cube = cube[cube[:,0].argsort()]
	#Split cube to remove redundant data.
	distance=cube[:,1]
	angle=cube[:,0]
	cube=cube[np.unique(cube[:,[0]],return_index=True,axis=0)[1]]
	for i in range (0, len(cube[:,0])):
		if i+1 != len(cube[:,0]):
			if cube[i+1,0]-cube[i,0] != 1:
				np.insert(cube, i, np.array((cube[i,0]+1,10)))


	Def_Camera=cube[:,1]
	Def_Camera = np.reshape(Def_Camera, (-1, 1))
	def_angle=cube[:,0]

    #Sonar data obtention
	midPoint = (def_angle.size)/2.0
	if UltSonarR > 3.5:
		UltSonarR = 10
	if UltSonarL > 3.5:
		UltSonarL = 10
	if len(def_angle) % 2 == 0:
		Sonar_R_Data = np.full(int(midPoint), round(UltSonarR, 3), dtype=float)
		Sonar_L_Data = np.full(int(midPoint), round(UltSonarL, 3), dtype=float)
	else:
		Sonar_R_Data = np.full(int(midPoint+1), round(UltSonarR, 3), dtype=float)
		Sonar_L_Data = np.full(int(midPoint), round(UltSonarL, 3), dtype=float)
	Sonar_Data = np.append(Sonar_R_Data, Sonar_L_Data, axis=0)
	Sonar_Data = np.reshape(Sonar_Data, (-1, 1))

	#Lidar data obtention
	LidarData = LidarData[:,np.newaxis]
	for i in range(len(LidarData)):
		if LidarData[i] == float('+inf') or LidarData[i] == 0:
			LidarData[i] = 10
		if i > 269:
			DefLidarData[i-270] = LidarData[i]
		if i < 270:
			DefLidarData[i+90] = LidarData[i]
	DefLidarData = np.array(DefLidarData)
	initial_sect=np.arange(0, def_angle[0])
	final_sect=np.arange(def_angle[0]+len(def_angle), 360)
	extra=np.append(initial_sect, final_sect)
	Def_Lidar_Data=np.delete(DefLidarData, extra.astype(int), axis=0)
	print(Def_Lidar_Data.shape)
	#if not os.path.exists('/home/andres/.ros/LidarData.csv'): np.savetxt('LidarData.csv', DefLidarData, delimiter=",")
	for i, data in enumerate(Def_Lidar_Data):
		with open('LidarData.csv', 'a') as file:
			writer = csv.writer(file)
			writer.writerow(data)
	print("Lidar data is saved!")
	print("Execution time for true data:", time.process_time() - start)
	quit()
