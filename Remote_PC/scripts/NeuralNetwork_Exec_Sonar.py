#! /usr/bin/env python3

# Libraries
#import io
import rospy
#from pandas import read_csv
from keras.models import load_model
# Import Numpy for easy array manipulation
import numpy as np
#Import csv to document stuff.
import time
import csv
#Import transformations stuff
#Check use of tf2 instead.
import tf
import tf.transformations
import tf2_ros
from tf2_ros import Buffer
from geometry_msgs.msg import Point, TransformStamped
from tf2_geometry_msgs import PointStamped
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
#Import math stuff
import math
# import ROS stuff
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField, LaserScan, Range
import sensor_msgs.point_cloud2 as pc2
import ros_numpy



# Load model
model = load_model('ANN_sonar_fusion2.h5')

# Summarize model
model.summary()

#Various flags.
XYZ = None
j=0
k=0
a=0
b=0
prev_angle = 0
#Array management variables.
Out_scan = []
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
XYZ



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

# Node initialization
rospy.init_node('NeuralNetwork_Exec')

# Topics subscription
subLidar = rospy.Subscriber('/full_data_scan', LaserScan, callback_Lidar)
subSonarL = rospy.Subscriber('/L_sonar_dist', Range, callback_Sonar_L)
subSonarR = rospy.Subscriber('/R_sonar_dist', Range, callback_Sonar_R)

#Waiting to receive data from topics.
rospy.wait_for_message('/L_sonar_dist', Range)
rospy.wait_for_message('/R_sonar_dist', Range)
rospy.wait_for_message('/full_data_scan', LaserScan)

# Topics publishing
scan_pub = rospy.Publisher('/scan', LaserScan, queue_size = 10)

# While loop
while not rospy.is_shutdown():
	start_overall=time.process_time()
	# Get laser values every second
	ros_rate = rospy.Rate(10) #10Hz
	ros_rate.sleep()

	#Transforming the coordinates from all sensors w.r.t. lidar r.f.
	sonarL_point=PointStamped()
	sonarL_point.header.frame_id = 'sonarL_link'
	sonarL_point.header.stamp =rospy.Time()
	sonarL_point.point.x= Sonar_L
	tf_buffer_SL = tf2_ros.Buffer()
	listenerSonarL = tf2_ros.TransformListener(tf_buffer_SL)
	if tf_buffer_SL.can_transform('base_scan','sonarL_link', rospy.Time(0), rospy.Duration(50)):
		UltSonarL=tf_buffer_SL.transform(sonarL_point, 'base_scan')
	UltSonarL=UltSonarL.point.x

	sonarR_point=PointStamped()
	sonarR_point.header.frame_id = 'sonarR_link'
	sonarR_point.header.stamp =rospy.Time()
	sonarR_point.point.x= Sonar_R
	tf_buffer_SR = tf2_ros.Buffer()
	listenerSonarR = tf2_ros.TransformListener(tf_buffer_SR)
	UltSonarR=tf_buffer_SR.transform(sonarR_point, 'base_scan')
	UltSonarR=UltSonarR.point.x

	def_angle = np.arange(60,120)
	def_angle = np.expand_dims(def_angle, axis=1)

    #Sonar data obtention
	midPoint = (def_angle.size)/2.0
	if UltSonarR > 3.5:
		UltSonarR = 3.6
	if UltSonarL > 3.5:
		UltSonarL = 3.6
	if len(def_angle) % 2 == 0:
		Sonar_R_Data = np.full(int(midPoint), round(UltSonarR, 3), dtype=float)
		Sonar_L_Data = np.full(int(midPoint), round(UltSonarL, 3), dtype=float)
	else:
		Sonar_R_Data = np.full(int(midPoint+1), round(UltSonarR, 3), dtype=float)
		Sonar_L_Data = np.full(int(midPoint), round(UltSonarL, 3), dtype=float)
	Sonar_Data = np.append(Sonar_R_Data, Sonar_L_Data, axis=0)
	Sonar_Data = np.reshape(Sonar_Data, (-1, 1))

	#Lidar data obtention
	lidarData = LidarData[:,np.newaxis]
	#print(lidarData)
	for i in range(len(lidarData)):
		if lidarData[i] == float('+inf') or lidarData[i] == 0:
			lidarData[i] = 3.6
		if i > 269:
			DefLidarData[i-270] = lidarData[i]
		if i < 270:
			DefLidarData[i+90] = lidarData[i]
	DefLidarData = np.array(DefLidarData)
	initial_sect=np.arange(0, def_angle[0])
	final_sect=np.arange(def_angle[0]+len(def_angle), 360)
	extra=np.append(initial_sect, final_sect)
	Def_Lidar_Data=np.delete(DefLidarData, extra.astype(int), axis=0)
	SensorData = np.concatenate((Def_Lidar_Data, Sonar_Data), axis = 1)


	ypredreal = model.predict(SensorData)
	Out_scan= LidarData

	#LiDARFusedDAta = np.concatenate((Def_Lidar_Data, ypredreal), axis = 1)
	#for i, data in enumerate(LiDARFusedDAta):
	#	with open('RMSEDataSonar1.csv', 'a') as file:
	#		writer = csv.writer(file)
	#		writer.writerow(data)
	#print("All data is saved!")

	ypredreal = ypredreal.flatten()
	ypredreal = np.where(ypredreal>3.5,0.0,ypredreal)

	for i in range(0, len(ypredreal)):
		if def_angle[i]<90:
			Out_scan[int(def_angle[i])-1+270]=ypredreal[i]
		else:
			Out_scan[int(def_angle[i])-1-90]=ypredreal[i]

	current_time = rospy.Time.now()

	scan = LaserScan()
	scan.header.stamp = current_time
	scan.header.frame_id = 'base_scan'
	scan.angle_min = 0
	scan.angle_max = 6.2657318115234375
	scan.angle_increment = 0.01745329238474369
	scan.time_increment = 2.990000029967632e-05
	scan.range_min = 0.12
	scan.range_max = 3.5
	scan.ranges = Out_scan
	scan_pub.publish(scan)
	duration=time.process_time()-start_overall
	print("Overall time=",(time.process_time() - start_overall))
	execTime=np.empty((0, 1), float)
	execTime = np.append(execTime, np.array([[duration]]), axis=0)
	for i, data in enumerate(execTime):
		with open('ExecTimesSonar.csv', 'a') as file:
			writer = csv.writer(file)
			writer.writerow(data)
			print("All data is saved!")
