#!/usr/bin/env python3

#Import libraries for Ultrasonic GPIO and TIME parameters.
import RPi.GPIO as GPIO
import tf2_ros
import time
import rospy
from datetime import datetime
from sensor_msgs.msg import Range


#Create array to store readings.
Dist=[]

#Kalman Filter Constants. 
F = 1
B = 0
H = 1
#Q = 0.00001
Q=0.1
#R= 0.14936120
R=0.4

x_hat_k_minus_1 = 1
p_k_minus_1 = 1

#Current temperature in Celsius. 

C_temp=20

#Set the pin numbers to be the "channel numbers.
GPIO.setmode(GPIO.BCM)

#Disable warnings if more circuits are connected to the raspberry Pi.
GPIO.setwarnings(False)

#Declare the trigger and echo pins of the dual sensor setup.
TRIG_L = 17
ECHO_L = 27

#Configure the GPIO as inputs and outputs.
GPIO.setup(TRIG_L,GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(ECHO_L,GPIO.IN)

def talker():
    distance_publisher = rospy.Publisher('/R_sonar_dist', Range, queue_size = 1)
    rospy.init_node('R_sonar', anonymous=True)
    min_range = 0.02
    max_range = 3.5
    while not rospy.is_shutdown():
        data = Range()
        data.header.stamp = rospy.Time.now()
        data.header.frame_id = "/sonarR_link"
        data.radiation_type = 0
        data.field_of_view = 0.26
        data.min_range = min_range
        data.max_range = max_range

        for i in range (20
):
            GPIO.output(TRIG_L, False)
            time.sleep(0.05)
            GPIO.output(TRIG_L, True)
            time.sleep(0.00001)
            GPIO.output(TRIG_L, False)

            start_time = datetime.now()
            while GPIO.input(ECHO_L)==0:
                pulse_start_S1 = time.time()
                time_delta = datetime.now() - start_time
                if time_delta.total_seconds()>=5:
                    break
            while GPIO.input(ECHO_L)==1:
                pulse_end_S1 = time.time()
                time_delta = datetime.now() - start_time
                if time_delta.total_seconds()>=5:
                    break
            pulse_duration_S1 = pulse_end_S1-pulse_start_S1

            distance_S1 = pulse_duration_S1*17150
            distance_S1 = round(distance_S1, 2)

            Dist.append(distance_S1)
            n = len(Dist)
            Dist.sort()
            if n % 2 == 0:
                Med1 = Dist[n//2]
                Med2 = Dist[n//2 -1]
                Med = (Med1+Med2)/2
            else:
                Med = Dist[n//2]
            Med=Med/100.0
            round(Med, 3)
        del Dist[:]
        #Kalman Filter
        #Prediction stage
        global x_hat_k_minus_1
        global p_k_minus_1
        x_hat_k_a_priori = x_hat_k_minus_1
        p_k_a_priori = p_k_minus_1 + Q
        #Innovation Stage
        K_k = p_k_a_priori * (p_k_a_priori + R)
        x_hat_k = x_hat_k_a_priori + K_k * (Med - x_hat_k_a_priori)
        p_k = (1 - K_k) * p_k_a_priori
        #Remember previous x and p values iterations
        x_hat_k_minus_1 = x_hat_k
        p_k_minus_1 = p_k

        FinalDist=x_hat_k
        round(FinalDist, 3)
        if (x_hat_k - Med) < 0.05 and (x_hat_k - Med) > -0.05: 
            data.range = FinalDist
            distance_publisher.publish(data)
            print ("____________")
            print ("right:", x_hat_k, "right unfiltered:", Med)
        else:
            print ("Kalman Filter for Right Sensor is Adjusting")
        rospy.sleep(1)
if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException: pass






