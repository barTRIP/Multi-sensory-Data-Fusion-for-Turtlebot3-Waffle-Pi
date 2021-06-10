#!/usr/bin/env python3

#Import libraries for Ultrasonic GPIO and TIME parameters.
import RPi.GPIO as GPIO
import time
import rospy
from datetime import datetime
from sensor_msgs.msg import Range


#Create array to store readings.
Dist=[]

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

        for i in range (13):
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
        print("Right", Med)
        data.range = Med
        distance_publisher.publish(data)
        rospy.sleep(1)
if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException: pass






