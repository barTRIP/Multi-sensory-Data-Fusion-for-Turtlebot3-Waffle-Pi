#!/usr/bin/env python3
#Import libraries for Ultrasonic GPIO and TIME parameters.
import RPi.GPIO as GPIO
import time
import sys
import signal
import rospy
from datetime import datetime
from std_msgs.msg import Float32

def signal_handler(signal, frame): #ctrl + c --> exit program
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

class sonar():
    def __init__(self):
        rospy.init_node('R_sonar', anonymous=True)
        self.distance_publisher = rospy.Publisher('/R_sonar_dist', Float32, queue_size = 1)
        self.r = rospy.Rate(10)
    def dist_sendor (self, dist):
        data = Float32()
        data.data = dist
        self.distance_publisher.publish(data)

#Create array to store readings.
Dist=[]

#Set the pin numbers to be the "channel numbers.
GPIO.setmode(GPIO.BCM)

#Disable warnings if more circuits are connected to the raspberry Pi.
GPIO.setwarnings(False)

#Declare the trigger and echo pins of the dual sensor setup.
TRIG_R = 17
ECHO_R = 27

#Configure the GPIO as inputs and outputs.
GPIO.setup(TRIG_R,GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(ECHO_R,GPIO.IN)

sensor_R = sonar()

i = 0
n = 0

try:
    while True:
        for i in range (5):
            GPIO.output(TRIG_R, False)
            time.sleep(0.05)
            GPIO.output(TRIG_R, True)
            time.sleep(0.00001)
            GPIO.output(TRIG_R, False)

            start_time = datetime.now()
            while GPIO.input(ECHO_R)==0:
                pulse_start_S1 = time.time()
                time_delta = datetime.now() - start_time
                if time_delta.total_seconds()>=5:
                    break
            while GPIO.input(ECHO_R)==1:
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
            round(Med, 2)
        print("Distance of right Sensor:", Med, "cm")
        del Dist[:]
        sensor_R.dist_sendor(Med)
        sensor_R.r.sleep()
except (KeyboardInterrupt, SystemExit):
    GPIO.cleanup()
    sys.exit(0)
except:
    GPIO.cleanup()




