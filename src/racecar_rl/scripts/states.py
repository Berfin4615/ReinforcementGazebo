#!/usr/bin/env python

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from threading import Thread
import math
#Publisher class
class CommandPublisher(Thread):
    def __init__(self):
        super(CommandPublisher, self).__init__()
        self.pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=10)
        self.rate = rospy.Rate(1) #1 Hz

    def run(self):
            rospy.loginfo("Command Publisher Node Started")
            while not rospy.is_shutdown():
                msg = AckermannDriveStamped()
                msg.drive.speed = 0.5
                msg.drive.steering_angle = 0.4189
                msg.drive.steering_angle_velocity = 1
                msg.drive.acceleration = 0.0
                msg.drive.jerk = 0.0
                self.pub.publish(msg)
                rospy.loginfo("Published drive command: Speed=%.2f, Steering Angle=%.2f", msg.drive.speed, msg.drive.steering_angle)
                self.rate.sleep()

#Subscriber class
class LidarSubscriber(Thread):
    def __init__(self):
        super(LidarSubscriber, self).__init__()
        rospy.Subscriber('/racecar/scan', LaserScan, self.callback)
        rospy.loginfo("LIDAR Subscriber Node Initialized")
        self.pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=10)
        self.rate = rospy.Rate(1)  # 1 Hz

    def callback(self, data):
        rospy.loginfo("LIDAR Data Received")
        
        # Process Lidar data
        ranges = data.ranges
        angle_min = data.angle_min
        angle_max = data.angle_max
        angle_increment = data.angle_increment
        rospy.loginfo("Ön: " + str(min(ranges[0:90])))
        rospy.loginfo("Sol ön: " + str(min(ranges[90:180])))
        rospy.loginfo("Sol yan: " + str(min(ranges[180:270])))
        rospy.loginfo("Sol arka: " + str(min(ranges[270:360])))
        rospy.loginfo("Arka: " + str(min(ranges[360:450])))
        rospy.loginfo("Sağ arka: " + str(min(ranges[450:540])))
        rospy.loginfo("Sağ orta: " + str(min(ranges[540:630])))
        rospy.loginfo("Sağ ön: " + str(min(ranges[630:720])))
        
        
        
    def run(self):
        rospy.loginfo("Starting LIDAR Subscriber Spin")
        rospy.spin()

if __name__ == "__main__":
    try:
        rospy.init_node('combined_node', anonymous=True)

        command_publisher = CommandPublisher()
        lidar_subscriber = LidarSubscriber()
        command_publisher.start()
        lidar_subscriber.start()

        command_publisher.join()
        lidar_subscriber.join()

    except rospy.ROSInterruptException:
        rospy.logerr("ROS Interrupt Exception")

