#!/usr/bin/env python3

import rospy
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from std_srvs.srv import Empty

class RacecarEnv:
    def __init__(self):
        self.goal_x, self.goal_y = 4.5, 4.5
        self.position = None
        self.heading = 0
        self.scan = None

        # Initialize ROS node
        rospy.init_node('racecar_env')
        
        # Publishers and Subscribers
        self.pub_cmd_vel = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=15)
        self.sub_odom = rospy.Subscriber('/vesc/odom', Odometry, self.getOdometry)
        self.sub_scan = rospy.Subscriber('/racecar/scan', LaserScan, self.getScanData)

        # Service proxies
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)

    def getScanData(self, scan_msg):
        self.scan = scan_msg

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation

    def getState(self):
        # Check if scan data is available
        if self.scan is not None:
            ranges = self.scan.ranges
            obstacle_min_range = min(ranges[0:90])
            done = (obstacle_min_range < 0.6)
        else:
            done = False  # Or handle this case as needed

        return done


    def step(self):
        self.rate = rospy.Rate(1)
        msg = AckermannDriveStamped()
        msg.drive.speed = 0.5
        msg.drive.steering_angle = 0.4189
        msg.drive.steering_angle_velocity = 1
        msg.drive.acceleration = 0.0
        msg.drive.jerk = 0.0
        self.pub_cmd_vel.publish(msg)
        rospy.loginfo("Published drive command: Speed=%.2f, Steering Angle=%.2f", msg.drive.speed, msg.drive.steering_angle)
        self.rate.sleep()
        
        done = self.getState()
        
        return done

    def reset(self, ep):
        self.time_step = 0
        
        rospy.wait_for_service('gazebo/reset_simulation')
        #time.sleep(0.1)
        try:
            self.reset_proxy()
            #time.sleep(0.1)
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")