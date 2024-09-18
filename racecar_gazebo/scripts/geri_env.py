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
            # Calculate the minimum range for left and right
            left_ranges = ranges[90:270]
            right_ranges = ranges[450:630]
            
            min_left_range = min(left_ranges)
            min_right_range = min(right_ranges)
            
            # Reward is based on how close the left and right ranges are to each other
            distance_diff = abs(min_left_range - min_right_range)
            reward = -distance_diff  # Negative distance difference is a reward, so closer is better
            
            done = (min_left_range < 0.6) or (min_right_range < 0.6)
        else:
            reward = 0
            done = False  # Or handle this case as needed

        return reward, done

    def step(self, action, ep):
        self.rate = rospy.Rate(0.2)
        msg = AckermannDriveStamped()
        msg.drive.speed = 0.5  # Linear velocity
        
        # Eğer action bir vektör değilse, doğrudan kullanabilirsiniz
        if isinstance(action, np.ndarray):
            action = action.squeeze()
        
        msg.drive.steering_angle = action # Steering angle
        msg.drive.steering_angle_velocity = 1
        msg.drive.acceleration = 0.0
        msg.drive.jerk = 0.0
        self.pub_cmd_vel.publish(msg)
        
        reward, done = self.getState()
        arrival = 0  # or calculate it as needed
        
        # Return state_prime, reward, done, arrival
        state_prime = self.getState()  # Adjust as needed
        return state_prime, reward, done, arrival



    def reset(self, ep):
        self.time_step = 0
        
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")
        
        rospy.sleep(1)  # Wait a bit for the simulation to reset
        state = self.getState()
        return state
