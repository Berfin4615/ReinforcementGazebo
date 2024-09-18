#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

# Q-Network tanımı
class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class RacecarEnv:
    def __init__(self):
        self.goal_x, self.goal_y = 4.5, 4.5
        self.position = None
        self.heading = 0
        self.scan = None

        rospy.init_node('racecar_env')

        # Publishers and Subscribers
        self.pub_cmd_vel = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=15)
        self.sub_odom = rospy.Subscriber('/vesc/odom', Odometry, self.get_odometry)
        self.sub_scan = rospy.Subscriber('/racecar/scan', LaserScan, self.get_scan_data)

        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)

    def get_scan_data(self, scan_msg):
        self.scan = scan_msg

    def get_odometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation

    def get_state(self):
        if self.scan is not None:
            left = min(self.scan.ranges[90:180])
            right = min(self.scan.ranges[540:630])
            return np.array([left, right], dtype=np.float32)
        else:
            return np.array([1.0, 1.0], dtype=np.float32)

    def step(self, action):
        # Araç hareket komutu
        msg = AckermannDriveStamped()
        if action == 0:
            msg.drive.steering_angle = -0.4  # Sağa dön
        else:
            msg.drive.steering_angle = 0.4  # Sola dön
        msg.drive.speed = 1.0  # Hız
        self.pub_cmd_vel.publish(msg)

        # Yeni durumu al
        state = self.get_state()

        # Done bayrağını çarpma durumuna göre ayarla
        done = False
        if min(state) < 0.5:  # LIDAR verilerine göre bir engele çarpma durumu
            done = True

        # Aracın sağ ve sol LIDAR farkına göre ödül hesapla
        reward = -abs(state[0] - state[1])  # Sağ ve sol LIDAR mesafesi farkı

        return state, reward, done
    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        self.reset_proxy()
        return self.get_state()