#!/usr/bin/env python
import pygame
import numpy as np
from z_ddqn import DDQNAgent
from collections import deque
import random, math

import rospy
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty



class RacecarEnv:
    def __init__(self):
        self.scan = None

        # Initialize ROS node
        rospy.init_node('racecar_env')
        
        # Publishers and Subscribers
        self.pub_cmd_vel = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=15)
        self.sub_scan = rospy.Subscriber('/racecar/scan', LaserScan, self.getScanData)

        # Service proxies
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        
        # Q-learning neural network agent
        self.actions = [-20,-10, 0, 10, 20]  # Steering angles (degrees)

    def getScanData(self, scan_msg):
        self.scan = scan_msg

    def getState(self):
        if self.scan is not None:
            ranges = self.scan.ranges
            
            # Taranan alanlar
            left_ranges = ranges[90:270]  # Sol taraftaki mesafeler
            right_ranges = ranges[450:630]  # Sağ taraftaki mesafeler
            front_ranges = ranges[0:90]  # Ön taraftaki mesafeler
            
            # Minimum mesafeleri bulma
            min_left_range = min(left_ranges) if left_ranges else float('inf')
            min_right_range = min(right_ranges) if right_ranges else float('inf')
            min_front_range = min(front_ranges) if front_ranges else float('inf')

            state = np.array([min_left_range, min_right_range, min_front_range])
            done = (min_left_range < 0.4) or (min_right_range < 0.4) or (min_front_range < 0.4)  # End episode if too close to a wall
            return state, done
        else:
            # Eğer `scan` verisi yoksa, üçlü state döndürün
            return np.array([0, 0, 0]), False

    def step(self, action):
        msg = AckermannDriveStamped()
        msg.drive.speed = 0.5  # Fixed speed
        msg.drive.steering_angle = np.deg2rad(self.actions[action])  # Convert action (index) to radians for steering angle
        self.pub_cmd_vel.publish(msg)

        # Small delay to allow action to take effect
        rospy.sleep(0.1)

        # Get new state and reward
        state, done = self.getState()
        reward = self.calculateReward(state)
        
        return state, reward, done

    def calculateReward(self, state):
        left_bin, right_bin, front_bin = state
        distance_diff = left_bin - right_bin
        if left_bin < 0.4 or right_bin < 0.4 or front_bin < 0.4:
            return -20  # Ceza, çok yakın
        # elif distance_diff < -0.3 and before_left_diff < 0.0:
        #     print("Sola yakınlaştı!")
        #     return -1
        # elif front_bin < 0.8 and before_front_diff < 0.0:
        #     print("Öne yakınlaştı!")
        #     return -1
        # elif distance_diff > 0.3 and before_right_diff > 0.0:
        #     print("Doğru karar alıp sola yakınlaştı!")
        #     return 1
        # elif distance_diff < -0.3 and before_left_diff > 0.0:
        #     print("Doğru karar alıp sağa yakınlaştı!")
        #     return 1
        # elif distance_diff > 0.3 and before_right_diff < 0.0:
        #     print("Sağa yakınlaştı!")
        #     return -1
        elif abs(distance_diff) < 0.3:
            return 1  # Ödül, duvarları ortalamış
        else:
            return -(abs(distance_diff)*10)  # Ceza, ortalama mesafeden sapmış


    def reset(self):
        self.time_step = 0
        
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")
        
        try:
            rospy.sleep(1)
        except rospy.exceptions.ROSTimeMovedBackwardsException:
            rospy.logwarn("ROSTime moved backwards, trying to reset the environment again.")
            rospy.sleep(1)
        state, _ = self.getState()
        return state


TOTAL_GAMETIME = 1000 # Max game time for one episode
N_EPISODES = 10000
REPLACE_TARGET = 50 

GameTime = 0 
GameHistory = []
renderFlag = False

ddqn_agent = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=5, epsilon=1.00, epsilon_end=0.10, epsilon_dec=0.9995, replace_target=REPLACE_TARGET, batch_size=512, input_dims=3)

# if you want to load the existing model uncomment this line.
# careful an existing model might be overwritten
#ddqn_agent.load_model()

ddqn_scores = []
eps_history = []

if __name__ == '__main__':

    for e in range(N_EPISODES):
        env = RacecarEnv()
        env.reset() #reset env 

        done = False
        score = 0
        counter = 0
        
        observation_, reward, done = env.step(0)
        observation = np.array(observation_)

        gtime = 0 # set game time back to 0
        
        renderFlag = False # if you want to render every episode set to true

        if e % 10 == 0 and e > 0: # render every 10 episodes
            renderFlag = True

        while not done:

            action = ddqn_agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            observation_ = np.array(observation_)

            # This is a countdown if no reward is collected the car will be done within 100 ticks
            if reward == 0:
                counter += 1
                if counter > 100:
                    done = True
            else:
                counter = 0

            score += reward

            ddqn_agent.remember(observation, action, reward, observation_, int(done))
            observation = observation_
            ddqn_agent.learn()
            
            gtime += 1

            if gtime >= TOTAL_GAMETIME:
                done = True

        eps_history.append(ddqn_agent.epsilon)
        ddqn_scores.append(score)
        avg_score = np.mean(ddqn_scores[max(0, e-100):(e+1)])

        if e % REPLACE_TARGET == 0 and e > REPLACE_TARGET:
            ddqn_agent.update_network_parameters()

        if e % 10 == 0 and e > 10:
            ddqn_agent.save_model()
            print("save model")
            
        print('episode: ', e,'score: %.2f' % score,
              ' average score %.2f' % avg_score,
              ' epsolon: ', ddqn_agent.epsilon,
              ' memory size', ddqn_agent.memory.mem_cntr % ddqn_agent.memory.mem_size)   

run()        