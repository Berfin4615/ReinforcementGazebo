#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import random
import gym
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import pygame

ep_reward_list   = []
number_of_actions      = 1
numberOfStates = 3
best_reward      = -10e8
delta_t             = 0.1
t_final             = 5000
clock               = pygame.time.Clock()
numberofepochs      = 100000
state               = []
newstate            = []
avg_reward_list = []
train               = True
def compile_model(model, optimizer):
    model.compile(optimizer=optimizer, loss='mse')  

class OUActionNoise:
    '''
    This class is used to define the Ornstein-Uhlenbeck process noise.
    '''
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta     = theta
        self.mean      = mean
        self.std_dev   = std_deviation
        self.dt        = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt \
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))
        # Store x into x_prev, Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
    #OUActionNoise(mean=np.zeros(1), std_deviation=float(0.2) * np.ones(1))

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64, num_states=None, num_actions=None):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        self.batch_size      = batch_size
        self.buffer_counter  = 0
        # Instead of list of tuples as the exp.replay concept go, We use different np.arrays for each tuple element
        self.state_buffer      = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer     = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer     = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded, replacing old records
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index]      = obs_tuple[0]
        self.action_buffer[index]     = obs_tuple[1]
        self.reward_buffer[index]     = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_counter           = self.buffer_counter + 1

class Agent(Buffer):
    def __init__(self, actor_network={'nn': [60, 30],
                                      'activation': 'relu',
                                      'initializer': glorot_normal,
                                      'optimizer': Adam(learning_rate=0.001)},
                 critic_network={'nn': [16, 32],
                                 'concat': [16, 60, 30],
                                 'activation': 'relu',
                                 'initializer': glorot_normal,
                                 'optimizer': Adam(learning_rate=0.002)},
                 gamma=0.99,
                 tau=0.005,
                 buffer_capacity=50000,
                 batch_size=64,
                 environment=None,
                 numberOfActions=1,
                 numberOfStates=3,
                 upperBound=300,
                 lowerBound=-300,
                 savelocation=os.getcwd() + "/",
                 loadsavedfile=False,
                 disablenoise=False,
                 annealing=250):
        
        Buffer.__init__(self, buffer_capacity=buffer_capacity, batch_size=batch_size,
                        num_actions=numberOfActions, num_states=numberOfStates)

        # Initialize properties
        self.numberOfStates = numberOfStates
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.gamma = gamma
        self.tau = tau
        self.noise = np.random.normal(0, 1.0)
        self.observation = (None, None, None, None)
        self.models = {}
        self.savelocation = savelocation
        self.loadsavedfile = loadsavedfile
        self.disablenoise = disablenoise
        self.numberOfActions = numberOfActions
        self.annealing = annealing
        self.maxtime = 0.0
        self.maxscore = 0.0
        self.mtd = False
        self.msd = False
        self.upperBound = upperBound
        self.lowerBound = lowerBound
        self.noisevariance = (self.upperBound - self.lowerBound) * 3.0
        self.main_actor_model = self.get_actor(name='MainActorModel')
        self.target_actor_model = self.get_actor(name='TargetActorModel')
        self.main_critic_model = self.get_critic(name='MainCriticModel')
        self.target_critic_model = self.get_critic(name='TargetCriticModel')
        
        # Making the weights equal initially
        self.target_actor_model.set_weights(self.main_actor_model.get_weights())
        self.target_critic_model.set_weights(self.main_critic_model.get_weights())

        # if self.loadsavedfile:
        #     for model_name in ['MainActorModel', 'TargetActorModel', 'MainCriticModel', 'TargetCriticModel']:
        #         if not os.path.exists(self.models[model_name]['model_path']):
        #             print('There is no model saved to the related directory!')
        #         else:
        #             print(f'\n******* loading model file for {model_name} *******\n')
        #             model_network = load_model(self.models[model_name]['model_path'])
        #             # Compile the model manually
        #             compile_model(model_network, self.actor_network['optimizer'] if 'Actor' in model_name else self.critic_network['optimizer'])
        #             self.models[model_name]['model_network'] = model_network
            
        #     # Set the weights of models
        #     self.main_actor_model.set_weights(self.models['MainActorModel']['model_network'].get_weights())
        #     self.target_actor_model.set_weights(self.models['TargetActorModel']['model_network'].get_weights())
        #     self.main_critic_model.set_weights(self.models['MainCriticModel']['model_network'].get_weights())
        #     self.target_critic_model.set_weights(self.models['TargetCriticModel']['model_network'].get_weights())

    def record_buffer(self):
        Buffer.record(self,self.observation)

    def get_actor(self, name='Actor'):
        inputs = Input(shape=(self.numberOfStates,), name=name+'_Stateinput')
        out = Dense(self.actor_network['nn'][0], activation=self.actor_network['activation'], kernel_initializer=glorot_normal())(inputs)
        
        for units in self.actor_network['nn'][1:]:
            out = Dense(units, activation=self.actor_network['activation'], kernel_initializer=glorot_normal())(out)
        
        outputs = Dense(self.numberOfActions, activation='tanh', kernel_initializer=glorot_normal())(out)
        outputs = outputs * self.upperBound
        model = Model(inputs, outputs)
        return model

    def get_critic(self, name='Critic'):
        state_input = Input(shape=(self.numberOfStates,), name=name+'_Stateinput')
        state_out = Dense(self.critic_network['nn'][0], activation=self.critic_network['activation'], kernel_initializer=glorot_normal())(state_input)
        
        for units in self.critic_network['nn'][1:]:
            state_out = Dense(units, activation=self.critic_network['activation'], kernel_initializer=glorot_normal())(state_out)

        action_input = Input(shape=(self.numberOfActions,), name=name+'_Actioninput')
        action_out = Dense(self.critic_network['concat'][0], activation="relu", kernel_initializer=glorot_normal())(action_input)

        concat = Concatenate()([state_out, action_out])
        for units in self.critic_network['concat'][1:]:
            concat = Dense(units, activation=self.critic_network['activation'], kernel_initializer=glorot_normal())(concat)

        outputs = Dense(1, kernel_initializer=glorot_normal())(concat)
        model = Model([state_input, action_input], outputs)
        return model
    def policy(self, state):
        sampled_actions = tf.squeeze(self.main_actor_model(state))
        if self.disablenoise:
            self.noise = np.zeros(1)
            self.action     = [np.squeeze(np.clip(sampled_actions.numpy(), self.lowerBound, self.upperBound))]
        else:
            self.noisevariance = self.noisevariance * 0.999995
            if self.noisevariance <= 0.02:
                self.noisevariance = 0.2
            # Adding noise to action
            self.action = np.random.normal(sampled_actions.numpy(),self.noisevariance)
            self.noise  = self.action - sampled_actions.numpy()
            # We make sure action is within bounds
            self.action     = np.squeeze(np.clip(self.action, self.lowerBound, self.upperBound))

        return np.array(self.action)


    # We compute the loss and update parameters
    def learn(self):
        '''
        This function is used to define the learning process of the agent. 
        The learning process is used to update the weights of the actor and critic models.
        '''
        if self.buffer_counter >= self.annealing:
            # Get sampling range
            record_range  = min(self.buffer_counter, self.buffer_capacity)
            # Randomly sample indices
            batch_indices = np.random.choice(record_range, self.batch_size)
            # Convert to tensors
            state_batch      = tf.convert_to_tensor(self.state_buffer[batch_indices])
            action_batch     = tf.convert_to_tensor(self.action_buffer[batch_indices])
            reward_batch     = tf.convert_to_tensor(self.reward_buffer[batch_indices])
            reward_batch     = tf.cast(reward_batch, dtype=tf.float32)
            next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
    
            with tf.GradientTape() as tape:
                target_actions = self.target_actor_model(next_state_batch)
                y              = reward_batch + self.gamma * self.target_critic_model([next_state_batch, target_actions])
                critic_value   = self.main_critic_model([state_batch, action_batch])
                critic_loss    = tf.math.reduce_mean(tf.math.square(y - critic_value))
            critic_grad = tape.gradient(critic_loss, self.main_critic_model.trainable_variables)
            self.critic_network['optimizer'].apply_gradients(zip(critic_grad, self.main_critic_model.trainable_variables))
            
            with tf.GradientTape() as tape: 
                actions      = self.main_actor_model(state_batch)
                critic_value = self.main_critic_model([state_batch, actions])
                actor_loss   = -tf.math.reduce_mean(critic_value) # Used `-value` as we want to maximize the value given by the critic for our actions
            actor_grad = tape.gradient(actor_loss, self.main_actor_model.trainable_variables)
            self.actor_network['optimizer'].apply_gradients(zip(actor_grad, self.main_actor_model.trainable_variables))
    
            self.update_target()

    def update_target(self):
        '''
        
        '''
        new_weights = []
        target_variables = self.target_critic_model.weights
        for i, variable in enumerate(self.main_critic_model.weights):
            new_weights.append(variable * self.tau + target_variables[i] * (1 - self.tau))

        self.target_critic_model.set_weights(new_weights)

        new_weights = []
        target_variables = self.target_actor_model.weights
        for i, variable in enumerate(self.main_actor_model.weights):
            new_weights.append(variable * self.tau + target_variables[i] * (1 - self.tau))

        self.target_actor_model.set_weights(new_weights)

    def save(self):
        for model in self.models.keys():
            self.models[model]['model_network'].save(self.models[model]['model_path'])
            
            if self.mtd:
                self.models[model]['maxtime']               = self.maxtime
                self.models[model]['model_network_maxtime'] = load_model(self.models[model]['model_path'])
                self.models[model]['model_network_maxtime'].save(self.models[model]['mtd_model_path'])

            if self.msd:
                self.models[model]['maxscore']               = self.maxscore
                self.models[model]['model_network_maxscore'] = load_model(self.models[model]['model_path'])
                self.models[model]['model_network_maxscore'].save(self.models[model]['msd_model_path'])
    def save_model(self):
        for model in self.models.keys():
            model_path = self.models[model]['model_path']
            self.models[model]['model_network'].save(model_path)

            if self.mtd:
                self.models[model]['maxtime'] = self.maxtime
                self.models[model]['model_network_maxtime'] = load_model(model_path)
                self.models[model]['model_network_maxtime'].save(self.models[model]['mtd_model_path'])

            if self.msd:
                self.models[model]['maxscore'] = self.maxscore
                self.models[model]['model_network_maxscore'] = load_model(model_path)
                self.models[model]['model_network_maxscore'].save(self.models[model]['msd_model_path'])

    # Modeli yükleme fonksiyonu
    def load_model(self):
        for model_name in ['MainActorModel', 'TargetActorModel', 'MainCriticModel', 'TargetCriticModel']:
            model_path = self.models[model_name]['model_path']
            if not os.path.exists(model_path):
                print(f'No saved model found for {model_name}!')
            else:
                print(f'Loading model for {model_name}')
                model_network = load_model(model_path)
                # Compile model with respective optimizers
                optimizer = self.actor_network['optimizer'] if 'Actor' in model_name else self.critic_network['optimizer']
                compile_model(model_network, optimizer)
                # Set the loaded network to the model
                self.models[model_name]['model_network'] = model_network
                # Set the weights to the main models
                if 'Main' in model_name:
                    self.main_actor_model.set_weights(model_network.get_weights())
                else:
                    self.target_actor_model.set_weights(model_network.get_weights())



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

    def step(self, action, before_state):
        msg = AckermannDriveStamped()
        msg.drive.speed = 0.5  # Fixed speed
        msg.drive.steering_angle = np.deg2rad(int(action))  # Convert action (index) to radians for steering angle
        self.pub_cmd_vel.publish(msg)

        # Small delay to allow action to take effect
        rospy.sleep(0.1)

        # Get new state and reward
        state, done = self.getState()
        reward = self.calculateReward(state, before_state)
        
        return state, reward, done


    def calculateReward(self, state, before_state):
        left_bin, right_bin, front_bin = state
        b_left_bin, b_right_bin, b_front_bin = before_state
        distance_diff = left_bin - right_bin
        before_left_diff = left_bin - b_left_bin
        before_right_diff = right_bin - b_right_bin
        before_front_diff = front_bin - b_front_bin
        if left_bin < 0.4 or right_bin < 0.4 or front_bin < 0.4:
            return -20  # Ceza, çok yakın
        elif distance_diff < -0.3 and before_left_diff < 0.0:
            print("Sola yakınlaştı!")
            return -1
        elif front_bin < 0.8 and before_front_diff < 0.0:
            print("Öne yakınlaştı!")
            return -1
        elif distance_diff > 0.3 and before_right_diff > 0.0:
            print("Doğru karar alıp sola yakınlaştı!")
            return 1
        elif distance_diff < -0.3 and before_left_diff > 0.0:
            print("Doğru karar alıp sağa yakınlaştı!")
            return 1
        elif distance_diff > 0.3 and before_right_diff < 0.0:
            print("Sağa yakınlaştı!")
            return -1
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


if __name__ == '__main__':
    env = RacecarEnv()
    episodes = 40
    myagent           = Agent(actor_network  = {'nn'          :[300,200],
                                            'activation'  :'relu',
                                            'initializer' :glorot_normal,
                                            'optimizer'   :Adam(learning_rate=0.001)}, 
                          critic_network = {'nn'          :[200,300],
                                            'concat'      :[100,200,50],
                                            'activation'  :'relu',
                                            'initializer' :glorot_normal,
                                            'optimizer'   :Adam(learning_rate=0.002)},
                          loadsavedfile=False,
                          disablenoise=False, environment = env,
                          lowerBound=-3,upperBound=3,
                          numberOfActions=number_of_actions,
                          numberOfStates=numberOfStates,
                          buffer_capacity= 50000, batch_size= 256,
                          tau= 0.005, gamma= 0.1, annealing= 1000)
    if myagent.loadsavedfile:
        myagent.load_model()
    for ep in range(episodes):
        state = env.reset()
        done = False
        action = myagent.policy(state.reshape(1,myagent.numberOfStates))
        before_state = env.reset()
        episodic_reward = 0
        while not done:
            next_state, reward, done = env.step(action, before_state)
            myagent.observation    = (state,action,reward,next_state)
            myagent.record_buffer()
            state = next_state
            episodic_reward = episodic_reward + reward
            myagent.learn()
        if done:
            
            print('\n----- New Epoch ----- Epoch: %s\n' % (ep+1))
        ep_reward_list.append(episodic_reward)
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)
        if avg_reward_list[-1]>best_reward:
            best_reward = avg_reward_list[-1]
            print('saving models')
            myagent.save_model()
        print('-----------------------------------------------------------------')
    # Plotting graph Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    print(best_reward)
    plt.show()
