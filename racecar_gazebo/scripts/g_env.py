#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import random
"""
Created on Wed Nov  8 15:13:04 2017
@author: ikaya
"""
import sys
import warnings
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os
warnings.filterwarnings("ignore")
import time
print('Done!')
delta_t             = 0.1
t_final             = 2000

class neuralnet():
    def __init__(self, numberofstate, numberofaction, 
                 activation_func, trainable_layer, initializer,
                 list_nn, load_saved_model, numberofmodels, dim):
        
        self.activation_func  = activation_func
        self.trainable_layer  = trainable_layer
        self.init             = initializer
        self.opt              = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.regularization   = 0.0
        self.description      = ''
        self.numberofstate    = numberofstate
        self.numberofaction   = numberofaction
        self.list_nn          = list_nn
        self.load_saved_model = load_saved_model
        self.total_layer_no   = len(self.list_nn)+1
        self.numberofmodels   = numberofmodels
        self.loss             = 'huber_loss'
        self.model            = {}
        self.input            = Input(shape=(self.numberofstate,), name='states')
        self.dim              = dim

        print('\nCreating RL Agents\n')
        LOut = {}
        for ii in range(self.numberofmodels):
            model_name = 'model'+str(ii+1)
            model_path = os.getcwd()+"/" + model_name + '.keras'
            print(model_path)
            L1 = Dense(self.list_nn[0], activation=self.activation_func,
                       kernel_initializer=self.init, trainable = self.trainable_layer)(self.input)

            for ii in range(1,len(self.list_nn)):
                L1 = Dense(self.list_nn[ii], activation=self.activation_func, trainable = self.trainable_layer,
                           kernel_initializer=self.init)(L1)    

            for dimension in range(self.dim):
                LOut['action'+str(dimension)]  = Dense(self.numberofaction, activation='linear', name='action'+str(dimension),
                            kernel_initializer=self.init)(L1)
            
            model = Model(inputs=self.input, outputs=[LOut['action'+str(dimension)] for dimension in range(self.dim)])
            # print('\n%s with %s params created' % (model_name,model.count_params()))

            optimizer = tf.keras.optimizers.Adam()

            model.compile(optimizer=optimizer, loss=self.loss, metrics=['mse'] * self.dim)

            self.model[model_name] = { 'model_name'    : model_name,
                                       'model_path'    : model_path,
                                       'model_network' : model,
                                       'numberofparams': model.count_params()}
                                       
            self.model['model1']['best'] = { 'model_path'    : {'maxscore' : os.getcwd()+"/" + 'best_model_msd' + '.keras',
                                                               'maxtime'  : os.getcwd()+"/" + 'best_model_mtd' + '.keras'},
                                            'model_network' : {'maxscore' : '','maxtime'  : ''},
                                            'mtd'           : False,
                                            'msd'           : False,
                                            'maxtime'       : 0,
                                            'maxscore'      : 0 }
            if self.load_saved_model:
                if not os.path.exists(self.model['model1']['model_path']):
                    print('There is no model saved to the related directory!')
                else:
                    self.model[model_name]['model_network'] = load_model(self.model['model1']['model_path'])
                    self.model['model1']['best']['model_network']['maxscore'] = load_model(self.model['model1']['model_path'])
                    self.model['model1']['best']['model_network']['maxtime']  = load_model(self.model['model1']['model_path'])
                                            
                    if os.path.exists(self.model['model1']['best']['model_path']['maxscore']):
                        self.model['model1']['best']['model_network']['maxscore'] = load_model(self.model['model1']['best']['model_path']['maxscore'])
                    if os.path.exists(self.model['model1']['best']['model_path']['maxtime']):
                        self.model['model1']['best']['model_network']['maxtime']  = load_model(self.model['model1']['best']['model_path']['maxscore'])
        #             print("Model is loaded.")
        # print('\n-----------------------')
        self.listOfmodels = [key for key in self.model.keys()]
    def __describe__(self):
        return self.description
     
    def summary(self):
        for key in self.model.keys():
            self.model[key]['model_network'].summary()
            print('\nModel Name is: ',self.model[key]['model_name'])
            print('\nModel Path is: ',self.model[key]['model_path'])
            print('\nActivation Function is: ',self.activation_func)
            print('\n*******************************************************************************')
        if self.description != '':
            print('\nModel Description: '+self.__describe__())

class agent(neuralnet):
    def __init__(self, numberofstate, numberofaction, dim, activation_func='selu', trainable_layer= True, 
                 initializer= 'he_normal', list_nn= [300,150, 50], 
                 load_saved_model= False, location='./', buffer= 10000, annealing= 5000, 
                 batchSize= 256, gamma= 0.9, tau= 0.03, numberofmodels= 2):
        
        super().__init__(numberofstate=numberofstate, numberofaction=numberofaction, activation_func=activation_func,
                         trainable_layer=trainable_layer, initializer=initializer,list_nn=list_nn, 
                         load_saved_model=load_saved_model, numberofmodels=numberofmodels, dim=dim)
        
        self.epsilon                  = 1.0
        self.location                 = location
        self.gamma                    = gamma
        self.batchSize                = batchSize
        self.buffer                   = buffer
        self.annealing                = annealing
        self.replay                   = []
        self.sayac                    = 0
        self.tau                      = tau
        self.state                    = []
        self.reward                   = None
        self.newstate                 = None
        self.done                     = False
        self.maxtime                  = 0
        self.maxscore                 = 0
        self.mtd                      = False
        self.msd                      = False
        
    def replay_list(self,actionn):
        if len(self.replay) < self.buffer: #if buffer not filled, add to it
            self.replay.append((self.state, actionn, self.reward, self.newstate, self.done))
        else: #if buffer full, overwrite old values
            if (self.sayac < (self.buffer-1)):
                self.sayac = self.sayac + 1
            else:
                self.sayac = 0
            self.replay[self.sayac] = (self.state, actionn, self.reward, self.newstate, self.done)

    def remember(self, main_model, target_model):
    
        model = self.model[main_model]
        target_model = self.model[target_model]
        minibatch = random.sample(self.replay, self.batchSize)

        states_old = np.array([memory[0] for memory in minibatch])
        actions = np.array([memory[1] for memory in minibatch])
        rewards = np.array([memory[2] for memory in minibatch])
        states_new = np.array([memory[3] for memory in minibatch])
        dones = np.array([memory[4] for memory in minibatch])

        Qval_old = model['model_network'].predict(states_old)
        Qval_new = model['model_network'].predict(states_new)
        Qval_trgt = target_model['model_network'].predict(states_new)

        y_train = np.copy(Qval_old)  # Ensure y_train is a copy of Qval_old, matching its shape

        for i in range(self.batchSize):
            if dones[i]:
                y_train[i] = rewards[i]
            else:
                maxQ = np.max(Qval_trgt[i])
                y_train[i] = rewards[i] + self.gamma * maxQ

        # print(f"y_train shape: {y_train.shape}")

        model['model_network'].fit(states_old, y_train, batch_size=self.batchSize, epochs=1, verbose=1)
        return model

    def train_model(self, epoch):
        '''
        This function is used to train the model. It is called after the agent is done with the episode.

        '''
        # print('\n%s and %s are main and target models, respectively' % ('model1','model2'))
        self.remember('model1','model2')
        # print('Training is done')
        if epoch % 10 == 0:
            counter1 = 1
            counter2 = counter1 + 1
            for _ in range(self.numberofmodels-1):
                if counter2 >= self.numberofmodels:
                    counter2 = 0
                # print('%s and %s are main and tardet models, respectively' % (self.listOfmodels[counter1],self.listOfmodels[counter2]))
                self.remember(self.listOfmodels[counter1],self.listOfmodels[counter2])
                counter1 = counter1 + 1
                counter2 = counter2 + 1    
            # print('Training is done for all models')      
     

        # if len(self.replay) >= self.annealing:        
        #     print('Training is done')
        # else:
        #     print('Training will begin after %d experience replay' % (self.annealing - len(self.replay)))

    def save_replay(self):
        return self.replay
        
    def save(self, target_time, score, target_score):
        self.model['model1']['model_network'].save(self.model['model1']['model_path'])

        if self.mtd:
            self.model['model1']['best']['mtd']                      = self.mtd
            self.model['model1']['best']['maxtime']                  = self.maxtime
            self.model['model1']['best']['model_network']['maxtime'] = load_model(self.model['model1']['model_path'])
            self.model['model1']['best']['model_network']['maxtime'].save(self.model['model1']['best']['model_path']['maxtime'])

        if self.msd:
            self.model['model1']['best']['msd']                       = self.msd
            self.model['model1']['best']['maxscore']                  = self.maxscore
            self.model['model1']['best']['model_network']['maxscore'] = load_model(self.model['model1']['model_path'])
            self.model['model1']['best']['model_network']['maxscore'].save(self.model['model1']['best']['model_path']['maxscore'])


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
        self.agent = agent(numberofstate=3, numberofaction=len(self.actions), dim=1, load_saved_model=False)

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
        msg.drive.steering_angle = np.deg2rad(self.actions[action])  # Convert action (index) to radians for steering angle
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

    for ep in range(episodes):
        state = env.reset()
        before_state = env.reset()
        done = False
        total_reward = 0
        while not done:

            randomnumber = np.random.random()
            if  randomnumber < env.agent.epsilon:
                action = np.random.randint(0,env.agent.numberofaction)
            else:
                action = np.argmax(env.agent.model['model1']['model_network'].predict(state.reshape(1, -1)))
            
            next_state, reward, done = env.step(action, before_state)

            env.agent.state = state
            env.agent.newstate = next_state
            env.agent.reward = reward
            env.agent.done = done
            env.agent.replay_list([action])

            env.agent.epsilon  = 0.05 if env.agent.epsilon<0.05 else env.agent.epsilon - 1 / env.agent.buffer
            print(env.agent.epsilon)
            before_state = state
            if len(env.agent.replay) > env.agent.batchSize:
                env.agent.train_model(ep)
            
            replay = env.agent.save_replay
            state = next_state
            total_reward += reward
            print(reward)
            env.agent.save(target_time=150, score=total_reward, target_score=100000)

        print(f"Episode: {ep}, Total Reward: {total_reward}")

