#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_srvs.srv import Empty
import random
import sys
import warnings
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt
import os
warnings.filterwarnings("ignore")
import time

print('Done!')
delta_t = 0.1
t_final = 2000

class NeuralNet:
    def __init__(self, number_of_state, number_of_action, activation_func, trainable_layer, initializer,
                 list_nn, load_saved_model, number_of_models, dim):
        
        self.activation_func = activation_func
        self.trainable_layer = trainable_layer
        self.init = initializer
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.regularization = 0.0
        self.description = ''
        self.number_of_state = number_of_state
        self.number_of_action = number_of_action
        self.list_nn = list_nn
        self.load_saved_model = load_saved_model
        self.total_layer_no = len(self.list_nn) + 1
        self.number_of_models = number_of_models
        self.loss = 'huber_loss'
        self.model = {}
        self.input = Input(shape=(self.number_of_state,), name='states')
        self.dim = dim

        print('\nCreating RL Agents\n')
        LOut = {}
        for ii in range(self.number_of_models):
            model_name = 'model' + str(ii + 1)
            model_path = os.getcwd() + "/" + model_name + '.keras'
            print(model_path)
            L1 = Dense(self.list_nn[0], activation=self.activation_func,
                       kernel_initializer=self.init, trainable=self.trainable_layer)(self.input)

            for ii in range(1, len(self.list_nn)):
                L1 = Dense(self.list_nn[ii], activation=self.activation_func, trainable=self.trainable_layer,
                           kernel_initializer=self.init)(L1)    

            for dimension in range(self.dim):
                LOut['action' + str(dimension)] = Dense(self.number_of_action, activation='linear', name='action' + str(dimension),
                            kernel_initializer=self.init)(L1)
            
            model = Model(inputs=self.input, outputs=[LOut['action' + str(dimension)] for dimension in range(self.dim)])

            model.compile(optimizer=self.opt, loss=self.loss, metrics=['mse'] * self.dim)

            self.model[model_name] = {
                'model_name': model_name,
                'model_path': model_path,
                'model_network': model,
                'number_of_params': model.count_params()
            }
                                       
            self.model['model1']['best'] = {
                'model_path': {
                    'maxscore': os.getcwd() + "/" + 'best_model_msd' + '.keras',
                    'maxtime': os.getcwd() + "/" + 'best_model_mtd' + '.keras'
                },
                'model_network': {
                    'maxscore': '',
                    'maxtime': ''
                },
                'mtd': False,
                'msd': False,
                'maxtime': 0,
                'maxscore': 0
            }
            if self.load_saved_model:
                if not os.path.exists(self.model['model1']['model_path']):
                    print('There is no model saved to the related directory!')
                else:
                    self.model[model_name]['model_network'] = load_model(self.model['model1']['model_path'])
                    self.model['model1']['best']['model_network']['maxscore'] = load_model(self.model['model1']['model_path'])
                    self.model['model1']['best']['model_network']['maxtime'] = load_model(self.model['model1']['model_path'])
                                            
                    if os.path.exists(self.model['model1']['best']['model_path']['maxscore']):
                        self.model['model1']['best']['model_network']['maxscore'] = load_model(self.model['model1']['best']['model_path']['maxscore'])
                    if os.path.exists(self.model['model1']['best']['model_path']['maxtime']):
                        self.model['model1']['best']['model_network']['maxtime'] = load_model(self.model['model1']['best']['model_path']['maxtime'])
        self.list_of_models = [key for key in self.model.keys()]

    def __describe__(self):
        return self.description
     
    def summary(self):
        for key in self.model.keys():
            self.model[key]['model_network'].summary()
            print('\nModel Name is: ', self.model[key]['model_name'])
            print('\nModel Path is: ', self.model[key]['model_path'])
            print('\nActivation Function is: ', self.activation_func)
            print('\n*******************************************************************************')
        if self.description != '':
            print('\nModel Description: ' + self.__describe__())

class Agent(NeuralNet):
    def __init__(self, number_of_state, number_of_action, dim, activation_func='selu', trainable_layer=True, 
                 initializer='he_normal', list_nn=[300, 150, 50], load_saved_model=False, location='./',
                 buffer=10000, annealing=5000, batch_size=256, gamma=0.9, tau=0.03, number_of_models=2):
        
        super().__init__(number_of_state=number_of_state, number_of_action=number_of_action,
                         activation_func=activation_func, trainable_layer=trainable_layer,
                         initializer=initializer, list_nn=list_nn, load_saved_model=load_saved_model,
                         number_of_models=number_of_models, dim=dim)
        
        self.epsilon = 1.0
        self.location = location
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = buffer
        self.annealing = annealing
        self.replay = []
        self.sayac = 0
        self.tau = tau
        self.state = []
        self.reward = None
        self.newstate = None
        self.done = False
        self.maxtime = 0
        self.maxscore = 0
        self.mtd = False
        self.msd = False
        
    def replay_list(self, actionn):
        if len(self.replay) < self.buffer:
            self.replay.append((self.state, actionn, self.reward, self.newstate, self.done))
        else:
            if self.sayac < (self.buffer - 1):
                self.sayac += 1
            else:
                self.sayac = 0
            self.replay[self.sayac] = (self.state, actionn, self.reward, self.newstate, self.done)

    def remember(self, main_model, target_model):
        model = self.model[main_model]
        target_model = self.model[target_model]
        minibatch = random.sample(self.replay, self.batch_size)

        states_old = np.array([memory[0] for memory in minibatch])
        actions = np.array([memory[1] for memory in minibatch])
        rewards = np.array([memory[2] for memory in minibatch])
        states_new = np.array([memory[3] for memory in minibatch])
        dones = np.array([memory[4] for memory in minibatch])

        Qval_old = model['model_network'].predict(states_old)
        Qval_new = model['model_network'].predict(states_new)
        Qval_trgt = target_model['model_network'].predict(states_new)

        y_train = np.copy(Qval_old)

        for i in range(self.batch_size):
            if dones[i]:
                y_train[i] = rewards[i]
            else:
                maxQ = np.max(Qval_trgt[i])
                y_train[i] = rewards[i] + self.gamma * maxQ

        model['model_network'].fit(states_old, y_train, batch_size=self.batch_size, epochs=1, verbose=1)
        return model

    def train_model(self, epoch):
        self.remember('model1', 'model2')
        if epoch % 10 == 0:
            counter1 = 1
            counter2 = counter1 + 1
            for _ in range(self.number_of_models - 1):
                if counter2 >= self.number_of_models:
                    counter2 = 0
                self.remember(self.list_of_models[counter1], self.list_of_models[counter2])
                counter1 += 1
                counter2 += 1    

    def save_replay(self):
        return self.replay
        
    def save(self, target_time, score, target_score):
        self.model['model1']['model_network'].save(self.model['model1']['model_path'])

        if score > self.maxscore:
            self.maxscore = score
            self.model['model1']['best']['model_network']['maxscore'] = self.model['model1']['model_network']
            self.model['model1']['best']['model_path']['maxscore'] = self.model['model1']['model_path']

        if target_time > self.maxtime:
            self.maxtime = target_time
            self.model['model1']['best']['model_network']['maxtime'] = self.model['model1']['model_network']
            self.model['model1']['best']['model_path']['maxtime'] = self.model['model1']['model_path']

        if self.maxscore > target_score:
            self.msd = True
        if self.maxtime > target_time:
            self.mtd = True
        return self.msd, self.mtd

    def act(self, state):
        return self.model['model1']['model_network'].predict(state)[0]

class Control:
    def __init__(self):
        self.ackermann_msg = AckermannDriveStamped()
        self.pub = rospy.Publisher('/vesc/ackermann_cmd_mux/input/nav_steer', AckermannDriveStamped, queue_size=10)
        rospy.init_node('drive', anonymous=True)
        rospy.Subscriber('/scan', LaserScan, self.callback)
        rospy.Timer(rospy.Duration(0.1), self.timer_callback)
        self.control_period = 0.1
        self.control_timer = rospy.Time.now()

    def callback(self, data):
        self.laser_data = data.ranges

    def timer_callback(self, event):
        if rospy.Time.now() - self.control_timer >= rospy.Duration(self.control_period):
            self.control_timer = rospy.Time.now()
            self.pub.publish(self.ackermann_msg)
    
    def send_command(self, speed, steering_angle):
        self.ackermann_msg.drive.speed = speed
        self.ackermann_msg.drive.steering_angle = steering_angle
        self.pub.publish(self.ackermann_msg)

if __name__ == "__main__":
    agent = Agent(number_of_state=10, number_of_action=2, dim=2)
    control = Control()

    for epoch in range(2000):
        state = np.random.rand(1, 10)  # Example state
        action = agent.act(state)
        control.send_command(action[0], action[1])

    rospy.spin()
