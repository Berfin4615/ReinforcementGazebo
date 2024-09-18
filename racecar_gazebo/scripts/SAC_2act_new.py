#!/usr/bin/env python3

import rospy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from tensorboardX import SummaryWriter
from env_SAC_new_test import RacecarEnv
from collections import deque
import random
import time
from threading import Thread
import sys
import os
import shutil
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import GPUtil
import psutil
class ReplayBuffer():
    def __init__(self, buffer_limit, DEVICE):
        self.buffer = deque(maxlen=buffer_limit)
        self.dev = DEVICE

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        s_batch = torch.tensor(s_lst, dtype=torch.float).to(self.dev)
        a_batch = torch.tensor(a_lst, dtype=torch.float).to(self.dev)
        r_batch = torch.tensor(r_lst, dtype=torch.float).to(self.dev)
        s_prime_batch = torch.tensor(s_prime_lst, dtype=torch.float).to(self.dev)
        done_batch = torch.tensor(done_mask_lst, dtype=torch.float).to(self.dev)

        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def size(self):
        return len(self.buffer)
        
class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            print("CPU percentage: ", psutil.cpu_percent())
            print('CPU virtual_memory used:', psutil.virtual_memory()[2], "\n")
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
        

if __name__ == '__main__':
    #rospy.init_node('mobile_robot_sac')
    
    GPU_CPU_monitor = Monitor(60)
    
    date = '0405'
    save_dir = "/root/rl-mobile/src/rl-mobile-robot-SAC/myrobot/src/SAC/saved_model/" + date 
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_dir += "/"

    writer = SummaryWriter('SAC_log/'+date)
    
    env = RacecarEnv()
    
    EPISODE = 2000
    MAX_STEP_SIZE = 3000
    
    sim_rate = rospy.Rate(100)   

    print_once = True

    for EP in range(EPISODE):
        state = env.reset(EP)
        done = False        
        
        for step in range(MAX_STEP_SIZE): #while not done:
            # action, log_prob = agent.choose_action(torch.FloatTensor(state))
            # action = action.detach().cpu().numpy()
            
            #print(a)
            action =  [5.5, 5.0] 
            
            done = env.step()
            
            
            if done:
                state = env.reset(EP)
   