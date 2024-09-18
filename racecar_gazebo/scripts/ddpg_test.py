#!/usr/bin/env python

import rospy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import random
import time
from threading import Thread
import GPUtil
import psutil
from DDPG import QNetwork, RacecarEnv  # Güncellenmiş kısımdan import

class ReplayBuffer:
    def __init__(self, buffer_limit, device):
        self.buffer = deque(maxlen=buffer_limit)
        self.device = device

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

        s_batch = torch.tensor(s_lst, dtype=torch.float).to(self.device)
        a_batch = torch.tensor(a_lst, dtype=torch.float).to(self.device)
        r_batch = torch.tensor(r_lst, dtype=torch.float).to(self.device)
        s_prime_batch = torch.tensor(s_prime_lst, dtype=torch.float).to(self.device)
        done_batch = torch.tensor(done_mask_lst, dtype=torch.float).to(self.device)

        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def size(self):
        return len(self.buffer)

class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay  # GPUtil arasında bekleme süresi
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            print("CPU kullanımı: ", psutil.cpu_percent())
            print("RAM kullanımı: ", psutil.virtual_memory()[2], "\n")
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True

if __name__ == '__main__':
    GPU_CPU_monitor = Monitor(60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_network = QNetwork(input_size=2, hidden_size=64, output_size=2).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    
    env = RacecarEnv()
    replay_buffer = ReplayBuffer(100000, device)

    EPISODE = 2000
    MAX_STEP_SIZE = 3000
    epsilon = 0.1
    gamma = 0.99
    batch_size = 64

    for episode in range(EPISODE):
        state = torch.tensor(env.reset(), dtype=torch.float32).to(device)  # Reset edildiğinde yeni episode başlıyor
        done = False
        total_reward = 0

        while not done:  # Bölüm (episode) bitene kadar devam et
            if random.random() > epsilon:
                action = q_network(state).argmax().item()  # Aksiyon seç
            else:
                action = random.choice([0, 1])  # Rastgele aksiyon seç

            next_state, reward, done = env.step(action)  # Adım at ve done bayrağını kontrol et
            next_state = torch.tensor(next_state, dtype=torch.float32).to(device)

            # Replay buffer'a ekle
            replay_buffer.put((state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done))

            state = next_state
            total_reward += reward

            # Replay buffer dolduğunda öğrenme yap
            if replay_buffer.size() > batch_size:
                s_batch, a_batch, r_batch, s_prime_batch, done_batch = replay_buffer.sample(batch_size)

                q_values = q_network(s_batch).gather(1, a_batch.long().unsqueeze(1)).squeeze(1)
                next_q_values = q_network(s_prime_batch).max(1)[0].detach()
                targets = r_batch + gamma * next_q_values * done_batch

                loss = F.mse_loss(q_values, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Episode {episode} finished. Total Reward: {total_reward}")