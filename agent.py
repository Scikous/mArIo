import torch.nn as nn
from model import Network
import torch.optim as optim

import torch
import numpy as np


#Deep-Q-Learning handler
class Agent:
    def __init__(self, frames, num_actions, batch_size=1):

        #learning parameters
        self.gamma = 1.0
        self.min_gamma = 0.005
        self.gamma_decay = 5e-5
        self.learning_rate = 5e-4
        #initialize networks
        self.online_network = Network(frames, num_actions, batch_size)
        self.target_network = Network(frames, num_actions, batch_size)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.learning_rate)
        self.optimizer.zero_grad()

        
    def deep_q_trainer(self, prev_states, next_states, actions, rewards, dones):
        pred_q_vals = self.online_network(prev_states)
        pred_q_vals = pred_q_vals[torch.arange(32), actions.squeeze()].unsqueeze(-1) #get corresponding q value to action taken
        target_actions_q_vals = self.target_network(next_states)
        target_actions_q_vals = target_actions_q_vals.max(dim=1, keepdim=True)[0]#get highest predicted q value for each sample
        target_actions_q_vals = rewards + self.gamma*target_actions_q_vals * (1 - dones)

        print(target_actions_q_vals.shape, pred_q_vals.shape)#, pred_actions_q_vals.shape)
        loss = self.loss(pred_q_vals, target_actions_q_vals)
        loss.backward()
        self.optimizer.zero_grad()
        self.optimizer.step()
