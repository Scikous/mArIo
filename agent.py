import torch.nn as nn
from model import Network
import torch.optim as optim

import torch
import numpy as np
import os


#Deep-Q-Learning handler
class Agent:
    def __init__(self, frames, num_actions, batch_size=1, eval=False):

        #learning parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gamma = 0.87
        self.epsilon = 0.97
        self.eps_decay = 0.995
        self.eps_min = 0.02
        self.learning_rate = 5e-4
        self.num_actions = num_actions

        #initialize networks
        self.online_network = Network(frames, num_actions, batch_size)
        self.target_network = Network(frames, num_actions, batch_size)
        #backprop initialization
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.learning_rate)
        #parameters for saving model
        self.load_model(eval)

    #train network using Deep-Q-Learning    
    def deep_q_trainer(self, prev_states, next_states, actions, rewards, dones):
        pred_q_vals = self.online_network(prev_states)
        action_q_vals = pred_q_vals[torch.arange(32), actions.squeeze()].unsqueeze(-1) #get corresponding q value to action taken
        target_actions_q_vals = self.target_network(next_states)
        target_actions_q_vals = target_actions_q_vals.max(dim=1, keepdim=True)[0]#get highest predicted q value for each sample
        target_actions_q_vals = rewards + self.gamma*target_actions_q_vals * (1 - dones)

        loss = self.loss(action_q_vals, target_actions_q_vals.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    #Update target network to online network, to 
    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def epsilon_decay(self):
        self.epsilon = max(self.epsilon*self.eps_decay, self.eps_min)
    
    #choose either a random action or a specific one based on learned knowledge
    def choose_action(self, state, eval=False):
        if np.random.rand() < self.epsilon and not eval:
            return np.random.randint(self.num_actions)
        else:
            pred_q_vals = self.online_network(state, 1)
            action = torch.argmax(pred_q_vals, dim=1).squeeze().item()
            return action
        
    #save the learning progress
    def save_model_checkpoint(self, episodes_reward=None):
        save_params = {"model_state":self.online_network.state_dict(), "optimizer_state": self.optimizer.state_dict()}
        total_average_reward = 0.0
        if episodes_reward:
            try:
                eps_rewards = np.load("episodes_rewards.npy")
                eps_rewards = np.append(eps_rewards, episodes_reward)
                total_average_reward = np.sum(eps_rewards)
            except Exception as e:
                print(e)
                eps_rewards = np.array([episodes_reward])
            finally:#check if model is saveable
                if eps_rewards[-1] > total_average_reward:
                    torch.save(save_params, f"mArIo_best.pt")
                    print("New Best Model Saved")
                np.save("episodes_rewards", eps_rewards)
        torch.save(save_params, f"mArIo_training.pt")
        print("Successfully saved model")

    def load_episodes_rewards(self):
        episodes_rewards = np.load("episodes_rewards.npy") if os.path.exists("episodes_rewards.npy") else None
        return episodes_rewards
    
    def load_model(self, eval=False):
        try:
            model_details = torch.load(f"mArIo_best.pt")
            self.online_network.load_state_dict(model_details["model_state"])
            if not eval:#optimizer only needed when training
                self.optimizer.load_state_dict(model_details["optimizer_state"])
            self.update_target_network()
            if eval:#avoids training mode
                self.online_network.eval()
            print("Successfully Loaded Model Checkpoint")
        except Exception as e:
            print(e)
    


    