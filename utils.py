import numpy as np
import random
import torch
from collections import deque
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
import matplotlib.pyplot as plt

class ReplayMem:
    def __init__(self, max_size):
        self.replay_memory = deque(maxlen=max_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def size(self):#knowing of the taken memory, very useful
        return len(self.replay_memory)
    #add experience to memory
    def append(self, experience):
        self.replay_memory.append(experience)

    #get a batch of random experiences
    def random_sample(self, batch_size):
        if batch_size > len(self.replay_memory):
            return None
        experiences = random.sample(self.replay_memory, batch_size) 
        prev_states, actions, rewards, next_states, dones = zip(*experiences)  # Unpack experience tuples

        # Convert all elements to tensors using list comprehension, numpy and torch.tensor
        prev_states = EnvUtils.lazyframe_to_tensor(prev_states)
        actions = torch.tensor(actions, dtype=torch.int).to(self.device).unsqueeze(-1)  # Assuming actions are continuous
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(-1)
        next_states = EnvUtils.lazyframe_to_tensor(next_states)
        dones = torch.tensor(dones, dtype=torch.int8).to(self.device).unsqueeze(-1)
        return prev_states, actions, rewards, next_states, dones
    

class EnvUtils:
    @staticmethod
    def wrapper(env):
        env = GrayScaleObservation(env)    # Convert to grayscale
        env = ResizeObservation(env, shape=(84, 84)) 
        env = FrameStack(env, num_stack=4)
        return env
    
    @staticmethod
    def frames_skipper(env, action, skip_frames=4):
        total_reward = 0.0
        for _ in range(skip_frames):
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        return next_state, total_reward, done, info
    
    @staticmethod
    def lazyframe_to_tensor(state):#can't do ops on lazyframes
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.tensor(np.asarray(state), dtype=torch.float32).to(device).squeeze(-1)
        return state
    @staticmethod
    def episodes_rewards_plotter(episodes_rewards):
        episodes = np.arange(0, len(episodes_rewards))
        plt.plot(episodes,episodes_rewards)
        plt.title("Reward over episodes")
        plt.xlabel("Episode")
        plt.ylabel("Episode Reward")
        plt.show()
