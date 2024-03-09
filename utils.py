import numpy as np
import random
import torch
from collections import deque
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation


class ReplayMem:
    def __init__(self, max_size):
        self.replay_memory = deque(maxlen=max_size)

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
        prev_states = torch.tensor(np.asarray([prev_state for prev_state in prev_states]), dtype=torch.float32).squeeze(-1)
        actions = torch.tensor(np.asarray([action for action in actions]), dtype=torch.int).unsqueeze(-1)  # Assuming actions are continuous
        rewards = torch.tensor(np.asarray([reward for reward in rewards]), dtype=torch.float32).unsqueeze(-1)
        next_states = torch.tensor(np.asarray([next_state for next_state in next_states]), dtype=torch.float32).squeeze(-1)
        dones = torch.tensor(np.asarray([done for done in dones]), dtype=torch.int8).unsqueeze(-1)
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
