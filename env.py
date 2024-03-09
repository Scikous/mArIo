from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
from model import cnn
import torch
import numpy as np

# Optionally apply wrappers in the desired order:
env = GrayScaleObservation(env)    # Convert to grayscale
env = ResizeObservation(env, shape=(84, 84)) 
env = FrameStack(env, num_stack=4)  # Stack 4 frames
obs = env
print(env.observation_space.shape)
test = cnn(env.observation_space.shape, len(SIMPLE_MOVEMENT))
done = True
# for step in range(10000):  # 5000 steps max, you can change this to any number you want
#     if done:
#         state = env.reset()
#     action = env.action_space.sample()  # take a random action
#     state, reward, done, info = env.step(action)
#     #state = torch.tensor()
#     d = torch.from_numpy((np.array(state))).unsqueeze(0).float().squeeze(-1)
#     print(info)#test(d))
#     env.render()  # display the environment state
#     if done:
#         break
env.close()
