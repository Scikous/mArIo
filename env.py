from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from model import Network
from agent import Agent
import utils
import torch
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = utils.EnvUtils.wrapper(env)


prev_state = env.reset()
print(env.observation_space.shape)
done = True

batch_size = 32
replay_memory = utils.ReplayMem(max_size=500)
test = Agent(env.observation_space.shape, len(SIMPLE_MOVEMENT), batch_size)



for step in range(10000):  # 5000 steps max, you can change this to any number you want
    if done:
        prev_state = env.reset()
    action = env.action_space.sample()  # take a random action
    next_state, reward, done, info = utils.EnvUtils.frames_skipper(env, action)
    experience = (prev_state, action, reward, next_state, done)
    replay_memory.append(experience)
    prev_state = next_state
    if replay_memory.size >= batch_size:
        prev_states, actions, rewards, next_states, dones = replay_memory.random_sample(batch_size=batch_size) 
    #print(len(states))
        test.deep_q_trainer(prev_states, next_states, actions, rewards, dones)
            #print(test.online_network(next_states))
    #print(states)

    env.render()  # display the environment state
    if done:
        break
env.close()
