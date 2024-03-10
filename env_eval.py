from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from agent import Agent
import utils
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = utils.EnvUtils.wrapper(env)
import numpy as np

start_state = utils.EnvUtils.lazyframe_to_tensor(env.reset())
prev_state = start_state
EPISODES = 50000
mArIo = Agent(env.observation_space.shape, len(SIMPLE_MOVEMENT), eval=True)
episodes_rewards = np.empty([])

for episode in range(EPISODES):  # 5000 steps max, you can change this to any number you want
    done = False
    prev_state = env.reset()
    episode_reward = 0.0
    while not done:
        action = mArIo.choose_action(utils.EnvUtils.lazyframe_to_tensor(prev_state), eval=True)  # take a random action
        next_state, reward, done, info = utils.EnvUtils().frames_skipper(env, action)
        prev_state = next_state
        episode_reward += reward
        env.render()  # display the environment state
    print(f"Episode: {episode}")
    episodes_rewards = np.append(episodes_rewards, episode_reward)
    if episode % 10 == 0:#update/sync target network with online every 10 episodes
        episodes_avg_reward = np.sum(episodes_rewards)/len(episodes_rewards)
        episodes_rewards = np.empty([])
        print(f"Average Episode Reward: {episodes_avg_reward}")

env.close()
