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

batch_size = 32
replay_memory = utils.ReplayMem(max_size=50000)
mArIo = Agent(env.observation_space.shape, len(SIMPLE_MOVEMENT), batch_size)
episodes_rewards = np.empty([])

utils.EnvUtils.episodes_rewards_plotter(mArIo.load_episodes_rewards())

for episode in range(EPISODES):  # 5000 steps max, you can change this to any number you want
    done = False
    prev_state = env.reset()
    episode_reward = 0.0
    while not done:
        action = mArIo.choose_action(utils.EnvUtils.lazyframe_to_tensor(prev_state))  # take a random action
        next_state, reward, done, info = utils.EnvUtils().frames_skipper(env, action)
        experience = (prev_state, action, reward, next_state, done)
        replay_memory.append(experience)
        prev_state = next_state
        if replay_memory.size >= batch_size:
            prev_states, actions, rewards, next_states, dones = replay_memory.random_sample(batch_size=batch_size) 
            mArIo.deep_q_trainer(prev_states, next_states, actions, rewards, dones)
        episode_reward += reward
        #env.render()  # display the environment state
    print(f"Episode: {episode}, Episode Reward: {episode_reward}")
    mArIo.epsilon_decay()
    episodes_rewards = np.append(episodes_rewards, episode_reward)
    if episode % 10 == 0:#update/sync target network with online every 10 episodes
        mArIo.update_target_network()
        mArIo.save_model_checkpoint(episodes_rewards)
        episodes_rewards = np.empty([])

env.close()
