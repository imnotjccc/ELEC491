import gym
import panda_gym
from stable_baselines3 import DDPG, TD3, SAC, HerReplayBuffer
from stable_baselines3.common.buffers import DictReplayBuffer

# env = gym.make("PandaReachOrnDense-v2", render=True)
env = gym.make("PandaReachOrnDense-v2")
log_dir = './tensorboard_log/panda_reach_v2_tensorboard/'

# SAC
model = SAC(policy="MultiInputPolicy", env=env, buffer_size=100000, replay_buffer_class=DictReplayBuffer, verbose=1, tensorboard_log=log_dir)
model.learn(total_timesteps=150000)
model.save("./training_log_sac/avoid_test")