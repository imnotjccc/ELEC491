import gym
import panda_gym
from stable_baselines3 import DDPG, TD3, SAC, HerReplayBuffer

env = gym.make("PandaReach-v2")
log_dir = './tensorboard_log/panda_reach_v2_tensorboard/'

# SAC
model = SAC(policy="MultiInputPolicy", env=env, buffer_size=100000, replay_buffer_class=HerReplayBuffer, verbose=1, tensorboard_log=log_dir)
model.learn(total_timesteps=80000)
model.save("./training_log_sac/sac_panda_reach_v2_2cm_0.1_dense_screen")