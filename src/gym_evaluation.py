import sys
sys.path.append("..")
from utils.argpaser import Singleton_argpaser as args
from utils.logger import Singleton_logger as logger

import numpy as np
import torch
import gym
#from a2c_ppo_acktr import utils
#from a2c_ppo_acktr.envs import make_vec_envs


def gym_evaluate(policy, env_name, device):
    eval_envs = gym.make(args.exp_name)
    eval_episode_rewards = []

    obs = eval_envs.reset()
    cum_reward = 0
    while len(eval_episode_rewards) < 10:
        obs = torch.tensor([[obs]],dtype= torch.float32).to(device)
        with torch.no_grad():
            action= policy(obs)
        action = torch.clamp(action,max = 1.0,min  = -1.0).cpu()
        # Obser reward and next obs
        obs, reward, done, _ = eval_envs.step(action)
        cum_reward += reward
        if done :
            eval_episode_rewards.append(cum_reward)
            cum_reward = 0
            obs = eval_envs.reset()
    return sum(eval_episode_rewards)/10.0
