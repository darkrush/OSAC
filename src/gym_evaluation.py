import sys
sys.path.append("..")
from utils.argpaser import Singleton_argpaser as args
from utils.logger import Singleton_logger as logger
from src.task_set import TaskSet
import numpy as np
import torch
import gym
#from a2c_ppo_acktr import utils
#from a2c_ppo_acktr.envs import make_vec_envs

def gym_evaluate(policy, env_name, device, times=1):
    if args.exp_name == 'swim':
        task_set = TaskSet(args.exp_name)
        task_set.set_coef(args)
        eval_envs = task_set.get_env()
    else:
        eval_envs = gym.make(args.exp_name)
    eval_episode_rewards = []

    obs = eval_envs.reset()
    cum_reward = 0
    while len(eval_episode_rewards) < times:
        action= policy(obs)
        # Obser reward and next obs
        obs, reward, done, _ = eval_envs.step(action)
        cum_reward += reward
        if done :
            eval_episode_rewards.append(cum_reward)
            cum_reward = 0
            obs = eval_envs.reset()
    eval_envs.close()
    return sum(eval_episode_rewards)/times
