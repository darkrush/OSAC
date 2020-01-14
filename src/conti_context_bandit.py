import numpy as np
import gym
from gym.spaces import Box

def c1(context,x_data,k = 1.3):
    return k*(x_data-context)**2

def c2(context,x_data,sigma = 0.15):
    return np.exp(-(x_data-context)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

class ContiContextBandit(gym.Env):
    def __init__(self,*args):
        p_list = [1.3, 0.15]
        for args_index in range(min(len(args),len(p_list))):
            p_list[args_index] = args[args_index]
        self.k = p_list[0]
        self.sigma = p_list[1]
        self.last_state = 0
        self.observation_space = Box(low = 0, high = 1, shape = [1])
        self.action_space = Box(low = -1, high = 1, shape = [1])

    def reset(self):
        self.last_state = np.random.uniform(0,1)
        return self.last_state

    def step(self,action):
        n_state = np.random.uniform(0,1)
        reward = self._calc_reward(action)
        done = False
        info = {}
        self.last_state = n_state
        return n_state,reward,done,info

    def _calc_reward(self,action,noise = 0.0):
        reward = c1(self.last_state, action, self.k)
        reward += c2(self.last_state, action, self.sigma)
        reward += np.random.randn()*noise
        return reward