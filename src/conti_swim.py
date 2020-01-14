import math
import gym
from gym import spaces
class ContiSwimEnv(gym.Env):
    def __init__(self,*args):
        #step_coef = 0.2,key_A = 0.5, key_f = 2.0, 
        #reward_width = 0.1, state_reward = 10.0, 
        #action_reward = 1.0, obs_coef = 1.0
        p_list = [0.2, 0.5, 1.0, 0.1, 10.0, 1.0, 1.0]
        for args_index in range(min(len(args),len(p_list))):
            p_list[args_index] = args[args_index]
        assert 0.0 < p_list[3] < 1.0
        self.step_coef = p_list[0]
        self.key_A = p_list[1]
        self.key_f = p_list[2]
        self.reward_yeta = 1.0/p_list[3]-1.0
        self.state_reward = p_list[4]
        self.action_reward = p_list[5]
        self.obs_coef = p_list[6]
        self.last_state = 0
        self.observation_space = spaces.Box(low = 0, high = 1*self.obs_coef, shape = [1])
        self.action_space = spaces.Box(low = -1, high = 1, shape = [1])

    def reset(self):
        self.last_state = 0
        return self._obs(self.last_state)

    def step(self,action):
        key = self.key_A * math.sin(2 * math.pi * self.key_f *self.last_state)
        effect = math.cos(math.pi * (action + key))
        n_state = min(max(self.last_state + self.step_coef * (effect ) , 0.0), 1.0)
        reward = self.state_reward * (self.last_state**self.reward_yeta) + self.action_reward * (effect)
        done = False
        info = {}
        self.last_state = n_state
        return self._obs(n_state),reward,done,info

    def _obs(self,state):
        return state*self.obs_coef