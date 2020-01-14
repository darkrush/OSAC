from gym.spaces import Box

class ContiEnv(object):
    def __init__(self,action_coef = 0.02):
        self.observation_space = Box(low = 0, high = 1, shape = [1])
        self.action_space = Box(low = -1, high = 1, shape = [1])
        self.last_state = 0
        self.step_cost = -0.2
        self.action_cost = -2.0
        self.reach_reward = 30
        self.action_coef = action_coef
        self.reach_state_bound = (0.9,1.1)
        self.reach_action_bound = (0.3,0.7)
        pass

    def reset(self):
        self.last_state = 0
        return self.last_state

    def step(self,action):
        n_state = self._state_trans(action)
        reward = self._calc_reward(action)
        done = False
        info = {}
        self.last_state = n_state
        return n_state,reward,done,info

    def _state_trans(self,action):
        #if (self.reach_state_bound[0] <= self.last_state <=self.reach_state_bound[1] and
        #    self.reach_action_bound[0] <= action <= self.reach_action_bound[1]):
        #    n_state = 0
        #else:
        n_state = self.last_state + action*self.action_coef
        n_state = max(0,(min(1,n_state)))
        return n_state

    def _calc_reward(self,action):
        if (self.reach_state_bound[0] <= self.last_state <=self.reach_state_bound[1] and
            self.reach_action_bound[0] <= action <= self.reach_action_bound[1]):
            re = self.reach_reward
        else:
            re = self.action_cost*action 
        re += self.step_cost
        return re