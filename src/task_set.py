from .conti_context_bandit import ContiContextBandit
from .conti_swim import ContiSwimEnv
from gym.wrappers import TimeLimit
class TaskSet(object):
    def __init__(self, task_name):
        self.task_name = task_name
        self.task_dict = {'ccb':ContiContextBandit,
                          'swim':ContiSwimEnv}
        assert task_name in self.task_dict.keys()
        self.env = None

    def set_coef(self, args):
        task_coef_key_list = [ key for key in args.__dict__.keys() if key.startswith('task_coef') ]
        task_coef_key_list = sorted(task_coef_key_list,key = lambda item:int(item.lstrip('task_coef')))
        task_coef_list = [args.__dict__[key] for key in task_coef_key_list]
        if self.env is None:
            self.env = self.task_dict[self.task_name](*task_coef_list)

    def get_env(self,max_episode_steps = 100):
        return TimeLimit(self.env, max_episode_steps=max_episode_steps)