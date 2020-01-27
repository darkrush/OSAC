from copy import deepcopy
import torch
import numpy
import os
import gym
import pybullet_envs
import sys
sys.path.append("..")
from utils.argpaser import Singleton_argpaser as args
from utils.logger import Singleton_logger as logger

from src.torch_memory import Memory
from src.task_set import TaskSet
from src.SAC_model import Q_phi_Network,GaussianPolicy,DeterministicPolicy
from src.gym_evaluation import gym_evaluate

DEFAULT_DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class SAC(object):
    def __init__(self, num_inputs, action_space):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.matrix = torch.zeros(args.hidden,args.hidden,dtype = torch.float64).to(DEFAULT_DEVICE)
        self.eig_value , self.eig_vector = torch.symeig(self.matrix, eigenvectors=True, )
        self.dirty_batch = [[],[]]

        self.critic = Q_phi_Network(num_inputs, action_space.shape[0],hidden_dim = args.hidden).to(DEFAULT_DEVICE)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = Q_phi_Network(num_inputs, action_space.shape[0]).to(DEFAULT_DEVICE)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning == True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(DEFAULT_DEVICE)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=DEFAULT_DEVICE)
                self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0],hidden_dim = args.hidden,action_space = action_space).to(DEFAULT_DEVICE)
            self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0],hidden_dim = args.hidden,action_space = action_space).to(DEFAULT_DEVICE)
            self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(DEFAULT_DEVICE).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        action = action.detach().cpu().numpy()[0]
        return action

    def update_matrix(self,state,action):
        self.dirty_batch[0].append([state])
        self.dirty_batch[1].append([action])
        
    def update_parameters(self, memory, updates):

        if len(self.dirty_batch[0])>=100:
            with torch.no_grad():
                state_batch = torch.tensor(self.dirty_batch[0],dtype=torch.float32).squeeze(1).to(DEFAULT_DEVICE)
                action_batch = torch.tensor(numpy.array(self.dirty_batch[1]),dtype=torch.float32)
                action_batch = action_batch.squeeze(1)
                action_batch = action_batch.to(DEFAULT_DEVICE)
                if len(action_batch.size()) == 1:
                    action_batch = action_batch.unsqueeze(1)
                _, _,phi1, _ = self.critic(state_batch, action_batch)
                double_phi = phi1.to(torch.float64)
                p_sum = torch.mm(double_phi.transpose(0,1),double_phi)
                self.matrix=self.matrix+p_sum
                self.eig_value , self.eig_vector = torch.symeig(self.matrix, eigenvectors=True, )
                self.dirty_batch = [[],[]]


        batch = memory.sample(args.batch_size)
        state_batch =      batch['obs0']
        action_batch =     batch['actions']
        next_state_batch = batch['obs1']
        reward_batch =     batch['rewards']
        mask_batch =       batch['terminals1']
        qf1, qf2, phi1, phi2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step

        with torch.no_grad():
            double_phi = phi1.to(torch.float64)
            temp_phi_squre = torch.mm(double_phi, self.eig_vector)**2
            temp_eig_value = self.eig_value+10.0
            UCB = torch.sum(temp_phi_squre/temp_eig_value,1,keepdim = True)
            UCB = args.beta*(UCB**0.5).to(torch.float32)
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target,_,_ = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + UCB + mask_batch * self.gamma * (min_qf_next_target)
            UCB_logs = UCB.mean().cpu()

        

        qf1_loss = torch.nn.functional.mse_loss(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = torch.nn.functional.mse_loss(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        
        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi,_,_ = self.critic(state_batch, pi)

        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(DEFAULT_DEVICE)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), UCB_logs.item()
    
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

def run():
    if args.exp_name == 'swim':
        task_set = TaskSet(args.exp_name)
        task_set.set_coef(args)
        env = task_set.get_env()
    else:
        env = gym.make(args.exp_name)
        if args.sparse_switch:
            env.set_control_coef(0.0)
        else:
            env.set_control_coef(1.0)

    memory = Memory(args.buffer_size,env.action_space.shape,env.observation_space.shape,DEFAULT_DEVICE)
    agent = SAC(env.observation_space.shape[0], env.action_space)

    data_idx = 0
    updates = 0
    last_state = env.reset()
    cum_reward = 0


    for epoch in range(args.epoch_number+1):
        for _ in range(args.rollout_step):
            action = agent.select_action(last_state,eval = False)
            state_n, reward, done, info = env.step(action)

            masks = 1.0
            if done :
                logger.append_data('cum_reward',data_idx,cum_reward)
                cum_reward = 0.0
                state_n = env.reset()
                if 'TimeLimit.truncated' in info.keys():
                    masks = 1.0 if info['TimeLimit.truncated'] else 0.0
                else:
                    masks = 0.0

            if args.exp_name == 'ccb':
                masks = 0.0

            agent.update_matrix(last_state,action)
            
            memory.add(last_state, action, reward, state_n, masks)

            last_state = state_n
            cum_reward+=reward
            data_idx+=1

        for i in range(args.updates_per_step):
            # Update parameters of all the networks
            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha,UCB = agent.update_parameters(memory, updates)
            updates += 1


        def policy(obs):
            action = agent.select_action(obs,eval = True)
            return action
        if (1+epoch)%1000 ==0:
            eval_return = gym_evaluate(policy, args.exp_name, DEFAULT_DEVICE)
            logger.append_data('eval_return',data_idx,eval_return)
            logger.add_log('eval_return: %f, '%eval_return)

        if (1+epoch)%100 ==0:
            logger.append_data('critic_1_loss',data_idx,critic_1_loss)
            logger.append_data('critic_2_loss',data_idx,critic_2_loss)
            logger.append_data('policy_loss',data_idx,policy_loss)
            logger.append_data('ent_loss',data_idx,ent_loss)
            logger.append_data('alpha',data_idx,alpha)
        
            logger.append_data('UCB',data_idx,UCB)
            logger.add_log('epoch %d, '%epoch + 
                           'UCB: %f, '%UCB +
                           'critic_1_loss: %f, '%critic_1_loss +
                           'critic_2_loss: %f, '%critic_2_loss +
                           'policy_loss: %f, '%policy_loss +
                           'ent_loss: %f, '%ent_loss +
                           'alpha: %f, '%alpha)
            logger.dump_log()
            logger.dump_data(True)

    env.close()