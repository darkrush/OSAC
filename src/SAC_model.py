import torch
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)



class QNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim = 256):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = torch.nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, 1)
        self.LN1 = torch.nn.LayerNorm(hidden_dim)
        self.LN2 = torch.nn.LayerNorm(hidden_dim)

        # Q2 architecture
        self.linear4 = torch.nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = torch.nn.Linear(hidden_dim, 1)
        self.LN3 = torch.nn.LayerNorm(hidden_dim)
        self.LN4 = torch.nn.LayerNorm(hidden_dim)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = self.LN1(torch.nn.functional.relu(self.linear1(xu)))
        x1 = self.LN2(torch.nn.functional.relu(self.linear2(x1)))
        x1 = self.linear3(x1)

        x2 = self.LN3(torch.nn.functional.relu(self.linear4(xu)))
        x2 = self.LN4(torch.nn.functional.relu(self.linear5(x2)))
        x2 = self.linear6(x2)

        return x1, x2

class Q_phi_Network(torch.nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim = 256):
        super(Q_phi_Network, self).__init__()

        # Q1 architecture
        self.linear1 = torch.nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, 1)
        self.LN1 = torch.nn.LayerNorm(hidden_dim)
        self.LN2 = torch.nn.LayerNorm(hidden_dim)

        # Q2 architecture
        self.linear4 = torch.nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = torch.nn.Linear(hidden_dim, 1)
        self.LN3 = torch.nn.LayerNorm(hidden_dim)
        self.LN4 = torch.nn.LayerNorm(hidden_dim)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = self.LN1(torch.nn.functional.relu(self.linear1(xu)))
        phi1 = self.LN2(torch.nn.functional.relu(self.linear2(x1)))
        x1 = self.linear3(phi1)

        x2 = self.LN3(torch.nn.functional.relu(self.linear4(xu)))
        phi2 = self.LN4(torch.nn.functional.relu(self.linear5(x2)))
        x2 = self.linear6(phi2)

        return x1, x2, phi1, phi2

class Q_phi_separate_Network(torch.nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim = 256):
        super(Q_phi_separate_Network, self).__init__()

        # Q1 architecture
        self.linear1 = torch.nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, 1)
        self.linear3_UCB = torch.nn.Linear(hidden_dim, 1)
        self.LN1 = torch.nn.LayerNorm(hidden_dim)
        self.LN2 = torch.nn.LayerNorm(hidden_dim)

        # Q2 architecture
        self.linear4 = torch.nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = torch.nn.Linear(hidden_dim, 1)
        self.linear6_UCB = torch.nn.Linear(hidden_dim, 1)
        self.LN3 = torch.nn.LayerNorm(hidden_dim)
        self.LN4 = torch.nn.LayerNorm(hidden_dim)

        self.apply(weights_init_)

    def forward(self, state, action, policy_inference = False):
        xu = torch.cat([state, action], 1)
        
        x1 = self.LN1(torch.nn.functional.relu(self.linear1(xu)))
        phi1 = self.LN2(torch.nn.functional.relu(self.linear2(x1)))
        x1 = self.linear3(phi1)
        if policy_inference:
            ucb1 = self.linear3_UCB(phi1)
        else:
            ucb1 = self.linear3_UCB(phi1.detach())

        x2 = self.LN3(torch.nn.functional.relu(self.linear4(xu)))
        phi2 = self.LN4(torch.nn.functional.relu(self.linear5(x2)))
        x2 = self.linear6(phi2)
        if policy_inference:
            ucb2 = self.linear6_UCB(phi2)
        else:
            ucb2 = self.linear6_UCB(phi2.detach())

        return x1, x2, phi1, phi2, ucb1, ucb2


class GaussianPolicy(torch.nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim = 256, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = torch.nn.Linear(num_inputs, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = torch.nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = torch.nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = torch.nn.functional.relu(self.linear1(state))
        x = torch.nn.functional.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(torch.nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim = 256, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = torch.nn.Linear(num_inputs, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.mean = torch.nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = torch.nn.functional.relu(self.linear1(state))
        x = torch.nn.functional.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)