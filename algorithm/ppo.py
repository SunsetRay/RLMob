import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
import time
import datetime
from environment import MobilityEnv
import numpy as np
from run import Parameter
from algorithm.eval import eval_on_test_dataset, eval_one_episode
from models.model import MLP_State, MLP2_State

d = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
gamma         = Parameter.gamma
lmbda         = Parameter.gae_lmbda
eps_clip      = 0.10
K_epoch       = 3
T_horizon     = 3
reward_scale  = Parameter.reward_scale

class PPO(nn.Module):
    def __init__(self, lr):
        super(PPO, self).__init__()
        self.data = []
        
        self.env = MobilityEnv()
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n  # discrete
        self.fc1   = nn.Linear(state_dim, 256).to(d)
        # self.fc_pi = nn.Linear(256, action_dim).to(d)
        self.fc_v  = nn.Linear(256, 1).to(d)
        
        self.pi = MLP2_State(state_dim, action_dim).to(d)
        pretrained_model_dict = torch.load('data/'+Parameter.dataset+'/cp/'+Parameter.prt_model_name)
        pi_dict = self.pi.state_dict()
        pi_new_dict = {k: v for k, v in pretrained_model_dict.items(
        ) if k in pi_dict.keys()}
        self.pi.load_state_dict(pi_new_dict)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0)
    
    def v(self, x):
        # x = F.relu(self.fc1(x))
        if Parameter.debug['share_v_pi_first_layer']:
            x = F.relu(self.pi.fc1(x))
        else:
            x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        # s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        a_lst, r_lst, prob_a_lst, done_lst = [], [], [], []
        # s_lst = torch.FloatTensor([]).to(d)
        # s_prime_lst  = torch.FloatTensor([]).to(d)
        s_lst, s_prime_lst = [], []

        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask,prob_a = torch.FloatTensor(s_lst).to(d), torch.LongTensor(a_lst).to(d), \
                                          torch.FloatTensor(r_lst).to(d), torch.FloatTensor(s_prime_lst).to(d), \
                                          torch.FloatTensor(done_lst).to(d), torch.FloatTensor(prob_a_lst).to(d)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.cpu().detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:  # reverse traverse
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(d)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

            self.optimizer.zero_grad()
            # loss.mean().backward(retain_graph=True)
            loss.mean().backward()
            self.optimizer.step()
        
def train(env_name,
          lr,
          hidden_sizes,
          epochs,
          batch_size,
          debug,):
    agent = PPO(lr)
    state_dim = agent.env.observation_space.shape[0]
    action_dim = agent.env.action_space.n  # discrete
    score = 0.0
    actions = []
    print_interval = Parameter.print_interval
    eval_interval = Parameter.eval_interval

    # make writer
    writer_filename = Parameter.log_dir+'ppo_'+datetime.datetime.now().strftime('%m-%d %H:%M:%S')
    writer = SummaryWriter(writer_filename)
    print('tb filename: {}'.format(writer_filename))

    for n_epi in range(epochs):
        s = agent.env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                # T_horizon > 5 means = 5, per episode update
                prob = agent.pi(torch.from_numpy(s).float().to(d))
                # prob = agent.pi(s)
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = agent.env.step(a)
                agent.put_data((s, a, r/reward_scale, s_prime, prob[a].item(), done))

                s = s_prime

                score += r
                actions.append(a)
                if done:
                    break

            if not Parameter.debug['eval_only']:
                agent.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("episode: {}, avg score: {:.1f}".format(n_epi, score/print_interval))
            # print('actions:', actions)
            writer.add_scalar(tag='Training-Episode-Reward', scalar_value=score/print_interval, global_step=n_epi)
            writer.add_text('training_batch_act:', str(actions), n_epi)
            score = 0.0
            actions = []
        
        if (n_epi % eval_interval == 0 and n_epi != 0) or Parameter.debug['eval_only']:
            _ = eval_on_test_dataset(n_epi, agent, writer)
    agent.env.close()
