import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
from environment import MobilityEnv
from models.model import mlp, MLP2_State
from tensorboardX import SummaryWriter
import time
import datetime
from run import Parameter
from algorithm.eval import eval_on_test_dataset, eval_one_episode


d = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VPG_Agent(nn.Module):
    def __init__(self, lr):
        super(VPG_Agent, self).__init__()
        # make environment, check spaces, get obs / act dims
        self.env = MobilityEnv()
        assert isinstance(self.env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
        assert isinstance(self.env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

        obs_dim = self.env.observation_space.shape[0]
        n_acts = self.env.action_space.n
        # self.pi = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts]).to(d)
        self.pi = MLP2_State(obs_dim, n_acts).to(d)
        pretrained_model_dict = torch.load('data/'+Parameter.dataset+'/cp/'+Parameter.prt_model_name)
        pi_dict = self.pi.state_dict()
        pi_new_dict = {k: v for k, v in pretrained_model_dict.items(
        ) if k in pi_dict.keys()}
        self.pi.load_state_dict(pi_new_dict)

        # make optimizer
        # self.optimizer = Adam([
        #                 {'params': self.pi.parameters(), 'lr': lr},
        #                 {'params': self.env.state_encode_net.parameters(), 'lr': lr}
        #                 ], lr=lr, weight_decay=0)
        self.optimizer = Adam(self.pi.parameters(), lr=lr, weight_decay=0)


def reward_to_go(rews, gamma=Parameter.gamma):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (gamma * rtgs[i+1] if i+1 < n else 0)
    return rtgs / Parameter.reward_scale

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, 
          epochs=50000, batch_size=5000, debug={}, render=False):
    # make writer
    writer_filename = Parameter.log_dir+'vpg_'+datetime.datetime.now().strftime('%m-%d %H:%M:%S')
    writer = SummaryWriter(writer_filename)
    print('tb filename: {}'.format(writer_filename))

    agent = VPG_Agent(lr)

    # make function to compute action distribution
    def get_policy(obs, dim=1):
        logits = agent.pi(obs, dim)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs, dim=0).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        # batch_obs = torch.FloatTensor([]).to(d)
        batch_obs = []
                                # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for reward-to-go weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = agent.env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:
            # save obs
            batch_obs.append(obs.copy())
            # batch_obs = torch.cat((batch_obs, obs.unsqueeze(0)), 0).detach()

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32).to(d))
            obs, rew, done, _ = agent.env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)
                # print('ep{} return: {}'.format(len(batch_rets), ep_ret))

                # the weight for each logprob(a_t|s_t) is reward-to-go from t
                batch_weights += list(reward_to_go(ep_rews))

                # reset episode-specific variables
                obs, done, ep_rews = agent.env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) >= batch_size:  # = max_step, one episode experiences make a batch
                    break

        # take a single policy gradient update step
        agent.optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32).to(d),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32).to(d),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32).to(d)
                                  )
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.pi.parameters(), 3)
        # torch.nn.utils.clip_grad_norm_(agent.env.state_encode_net.parameters(), 3)
        agent.optimizer.step()
        torch.cuda.empty_cache()
        # print('cuda memory:', torch.cuda.memory_allocated())
        # print('batch_act:', batch_acts)
        return batch_loss, batch_rets, batch_lens, batch_acts


    print_interval = 50
    eval_interval = 5000
    # training loop
    for i in range(epochs):
        if not Parameter.debug['eval_only']:
            batch_loss, batch_rets, batch_lens, batch_acts = train_one_epoch()
            if i % print_interval == 0 and i != 0:
                print("episode: {}, avg score: {:.1f}".format(i, np.mean(batch_rets)))
                # print('actions:', actions)
                writer.add_scalar(tag='Training-Episode-Reward', scalar_value=np.mean(batch_rets), global_step=i)
                writer.add_text('batch_act:', str(batch_acts), i)

        if (i % eval_interval == 0 and i != 0) or Parameter.debug['eval_only']:
            _ = eval_on_test_dataset(i, agent, writer)
    
    writer.close()