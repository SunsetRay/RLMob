import numpy as np
import gym
from gym import Env
from gym.envs.registration import EnvSpec
import pickle
import codecs
import logging
import torch
from models.model import StateEncodeNetwork, StateEncodeNetworkFixed
from models.model_baseline import RNN4RecNetwork
from run import Parameter
from copy import deepcopy


d = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MobilityEnv(Env):
    """
    Environment for human mobility prediction simulated by static data
    """
    def __init__(self, l2_dist_scale=Parameter.l2_dist_scale):
        super(MobilityEnv, self).__init__()
        self.action_space = Action.space
        if Parameter.algorithm == 'SAC':
            self.action_space = ContinuousAction.space
        self.observation_space = State.space
        self.spec = EnvSpec('MobilityEnv-v0')

        self.dataset = Dataset(Parameter.dataset, 
                               Parameter.loc_emb_name, 
                               Parameter.tim_emb_name, 
                               Parameter.uid_emb_name)

        self.ground_truth_pid = []
        self.ground_truth_district = []
        self.ground_truth_tid = []
        self.history_pids = []
        self.history_pids_emb = []
        self.baseline_history = []
        self.uid = None

        dim_emb = Parameter.loc_emb_size
        self.state_encode_net_fixed = StateEncodeNetworkFixed(dim_emb).to(d)

        self.baseline_net = RNN4RecNetwork().to(d)
        pretrained_model_dict = torch.load('data/'+Parameter.dataset+'/cp/'+Parameter.prt_model_name)
        baseline_model_dict = torch.load('data/'+Parameter.dataset+'/cp/'+Parameter.bsl_model_name)
        
        fixed_net_dict = self.state_encode_net_fixed.state_dict()
        fixed_net_new_dict = {k: v for k, v in pretrained_model_dict.items(
        ) if k in fixed_net_dict.keys()}
        baseline_net_dict = self.baseline_net.state_dict()
        baseline_net_new_dict = {k: v for k, v in baseline_model_dict.items(
        ) if k in baseline_net_dict.keys()}

        self.state_encode_net_fixed.load_state_dict(fixed_net_new_dict)
        self.state_encode_net_fixed.eval()

        self.baseline_net.load_state_dict(baseline_net_new_dict)
        self.baseline_net.eval()  # for disabling bn and dropout
        self.l2_dist_scale = l2_dist_scale

        self.step_num = 0
        self.last_pid = 0

        self.batch_size = 50
        self.count = 0
        self.is_eval = False
        self.batch_baseline = []

    def set_eval(self):
        self.dataset.set_eval()
        self.is_eval = True
    
    def set_train(self):
        self.is_eval = False

    def step(self, action):
        # step and init
        self.step_num += 1
        this_step_target_pid = self.ground_truth_pid[self.step_num-1]
        this_step_target_pid_emb = np.array(self.dataset.id_to_loc_emb(this_step_target_pid))
        this_step_target_dist_id = self.ground_truth_district[self.step_num-1]
        info = {'ccorrect_bsl': 0,
                'ccorrect_ours': 0,
                'pcorrect_bsl': 0,
                'pcorrect_ours': 0,
                'l2dist_bsl': 0,
                'l2dist_ours': 0,
                'action_bsl': 0,
                'action_ours': 0,
                'grdtruth':0,
                }

        # done
        done = True if self.step_num == len(self.ground_truth_pid) else False

        # state encode
        self.history_pids_emb.append(self.dataset.id_to_loc_emb(action))
        self.history_pids.append(action)

        hx, cx = self.state_encode_net_fixed.init_hidden_states()
        self.history_pids_emb_v = torch.FloatTensor([self.history_pids_emb])  # unsqueeze
        traj_emb, _ = self.state_encode_net_fixed(self.history_pids_emb_v, hx, cx)
        traj_emb = traj_emb[0]

        if not done:
            next_target_time = self.ground_truth_tid[self.step_num]
            next_target_time_emb = self.dataset.id_to_time_emb(next_target_time)
            next_target_time_emb = torch.FloatTensor(next_target_time_emb).to(d)
            state = torch.cat((traj_emb, next_target_time_emb), 0)
            state = torch.cat((state, self.uid_emb), 0) if self.uid is not None else state
        else:
            next_target_time = 0
            next_target_time_emb = torch.zeros(Parameter.loc_emb_size).to(d)
            state = torch.cat((traj_emb, next_target_time_emb), 0)
            state = torch.cat((state, self.uid_emb), 0) if self.uid is not None else state

        # reward
            # reward definition
        def reward_calculation(a, obj):
            """r: reward, a: action"""
                # precision score
            r = Parameter.precision_score if this_step_target_pid == a else 0
            info['pcorrect_'+obj] = 1 if this_step_target_pid == a else 0

                # category score
            if r == 0:
                r = Parameter.category_score if this_step_target_dist_id == self.dataset.get_district_id(a) else 0
                info['ccorrect_'+obj] = 1 if this_step_target_dist_id == self.dataset.get_district_id(a) else 0
            
                # l2 dist punishment
            a_emb = np.array(self.dataset.id_to_loc_emb(a))
            l2_dist = np.linalg.norm(a_emb - this_step_target_pid_emb)  # compute l2 distance
            scaled_l2_dist = self.l2_dist_scale * l2_dist
            info['l2dist_'+obj] = scaled_l2_dist
            r -= scaled_l2_dist

                # other rules
            if obj == 'bsl':
                info['action_bsl'] = a
            else:
                info['action_ours'] = a
            return r

            # RL reward
        rl_reward = reward_calculation(action, 'ours')

            # Baseline reward
        baseline_history_v = torch.LongTensor([self.baseline_history]).to(d)
        target_time = self.ground_truth_tid[self.step_num-1]
        target_time_v = torch.LongTensor([target_time]).to(d)
        with torch.no_grad():
            dl_pid_logits = self.baseline_net(baseline_history_v, target_time_v, self.uid)

        _, idx = dl_pid_logits.data.topk(1)
        baseline_action = int(idx.cpu().numpy()[0][0])
        baseline_reward = reward_calculation(baseline_action, 'bsl')
        self.baseline_history.append(baseline_action)

        self.count += 1

            #  final reward
        reward = rl_reward - baseline_reward if self.is_eval or Parameter.reward_use_bsl else rl_reward

        self.last_pid = action
        info['grdtruth'] = this_step_target_pid
        state = state.cpu().numpy() if Parameter.debug['mode'] != 'end2end' else state
        return (state, reward, done, info)

    def reset(self):
        # init
        self.step_num = 0
        init_data = self.dataset.get(self.is_eval)
        self.ground_truth_pid = deepcopy(init_data['pid_label'][0])
        self.ground_truth_district = deepcopy(init_data['districts_label'][0])
        self.ground_truth_tid = deepcopy(init_data['tid_label'][0])
        self.history_pids = deepcopy(init_data['present_pid'][0])
        self.history_pids_emb = []
        self.baseline_history = deepcopy(init_data['present_pid'][0])
        self.uid = deepcopy(init_data['uids']) if Parameter.use_user else None
        self.uid_emb = self.dataset.id_to_uid_emb(self.uid[0]) if self.uid is not None else None
        self.uid = torch.LongTensor(self.uid).to(d) if Parameter.use_user else None
        self.uid_emb = torch.FloatTensor(self.uid_emb).to(d) if self.uid_emb is not None else None

        self.last_pid = self.history_pids[-1]

        for h_pid in self.history_pids:
            self.history_pids_emb.append(self.dataset.id_to_loc_emb(h_pid))

        # state encode
        hx, cx = self.state_encode_net_fixed.init_hidden_states()
        self.history_pids_emb_v = torch.FloatTensor([self.history_pids_emb])  # unsqueeze
        traj_emb, _ = self.state_encode_net_fixed(self.history_pids_emb_v, hx, cx)

        traj_emb = traj_emb[0]
        target_time_emb = self.dataset.id_to_time_emb(self.ground_truth_tid[self.step_num])
        target_time_emb = torch.FloatTensor(target_time_emb).to(d)

        init_state = torch.cat((traj_emb, target_time_emb), 0)
        init_state = torch.cat((init_state, self.uid_emb), 0) if self.uid is not None else init_state
        
        init_state = init_state.cpu().numpy() if Parameter.debug['mode'] != 'end2end' else init_state
        return init_state
    
    def render(self, mode='human'):
        """ no screen for Human Mobility env to render """
        print("[not supporting rendering]")
        raise NotImplementedError
    
    def rank_and_select_loc(self, action):
        """rank the scores of pids and select the argmax (or by other stretegies)"""
        max_pid = 0
        max_score = -1e6
        score_list = []
        for pid, loc_emb in enumerate(self.dataset.loc_emb_data):
            score = np.dot(action, loc_emb)
            if score > max_score:
                max_pid = pid + 1 - 1
                max_score = score + 1 - 1
            score_list.append(score)
        return max_pid


class Dataset:
    def __init__(self, dataset_name: str, loc_emb_name: str, time_emb_name: str, uid_emb_name: str):
        self.data_train = pickle.load(open('data/'+dataset_name+'/bsl/train.pk', 'rb'), encoding='iso-8859-1')
        self.data_test = pickle.load(open('data/'+dataset_name+'/bsl/test.pk', 'rb'), encoding='iso-8859-1')
        print('{} size: {}, {}'.format(dataset_name, len(self.data_train), len(self.data_test)))

        self.emb_data = torch.load('data/'+dataset_name+'/cp/'+Parameter.bsl_model_name)
        
        self.time_emb_data = self.emb_data['emb_time.weight'].cpu().numpy()
        self.loc_emb_data = self.emb_data['emb_loc.weight'].cpu().numpy()
        self.pid_district_map = pickle.load(open('data/'+dataset_name+'/pid_district_map.pk', 'rb'), encoding='iso-8859-1')
        if Parameter.use_user:
            self.uid_emb_data = self.emb_data['emb_uid.weight'].cpu().numpy()

        self.i = 0  # idx

    def get(self, is_eval: bool=False):
        """randomly choose an index < 80% in training, and > 80% in eval"""
        self.i = np.random.randint(low=0, high=len(self.data_train)) if not is_eval else self.i + 1
        return self.data_test[self.i] if is_eval else self.data_train[self.i]
    
    def set_eval(self):
        self.i = 0
    
    def id_to_loc_emb(self, id:int):
        return self.loc_emb_data[id]
    
    def id_to_time_emb(self, id:int):
        return self.time_emb_data[id]

    def id_to_uid_emb(self, id:int):
        return self.uid_emb_data[id]

    def get_district_id(self, pid:int):
        return self.pid_district_map[pid]
    
    def is_eval_finished(self):
        return True if self.i >= len(self.data_test) - 1 else False


class Action:
    """discrete action space: all pids"""
    num_action = Parameter.loc_size
    space = gym.spaces.Discrete(n=num_action)


class ContinuousAction:
    """continuous action space: dim == loc_emb_size"""
    dim_action = Parameter.loc_emb_size
    space = gym.spaces.Box(
        low=-np.ones(dim_action), high=np.ones(dim_action))


class State:
    """trajectory representation + target tid emb"""
    dim_state = Parameter.traj_emb_size + Parameter.loc_emb_size  # traj_emb + time_emb
    dim_state += Parameter.uid_emb_size if Parameter.use_user else 0
    space = gym.spaces.Box(low=-np.ones(dim_state) * np.inf,
                           high=np.ones(dim_state) * np.inf)