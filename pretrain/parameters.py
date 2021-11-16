import torch
import numpy as np
import pickle

class Parameters(object):
    def __init__(self):
        # global settings
        self.seed = 1
        self.use_cuda = True
        self.cuda_no = 4
        self.is_debug = False
        self.data_split_rate = 0.005
        self.model = 'MLP'
        self.batch_size = 32
        self.early_stop_lr = 0.4 * 1e-6
        self.is_baseline = True
        self.dataset = 'T'

        # Ablation study
        self.use_time = False
        self.use_user = True
        self.use_target_time = True

        # Fine-tuning settings
        self.use_pretrained_model = False
        self.pretrained_model_epoch = 5
        self.pretrained_model_folder = 'cp'
        self.save_model = True
        # if not save_model, no optimal model would be loaded during training
        self.save_emb_pk = False

        # dataset
        self.window_size = 7
        self.tim_size = 168
        # valid time_size = 168 or 48 or 24
        self.target_num = 5
    
        # model params
        self.loc_emb_size = 64
        self.uid_emb_size = 64

        # norm params
        self.L2 = 1e-6
        self.use_dropout = True
        self.dropout_p = 0.5

        # learning rate
        self.lr_step = 3
        self.lr_decay = 0.5
        self.epoch = 80
        self.clip = 4.0

        # rnn params
        self.rnn_type = 'LSTM'
        self.hidden_size = 128
        self.lr_single = 1e-3
        self.num_layers = 1

        # -------------------init dataset-------------------
        if self.dataset == 'THU':
            self.loc_size = 114
            self.uid_size = 10966
            self.use_user = False
            self.save_emb_name = 'THU'
        if self.dataset == 'T_original':
            self.loc_size = 24321
            self.uid_size = 2293
            self.use_user = True
            self.save_emb_name = 'T_original'
        if self.dataset == 'T':
            self.loc_size = 300
            self.uid_size = 1729
            self.use_user = True
            self.save_emb_name = 'T'
        if self.dataset == 'F':
            self.loc_size = 500
            self.uid_size = 532
            self.use_user = True
            self.save_emb_name = 'F'
        if self.model != 'RNN':
            self.use_target_time = False
        
        with open('data/'+self.dataset+'/pid_district_map.pk', 'rb') as f:
            self.map = pickle.load(f)
        self.emb_data = torch.load('data/'+self.dataset+'/cp/baseline_mlp2.m')
        self.loc_emb_data = self.emb_data['emb_loc.weight'].cpu().numpy()
        
    def is_same_category(self, pid1, pid2):
        if self.map[pid1] == self.map[pid2]:
            return True
        return False
    
    def l2_dist(self, pid1, pid2):
        emb1 = self.loc_emb_data[pid1]
        emb2 = self.loc_emb_data[pid2]
        l2_d = np.linalg.norm(emb1 - emb2)
        assert l2_d >= 0
        return l2_d
        
