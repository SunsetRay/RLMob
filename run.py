import torch
import click
import warnings

import os

class Parameter:
    algorithm = 'PPO'
    env_name = 'Mob'
    lr = 5e-6
    gamma = 0.9  # 1, 0.98, 0.95, 0.9, 0.8, 0.5, 0
    gae_lmbda = 1  # GAE lmbda 0, 0.5, 0.8, 0.9, 0.95
    batch_size = 5
    hidden_sizes = [256]  # for vpg mlp
    traj_emb_size = 128
    episodes = int(2e6)
    target_session_len = 5
    print_interval = 50
    eval_interval = 5000
    log_dir = '/data/tb_0811/'
    debug = {
        'mode': 'pretrain',  # end2end, pretrain
        'eval_only': False,
        'share_v_pi_first_layer':False,
    }

    # reward
    category_score = 3
    precision_score = 20
    l2_dist_scale = 0.1
    reward_scale = 10.0

    # data
    dataset = 'F'  # T, F, THU (Foursquare-TKY, Foursquare-NYK, Univ-WIFI respectively)
    use_user = True
    reward_use_bsl = True

    # init
    loc_emb_size = 64
    uid_emb_size = 64
    if dataset == 'T':
        loc_size = 300
        uid_size = 1729
        loc_emb_name = 'T_07-30_loc'
        tim_emb_name = 'T_07-30_tim'
        uid_emb_name = 'T_07-30_uid'
        bsl_model_name = 'baseline_mlp2.m'
        prt_model_name = 'baseline_mlp2.m'
    if dataset == 'THU':
        loc_size = 114
        uid_size = 9887
        loc_emb_name = None
        tim_emb_name = None
        uid_emb_name = None
        use_user = False
        bsl_model_name = 'baseline_mlp2.m'
        prt_model_name = 'baseline_mlp2.m'
    if dataset == 'F':
        loc_size = 500
        uid_size = 532
        loc_emb_name = 'F_08-01_loc'
        tim_emb_name = 'F_08-01_tim'
        uid_emb_name = 'F_08-01_uid'
        bsl_model_name = 'baseline_mlp2.m'
        prt_model_name = 'baseline_mlp2.m'


if __name__ == '__main__':
    print('******PID:'+str(os.getpid())+'******')

    # seed = 0  # use fixed random seed to make it reproducible
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    print('=================Parameters=================')
    print('\n'.join(['%s:%s' % item for item in Parameter.__dict__.items()]))
    print('=================Parameters=================')

    gpu_id = 0  # change here to alter GPU id
    print('GPU id:'+str(gpu_id))
    if Parameter.algorithm == 'VPG':  # REINFORCE
        from algorithm.rtg_vpg import train
    if Parameter.algorithm == 'PPO':
        from algorithm.ppo import train

    with torch.cuda.device(gpu_id):
        warnings.filterwarnings("ignore")
        
        train(env_name=Parameter.env_name, 
              lr=Parameter.lr, 
              hidden_sizes=Parameter.hidden_sizes, 
              epochs=Parameter.episodes, 
              batch_size=Parameter.batch_size,
              debug=Parameter.debug)

    print('============Done============')
    # nohup python -u run.py >log/F_ppo.log 2>&1 &