import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
import time
from environment import MobilityEnv
import numpy as np
from run import Parameter
from sklearn.metrics import f1_score, precision_score, recall_score

d = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_one_episode(agent, max_steps=5):
    """ evaluation episode (set exploration to none & use original query) """
    state = agent.env.reset()
    final_reward, episode_reward, episode_objectives = 0., 0., np.zeros(6)
    info_list = {'ccorrect_bsl': [],
                'ccorrect_ours': [],
                'pcorrect_bsl': [],
                'pcorrect_ours': [],
                'l2dist_bsl': [],
                'l2dist_ours': [],
                'action_bsl': [],
                'action_ours': [],
                'grdtruth': [],
                }

    for step in range(max_steps):
        state = torch.FloatTensor(state).to(d)
        prob = agent.pi(state).detach().cpu().numpy()
        # m = Categorical(prob)
        # action = m.sample().item()
        action = np.argmax(prob)
        # plot_action_histogram("Eval-Actions-histogram", action)
        next_state, reward, done, info = agent.env.step(action)
        episode_reward += reward

            # info processing
        episode_objectives[0] += info['ccorrect_bsl']
        episode_objectives[1] += info['ccorrect_ours']
        episode_objectives[2] += info['pcorrect_bsl']
        episode_objectives[3] += info['pcorrect_ours']
        episode_objectives[4] += info['l2dist_bsl']
        episode_objectives[5] += info['l2dist_ours']

        for k, _ in info_list.items():
            info_list[k].append(info[k])

        if done:
            break
        state = next_state  # update next state
    return episode_reward, episode_objectives, info_list

def eval_on_test_dataset(episode, agent, writer, max_steps=5):
    """Eval: use 20% data at the tail of the dataset as Testset"""
    print("Eval Test Dataset in episodes:", episode)
    eval_delta_list = []
    better_eval, worse_eval, equal_eval = 0, 0, 0
    agent.env.set_eval()

    eval_print_count = 0
    if Parameter.eval_mode == '5k':
        test_data_len = 1 + agent.env.dataset.data_len - agent.env.dataset.test_startpoint
    else:
        test_data_len = len(agent.env.dataset.data_test)

    acc_num_ours = 0
    acc_num_bsl = 0
    ccor_num_ours = 0
    ccor_num_bsl = 0
    l2_sum_ours = 0
    l2_sum_bsl = 0

    grdtruth_all = []
    ours_action_all = []
    bsl_action_all = []

    while not agent.env.dataset.is_eval_finished():
        with torch.no_grad():
            final_reward_eval, ours_objectives, info_list = eval_one_episode(agent, max_steps)
        eval_delta_list.append(final_reward_eval)

            # better, worse and equal
        if final_reward_eval > 0:
            better_eval += 1
        elif final_reward_eval < 0:
            worse_eval += 1
        else:
            equal_eval += 1

            # acc@1, cocin, l2 appending
        acc_num_bsl += ours_objectives[2]
        acc_num_ours += ours_objectives[3]
        ccor_num_bsl += ours_objectives[0]
        ccor_num_ours += ours_objectives[1]
        l2_sum_bsl += ours_objectives[4]
        l2_sum_ours += ours_objectives[5]

            # collecting info for (macro-) f1, precision, recall
        grdtruth_all.extend(info_list['grdtruth'])
        bsl_action_all.extend(info_list['action_bsl'])
        ours_action_all.extend(info_list['action_ours'])

        if eval_print_count % 1000 == 0:
            print('Eval Progress: {}/{}'.format(eval_print_count, int(test_data_len)))
        eval_print_count += 1

    mean_value = sum(eval_delta_list)/(len(eval_delta_list))

        # acc@1, ccor, l2 computing
    acc_num_bsl /= test_data_len * Parameter.target_session_len
    acc_num_ours /= test_data_len * Parameter.target_session_len
    ccor_num_bsl /= test_data_len * Parameter.target_session_len
    ccor_num_ours /= test_data_len * Parameter.target_session_len
    l2_sum_bsl /= test_data_len * Parameter.target_session_len
    l2_sum_ours /= test_data_len * Parameter.target_session_len

        # (macro-) f1, precision, recall computing
    f1_bsl = f1_score(grdtruth_all, bsl_action_all, average='macro')
    f1_ours = f1_score(grdtruth_all, ours_action_all, average='macro')
    prec_bsl = precision_score(grdtruth_all, bsl_action_all, average='macro')
    prec_ours = precision_score(grdtruth_all, ours_action_all, average='macro')
    rec_bsl = recall_score(grdtruth_all, bsl_action_all, average='macro')
    rec_ours = recall_score(grdtruth_all, ours_action_all, average='macro')

    writer.add_scalar(
        tag='Eval-Delta-Mean', scalar_value=mean_value, global_step=episode)  # log
    writer.add_scalar(
        tag='Eval-Better', scalar_value=better_eval, global_step=episode)
    writer.add_scalar(
        tag='Eval-Worse', scalar_value=worse_eval, global_step=episode)

    print("[Eval] better: {}, worse: {}, equal: {}, Eval-Delta-Mean: {:.4f}".format(
        better_eval, worse_eval, equal_eval, mean_value))
    print("acc@1_bsl: {:.4f}, acc@1_ours: {:.4f}, ccor_bsl: {:.4f}, ccor_ours: {:.4f}".format(
        acc_num_bsl, acc_num_ours, ccor_num_bsl, ccor_num_ours))
    print("l2_avg_bsl: {:.4f}, l2_avg_ours: {:.4f}, f1_bsl: {:.4f}, f1_ours: {:.4f}".format(
        l2_sum_bsl, l2_sum_ours, f1_bsl, f1_ours))
    print("prec_bsl: {:.4f}, prec_ours: {:.4f}, rec_bsl: {:.4f}, rec_ours: {:.4f}".format(
        prec_bsl, prec_ours, rec_bsl, rec_ours))

    writer.add_scalar(
        tag='Eval-Acc1-Baseline', scalar_value=acc_num_bsl, global_step=episode)
    writer.add_scalar(
        tag='Eval-Acc1-Ours', scalar_value=acc_num_ours, global_step=episode)
    writer.add_scalar(
        tag='Eval-CCor-Baseline', scalar_value=ccor_num_bsl, global_step=episode)
    writer.add_scalar(
        tag='Eval-CCor-Ours', scalar_value=ccor_num_ours, global_step=episode)
    writer.add_scalar(
        tag='Eval-L2-Baseline', scalar_value=l2_sum_bsl, global_step=episode)
    writer.add_scalar(
        tag='Eval-L2-Ours', scalar_value=l2_sum_ours, global_step=episode)

    writer.add_scalar(
        tag='Eval-F1-Baseline', scalar_value=f1_bsl, global_step=episode)
    writer.add_scalar(
        tag='Eval-F1-Ours', scalar_value=f1_ours, global_step=episode)
    writer.add_scalar(
        tag='Eval-Precision-Baseline', scalar_value=prec_bsl, global_step=episode)
    writer.add_scalar(
        tag='Eval-Precision-Ours', scalar_value=prec_ours, global_step=episode)
    writer.add_scalar(
        tag='Eval-Recall-Baseline', scalar_value=rec_bsl, global_step=episode)
    writer.add_scalar(
        tag='Eval-Recall-Ours', scalar_value=rec_ours, global_step=episode)

    # writer.add_scalars('Eval-Metrics', tag_scalar_dict, episode)
    
    writer.add_text('Test_Info:', str(info_list), episode)
    # only use the last episode at test phase as a reference

    print('============================================')

    agent.env.set_train()
    if Parameter.debug['eval_only']:
        exit(0)
    return mean_value
