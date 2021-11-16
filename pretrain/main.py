# coding=utf-8
# import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import pickle
import numpy as np
import os
import time
import datetime
import copy

from model import SimpleRNN, MLP
from parameters import Parameters


def run(parameters):
    with open('data/'+parameters.dataset+'/bsl/train.pk', 'rb') as f:
        data_train = pickle.load(f)

    with open('data/'+parameters.dataset+'/bsl/test.pk', 'rb') as f:
        data_test = pickle.load(f)

    if parameters.is_debug:
        percent_split = int(len(data_test) * parameters.data_split_rate)
        data_test = data_test[:percent_split]
        percent_split = int(len(data_train) * parameters.data_split_rate)
        data_train = data_train[:percent_split]
    
    print('data length:', len(data_train), len(data_test))

    if parameters.model == 'RNN':
        model = SimpleRNN(parameters)
        from train_teacher_forcing import Train, Test
        if parameters.is_baseline:
            data_train = dataset_preprocessing(data_train, parameters)
    if parameters.model == 'MLP':
        model = MLP(parameters)
        from train import Train, Test

    # for name in model.state_dict():
    #     print(name)

    print('======Train & Test======')
    criterion = nn.NLLLoss()

    if parameters.use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    single_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=parameters.lr_single,
                                        weight_decay=parameters.L2)
    single_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(single_optimizer,
                                                                  'max',
                                                                  patience=parameters.lr_step,
                                                                  factor=parameters.lr_decay,
                                                                  threshold=1e-3)
    lr_single = parameters.lr_single
    save_path = './cptemp/'
    if (os.path.exists(save_path) == False):
        os.system('mkdir '+save_path)
    accuracy = []
    save_name_tmp_list = []

    for epoch in range(parameters.epoch):
        start_time = time.time()

        train = Train(parameters)
        test = Test(parameters)

        # train
        model, avg_loss = train.train(
            data_train, model, single_optimizer, lr_single, criterion)
        print('==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{}'.format(
            epoch, avg_loss, lr_single))

        # test
        with torch.no_grad():
            avg_acc, test_avg_loss = test.test(
                data_test, model, criterion)
        accuracy.append(avg_acc[2])
        print('==>Test Acc 10:{:.4f} Acc 5:{:.4f} Acc 1:{:.4f} Loss:{:.4f}'.format(
            avg_acc[0], avg_acc[1], avg_acc[2], test_avg_loss))
        print('==>Best Epoch:{:0>2d} Best Acc 1:{:.4f}'.format(
            np.argmax(accuracy), np.max(accuracy)))

        # output time
        end_time = time.time()
        print('==>Epoch time:{}'.format(end_time-start_time))

        single_scheduler.step(avg_acc[2])
        lr_last = lr_single + 1 - 1
        lr_single = single_optimizer.param_groups[0]['lr']
        if parameters.save_model:
            save_name_tmp = str(start_time)+'ep_' + \
                str(epoch) + '.m'
            save_name_tmp_list.append(save_name_tmp)
            torch.save(model.state_dict(),
                       save_path + 'single_' + save_name_tmp)

            if parameters.save_emb_pk:
                save_emb_prefix = 'data/' + parameters.save_emb_name + '_' + datetime.datetime.now().strftime('%m-%d')
                pickle.dump(model.state_dict()['emb_loc.weight'].cpu().numpy(),
                            open(save_emb_prefix+'_loc.pk','wb'), protocol=2)
                pickle.dump(model.state_dict()['emb_time.weight'].cpu().numpy(),
                            open(save_emb_prefix+'_tim.pk','wb'), protocol=2)
                pickle.dump(model.state_dict()['emb_uid.weight'].cpu().numpy(),
                            open(save_emb_prefix+'_uid.pk','wb'), protocol=2)


        if lr_single < parameters.early_stop_lr:
            # early stop
            break

        if lr_last > lr_single and epoch != 0 and parameters.save_model:
            load_epoch0 = np.argmax(accuracy)
            if load_epoch0 != epoch:
                try:
                    load_epoch = save_name_tmp_list[load_epoch0]

                    load_name_tmp = load_epoch
                    print('load epoch={} model state, name={}'.format(
                        load_epoch0, load_name_tmp))

                    model.load_state_dict(torch.load(
                        save_path + 'single_' + load_name_tmp))
                except:
                    print('load error, load epoch={}'.format(load_epoch0))

def dataset_preprocessing(data, parameters):
    new_data = []
    for d in data:
        for i in range(1, parameters.target_num):
            new_d = {}
            new_d['present_pid'] = copy.deepcopy(d['present_pid'])
            new_d['present_pid'][0].extend(d['pid_label'][0][:i])
            new_d['present_pid'][0].extend([0 for j in range(parameters.target_num-1-i)])  # padding
            new_d['pid_label'] = [copy.deepcopy(d['pid_label'][0][i:])]

            new_d['present_tid'] = copy.deepcopy(d['present_tid'])
            new_d['present_tid'][0].extend(d['tid_label'][0][:i])
            new_d['present_tid'][0].extend([0 for j in range(parameters.target_num-1-i)])  # padding
            new_d['tid_label'] = [copy.deepcopy(d['tid_label'][0][i:])]

            new_d['uids'] = copy.deepcopy(d['uids'])
            new_d['seq_len'] = parameters.window_size + i
            new_data.append(new_d)
        d['seq_len'] = parameters.window_size
        d['present_pid'][0].extend([0 for j in range(parameters.target_num-1)])
        d['present_tid'][0].extend([0 for j in range(parameters.target_num-1)])  # padding
        new_data.append(d)
    return new_data


if __name__ == "__main__":
    print('******PID:'+str(os.getpid())+'******')
    para = Parameters()
    np.random.seed(para.seed)
    torch.manual_seed(para.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if para.use_cuda:
        torch.cuda.manual_seed(para.seed)
    # print(torch.cuda.is_available())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(para.cuda_no)
    print('======All Parameters======')
    print('\n'.join(['%s:%s' % item for item in para.__dict__.items()]))
    run(para)
    print('============Done============')
    # nohup python -u pretrain/main.py >pretrain/log/out_200809_T_mlp.log 2>&1 &
