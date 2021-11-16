# coding=utf-8
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def time_converter(tim_size, time, target_time):
    if tim_size == 168:
        return time, target_time
    if tim_size == 48:
        for i, time_window in enumerate(time):
            if target_time[i] < 120:
                target_time[i] = target_time[i] % 24
            else:
                target_time[i] = 24 + target_time[i] % 24
            for j, time_point in enumerate(time_window):
                if time_point < 120:
                    time[i][j] = time[i][j] % 24
                else:
                    time[i][j] = 24 + time[i][j] % 24

    # weekdays & weekends are the same
    if tim_size == 24:
        for i, time_window in enumerate(time):
            target_time[i] = target_time[i] % 24
            for j, time_point in enumerate(time_window):
                time[i][j] = time[i][j] % 24

    return time, target_time


class SimpleRNN(nn.Module):
    def __init__(self, parameters):
        super(SimpleRNN, self).__init__()
        self.use_time = parameters.use_time
        self.use_cuda = parameters.use_cuda
        self.use_user = parameters.use_user
        self.use_target_time = parameters.use_target_time
        self.use_dropout = parameters.use_dropout
        self.loc_size = parameters.loc_size
        self.tim_size = parameters.tim_size
        self.window_size = parameters.window_size
        self.uid_size = parameters.uid_size
        self.loc_emb_size = parameters.loc_emb_size
        self.uid_emb_size = parameters.uid_emb_size
        self.batch_size = parameters.batch_size
        self.hidden_size = parameters.hidden_size

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_time = nn.Embedding(self.tim_size, self.loc_emb_size)
        self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)
        self.rnn_type = parameters.rnn_type
        self.num_layers = parameters.num_layers

        input_size = self.loc_emb_size * 2 if self.use_time else self.loc_emb_size
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, self.hidden_size, num_layers=self.num_layers, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, self.hidden_size, self.num_layers, batch_first=True)

        fc_in_size = self.hidden_size
        fc_in_size += self.loc_emb_size if self.use_target_time else 0
        fc_in_size += self.uid_emb_size if self.use_user else 0
        # self.fc = nn.Linear(fc_in_size, self.loc_size)
        self.fc1 = nn.Linear(fc_in_size, 256)
        self.fc2 = nn.Linear(256, self.loc_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)

    def forward(self, loc, u_id, time, target_time, seq_len=None, category=None):
        time, target_time = time_converter(self.tim_size, time, target_time)
        if self.use_user:
            uid_emb = self.emb_uid(u_id)
        if self.use_time:
            time_emb = self.emb_time(time)
        if self.use_target_time:
            target_time_emb = self.emb_time(target_time)

        h1 = Variable(torch.zeros(self.num_layers, loc.size(0), self.hidden_size))
        c1 = Variable(torch.zeros(self.num_layers, loc.size(0), self.hidden_size))
        if self.use_cuda:
            h1 = h1.cuda()
            c1 = c1.cuda()

        x = self.emb_loc(loc)  # [window_size, batch_size, emb_size]

        if self.use_time:
            x = torch.cat((x, time_emb), 2)

        if self.use_dropout:
            x = self.dropout(x)

        if seq_len is not None:
            x = pack_padded_sequence(x, seq_len, enforce_sorted=False, batch_first=True)

        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            # rnn outputs (all), the last hidden state
            out, h1 = self.rnn(x, h1)
            # out:[window_size, batch_size, hidden_size]
        elif self.rnn_type == 'LSTM':
            out, (h1, c1) = self.rnn(x, (h1, c1))

        out, seq_len = pad_packed_sequence(out, batch_first=True)
        seq_len -= 1
        out_new = torch.zeros((out.size(0), out.size(2)))
        out_new = out_new.cuda() if self.use_cuda else out_new
        for i in range(out.size(0)):  # get the last output
            out_new[i] = out[i][seq_len[i]]
        # loc=[window_size, batch_size, hidden_size]-> loc=[batch_size, window_size, hidden_size]
        
        out = out_new
        out = out.view(out.size(0), -1)
        out = torch.cat((out, target_time_emb), 1) if self.use_target_time else out
        out = torch.cat((out, uid_emb), 1) if self.use_user else out
        if self.use_dropout:
            out = self.dropout(out)
        y = self.fc2(F.relu(self.fc1(out)))
        score = F.log_softmax(y, dim=1)
        return score


class MLP(nn.Module):
    def __init__(self, parameters):
        super(MLP, self).__init__()
        self.target_num = parameters.target_num
        self.use_cuda = parameters.use_cuda
        self.use_user = parameters.use_user
        self.use_target_time = parameters.use_target_time
        self.use_dropout = parameters.use_dropout
        self.loc_size = parameters.loc_size
        self.tim_size = parameters.tim_size
        self.window_size = parameters.window_size
        self.uid_size = parameters.uid_size
        self.loc_emb_size = parameters.loc_emb_size
        self.uid_emb_size = parameters.uid_emb_size
        self.batch_size = parameters.batch_size

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_time = nn.Embedding(self.tim_size, self.loc_emb_size)
        self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)

        input_size = self.loc_emb_size * self.window_size
        input_size += self.uid_emb_size if self.use_user else 0
        input_size += self.loc_emb_size * self.target_num if self.use_target_time else 0
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            # nn.ReLU(),
            # nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.target_num * self.loc_size),
        )
        self.dropout = nn.Dropout(p=parameters.dropout_p)
    
    def forward(self, loc, u_id, time, target_time, category_all):
        x = self.emb_loc(loc)
        x = x.view(x.size(0), -1)

        if self.use_target_time:
            time, target_time = time_converter(self.tim_size, time, target_time)
            target_time_emb = self.emb_time(target_time)
            target_time_emb = target_time_emb.view(target_time_emb.size(0), -1)
            x = torch.cat((x, target_time_emb), 1)

        if self.use_user:
            uid_emb = self.emb_uid(u_id)
            uid_emb = uid_emb.view(uid_emb.size(0), -1)
            x = torch.cat((x, uid_emb), 1)

        if self.use_dropout:
            x = self.dropout(x)

        x = self.fc(x)

        x = x.view(x.size(0) * self.target_num, -1)

        score = F.log_softmax(x, dim=1)
        return score
