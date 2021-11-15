import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.autograd import Variable
from torch.distributions import Categorical
from run import Parameter


d = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class parameters:
    use_dropout = True
    dropout_p = 0.5
    loc_size = Parameter.loc_size
    uid_size = Parameter.uid_size
    tim_size = 168
    window_size = 7
    loc_emb_size = 64
    uid_emb_size = 64
    hidden_size = 128
    batch_size = 1
    rnn_type = 'LSTM'
    num_layers = 1



class RNN4RecNetwork(nn.Module):
    def __init__(self):
        super(RNN4RecNetwork, self).__init__()
        self.use_dropout = parameters.use_dropout
        self.loc_size = parameters.loc_size
        self.uid_size = parameters.uid_size
        self.tim_size = parameters.tim_size
        self.uid_emb_size = parameters.uid_emb_size
        self.loc_emb_size = parameters.loc_emb_size
        self.batch_size = parameters.batch_size
        self.hidden_size = parameters.hidden_size

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_time = nn.Embedding(self.tim_size, self.loc_emb_size)
        self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)
        self.rnn_type = parameters.rnn_type
        self.num_layers = parameters.num_layers

        # hidden_size=[batch_size, 2*window_size, loc_size]
        input_size = self.loc_emb_size
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, self.hidden_size, self.num_layers, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, self.hidden_size, self.num_layers, batch_first=True)

        input_size = self.hidden_size + self.loc_emb_size
        input_size += self.uid_emb_size if Parameter.use_user else 0
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, self.loc_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)

    def forward(self, loc, target_time, uid=None):
        target_time_emb = self.emb_time(target_time)
        uid_emb = self.emb_uid(uid) if uid is not None else None

        h1 = Variable(torch.zeros(self.num_layers, loc.size(0), self.hidden_size)).to(d)
        c1 = Variable(torch.zeros(self.num_layers, loc.size(0), self.hidden_size)).to(d)

        # loc=[batch_size, window_size]-> loc=[window_size, batch_size]
        x = self.emb_loc(loc)  # [window_size, batch_size, emb_size]

        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            # rnn outputs (all), the last hidden state
            out, h1 = self.rnn(x, h1)
            # out:[window_size, batch_size, hidden_size]
        elif self.rnn_type == 'LSTM':
            out, (h1, c1) = self.rnn(x, (h1, c1))

        out = out[:, -1, :]
        # loc=[window_size, batch_size, hidden_size]-> loc=[batch_size, window_size, hidden_size]
        
        out = torch.cat((out, target_time_emb), 1)
        out = torch.cat((out, uid_emb), 1) if uid_emb is not None else out
        y = self.fc2(F.relu(self.fc1(out)))
        score = F.softmax(y, dim=1)
        return score