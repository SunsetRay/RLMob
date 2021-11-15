import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.autograd import Variable
from torch.distributions import Categorical
from run import Parameter


d = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save(model, dir):
    torch.save(model, open('data/'+dataset_name+'/'+dataset_name+'.pk', 'rb'))


class StateEncodeNetwork(nn.Module):
    """
    input: [batch_size, seq_len, dim_emb]
    output: [batch_size, dim_state]
    """
    def __init__(self, dim_emb, dim_out, hidden_size=256):
        super(StateEncodeNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(dim_emb, hidden_size, bias=True),
            nn.ReLU(True),

            # nn.Dropout(p=0.5),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.ReLU(True)
        )

        # recurrent layer
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            num_layers=1, batch_first=True, bias=True)

        # output layer
        self.value_fc = nn.Sequential(
            nn.Linear(hidden_size, dim_out, bias=True)
        )

    def init_hidden_states(self, batch_size=1):
        hx = torch.zeros(1, batch_size, self.hidden_size).float().to(d)
        cx = torch.zeros(1, batch_size, self.hidden_size).float().to(d)
        return hx, cx

    def forward(self, loc_emb, hidden_state, cell_state):
        loc_emb = loc_emb.to(d)
        hidden = self.net(loc_emb)

        lstm_out = self.lstm(hidden, (hidden_state, cell_state))
        out = lstm_out[0][:, -1, :]
        hx = lstm_out[1][0]
        cx = lstm_out[1][1]

        out = self.value_fc(out)
        return out, (hx, cx)


class StateEncodeNetworkFixed(nn.Module):
    """
    pretrained parameters,
    input: [batch_size, seq_len, dim_emb]
    output: [batch_size, hidden_size]
    """
    def __init__(self, dim_emb, hidden_size=Parameter.traj_emb_size):
        super(StateEncodeNetworkFixed, self).__init__()
        for p in self.parameters():
            p.requires_grad = False
        self.hidden_size = hidden_size

        # recurrent layer
        self.rnn = nn.LSTM(input_size=dim_emb, hidden_size=hidden_size,
                           num_layers=1, batch_first=True)

    def init_hidden_states(self, batch_size=1):
        hx = torch.zeros(1, batch_size, self.hidden_size).float().to(d)
        cx = torch.zeros(1, batch_size, self.hidden_size).float().to(d)
        return hx, cx

    def forward(self, loc_emb, hidden_state, cell_state):
        loc_emb = loc_emb.to(d)

        lstm_out = self.rnn(loc_emb, (hidden_state, cell_state))
        out = lstm_out[0][:, -1, :]
        hx = lstm_out[1][0]
        cx = lstm_out[1][1]

        out = out.detach() if Parameter.debug['mode'] != 'end2end' else out

        return out, (hx, cx)


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class MLP_State(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(MLP_State, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(state_dim, action_dim).to(d)
    def forward(self, x, softmax_dim=0):
        x = self.fc(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob


class MLP2_State(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(MLP2_State, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size).to(d)
        self.fc2 = nn.Linear(hidden_size, action_dim).to(d)
    def forward(self, x, softmax_dim=0):
        x = self.fc2(F.relu(self.fc1(x)))
        prob = F.softmax(x, dim=softmax_dim)
        return prob


# test
class parameters:
    use_dropout = True
    loc_size = Parameter.loc_size
    tim_size = 168
    window_size = 7
    loc_emb_size = 64
    hidden_size = 128