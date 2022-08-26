import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import dill

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SrnnNet(nn.Module):
    """
    未修改的网络，目前无效
    """
    def __init__(self, input_size, hidden_size1, hidden_size2, embw):
        super(SrnnNet, self).__init__()
        self.shared_layer = nn.Sequential(
            RNNLayer(input_size=input_size, hidden_size=hidden_size1)
        )
        self.tow2 = nn.Sequential(
            RNNLayer(input_size=input_size, hidden_size=hidden_size2),
            # neuros related to 2 rnns will be used for sequense classification
            Selectitem(0),  # get the RNN output
            nn.Linear(in_features=hidden_size2, out_features=5),
        )  # for sequence
        self.tow1 = nn.Sequential(
            Selectitem(0),  # get the RNN output
            nn.Linear(in_features=hidden_size1, out_features=2),
        )  # for sentence
        self.dropout = nn.Dropout(0.5)
        self.embedding = nn.Parameter(embw)

    def forward(self, x):
        """返回的是y_pred， z_pred"""
        x = nn.functional.embedding(torch.tensor(x[0], dtype=torch.long).to(device), self.embedding)
        out = self.shared_layer(x)  # out is a tuple
        model_out1 = self.tow1(out)
        model_out2 = self.tow2(out)
        return model_out1, model_out2


class Selectitem(nn.Module):
    def __init__(self, index):
        super(Selectitem, self).__init__()
        self.index = index

    def forward(self, inputs):
        return inputs[self.index]


class RNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNLayer, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

    def forward(self, inputs):
        inputsX = self.dropout(inputs[0])
        if isinstance(inputs, tuple):
            return (self.rnn(inputsX, torch.zeros(list(inputs[1].size())).to(device)))
        else:
            return self.rnn(inputs)    # 默认0初始化
