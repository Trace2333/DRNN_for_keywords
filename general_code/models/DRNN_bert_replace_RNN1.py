import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DRNNBertReplaceRNN(nn.Module):
    """
    主试验网路
    （已经证明是有效的）
    """
    def __init__(self, inputsize, hiddensize1, hiddensize2, inchanle, outchanle1, outchanle2, batchsize):
        super(DRNNBertReplaceRNN, self).__init__()
        self.hiddensize = hiddensize1
        self.batchsize = batchsize
        #self.RNN1 = nn.RNN(inputsize1, hiddensize1, batch_first=True, bidirectional=False)
        self.layer1 = BertModel.from_pretrained("C:\\Users\\Trace\\Desktop\\Projects\\bert-base-uncased")
        self.RNN2 = nn.RNN(inputsize, hiddensize2, batch_first=True, bidirectional=False)
        self.Linear1 = nn.Linear(in_features=inchanle, out_features=outchanle1)
        self.Linear2 = nn.Linear(in_features=inchanle, out_features=outchanle2)
        self.dropout = nn.Dropout(0.5)
        self.tokenizer = BertTokenizer.from_pretrained("C:\\Users\\Trace\\Desktop\\Projects\\bert-base-uncased")
        self.weight_init()

    def forward(self, inputs):
        """前向计算"""
        state1 = self.init_state(self.batchsize)
        state2 = self.init_state(self.batchsize)
        if isinstance(inputs, tuple):
            max_length = max(list(len(x.split()) for x in inputs[0]))
            tokenized_inputs = self.tokenizer(inputs[0], return_tensors='pt', truncation=True, max_length=max_length, padding=True).to(device)
            output1 = self.layer1(**tokenized_inputs)
            rnnout2, state2 = self.RNN2(output1['last_hidden_state'])
            y = self.Linear1(output1['last_hidden_state'])
            z = self.Linear2(rnnout2)
            return y, z
        else:
            max_length = max(list(len(x.split()) for x in inputs))
            tokenized_inputs = self.tokenizer(inputs, return_tensors='pt', truncation=True, max_length=max_length,
                                              padding=True).to(device)
            output1 = self.layer1(**tokenized_inputs)

            rnnout2, state2 = self.RNN2(output1['last_hidden_state'])
            y = self.Linear1(output1['last_hidden_state'])
            z = self.Linear2(rnnout2)
            return y, z

    def init_state(self, batchsize):
        """提供零初始化"""
        return torch.zeros((1, batchsize, self.hiddensize))

    def weight_init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1)

