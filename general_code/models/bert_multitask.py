import torch.nn as nn
from transformers import BertModel


"""输入沿袭微调bert的输入"""
class BertForMultiTask(nn.Module):
    def __init__(self, pretrain_path, hidden_size, out_size1, out_size2):
        super(BertForMultiTask, self).__init__()
        self.bert_layer = BertModel.from_pretrained(pretrain_path)
        self.linear1 = nn.Linear(
            hidden_size,
            out_size1
        )
        self.linear2 = nn.Linear(
            hidden_size,
            out_size2
        )
        self.weight_init()

    def forward(self, inputs):
        bert_out = self.bert_layer(inputs)   # inputs只包含了ids
        out1 = self.linear1(bert_out['last_hidden_state'])
        out2 = self.linear2(bert_out['last_hidden_state'])
        return (out1, out2)

    def weight_init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1)