import torch
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RNNdatasetForBertReplaceRNN(Dataset):
    """
    dataset
    """
    def __init__(self, input_data):
        super(RNNdatasetForBertReplaceRNN, self).__init__()
        self.data = input_data

    def __getitem__(self, item):
        """标准gettiem格式"""
        y1 = list(self.data[1][item])
        y1_tensor = torch.tensor(y1)
        y2 = list(self.data[2][item])
        y2_tensor = torch.tensor(y2)
        return self.data[0][item], (y1, y2)

    def __len__(self):
        """标准len格式"""
        return len(self.data[0])


def collate_fun_for_bert_replace_rnn(batch):
    batchX = [i[0] for i in batch]
    batchY = [i[1][0] for i in batch]
    batchZ = [i[1][1] for i in batch]
    Y = padding_Y_Z(batchY)
    Z = padding_Y_Z(batchZ)
    return (batchX, Y, Z)


def padding_Y_Z(x):
    output = []
    max_length = max(len(list(i)) for i in x)
    for sentence in x:
        while len(sentence) < max_length:
            sentence.append(0)
        output.append(sentence)
    tensor_out = torch.tensor(output[0], dtype=torch.int8).unsqueeze(0)
    for i in output[1:]:
        t = torch.tensor(i).unsqueeze(0)
        tensor_out = torch.cat((tensor_out, t), dim=0)
    return tensor_out.to(device)



