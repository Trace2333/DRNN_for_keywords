from torch.utils.data import Dataset


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


