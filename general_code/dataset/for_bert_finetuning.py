import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


class dataset_bert_finetuning(Dataset):
    def __init__(self, sentence, labels1, labels2):
        super(dataset_bert_finetuning, self).__init__()
        self.sentence_labels = labels1
        self.sequence_labels = labels2
        self.sentence = sentence

    def __getitem__(self, index):
        return self.sentence[index], (self.sentence_labels[index], self.sequence_labels[index])

    def __len__(self):
        return len(self.sentence)


def collate_fun(batch):
    ids = [x[0].to(device) for x in batch]
    ids = pad_sequence(ids).T
    y = [torch.tensor(x[1][0]).to(device) for x in batch]
    y = pad_sequence(y).T
    z = [torch.tensor(x[1][1]).to(device) for x in batch]
    z = pad_sequence(z).T
    return (ids, (y, z))
