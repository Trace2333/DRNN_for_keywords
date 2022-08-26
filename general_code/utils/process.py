import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import dill


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_model = open("embedding_origin.pkl", "rb")
matrix = dill.load(embedding_model)
matrix = torch.tensor(matrix)


class lossfun(nn.Module):
    """废用函数"""
    def __init__(self):
        super().__init__()

    def forward(self, out, y):
        loss = torch.sum((out * torch.log(y)))
        return loss


def sen_process(sen_list, embedding_model):
    """
    将句子列表处理成embedding，没有涉及到nn.embedding

    Note:
        embedding_file:contains vectors--->vectors:[1,1,1,1,1...]
    """
    if sen_list[0] in embedding_model:
        sen_embedding = torch.tensor(embedding_model[sen_list[0]]).unsqueeze(0)
    else:
        sen_embedding = torch.randn([1, 300])
    for i in sen_list[1:]:
        if i in embedding_model:  # For the trained word2vec model
            sen_embedding = torch.cat((sen_embedding, torch.tensor(embedding_model[i]).unsqueeze(0)), dim=0)
        else:
            sen_embedding = torch.cat((sen_embedding, torch.randn([1, 300])), dim=0)  # For the trained word2vec model
    return sen_embedding

def padding_sentence(inputs):
    """
    Padding句子并输出

    Args:
        :params inputList:list of lists
        :params forced_length:none if no input,if the parameter is not given,we will pad tne sentences with the maxlength of the list
    Return:
        padding后的句子对
    """
    inputListX = []
    inputListY = []
    inputListZ = []
    for i in inputs:
        inputListX.append(list(i)[0])
        inputListY.append(list(i[1])[0])
        inputListZ.append(list(i[1])[1])
    max_length = max(len(list(x)) for x in inputListY)
    return padding(inputListX, max_length), (padding(inputListY, max_length), padding(inputListZ, max_length))


def padding(inputList, max_length, forced_length=None):
    """
    padding句子

    Args:
        :params inputList:输入的嵌套列表
        :params max_length:最大的长度，用于在指定长度时使用
        :params forced_length:是否强制长度
    Return:
        padding后的嵌套列表
    """
    if forced_length is None:
        num_padded_length = max_length  # padding to the curant max length
        padded_list = []
        for sentence in inputList:
            padded_sentence = sentence.tolist()
            while len(padded_sentence) < num_padded_length:
                padded_sentence.append(0)  # is the max length indefinitely
            padded_list.append(padded_sentence)
        return padded_list
    else:
        if max_length < forced_length:
            num_padded_length = forced_length
            padded_list = []
            for sentence in inputList:
                padded_sentence = sentence.tolist()
                while len(padded_sentence) < num_padded_length:
                    padded_sentence.append(0)  # is the max length indefinitely
                    padded_list.append(padded_sentence)
            return padded_list
        else:
            num_padded_length = forced_length    # some sentences shoule be cut.
            padded_list = []
            for sentence in inputList:
                padded_sentence = sentence.tolist()
                while len(padded_sentence) < num_padded_length:
                    padded_sentence.append(0)  # is the max length indefinitely
                    padded_list.append(padded_sentence)
                while len(padded_sentence) > num_padded_length:
                    padded_sentence.pop()
                    padded_list.append(padded_sentence)
            return padded_list


def contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)

    lpadded = int(win / 2) * [0] + l + int(win / 2) * [0]
    out = [lpadded[i:i + win] for i in range(len(l))]

    assert len(out) == len(l)
    return out


def contextwin_2(ls, win):
    assert (win % 2) == 1
    assert win >= 1
    outs = []
    for l in ls:
        outs.append(contextwin(l, win))
    return outs


def getKeyphraseList(l):
    res, now = [], []
    for i in range(len(l)):
        if l[i] != 0:
            now.append(str(i))
        if l[i] == 0 or i == len(l) - 1:
            if len(now) != 0:
                res.append(' '.join(now))
            now = []
    return set(res)


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

