import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import dill


class RNNdataset(Dataset):
    """
    dataset
    """
    def __init__(self, input_data):
        super(RNNdataset, self).__init__()
        self.data = (input_data)

    def __getitem__(self, item):
        """标准gettiem格式"""
        y1 = list(self.data[1][item])
        y1_tensor = torch.tensor(y1)
        y2 = list(self.data[2][item])
        y2_tensor = torch.tensor(y2)
        return torch.tensor(list(self.data[0][item])), (y1_tensor, y2_tensor)


def collate_fun1(batch_data):
    """
     自定义collate方法1，返回已经完成embedding后的batch

     Args:
         batch_data:list
     Return:
         完成处理的batch
    """
    # NO EXTRA PARAMETERS.....THE lENGTH of PADDED SHOULD BE SET IN FUN:padding_sentence()
    padded_list = padding_sentence(batch_data)
    # batched_dataX = []
    # for listX in padded_list[0]:
    #    batched_dataX.append(sen_process(listX, embedding_model))
    # When using the self-trained word2vec embedding, it is available
    batched_dataX = nn.functional.embedding(torch.tensor(contextwin_2(padded_list[0], 3), dtype=torch.int32),
                                            matrix).flatten(2)
    # When using given embedding
    batched_dataY = padded_list[1][0]
    batched_dataZ = padded_list[1][1]
    dataX = batched_dataX[0].unsqueeze(0)
    dataY = torch.tensor(batched_dataY[0]).unsqueeze(0)
    dataZ = torch.tensor(batched_dataZ[0]).unsqueeze(0)
    for i in batched_dataX[1:]:
        dataX = torch.cat((dataX, i.unsqueeze(0)), dim=0)
    for i in batched_dataY[1:]:
        dataY = torch.cat((dataY, torch.tensor(i).unsqueeze(0)), dim=0)
    for i in batched_dataZ[1:]:
        dataZ = torch.cat((dataZ, torch.tensor(i).unsqueeze(0)), dim=0)
    dataX.to(device)
    dataY.to(device)
    dataZ.to(device)
    return (dataX, (dataY, dataZ))

def collate_fun2(batch_data):
    """
    自定义的collate方法2，得到未经过embedding的x输入和经过cat的标签对

    Args：
        :params batch_data:由dataloader自动输出的batch data
    Return:
        元组，（未embedding的x， （tensorY， tensorZ））
    """
    padded_list = padding_sentence(batch_data)
    batched_dataX = padded_list[0]
    batched_dataY = padded_list[1][0]
    batched_dataZ = padded_list[1][1]
    dataY = (torch.tensor(batched_dataY[0]).to(device)).unsqueeze(0)
    dataZ = (torch.tensor(batched_dataZ[0]).to(device)).unsqueeze(0)
    for i in batched_dataY[1:]:
        dataY = torch.cat((dataY, (torch.tensor(i).to(device)).unsqueeze(0)), dim=0)
    for i in batched_dataZ[1:]:
        dataZ = torch.cat((dataZ, (torch.tensor(i).to(device)).unsqueeze(0)), dim=0)
    return (batched_dataX, (dataY, dataZ))

