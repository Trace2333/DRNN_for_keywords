import pickle
from data_process_for_bert import sen_to_list, get_list, dict_to_list
import dill
import numpy as np
import torch
import os
from transformers import BertModel, BertTokenizer
"""
    train_x = pickle.load(fx)
print(train_x)
"""
"""a = np.zeros([1, 768])
b = a.squeeze(0)
print(b)"""

"""os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda')
output = torch.zeros([1, 768]).to(device)
bert_filename = "C:\\Users\\Trace\\Desktop\\Projects\\bert-base-uncased"
tokens = ["how", "are", "youareyou"]
model = BertModel.from_pretrained(bert_filename).to(device)
tokenizer = BertTokenizer.from_pretrained(bert_filename)
for token in tokens:
    encoded_token = tokenizer.batch_encode_plus(
                [token],
                return_tensors='pt',
            )['input_ids'].to(device)
    out = model(encoded_token)['last_hidden_state'].mean(1)
    output =torch.cat((output, out), dim=0)
output = output[1:]
print(output.size())"""

#with open("data_set.pkl", "rb") as f:
#    dataset = dill.load(f)
"""train_x = sen_to_list(get_list(".\\data\\trnTweet")[0])
length_with_train = len(train_x) - len(dataset[0][1])
Y_train = dataset[0][1]
Z_train = dataset[0][2]
Y_test = dataset[1][1]
Z_test = dataset[1][2]
temp = []
print(length_with_train)
for i in range(length_with_train):
    length = len(train_x[len(dataset[0][1])])
    while len(temp) != length:
        temp.append(0)
    Y_train.append(temp)
    Z_train.append(temp)
    temp = []
print(temp)"""

"""
#with open("data_set.pkl", "rb") as f:
    #dataset = dill.load(f)
token_list = list(dataset[2]['words2idx'].keys())
tokens = []
output = []
for sen in dataset[0][0]:
    ids = list(sen)
    for i in ids:
        token = token_list[i - 1]
        tokens.append(token)
    output.append(tokens)
    tokens = []
#with open("train_add.pkl", "wb") as f:
    #pickle.dump(output, f)

output = []

for sen in dataset[1][0]:
    ids = list(sen)
    for i in ids:
        token = token_list[i - 1]
        tokens.append(token)
    output.append(tokens)
    tokens = []
#with open("test_add.pkl", "wb") as f:
    #pickle.dump(output, f)"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""a = np.array(torch.randn([300000, 768]))
b = torch.tensor(a).to(device)
print(a)"""

#with open(".\\data\\embedding.pkl", "rb") as f:
#    embw = pickle.load(f)
"""embw = np.array(dict_to_list(embw))
embw = torch.tensor(embw).to(device)
print(embw.size())

dataset_file = open("data_set.pkl", 'rb')
train, test, dict = dill.load(dataset_file)
dict = dict['words2idx']
out_list = list(dict.keys())
print(dict)"""

"""a = torch.randn([2, 4])
b = torch.randn([2, 4])
c = torch.matmul(b, a.permute(1, 0))
print(c.size())"""


"""
from transformers import BertTokenizer,AutoModel
def getdata(file_path):
    data_list,tag_list = [],[]
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            data_list.append(line.split('\t')[0])
            tag_list.append(line.split('\t')[1])
    return data_list,tag_list


file_path = ".\\data\\trnTweet"
data_list , tag_list = getdata(file_path)
sample = data_list[:16]
max_length = max(list(len(x.split()) for x in sample))
tokenizer = BertTokenizer.from_pretrained('C:\\Users\\Trace\\Desktop\\Projects\\bert-base-uncased')
token_example = tokenizer(sample,padding="max_length", max_length=max_length, truncation=True, return_tensors='pt')
transformer_model = AutoModel.from_pretrained('C:\\Users\\Trace\\Desktop\\Projects\\bert-base-uncased')
transformer_model.train()
transformer_outputs = transformer_model(**(token_example))
print(transformer_outputs.keys())
print(transformer_outputs['last_hidden_state'].size())
print(transformer_outputs['pooler_output'].size())
"""
"""

#with open("data_set.pkl", "rb") as f:
    #dataset = dill.load(f)
token_list = list(dataset[2]['words2idx'].keys())
tokens = []
output = []
for sen in dataset[0][0]:
    ids = list(sen)
    for i in ids:
        token = token_list[i - 1]
        tokens.append(token)
    output.append(tokens)
    tokens = []
#with open("train_add_sen.pkl", "wb") as f:
    #pickle.dump(output, f)

output = []

for sen in dataset[1][0]:
    ids = list(sen)
    for i in ids:
        token = token_list[i - 1]
        tokens.append(token)
    output.append(tokens)
    tokens = []
#with open("test_add_sen.pkl", "wb") as f:
    #pickle.dump(output, f)
"""
"""
import time
t = time.strftime("%y-%m-%d-%H:%M:%S", time.localtime()) + "test"
print(t)
"""
"""device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

a = [[1,2,3,4],
     [4,5,6,7],
     [5,6,7,8]]
b = torch.tensor(a)
print(b)


"""
import random
a = random.uniform(1, 3)
print(a)
