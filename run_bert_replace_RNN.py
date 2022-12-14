import torch.cuda
from general_code.utils.args import ArgsParse
from train_bert_replace_RNN import trainer_bert_replace_RNN
from general_code.utils.sh_create import json_create, sh_create

args = ArgsParse()    #实例化
args.train_file = "train_bert_replace_RNN.py"
args.device = torch.device("cuda" if torch.cuda.is_available() is True else "cpu")
args.seed = 42
args.batch_size = 32
args.input_size = 768
args.hidden_size1 = 768
args.hidden_size2 = 768
args.epochs = 1
args.lr = 1e-3
args.evaluation_epochs = 1
args.optim = "Adam"
args.loss_fun = "CrossEntropyLoss"
args.project = "DRNN-bert-replace-RNN"

json_args = json_create(args)
sh_create("run_bert_replace_RNN.sh", json_args)

trainer_bert_replace_RNN(args)

