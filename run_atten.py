from train_atten import trainer_atten
import torch.cuda
from general_code.utils.args import ArgsParse
from general_code.utils.sh_create import json_create, sh_create

args = ArgsParse()    #实例化
args.train_file = "train_atten.py"
args.device = torch.device("cuda" if torch.cuda.is_available() is True else "cpu")
args.seed = 42
args.batch_size = 128
args.input_size = 300
args.hidden_size1 = 300
args.hidden_size2 = 300
args.epochs = 1
args.lr = 1e-7
args.evaluation_epochs = 1
args.optim = "Adam"
args.loss_fun = "CrossEntropyLoss"
args.project = "DRNN-Attention"
args.if_load = True
args.if_save = True
args.notes = "epoch=5;saving;lr=1e-7;eval_epoch=1;default"
args.save_name= 'epoch=5-lr=1e-7.pth'
args.load_para = "epoch=4.pth"
json_args = json_create(args)
sh_create("run_atten.sh", json_args)

trainer_atten(args)

