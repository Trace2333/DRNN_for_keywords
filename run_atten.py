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
args.lr = 1e-3
args.evaluation_epochs = 1
args.optim = "Adam"
args.loss_fun = "CrossEntropyLoss"
args.project = "DRNN-Attention"

args.if_load = False
args.if_save = False
args.save_name= 'epoch=5-lr=2e-5.pth'
args.load_para = "epoch=4.pth"

args.notes = "测试新代码"

args.debug = False

json_args = json_create(args)
sh_create("run_atten.sh", json_args)
trainer_atten(args)

