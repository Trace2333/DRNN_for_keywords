from train import trainer_basic
import torch.cuda
from general_code.utils.args import ArgsParse
from general_code.utils.sh_create import json_create, sh_create

args = ArgsParse()    #实例化
args.train_file = "train.py"
args.device = torch.device("cuda" if torch.cuda.is_available() is True else "cpu")
args.seed = 42
args.batch_size = 64
args.input_size = 300
args.hidden_size1 = 300
args.hidden_size2 = 300
args.epochs = 100
args.lr = 3e-3
# 超参数

args.evaluation_epochs = 1
args.optim = "Adam"
args.loss_fun = "CrossEntropyLoss"
# 模块

args.if_load = False
args.load_para = ""
args.save_name = "Test.pth"
args.if_save = False

args.project = "DRNN"
args.notes = "长epoch实验,epoch=100,综合记录,alpha=0.4,gamma=0.91,lr=3e-3"

args.debug = False

json_args = json_create(args)
sh_create("run_base.sh", json_args)
trainer_basic(args)

