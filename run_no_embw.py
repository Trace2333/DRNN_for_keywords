import torch.cuda
from train_no_embw import trainer_no_embw
from general_code.utils.args import ArgsParse
from general_code.utils.sh_create import json_create, sh_create

args = ArgsParse()    #实例化
args.train_file = "train_no_embw.py"
args.device = torch.device("cuda" if torch.cuda.is_available() is True else "cpu")
args.seed = 42
args.batch_size = 64
args.input_size = 300
args.hidden_size1 = 300
args.hidden_size2 = 300
args.epochs = 200
args.lr = 3e-3

args.optim = "Adam"
args.loss_fun = "CrossEntropyLoss"
args.project = "DRNN-No-Embw"
args.notes = "First experiment"

args.load_para = "init.pth"
args.if_load = False
args.if_save = False
args.save_name = "init.pth"

args.notes = "测试初始较大学习率，学习率策略为指数缩减，gamma为0.95，epoch=200的大epoch训练，优化器为adam，alpha=0.3"

args.debug = False   # 负责添加权重可视化

json_args = json_create(args)
sh_create("run_no_embw.sh", json_args)
trainer_no_embw(args)

