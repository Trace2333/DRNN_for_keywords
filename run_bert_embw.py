from train import trainer_basic
import torch.cuda
from general_code.utils.args import ArgsParse
from general_code.utils.sh_create import json_create, sh_create

args = ArgsParse()    #实例化
args.train_file = "train_bert_embw.py"
args.device = torch.device("cuda" if torch.cuda.is_available() is True else "cpu")
args.seed = 42
args.batch_size = 32
args.input_size = 300
args.hidden_size1 = 300
args.hidden_size2 = 300
args.epochs = 1
args.lr = 1e-4
args.evaluation_epochs = 1
args.optim = "Adam"
args.loss_fun = "CrossEntropyLoss"
args.project = "DRNN-bert-embw"

json_args = json_create(args)
sh_create("run_bert_embw.sh", json_args)

trainer_basic(args)

