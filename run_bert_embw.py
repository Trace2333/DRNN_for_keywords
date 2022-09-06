from train_bert_embw import trainer_bert_embw
import torch.cuda
from general_code.utils.args import ArgsParse
from general_code.utils.sh_create import json_create, sh_create

args = ArgsParse()    #实例化
args.train_file = "train_bert_embw.py"
args.device = torch.device("cuda" if torch.cuda.is_available() is True else "cpu")
args.seed = 42
args.batch_size = 16
args.input_size = 768
args.hidden_size1 = 768
args.hidden_size2 = 768
args.epochs = 1
args.lr = 1e-3
args.evaluation_epochs = 1
args.optim = "Adam"
args.loss_fun = "CrossEntropyLoss"
args.project = "DRNN-bert-embw"

args.notes = "epoch=1;lr=1e-3;save;default;softmax_activation"

args.if_save = True
args.if_load = False
args.load_para = "epoch=3_lr=1e-3.pth"
args.save_path = "epoch=1_lr=1e-3_softmax_activation.pth"

json_args = json_create(args)
sh_create("run_bert_embw.sh", json_args)

trainer_bert_embw(args)

