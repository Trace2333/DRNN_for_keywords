import torch.cuda
from general_code.utils.args import ArgsParse
from general_code.utils.sh_create import json_create, sh_create
from Bert_finetuing import trainer_bert_finetuning, trainer_multitask_bert


args = ArgsParse()    #实例化
args.train_file = "Bert_finetuing.py"
args.device = torch.device("cuda" if torch.cuda.is_available() is True else "cpu")
args.seed = 42
args.batch_size = 32
args.epochs = 15
args.lr = 1e-5
args.evaluation_epochs = 1
args.optim = "Adam"
args.loss_fun = "CrossEntropyLoss"
args.project = "Bert-finetuning"
args.bert_path = "/data1/trace/Github_code/bert-base-uncased"
args.input_ids_filename = "./hot_data/input_ids.pkl"
args.eval_ids_filename = "./hot_data/eval_ids.pkl"
args.train_sentences_path = "./hot_data/train_add.pkl"
args.test_sentence_path = "./hot_data/test_add.pkl"
args.labels_path = "./hot_data/data_set.pkl"
args.alpha = 0.5
args.hidden_size = 768

args.notes = "提高学习率到1e-5,epoch=15"

json_args = json_create(args)
sh_create("run_bert_finetuning.sh", json_args)

#trainer_bert_finetuning(args)
trainer_multitask_bert(args)
