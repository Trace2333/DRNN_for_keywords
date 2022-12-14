import os
import wandb
import torch
import dill
import pickle
from numpy import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from general_code.utils.evalTools import acc_metrics, f1_metrics, recall_metrics
from transformers import BertForTokenClassification, BertTokenizer
from general_code.models.bert_multitask import BertForMultiTask
from general_code.dataset.for_bert_finetuning import dataset_bert_finetuning, collate_fun


def data_reprocess(inputs, tokenizer, target_path):
    """用第一个token作为wordpiece的结果"""
    if os.path.exists("./hot_data/" + target_path):
        with open("./hot_data/" + target_path, "rb") as f:
            output = pickle.load(f)
        return output
    output = []
    for sen in tqdm(inputs, desc="Reprocessing Data..."):
        tokens = sen.split()
        sen_ids = torch.tensor([101])
        for token in tokens:    # 开始编码
            out_ids = tokenizer(
                token,
                return_tensors="pt",
                add_special_tokens=False,
                padding=False,
            )['input_ids']
            if out_ids.size()[1] is not 1:
                out_ids = out_ids[0][0].unsqueeze(0).unsqueeze(0)
            sen_ids = torch.cat((sen_ids, out_ids.squeeze(0)), dim=0)
        sen_ids = torch.cat((sen_ids, torch.tensor([102])), dim=0)
        sen_ids = sen_ids[1:-1]
        output.append(sen_ids)
    with open("./hot_data/" + target_path, "wb") as f:
        pickle.dump(output, f)
    return output


def labels_extrace(dataset_path):
    """从dataset文件中抽取出标签"""
    with open(dataset_path, "rb") as f:
        train, test, dict = dill.load(f)
        train_labels = (train[1], train[2])
        test_labels = (test[1], test[2])
    return train_labels, test_labels


def sentence_reshape(train_path, test_path):
    """从token还原句子，留做模型输入"""
    with open(train_path, "rb") as f:
        train_data_pre = pickle.load(f)
    with open(test_path, "rb") as f:
        test_data_pre = pickle.load(f)
    train_data = []
    test_data = []
    for tokens in train_data_pre:
        temp = tokens[0]
        for token in tokens[1:]:
            temp += (" " + token)
        train_data.append(temp)
        temp = ""

    for tokens in test_data_pre:
        temp = tokens[0]
        for token in tokens[1:]:
            temp += (" " + token)
        test_data.append(temp)
        temp = ""
    return train_data, test_data


def trainer_bert_finetuning(args):
    """
    需要准备的数据只有两个，第一个是句子，第二个是标签即可

    Args:
        args: 参数类

    Returns:
       无
    """
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    wandb.login(host="http://47.108.152.202:8080",
                key="local-86eb7fd9098b0b6aa0e6ddd886a989e62b6075f0")
    wandb.init(project=args.project,
               notes=args.notes)
    model = BertForTokenClassification.from_pretrained(args.bert_path).to(device)
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    train, test = sentence_reshape(
        train_path=args.train_sentences_path,
        test_path=args.test_sentence_path,
    )
    train_labels, test_labels = labels_extrace(args.labels_path)
    if not os.path.exists(args.input_ids_filename):
        input_ids = data_reprocess(train, tokenizer)
    else:
        with open(args.input_ids_filename, "rb") as f:
            input_ids = pickle.load(f)
    dataset_for_fine = dataset_bert_finetuning(input_ids, train_labels[0], train_labels[1])
    train_loader = DataLoader(
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        dataset=dataset_for_fine,
        collate_fn=collate_fun
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        iteration = tqdm(train_loader, desc="Running train process:")
        model.train()
        for step, batch in enumerate(iteration):
            ids = torch.tensor(batch[0])
            y, z = batch[1]
            out = model(ids,
                        labels=y.contiguous(),
                        output_hidden_states=True)
            loss = out['loss']
            wandb.log({"Loss1:": loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def trainer_multitask_bert(args):
    """直接用bert作为底层完成多任务"""

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    wandb.login(host="http://47.108.152.202:8080",
                key="local-86eb7fd9098b0b6aa0e6ddd886a989e62b6075f0")
    #os.system("wandb offline")   # debug
    os.system("wandb online")
    wandb.init(project=args.project)
    model = BertForMultiTask(
        pretrain_path=args.bert_path,
        hidden_size=args.hidden_size,
        out_size1=2,
        out_size2=5,
    ).to(device)
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    train, test = sentence_reshape(
        train_path=args.train_sentences_path,
        test_path=args.test_sentence_path,
    )
    train_labels, test_labels = labels_extrace(args.labels_path)
    if not os.path.exists(args.input_ids_filename) or os.path.exists(args.eval_ids_filename):
        input_ids = data_reprocess(train, tokenizer, "input_ids.pkl")
        eval_ids = data_reprocess(test, tokenizer, "eval_ids.pkl")
    else:
        with open(args.input_ids_filename, "rb") as f:
            input_ids = pickle.load(f)
        with open(args.eval_ids_filename, "rb") as f2:
            eval_ids = pickle.load(f2)
    dataset_for_fine = dataset_bert_finetuning(input_ids, train_labels[0], train_labels[1])
    dataset_for_eval = dataset_bert_finetuning(eval_ids, test_labels[0], test_labels[1])
    train_loader = DataLoader(
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        dataset=dataset_for_fine,
        collate_fn=collate_fun
    )
    eval_loader = DataLoader(
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        dataset=dataset_for_eval,
        collate_fn=collate_fun
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossfun = torch.nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        iteration = tqdm(train_loader, desc="Running train process:")
        model.train()
        for step, batch in enumerate(iteration):
            ids = torch.tensor(batch[0])
            y, z = batch[1]
            out1, out2 = model(ids)
            loss1 = lossfun(out1.permute(0, 2, 1), y)
            loss2 = lossfun(out2.permute(0, 2, 1), z)
            loss = loss1 * args.alpha + loss2 * (1 - args.alpha)
            sen_train = out1.argmax(-1)
            seq_train = out2.argmax(-1)

            sen_acc_train = acc_metrics(sen_train, y)
            seq_acc_train = acc_metrics(seq_train, z)
            sen_recall_train = recall_metrics(sen_train, y)
            seq_recall_train = recall_metrics(seq_train, z)
            sen_f1_train = f1_metrics(sen_acc_train, sen_recall_train)
            seq_f1_train = f1_metrics(seq_acc_train, seq_recall_train)

            wandb.log({"loss1:": loss1.item()})
            wandb.log({"loss2:": loss2.item()})
            wandb.log({"loss:": loss.item()})
            wandb.log({"train sentence precision:": sen_acc_train})
            wandb.log({"train sequence precision:": seq_acc_train})
            wandb.log({"train sentence recall:": sen_recall_train})
            wandb.log({"train sequence recall:": seq_recall_train})
            wandb.log({"train sentence f1-score:": sen_f1_train})
            wandb.log({"train sequence f1-score:": seq_f1_train})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # For eval
        model.eval()
        sen_acc_eval = []
        seq_acc_eval = []
        sen_recall_eval = []
        seq_recall_eval = []
        iteration = tqdm(eval_loader, desc="Running eval process:")
        for step, batch in enumerate(iteration):
            ids = torch.tensor(batch[0])
            y, z = batch[1]
            out1, out2 = model(ids)
            sen_pred = out1.argmax(-1)
            seq_pred = out2.argmax(-1)
            sen_acc_eval.append(acc_metrics(sen_pred, y))
            seq_acc_eval.append(acc_metrics(seq_pred, z))
            sen_recall_eval.append(recall_metrics(sen_pred, y))
            seq_recall_eval.append(recall_metrics(seq_pred, z))
            sen_f1_eval = f1_metrics(mean(sen_acc_eval), mean(sen_recall_eval))
            seq_f1_eval = f1_metrics(mean(seq_acc_eval), mean(seq_recall_eval))
            
        wandb.log({"Sentence Precision": mean(sen_acc_eval)})
        wandb.log({"Sequence Precision": mean(seq_acc_eval)})
        wandb.log({"Sentence Recall": mean(sen_recall_eval)})
        wandb.log({"Sequence Recall": mean(seq_recall_eval)})
        wandb.log({"Sentence F1 Score": sen_f1_eval})
        wandb.log({"Sequence F1 Score": seq_f1_eval})
        
        