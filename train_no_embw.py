import logging
import wandb
import torch
import os
import dill
from tqdm import tqdm
from numpy import *
from general_code.utils.config_fun import load_config
from general_code.utils.args import get_parameter
from torch.utils.data import DataLoader
from general_code.models.DRNN import DRNNNoEmbw
from general_code.dataset.for_word2vec_embw import RNNdataset, collate_fun2
from general_code.utils.evalTools import acc_metrics, recall_metrics, f1_metrics


def trainer_no_embw(args=None):
    if args is None:
        args = get_parameter()
    torch.manual_seed(args.seed)
    wandb.login(host="http://47.108.152.202:8080",
                key="local-86eb7fd9098b0b6aa0e6ddd886a989e62b6075f0")
    os.system("wandb online")
    wandb_config = {'lr': args.lr,
                    'batch_size': args.batch_size,
                    'epochs': args.epochs,
                    'optimizer': args.optim,
                    'lossfun': args.loss_fun}
    wandb.init(project=args.project,
               notes=args.notes,
               config=wandb_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size    # 基本参数
    input_size = args.input_size
    hiden_size1 = args.hidden_size1
    hiden_size2 = args.hidden_size2
    epochs = args.epochs
    lr = args.lr

    model = DRNNNoEmbw(inputsize=input_size,
                       inputsize1=900,
                       hiddensize1=hiden_size1,
                       hiddensize2=hiden_size2,
                       inchanle=hiden_size2,
                       outchanle1=2,
                       outchanle2=5,
                       batchsize=batch_size,
                       embw_size=240057,
                       vec_size=300
                       ).to(device)
    load_config(
        model,
        target_path="/DRNN-No-Embw/",
        para_name=args.load_para,
        if_load_or_not=args.if_load
    )
    dataset_file = open("./hot_data/data_set.pkl", 'rb')
    train, test, dict = dill.load(dataset_file)

    dataset = RNNdataset(train)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0,
                              drop_last=True,
                              collate_fn=collate_fun2
                              )

    evaluation_dataset = RNNdataset(test)
    evaluation_loader = DataLoader(dataset=evaluation_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=0,
                                   drop_last=True,
                                   collate_fn=collate_fun2
                                   )
    if args.optim == "CrossEntropyLoss":
        lossfunction = torch.nn.CrossEntropyLoss()    # 优化器、损失函数选择
    else:
        lossfunction = torch.nn.CrossEntropyLoss()
    if args.optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12000, gamma=0.95)

    logging.info("Start Iteration")
    for epoch in range(epochs):  # the length of padding is 128
        iteration = tqdm(train_loader, desc=f"TRAIN on epoch {epoch}")
        model.train()
        for step, inputs in enumerate(iteration):
            output1, output2 = model(
                (inputs[0], torch.randn([1, batch_size, hiden_size1])))  # 模型计算
            sentence_preds = output1.argmax(axis=2)
            sequence_preds = output2.argmax(axis=2)

            sen_acc = acc_metrics(sentence_preds, inputs[1][0])    #  指标计算
            seq_acc = acc_metrics(sequence_preds, inputs[1][1])
            sen_recall = recall_metrics(sentence_preds, inputs[1][0])
            seq_recall = recall_metrics(sentence_preds, inputs[1][0])
            sen_f1 = f1_metrics(sen_acc, sen_recall)
            seq_f1 = f1_metrics(seq_acc, seq_recall)

            wandb.log({"Train Sentence Precision": sen_acc})    #  指标可视化
            wandb.log({"Train Sequence Precision": seq_acc})
            wandb.log({"Train Sentence Recall": sen_recall})
            wandb.log({"Train Sequence Recall": seq_recall})
            wandb.log({"Train Sentence F1 Score": sen_f1})
            wandb.log({"Train Sequence F1 Score": seq_f1})

            loss1 = lossfunction(output1.permute(0, 2, 1), inputs[1][0])    # loss计算,按照NER标准
            loss2 = lossfunction(output2.permute(0, 2, 1), inputs[1][1])
            loss = loss2 * 0.7 + loss1 * 0.3

            iteration.set_postfix(loss1='{:.4f}'.format(loss1), loss2='{:.4f}'.format(loss2), p='{:.3f}'.format(sen_acc))
            wandb.log({"train loss1": loss1})
            wandb.log({"train loss2": loss2})
            wandb.log({"train Totalloss": loss})
            wandb.log({"lr:": optimizer.state_dict()['param_groups'][0]['lr']})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if args.debug is not None and args.debug is True:
                for name, parms in model.named_parameters():    # debug时使用，可视化每一个层的grad与weight
                    wandb.log({f"{name} Weight:" : torch.mean(parms.data)})
                    if parms.grad is not None:
                        wandb.log({f"{name} Grad_Value:" : torch.mean(parms.grad)})

        evaluation_iteration = tqdm(evaluation_loader, desc=f"eval on {epoch} parameters...")
        model.eval()
        sen_acc_eval = []
        seq_acc_eval = []
        sen_recall_eval = []
        seq_recall_eval = []

        for step, evaluation_input in enumerate(evaluation_iteration):
            with torch.no_grad():
                output1, output2 = model((evaluation_input[
                    0]))  # 模型计算
                sentence_preds = output1.argmax(axis=2)
                sequence_preds = output2.argmax(axis=2)

                sen_acc_eval.append(acc_metrics(sentence_preds, evaluation_input[1][0]))  # 参数计算
                seq_acc_eval.append(acc_metrics(sequence_preds, evaluation_input[1][1]))
                sen_recall_eval.append(recall_metrics(sentence_preds, evaluation_input[1][0]))
                seq_recall_eval.append(recall_metrics(sentence_preds, evaluation_input[1][0]))
                sen_f1_eval = f1_metrics(mean(sen_acc_eval), mean(sen_recall_eval))
                seq_f1_eval = f1_metrics(mean(seq_acc_eval), mean(seq_recall_eval))

        wandb.log({"Sentence Precision": mean(sen_acc_eval)})
        wandb.log({"Sequence Precision": mean(seq_acc_eval)})
        wandb.log({"Sentence Recall": mean(sen_recall_eval)})
        wandb.log({"Sequence Recall": mean(seq_recall_eval)})
        wandb.log({"Sentence F1 Score": sen_f1_eval})
        wandb.log({"Sequence F1 Score": seq_f1_eval})

    if args.if_save is True:
        torch.save(model.state_dict(), "./check_points/DRNN-No-Embw/" + args.save_name)

