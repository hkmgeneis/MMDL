import torch
import time
import copy
from utils.common import logger
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms

import pandas as pd
import numpy as np
import json
from sklearn import preprocessing


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tb = SummaryWriter('/home/yangjy/HE_files/resnet18withclinical6_3')  # 改动1


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=50, cnv=True):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())  # model.state_dict()存放模型训练的参数
    best_acc = 0.0

    if cnv:
        cnv_feature=pd.read_csv('/home/yangjy/msipredictor-master/clin6-aftercorrelation1.csv')  # 改动2
        peoples=[i for i in cnv_feature.id]
        features=[cnv_feature[i] for i in cnv_feature.columns[1:]]
        min_max_scaler = preprocessing.MinMaxScaler()  # 归一化
        cnv_features = min_max_scaler.fit_transform(features)  # 数据标准化
    
    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))

        phase = 'train'
        model.train()  # 启用batch normalization和drop out

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.遍历数据
        for inputs_, labels_, names_, _ in dataloaders[phase]:
            inputs_ = inputs_.to(device)
            labels_ = labels_.to(device)

            # zero the parameter gradients#参数梯度设置为0
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):#影响自动求导，前向传播后不会进行求导和进行反向传播
                if cnv:
                    X_train_minmax = [cnv_features[:,peoples.index(i)] for i in names_]
                    outputs_ = model(inputs_, torch.from_numpy(np.array(X_train_minmax, dtype=np.float32)).to(device))#model(图像+临床数据)
                else:
                    outputs_ = model(inputs_)
                _, preds = torch.max(outputs_, 1)#按维度输出最大值并返回索引，1代表行
                loss = criterion(outputs_, labels_)#计算损失

                # backward + optimize only if in training phase
                loss.backward()#反向传播，计算当前梯度
                optimizer.step()#根据梯度更新网络参数

            # statistics
            running_loss += loss.item() * inputs_.size(0)#计算所有batch的损失和
            running_corrects += torch.sum((preds == labels_.data).int())#计算所有batch的精确度

        epoch_loss = running_loss / dataset_sizes[phase]#除以样本总数得到平均的loss
        epoch_acc = running_corrects / dataset_sizes[phase]#平均精确度

        scheduler.step()#更新学习率
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))
        tb.add_scalar("Train/Loss", epoch_loss, epoch)
        tb.add_scalar("Train/Accuracy", epoch_acc, epoch)
        tb.flush()

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}h {:.0f}m'.format(
        time_elapsed // 3600, (time_elapsed-time_elapsed // 3600) * 60))
    logger.info('Best train Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, tb
