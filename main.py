from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.common import logger
from utils.custom_dset import CustomDset
# from utils.analytics import draw_roc, draw_roc_for_multiclass

# import train_test_splitter
from train1 import train_model
from test1 import test

from Net import Net, Cnn_With_Clinical_Net

plt.ion()   # interactive mode 交互模式：展示动态图或多个窗口，使matplotlib的显示模式转换为交互模式

# Data augmentation and normalization for training
# Just normalization for validation，验证集只需要颜色归一化
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

def generative_model(model, k, cnv=True):   # 加入临床信息
    image_datasets = {x: CustomDset(os.getcwd()+f'/database6_3/{x}_{k}.csv',  # 按照交叉验证的划分数据集，生成csv文件，然后将数据导入进trian中。 改动
                        data_transforms[x]) for x in ['train']}  # 没有独立测试集，所有的都导入进trian中。

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,  # 载入数据
                    shuffle=True, num_workers=4) for x in ['train']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}  # 数量
    class_names = image_datasets['train'].classes

    logger.info(f'model {model} / 第 {k+1} 折')  # K=0-4,

    available_policies = {"resnet18": models.resnet18, "vgg16": models.vgg16, "vgg19": models.vgg19, 
            "alexnet": models.alexnet, "inception": models.inception_v3}
    
    model_ft = available_policies[model](pretrained=True)  # 加载预训练网络(迁移学习）ft：可能是finetune意思。
    
    if cnv:
        model_ft = Cnn_With_Clinical_Net(model_ft)  # 加入临床信息：模型选择Cnn_With_Clinical_Net；否则模型为Net。
    else:
        model_ft = Net(model_ft)
    model_ft = model_ft.to(device)  # 模型传给设备
    
    criterion = nn.CrossEntropyLoss()  # 损失函数
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)  # 优化器
    # Decay LR by a factor of 0.1 every 7 epochs    每7个epoch将LR衰减0.1倍。
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  # exp_lr_scheduler是：学习率的步数

    model_ft, tb = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, 
        dataset_sizes, num_epochs=50, cnv=cnv)
    tb.close()  # 释放空间。     tb：tensorboard

    save_model = os.getcwd()+f'/resultswithclin6_3/{model}_{k}'  #用.pkl文件来保存模型的权重参数等。，改动
    if cnv:
        save_model = save_model + '_cnv'
    save_model = save_model + '.pkl'  # 有临床信息则..,否则保存成.pkl。
    
    torch.save(model_ft, save_model)


def main(ocs, classification, K, cnv):
     
    # train_test_splitter.main("/media/zw/Elements1/tiles_cn", "/home/xisx/tmbpredictor/labels/uteri.csv")

    for k in range(K):
        generative_model("resnet18", k, cnv=cnv)  # 这里注释掉就是不训练模型，只跑test。
        path = os.getcwd()+f'/resultswithclin6_3/resnet18_{k}'  # 路径是保存的模型的权重参数。
        if cnv:
            path = path + '_cnv'
        model_ft = torch.load(path + '.pkl')
        test(model_ft, "resnet18", k)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script',
                                     epilog="authorized by geneis ")
    parser.add_argument('--classification', type=int, default=2)  # 分两类
    parser.add_argument('--K', type=int, default=5)  # 5折交叉验证
    parser.add_argument('--cnv', type=bool, default=True)  # 使用临床信息
    args = parser.parse_args()

    origirn_classfication_set = None  # 原始分类设置为空

    main(origirn_classfication_set, args.classification, args.K, args.cnv)
