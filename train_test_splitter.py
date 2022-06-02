# -*- coding: UTF-8 -*-
import os
import random
import shutil
from glob import glob  # 查找符合特定规则的文件路径名，相当于文件搜索。查找文件只用到三个匹配符：””, “?”, “[]”
import pandas as pd
import numpy as np
import argparse
import yaml
from utils.common import logger
from pathlib import Path  # 路径
import time

from sklearn.model_selection import StratifiedKFold  # 交叉验证

with open(os.path.join(os.getcwd(), 'config/config.yml'), 'r', encoding='utf8') as fs:  # 获取当前目录，并组合成新目录
    cfg = yaml.load(fs, Loader=yaml.FullLoader)  # 读取yaml文件。默认加载器（FullLoader）
K = cfg['k']  # 将yaml中的k值赋值给K

# available_policies = {'MSI':1, 'MSS':0}
available_policies = {}  # 存放分类的类别？
#多阈值
save_paths ={"halves":os.getcwd()+f"/database6_3/",
            "trisection":os.getcwd()+f"database3/"}


def allDataToTrain(X, y, divisionMethod):  # 未进行交叉验证的
    train_data = []
    for (p, label) in zip(X, y): # 并行遍历   p 和label是将要被赋值的，X，y的值传给P,label。
        for img in glob(p+"/*"):  # 文件路径查找匹配，匹配p到img中。
            train_data.append((img, label))  # 将图片和标签放进train_data中
    
    pdf = pd.DataFrame(train_data, columns=['img', 'label']).sort_values(by=['label'], ascending=True).reset_index(drop = True) # .reset_index(drop = True防止index改变。
    # pdf是train_data，train_data中包括img和label，按照其中的label进行升序排序。
    # Get the smallest number of image in each category
    min_num = min(pdf['label'].value_counts())  # value_counts计算“MSI”和“MSS”在Label这一列中的数量——61个。
    
    # Random downsampling 下采样
    index = []
    for i in range(2):  # 2=两类，i=0或者1，（取不到2）.  数据预处理时会将样本数量多的标记为1（MSS标记为1），MSI标记为0？
        if i == 0:  # 0 是MSS,
            start = 0
            end = pdf['label'].value_counts()[i]  # end是：
        else:
            start = end
            end = end + pdf['label'].value_counts()[i]  # value_counts（）是计数的
        index = index + random.sample(range(start, end), min_num)
        
    pdf = pdf.iloc[index].reset_index(drop = True)  # 最好加这一句，防止index乱了， drop=True: 把原来的索引index列去掉，丢掉
    # Shuffle
    pdf = pdf.reindex(np.random.permutation(pdf.index)).reset_index(drop = True)  # np.random.permutation 生成随机序列
    pdf.to_csv(save_paths[divisionMethod] + f"train.csv", index=None, header=None)  # 保存到save_path路径下的“train.csv”，index=None，header=None：无表头时候要这么设置，因为to_csv会自动读取表头。
    # 所有的数据都做train？


def useCrossValidation(X, y, divisionMethod):  # 定义进行了交叉验证的，divisionMethod分类方法
    print(divisionMethod)  # 打印：halves
    skf = StratifiedKFold(n_splits=K, shuffle=True)  # n_splits 是进行几折的，     shuffle=True是每次划分的结果都不一样，表示经过洗牌。分层交叉验证

    for fold, (train, test) in enumerate(skf.split(X, y)):
        train_data = []
        test_data = []

        train_set, train_label = pd.Series(X).iloc[train].tolist(), pd.Series(y).iloc[train].tolist()  # tolist矩阵转化成列表
        test_set, test_label = pd.Series(X).iloc[test].tolist(), pd.Series(y).iloc[test].tolist()
       
        
        for (data, label) in zip(train_set, train_label):
           # print(data, label)
            for img in glob(data+'/*'):
                train_data.append((img, label)) 
        for (data, label) in zip(test_set, test_label):
            for img in glob(data+'/*'):
                test_data.append((img, label))

        pdf = pd.DataFrame(train_data, columns=['img', 'label']).sort_values(by=['label'], ascending=True).reset_index(drop = True)  # pdf中存的是train_data（包括in=mg ,label)
        
        # Get the smallest number of image in each category
        min_num = min(pdf['label'].value_counts())
        
        # Random downsampling
        index = []
        for i in range(2):  # 2=2类
            if i == 0:
                start = 0
                end = pdf['label'].value_counts()[i]
            else:
                start = end
                end = end + pdf['label'].value_counts()[i]
            index = index + random.sample(range(start, end), min_num)
            
        pdf = pdf.iloc[index].reset_index(drop = True)

        # Shuffle
        pdf = pdf.reindex(np.random.permutation(pdf.index)).reset_index(drop = True)
        print(save_paths[divisionMethod] + f"train_{fold}.csv")
        pdf.to_csv(save_paths[divisionMethod] + f"train_{fold}.csv", index=None, header=None)

        pdf1 = pd.DataFrame(test_data)
        pdf1.to_csv(save_paths[divisionMethod] + f"test_{fold}.csv", index=None, header=None)


def main(srcImg, label, divisionMethod, isCrossValidation=True, shuffle=True):
    assert os.path.exists(label), "Error: 标签文件不存在"  # 判断label是否存在
    assert Path(label).suffix == '.csv', "Error: 标签文件需要是csv文件"  # 查看文件的后缀

    try:
        df = pd.read_csv(label, usecols=["TCGA_ID", "MSI_result"])  # usecols函数实现读取指定列
    except :
        print("Error: 未在文件中发现TCGA_ID或MSI_result列信息")
    
    img_dir = glob(os.path.join(srcImg, '*'))
    xml_file_seq = [img.split('/')[-2] for img in img_dir]

    msi_label_seq = [getattr(row, 'TCGA_ID') for row in df.itertuples() if getattr(row, 'MSI_result') == "MSI"]
    mss_label_seq = [getattr(row, 'TCGA_ID') for row in df.itertuples() if getattr(row, 'MSI_result') == "MSS"]
    
    assert msi_label_seq != 0, "Error: 数据分布异常"
    assert mss_label_seq != 0, "Error: 数据分布异常"

    X  = []
    y = []

    for msi in msi_label_seq:
        if os.path.join(srcImg, msi) in img_dir:
            # print(os.path.join(srcImg, msi))
            X.append(os.path.join(srcImg, msi))
            y.append(1)
    for mss in mss_label_seq:
        if os.path.join(srcImg, mss) in img_dir:
            X.append(os.path.join(srcImg, mss))
            y.append(0)

    if isCrossValidation:
        useCrossValidation(X, y, divisionMethod)  # useCrossValidation 和 allDataToTrain只需要一个就行，但是我们肯定做了交叉验证，所以只用useCrossValidation。
    else:
        allDataToTrain(X, y, divisionMethod)
 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script',
                                     epilog="authorized by geneis ")
    parser.add_argument('--stained_tiles_home', type=str, default="/home/yangjy/CRC1/")
    parser.add_argument('--label_dir_path', type=str, default="/home/yangjy/msipredictor-master/labels/clintzheng.csv")
    parser.add_argument('--divisionMethod', type=str, default="halves")
    parser.add_argument("--isCrossValidation", type=bool, default=True)
    args = parser.parse_args()
    main(args.stained_tiles_home, args.label_dir_path,args.divisionMethod, args.isCrossValidation)
    
