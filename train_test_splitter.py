# -*- coding: UTF-8 -*-
import os
import random
import shutil
from glob import glob  
import pandas as pd
import numpy as np
import argparse
import yaml
from utils.common import logger
from pathlib import Path  
import time

from sklearn.model_selection import StratifiedKFold  

#with open(os.path.join(os.getcwd(), 'config/config.yml'), 'r', encoding='utf8') as fs:  
    #cfg = yaml.load(fs, Loader=yaml.FullLoader)  
#K = cfg['k']  
K=2
available_policies = {}  

save_paths ={"halves":os.getcwd()+f"/data/",
            "trisection":os.getcwd()+f"/data/"}


def allDataToTrain(X, y, divisionMethod):  
    train_data = []
    for (p, label) in zip(X, y): 
        for img in glob(p+"/*"):  
            train_data.append((img, label)) 
    
    pdf = pd.DataFrame(train_data, columns=['img', 'label']).sort_values(by=['label'], ascending=True).reset_index(drop = True) 
    # Get the smallest number of image in each category
    min_num = min(pdf['label'].value_counts())  
    
    # Random downsampling 
    index = []
    for i in range(2): 
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
    pdf.to_csv(save_paths[divisionMethod] + f"train.csv", index=None, header=None)  

def useCrossValidation(X, y, divisionMethod):   
    skf = StratifiedKFold(n_splits=K, shuffle=True)  

    for fold, (train, test) in enumerate(skf.split(X, y)):
        train_data = []
        test_data = []

        train_set, train_label = pd.Series(X).iloc[train].tolist(), pd.Series(y).iloc[train].tolist()  
        test_set, test_label = pd.Series(X).iloc[test].tolist(), pd.Series(y).iloc[test].tolist()
       
        
        for (data, label) in zip(train_set, train_label):
            #print(data, label)
            for img in glob(data+'/*'):
                train_data.append((img, label)) 
        for (data, label) in zip(test_set, test_label):
            for img in glob(data+'/*'):
                test_data.append((img, label))

        pdf = pd.DataFrame(train_data, columns=['img', 'label']).sort_values(by=['label'], ascending=True).reset_index(drop = True)  
        
        # Get the smallest number of image in each category
        min_num = min(pdf['label'].value_counts())
        
        # Random downsampling
        index = []
        for i in range(2):  
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
    assert os.path.exists(label), "Error: 标签文件不存在"  
    assert Path(label).suffix == '.csv', "Error: 标签文件需要是csv文件"  

    try:
        df = pd.read_csv(label, usecols=["TCGA_ID", "TMB20"])  
    except :
        print("Error: 未在文件中发现TCGA_ID或TMB20列信息")
    
    img_dir = glob(os.path.join(srcImg, '*'))
    xml_file_seq = [img.split('/')[-2] for img in img_dir]

    msi_label_seq = [getattr(row, 'TCGA_ID') for row in df.itertuples() if getattr(row, 'TMB20') == 1]
    mss_label_seq = [getattr(row, 'TCGA_ID') for row in df.itertuples() if getattr(row, 'TMB20') == 0]
    
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
        useCrossValidation(X, y, divisionMethod)  
    else:
        allDataToTrain(X, y, divisionMethod)
 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script',
                                     epilog="authorized by geneis ")
    parser.add_argument('--stained_tiles_home', type=str, default="/home/zyj/Desktop/hkm/code/crctcga/tiles_cn/")
    parser.add_argument('--label_dir_path', type=str, default="/home/zyj/Desktop/hkm/code/mmdl/label/colo_tmb_label4.csv")
    parser.add_argument('--divisionMethod', type=str, default="halves")
    parser.add_argument("--isCrossValidation", type=bool, default=True)
    args = parser.parse_args()
    main(args.stained_tiles_home, args.label_dir_path,args.divisionMethod, args.isCrossValidation)
    
