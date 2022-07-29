import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.nn import functional as F
from sklearn import metrics
import os
from utils.custom_dset import CustomDset
from utils.common import logger
import csv
import sys
from sklearn import preprocessing
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), # 从中心裁剪
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def test(model, model_name, k=0, K=2, types=0, cnv=False): # cnv=linincal 
    model.eval()

    if cnv:
        cnv_feature=pd.read_csv('/home/zyj/Desktop/hkm/code/mmdl/label/clin6-aftercorrelation1.csv')  
        peoples=[i for i in cnv_feature.id]
        features=[cnv_feature[i] for i in cnv_feature.columns[1:]]
        min_max_scaler = preprocessing.MinMaxScaler()
        cnv_features = min_max_scaler.fit_transform(features)
    
    testset = CustomDset(os.getcwd()+f'/data/test_{k}.csv'.format(k), data_transforms['test'])  
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=4)

    person_prob_dict = dict()
    with torch.no_grad():
        for data in testloader:
            images, labels, names_, images_names = data
            if cnv:
                X_train_minmax = [cnv_features[:,peoples.index(i)] for i in names_]
                outputs = model(images.to(device), torch.from_numpy(np.array(X_train_minmax, dtype=np.float32)).to(device))
            else:
                outputs = model(images.to(device))
            probability = F.softmax(outputs, dim=1).data.squeeze()
            probs = probability.cpu().numpy()
            for i in range(labels.size(0)):
                p = names_[i]
                if p not in person_prob_dict.keys():
                    person_prob_dict[p] = {
                        'prob_0': 0, 
                        'prob_1': 0,
                        'label': labels[i].item(),
                        'img_num': 0}
                if probs.ndim == 2:
                    person_prob_dict[p]['prob_0'] += probs[i, 0]
                    person_prob_dict[p]['prob_1'] += probs[i, 1]
                    person_prob_dict[p]['img_num'] += 1
                else:
                    person_prob_dict[p]['prob_0'] += probs[0]
                    person_prob_dict[p]['prob_1'] += probs[1]
                    person_prob_dict[p]['img_num'] += 1

    
    y_true = []
    y_pred = []
    score_list = []

    total = len(person_prob_dict)
    correct = 0
    for key in person_prob_dict.keys():
        predict = 0
        if person_prob_dict[key]['prob_0'] < person_prob_dict[key]['prob_1']:
            predict = 1
        if person_prob_dict[key]['label'] == predict:
            correct += 1
        y_true.append(person_prob_dict[key]['label'])
        score_list.append([person_prob_dict[key]['prob_0']/person_prob_dict[key]["img_num"],person_prob_dict[key]['prob_1']/person_prob_dict[key]["img_num"]])
        y_pred.append(predict)
        open(f'{model_name}_confusion_matrix_classification_{types}.txt', 'a+').write(str(person_prob_dict[key]['label'])+"\t"+str(predict)+'\n')
    
    np.save(os.getcwd()+f'/results/y_true_{k}.npy', np.array(y_true))  
    np.save(os.getcwd()+f'/results/score_{k}.npy', np.array(score_list))
    np.save(os.getcwd()+f'/results/y_pred_{k}.npy', np.array(y_pred))
    logger.info('Accuracy of the network on test images: %d %%' % (
        100 * correct / total))


    
