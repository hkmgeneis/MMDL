# !/usr/bin/python
# coding: utf-8

"""
Created on Feb 4, 2021
Update  on
__Author__: xisx
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
from scipy import interp
from itertools import cycle
from utils.common import logger
import argparse
import torch
import os
import sys
sys.path.append('..')

y_pred0=np.load(os.getcwd() + f'/results/y_pred_0.npy')
y_pred1=np.load(os.getcwd() + f'/results/y_pred_1.npy')
#y_pred2=np.load('/home/ubuntu/hkm/results/tmb5/top20/top201/y_pred_2.npy')
#y_pred3=np.load('/home/ubuntu/hkm/results/tmb5/top20/top201/y_pred_3.npy')
#y_pred4=np.load('/home/ubuntu/hkm/results/tmb5/top20/top201/y_pred_4.npy')
predict=[]
predict=np.append(y_pred0,y_pred1)
#predict=np.append(y_pred2,predict)
#predict=np.append(predict,y_pred3)
#predict=np.append(predict,y_pred4)
predict=np.array(predict)
np.save(os.getcwd() + f'/results/pred.npy',predict)

y_true0=np.load(os.getcwd() + f'/results/y_true_0.npy')
y_true1=np.load(os.getcwd() + f'/results/y_true_1.npy')
#y_true2=np.load('/home/ubuntu/hkm/results/tmb5/top20/top201/y_true_2.npy')
#y_true3=np.load('/home/ubuntu/hkm/results/tmb5/top20/top201/y_true_3.npy')
#y_true4=np.load('/home/ubuntu/hkm/results/tmb5/top20/top201/y_true_4.npy')
true=[]
true=np.append(y_true0, y_true1)
#true=np.append(true,y_true2)
#true=np.append(true,y_true3)
#true=np.append(true,y_true4)
true=np.array(true)
np.save(os.getcwd() + f'/results/true.npy',true)

# plot confusion-matrix
sns.set()
f, ax = plt.subplots(figsize=(9,6))
c2 = confusion_matrix(np.array(predict),np.array(true))
sns.heatmap(c2, annot=True, fmt="d", ax=ax, cmap='Blues')  
ax.set_title('confusion matric')   
ax.set_xlabel('true')
ax.set_ylabel('predict')     
plt.show()
plt.savefig(os.getcwd() + f'/results/cm-tmb2.svg')
    
p = precision_score(true,predict, average='macro')
r = recall_score(true,predict, average='macro')
f1score= f1_score(true,predict, average='macro')
print('precision:',p)
print('recall:',r )
print('f1score:',f1score)
