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
tb = SummaryWriter('/home/zyj/Desktop/hkm/code/mmdl/runs/')  


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=50, cnv=False):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())  
    best_acc = 0.0

    if cnv:
        cnv_feature=pd.read_csv('/home/zyj/Desktop/hkm/code//mmdl/label/clin6-aftercorrelation1.csv')  
        peoples=[i for i in cnv_feature.id]
        features=[cnv_feature[i] for i in cnv_feature.columns[1:]]
        min_max_scaler = preprocessing.MinMaxScaler()  
        cnv_features = min_max_scaler.fit_transform(features)  
    
    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))

        phase = 'train'
        model.train()  

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data
        for inputs_, labels_, names_, _ in dataloaders[phase]:
            inputs_ = inputs_.to(device)
            labels_ = labels_.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                if cnv:
                    X_train_minmax = [cnv_features[:,peoples.index(i)] for i in names_]
                    outputs_ = model(inputs_, torch.from_numpy(np.array(X_train_minmax, dtype=np.float32)).to(device))
                else:
                    outputs_ = model(inputs_)
                _, preds = torch.max(outputs_, 1)
                loss = criterion(outputs_, labels_)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs_.size(0)
            running_corrects += torch.sum((preds == labels_.data).int())

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects / dataset_sizes[phase]

        scheduler.step()
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
