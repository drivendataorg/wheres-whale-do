import sys 
from tqdm import tqdm
import math
import random
from pathlib import Path
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import cv2
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import torch
import timm
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torch.optim import Adam, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from utils import *
from config import CFG
from dataset import *
from augs import *
from model import *
from scheduler import *

PROJ_DIRECTORY = Path.cwd().parent 
DATA_DIRECTORY = PROJ_DIRECTORY / "data"
metadata = pd.read_csv(DATA_DIRECTORY / "metadata.csv", index_col="image_id")
query_scenarios = pd.read_csv(DATA_DIRECTORY / "query_scenarios.csv", index_col="scenario_id")
seed_torch(CFG.SEED)
labels = set(metadata['whale_id'].values)
labels = list(labels)
labels = sorted(labels)
classes = []
for i in metadata['whale_id'].values:
    classes.append(labels.index(i))   
metadata["label"] = classes
metadata = metadata[metadata["viewpoint"] == "top"]
folds = metadata.copy().reset_index()
Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.SEED)
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds['label'])):
    folds.loc[val_index, 'fold'] = int(n)
folds['fold'] = folds['fold'].astype(int)


def train_fn(dataloader,model,criterion,optimizer,device,scheduler,epoch):
    model.train()
    loss_score = AverageMeter()
    criterion2 = torch.nn.MSELoss()
    tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
    for bi,d in tk0:        
        batch_size = d[0].shape[0]
        images = d[0]
        targets = d[1]
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        output = model(images,targets)        
        loss = criterion(output,targets)        
        loss.backward()
        optimizer.step()        
        loss_score.update(loss.detach().item(), batch_size)
        tk0.set_postfix(Train_Loss=loss_score.avg,Epoch=epoch,LR=optimizer.param_groups[0]['lr'])        
    if scheduler is not None:
            scheduler.step()        
    return loss_score

def eval_fn(data_loader,model,criterion,device):    
    loss_score = AverageMeter()    
    model.eval()
    tk0 = tqdm(enumerate(data_loader), total=len(data_loader))    
    with torch.no_grad():        
        for bi,d in tk0:
            batch_size = d[0].size()[0]
            image = d[0]
            targets = d[1]
            image = image.to(device)
            targets = targets.to(device)
            output = model(image,targets)
            loss = criterion(output,targets)            
            loss_score.update(loss.detach().item(), batch_size)
            tk0.set_postfix(Eval_Loss=loss_score.avg)            
    return loss_score

def run(fold = 0, model_name = "effb5"):        
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index
    train = folds.loc[trn_idx].reset_index(drop=True)
    valid = folds.loc[val_idx].reset_index(drop=True)
    # Defining DataSet
    if model_name == "effv2":
        train_dataset = TrainDataset(
        csv=train,
        transforms=get_train_transforms2(),
        transforms2=get_valid_transforms(),
        )
    else:
        train_dataset = TrainDataset(
            csv=train,
            transforms=get_train_transforms(),
            transforms2=get_valid_transforms(),
        )
        
    valid_dataset = TrainDataset(
        csv=valid,
        transforms=get_valid_transforms(),
    )
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CFG.TRAIN_BATCH_SIZE,
        pin_memory=True,
        drop_last=True,
        num_workers=CFG.NUM_WORKERS
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=CFG.VALID_BATCH_SIZE,
        num_workers=CFG.NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    
    # Defining Device
    device = torch.device("cuda")    
    # Defining Model for specific fold
    if model_name == "effv2":
        CFG.model_name = 'efficientnetv2_rw_m'
        model = WhaleNet2(**CFG.model_params2)
    else:
        model = WhaleNet(**CFG.model_params)
    model.to(device)    
    #DEfining criterion
    criterion = fetch_loss()
    criterion.to(device)        
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.scheduler_params['lr_start'])    
    #Defining LR SCheduler
    scheduler = WhaleScheduler(optimizer,**CFG.scheduler_params)
        
    # THE ENGINE LOOP
    best_loss = 10000
    for epoch in range(CFG.EPOCHS):
        train_loss = train_fn(train_loader, model,criterion, optimizer, device,scheduler=scheduler,epoch=epoch)        
        valid_loss = eval_fn(valid_loader, model, criterion,device)        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(),f'model_{CFG.model_name}_IMG_SIZE_{CFG.DIM[0]}_{CFG.loss_module}.bin')
            print('best model found for epoch {}'.format(epoch))