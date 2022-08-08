import sys
import os
path = os.path.dirname(os.path.realpath(__file__))
print(path)
sys.path.append(f'{path}/whale_chen/')

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader, ConcatDataset

import pytorch_lightning as pl

import pandas as pd
import numpy as np
import gc
import json

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm
from functools import partial
import time
import datetime
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


from data import *
from models import *
from utils import *
from loss import *
import glob


now = datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
print(f'[ ! ] Start is : {now}')


class Config:
    data_dir = '/code_execution/data/'
    # train_dir = data_dir + '/train_images/' # images 51033
    train_dir = data_dir + '/train_images_new/' # images 51033
    test_dir = data_dir + '/test_images/' # images 27956
    crop_train_dir = data_dir + '/images/' # images 51033
    crop_test_dir = data_dir + '/test_images_yolo_x6_public_ensemble_crop_640/' # images 27956
    
    resample_train_dir = data_dir + '/train_resample/'
    # for new data
    folds_csv_path = data_dir + 'train_5folds_new.csv'
    test_csv_path = data_dir + 'sample_submission.csv'
    
    do_crop = True
    do_mask = False
    do_resample = False
    labels_json_path = f'{data_dir}/labels.json'
    species_json_path = f'{data_dir}/species.json'
    ids_json_path = f'{path}/individual_ids_new.json'
    

    with open(ids_json_path,'r') as f:
        individual_ids_mapping = json.load(f)
    # individual_ids_mapping = None
        
    
    individual_ids_num_classes = len(individual_ids_mapping)
    
    model_name = 'convnext_large'
    
    image_size = 320           ### input size in training
    embedding_size = 512
    dropout = 0.4
    
    device = 'cuda'             ### set gpu or cpu mode
    debug = False              ### debug flag for checking your modify code
    
    gpus = 2                 ### gpu numbers
    precision = 16             ### training precision 8, 16,32, etc
    batch_size = 32            ### total batch size
    

    lr = 3e-4     ### learning rate default 1e-4,effnet, 2.5e-5,swin transformer
    min_lr = 1e-6              ### min learning rate
    weight_decay = 1e-6
    num_workers = 24            ### number workers
    print_freq = 100            ### print log frequency

    seed = 42
    n_fold = 5
    trn_fold = [0,1,2,3,4]
    
    optimizer = 'Adam'
    scheduler = 'GradualWarmupSchedulerV2' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'GradualWarmupSchedulerV2']
    
    epochs = 50
    warmup_epochs = 3
    cosine_epochs = epochs - warmup_epochs
#     epochs = warmup_epochs + cosine_epochs ### total training epochs
    multiplier = 10
   
    freeze_epochs = 5
  
    ### augment
    do_cutout = False
    cutout_prob = 0.5
    
    mixup_epochs = 0
    do_mixup = False
    mixup_prob = 0.5
    mixup_alpha = 1.0

    do_fmix = False
    fmix_prob = 0.25

    do_cutmix = False
    cutmix_prob = 0.25

    do_snapmix = False
    snapmix_prob = 0.5
    snapmix_beta = 1.0

    criterion = 'LabelSmoothingCrossEntropy'  ##ClassBalancedLabelSmoothingCrossEntropy
    label_smoothing = 0.2 #best
    
    fl_gamma = 6.0
    mt_type='arcface' #['cls','arcface','arcface_adaptive']
    # ArcFace Hyperparameters
    s = 20.0
    m = 0.30
    ls_eps = 0.0
    easy_margin = False
    
    s_p4 = s
    m_p4 = m
    
    s_p3 = s
    m_p3 = m
    
    save_row_num = 8
    
    save_dir = f'{model_name}_folds_aug5_{image_size}_{epochs}e_{optimizer}_{scheduler}\
lr{lr}_{criterion}_ls{label_smoothing}_{mt_type}_margin{m}_dropout{dropout}_3head_yolo_x6_public_ensemble_fl_rp/'
        
CFG2 = Config


class Config:
    data_dir = '/code_execution/data/'
    # train_dir = data_dir + '/train_images/' # images 51033
    train_dir = data_dir + '/train_images_new/' # images 51033
    test_dir = data_dir + '/test_images/' # images 27956
    crop_train_dir = data_dir + '/images/' # images 51033
    crop_test_dir = data_dir + '/test_images_yolo_x6_public_ensemble_crop_640/' # images 27956
    
    resample_train_dir = data_dir + '/train_resample/'
    # for new data
    folds_csv_path = data_dir + 'train_5folds_new.csv'
    test_csv_path = data_dir + 'sample_submission.csv'
    
    do_crop = True
    do_mask = False
    do_resample = False
    labels_json_path = f'{data_dir}/labels.json'
    species_json_path = f'{data_dir}/species.json'
    ids_json_path = f'{path}/individual_ids_new.json'
    

    with open(ids_json_path,'r') as f:
        individual_ids_mapping = json.load(f)
    # individual_ids_mapping = None
        
    
    individual_ids_num_classes = len(individual_ids_mapping)
    
    model_name = 'tf_efficientnet_b5_ns'
    
    image_size = 384           ### input size in training
    embedding_size = 512
    dropout = 0.2
    
    device = 'cuda'             ### set gpu or cpu mode
    debug = False              ### debug flag for checking your modify code
    
    gpus = 2                 ### gpu numbers
    precision = 16             ### training precision 8, 16,32, etc
    batch_size = 32            ### total batch size
    

    lr = 3e-4     ### learning rate default 1e-4,effnet, 2.5e-5,swin transformer
    min_lr = 1e-6              ### min learning rate
    weight_decay = 1e-6
    num_workers = 24            ### number workers
    print_freq = 100            ### print log frequency

    seed = 42
    n_fold = 5
    trn_fold = [0,1,2,3,4]
    
    optimizer = 'Adam'
    scheduler = 'GradualWarmupSchedulerV2' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'GradualWarmupSchedulerV2']
    
    epochs = 50
    warmup_epochs = 3
    cosine_epochs = epochs - warmup_epochs
#     epochs = warmup_epochs + cosine_epochs ### total training epochs
    multiplier = 10
   
    freeze_epochs = 5
  
    ### augment
    do_cutout = False
    cutout_prob = 0.5
    
    mixup_epochs = 0
    do_mixup = False
    mixup_prob = 0.5
    mixup_alpha = 1.0

    do_fmix = False
    fmix_prob = 0.25

    do_cutmix = False
    cutmix_prob = 0.25

    do_snapmix = False
    snapmix_prob = 0.5
    snapmix_beta = 1.0

    criterion = 'LabelSmoothingCrossEntropy'  ##ClassBalancedLabelSmoothingCrossEntropy
    label_smoothing = 0.2 #best
    
    fl_gamma = 6.0
    mt_type='arcface' #['cls','arcface','arcface_adaptive']
    # ArcFace Hyperparameters
    s = 20.0
    m = 0.30
    ls_eps = 0.0
    easy_margin = False
    
    s_p4 = s
    m_p4 = m
    
    s_p3 = s
    m_p3 = m
    
    save_row_num = 8
    
    save_dir = f'{model_name}_folds_aug_{image_size}_{epochs}e_{optimizer}_{scheduler}\
lr{lr}_{criterion}_ls{label_smoothing}_{mt_type}_margin{m}_3head_yolo_x6_public_ensemble_fl_rp/'
    
CFG3 = Config


class Config:
    data_dir = '/code_execution/data/'
    # train_dir = data_dir + '/train_images/' # images 51033
    train_dir = data_dir + '/train_images_new/' # images 51033
    test_dir = data_dir + '/test_images/' # images 27956
    crop_train_dir = data_dir + '/images/' # images 51033
    crop_test_dir = data_dir + '/test_images_yolo_x6_public_ensemble_crop_640/' # images 27956
    
    resample_train_dir = data_dir + '/train_resample/'
    # for new data
    folds_csv_path = data_dir + 'train_5folds_new.csv'
    test_csv_path = data_dir + 'sample_submission.csv'
    
    do_crop = True
    do_mask = False
    do_resample = False
    labels_json_path = f'{data_dir}/labels.json'
    species_json_path = f'{data_dir}/species.json'
    ids_json_path = f'{path}/individual_ids_new.json'
    

    with open(ids_json_path,'r') as f:
        individual_ids_mapping = json.load(f)
    # individual_ids_mapping = None
        
    
    individual_ids_num_classes = len(individual_ids_mapping)
    
    model_name = 'tf_efficientnet_b7_ns'
    
    image_size = 320           ### input size in training
    embedding_size = 512
    dropout = 0.4
    
    device = 'cuda'             ### set gpu or cpu mode
    debug = False              ### debug flag for checking your modify code
    
    gpus = 2                 ### gpu numbers
    precision = 16             ### training precision 8, 16,32, etc
    batch_size = 32            ### total batch size
    

    lr = 3e-4     ### learning rate default 1e-4,effnet, 2.5e-5,swin transformer
    min_lr = 1e-6              ### min learning rate
    weight_decay = 1e-6
    num_workers = 24            ### number workers
    print_freq = 100            ### print log frequency

    seed = 42
    n_fold = 5
    trn_fold = [0,1,2,3,4]
    
    optimizer = 'Adam'
    scheduler = 'GradualWarmupSchedulerV2' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'GradualWarmupSchedulerV2']
    
    epochs = 50
    warmup_epochs = 3
    cosine_epochs = epochs - warmup_epochs
#     epochs = warmup_epochs + cosine_epochs ### total training epochs
    multiplier = 10
   
    freeze_epochs = 5
  
    ### augment
    do_cutout = False
    cutout_prob = 0.5
    
    mixup_epochs = 0
    do_mixup = False
    mixup_prob = 0.5
    mixup_alpha = 1.0

    do_fmix = False
    fmix_prob = 0.25

    do_cutmix = False
    cutmix_prob = 0.25

    do_snapmix = False
    snapmix_prob = 0.5
    snapmix_beta = 1.0

    criterion = 'LabelSmoothingCrossEntropy'  ##ClassBalancedLabelSmoothingCrossEntropy
    label_smoothing = 0.2 #best
    
    fl_gamma = 6.0
    mt_type='arcface' #['cls','arcface','arcface_adaptive']
    # ArcFace Hyperparameters
    s = 20.0
    m = 0.30
    ls_eps = 0.0
    easy_margin = False
    
    s_p4 = s
    m_p4 = m
    
    s_p3 = s
    m_p3 = m
    
    save_row_num = 8
    
    save_dir = f'{model_name}_folds_aug_{image_size}_{epochs}e_{optimizer}_{scheduler}\
lr{lr}_{criterion}_ls{label_smoothing}_{mt_type}_margin{m}_dropout{dropout}_3head_yolo_x6_public_ensemble_fl_rp/'
    
CFG4 = Config

class CustomPLModel(pl.LightningModule):
    def __init__(self):
        super(CustomPLModel,self).__init__()
        self.model = MultiHeadNet(CFG, pretrained=False)
#         self.model = DOLGNet(CFG, pretrained=False)


    def forward(self, x):
        return self.model.predict(x)
    
def apply_tta(input):
    inputs = []
    inputs.append(input)
#     inputs.append(torch.flip(input, dims=[2]))
    inputs.append(torch.flip(input, dims=[3]))
#     inputs.append(torch.rot90(input, k=1, dims=[2, 3]))
#     inputs.append(torch.rot90(input, k=2, dims=[2, 3]))
#     inputs.append(torch.rot90(input, k=3, dims=[2, 3]))
#     inputs.append(torch.rot90(torch.flip(input, dims=[2]), k=1, dims=[2, 3]))
#     inputs.append(torch.rot90(torch.flip(input, dims=[2]), k=3, dims=[2, 3]))
    return inputs


def do_predict_models(models, loader, tta=['']):
    total_embeddings = [[] for i, _ in enumerate(models)]
    total_targets = []
    total_image_ids = []
    total_num = 0
    for model in models:
        model.eval()
    for t, (image, targets, image_ids) in enumerate(loader):
        batch_size = image.size(0)
        image = image.to(device)
        for i, model in enumerate(models):
            with torch.no_grad():
                features_list = []
                for image in apply_tta(image):
                    features = model(image)
                    features_list.append(features)

            features = torch.stack(features_list,0).mean(0).detach().cpu().numpy()
            total_embeddings[i].append(features)

        total_num += batch_size
        total_targets.append(targets)
        total_image_ids.append(image_ids)
    assert(total_num == len(loader.dataset))
    total_embeddings = [np.concatenate(x) for x in total_embeddings]
    total_targets = np.concatenate(total_targets)
    total_image_ids = np.concatenate(total_image_ids)
    return total_embeddings,total_targets,total_image_ids

scenarios = pd.read_csv('/code_execution/data/query_scenarios.csv')

todo_images = []
query_paired = []
for i, x in scenarios.iterrows():
    qu = pd.read_csv(f'/code_execution/data/{x.queries_path}')
    todo_images.extend(qu.query_image_id.unique())
    db = pd.read_csv(f'/code_execution/data/{x.database_path}')
    todo_images.extend(db.database_image_id.unique())
    query_paired.append((qu, db))
    
todo_images = list(set(todo_images))

train = pd.DataFrame({'image_id': todo_images})
train['path'] = train['image_id'] + '.jpg'
train['whale_id'] = 'whale000'
train['image'] = train['image_id'] + '.jpg'
train['individual_id'] = 'whale000'

target_encodings = {CFG3.individual_ids_mapping[x]:x for x in CFG3.individual_ids_mapping}

def predict_cfg_with_paths(df, CFG, paths):
    # embbedding dfs    
    train_embedding_dfs = []
    train_dataset =ValidMultiHeadDataset(
            CFG, 
            train, 
            transforms=get_val_transforms(CFG)) 

    train_dataloader = DataLoader(
        train_dataset,
        8, 
        num_workers=0, 
        shuffle=False,
        pin_memory=True)

    pl_models = []
    for path in paths:
        pl_model = CustomPLModel()
        print(path)
        pl_model = pl_model.load_from_checkpoint(path)
        pl_model.to(device)
        pl_models.append(pl_model)
        
    train_embeddings, train_targets, train_image_ids = do_predict_models(pl_models, train_dataloader)
    for train_embedding in train_embeddings:
        train_embedding = torch.tensor(train_embedding)
        train_embedding = torch.nn.functional.normalize(train_embedding).numpy()
        train_embedding = torch.tensor(train_embedding)
        train_embedding = torch.nn.functional.normalize(train_embedding).numpy()
        train_embeddings_df = pd.DataFrame(train_embedding, index=train['image_id'])
        train_embedding_dfs.append(train_embeddings_df)
    return train_embedding_dfs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# LOGGER = get_log(file_name=f'{path}/valid.log')
# LOGGER.info(f'Validing model {CFG.model_name}, \
# params with batch_size={CFG.batch_size}, image_size={CFG.image_size}, \
# scheduler={CFG.scheduler}, init_lr={CFG.lr}, mt={CFG.mt_type}, m={CFG.m},s={CFG.s}.')


train_embedding_dfs = []

start = time.time()
# config2
CFG = CFG2
# for path in glob.glob(f'{CFG2.save_dir}/*/*/mAP5_best.ckpt'):
#     train_embedding_dfs.append(predict_cfg(train, CFG2, path))

train_embedding_dfs.extend(predict_cfg_with_paths(train, CFG2, glob.glob(f'{CFG2.save_dir}/*/*/mAP5_best.ckpt')))

now = datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
print(f'[ ! ] Now is : {now}')

CFG = CFG3
# for path in glob.glob(f'{CFG3.save_dir}/*/*/mAP5_best.ckpt'):
#     train_embedding_dfs.append(predict_cfg(train, CFG3, path))

train_embedding_dfs.extend(predict_cfg_with_paths(train, CFG3, glob.glob(f'{CFG3.save_dir}/*/*/mAP5_best.ckpt')))

now = datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
print(f'[ ! ] Now is : {now}')


CFG = CFG4

train_embedding_dfs.extend(predict_cfg_with_paths(train, CFG4, glob.glob(f'{CFG4.save_dir}/*/*/mAP5_best.ckpt')))

# for path in glob.glob(f'{CFG4.save_dir}/*/*/mAP5_best.ckpt'):
#     train_embedding_dfs.append(predict_cfg(train, CFG4, path))

now = datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
print(f'[ ! ] Time used: {time.time() - start}, now is : {now}')


weights = [1] * 9 + [0.75] * 4

# # for fold in CFG.trn_fold:    
# train_dataset =ValidMultiHeadDataset(
#         CFG, 
#         train, 
#         transforms=get_val_transforms(CFG)) 
    
# # valid_dataset =ValidMultiHeadDataset(
# #         CFG, 
# #         df_valid, 
# #         transforms=get_val_transforms(CFG)) 
    
# train_dataloader = DataLoader(
#     train_dataset,
#     8, 
#     num_workers=0, 
#     shuffle=False)
    
# # valid_dataloader = DataLoader(
# #     valid_dataset,
# #     CFG.batch_size*2, 
# #     num_workers=4, 
# #     shuffle=False) 



# # pl_model = pl_model.load_from_checkpoint(f'{path}/mAP5_best.ckpt')
# # pl_model.to(device)
# # train_embeddings, train_targets, train_image_ids = do_predict(pl_model, train_dataloader)
# # # valid_embeddings, valid_targets, valid_image_ids = do_predict(pl_model, valid_dataloader)

# # train_embeddings = torch.tensor(train_embeddings)
# # train_embeddings = torch.nn.functional.normalize(train_embeddings).numpy()
# # train_embeddings_df = pd.DataFrame(train_embeddings, index=train['image_id'])
# # train_embeddings_df.head(2)

# train_embedding_dfs = []
# for f in os.listdir('./tf_efficientnet_b5_ns/'):
#     ckpt_name = f'tf_efficientnet_b5_ns/{f}/mAP5_best.ckpt'
#     print(ckpt_name)
#     pl_model = pl_model.load_from_checkpoint(f'{path}/{ckpt_name}')
#     pl_model.to(device)
#     train_embeddings, train_targets, train_image_ids = do_predict(pl_model, train_dataloader)
#     # valid_embeddings, valid_targets, valid_image_ids = do_predict(pl_model, valid_dataloader)
#     train_embeddings = torch.tensor(train_embeddings)
#     train_embeddings = torch.nn.functional.normalize(train_embeddings).numpy()
#     # train_embeddings_all.append(train_embeddings)
    
#     # train_embeddings = np.stack(train_embeddings_all, -1).mean(2)
#     train_embeddings = torch.tensor(train_embeddings)
#     train_embeddings = torch.nn.functional.normalize(train_embeddings).numpy()
#     train_embeddings_df = pd.DataFrame(train_embeddings, index=train['image_id'])
#     # train_embeddings_df.head(2)
#     train_embedding_dfs.append(train_embeddings_df)



subs = []

for query, database in query_paired:
    sims = []
    for idf, train_embeddings_df in enumerate(train_embedding_dfs):
        db_feature_df = train_embeddings_df.loc[database.database_image_id].copy()
        query_image_df = train_embeddings_df.loc[query.query_image_id].copy()
        sim = pd.DataFrame(np.dot(query_image_df.values, db_feature_df.values.T),
                 index=query_image_df.index, columns=db_feature_df.index).T
        sims.append(sim * weights[idf])
    sim = pd.concat(sims).groupby(level=0).mean()
    for c in sim.columns:
        qid = query[query.query_image_id == c].iloc[0].query_id
        # pr = sim[c].sort_values(ascending=False).head(20)
        if c in sim.index:
            pr = sim.drop(c)[c].sort_values(ascending=False).head(20)
        else:
            pr = sim[c].sort_values(ascending=False).head(20)
        for k, v in dict(pr).items():
            subs.append({
                'query_id': qid,
                'database_image_id': k,
                'score': max(min((v + 1)  * 0.49, 0.9999), 0)
            })     
            
sub = pd.DataFrame(subs)

sub.to_csv('/code_execution/submission/submission.csv', index=False)