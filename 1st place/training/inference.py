# PLEASE NOTICE THIS INFERENCE IS REFACTORED A LITTLE AND IT IS NOT THE ONE USED ON THE COMP
# IN CASE YOU GOT DIFFERENT RESULTS USE THIS SCRIPT
# https://github.com/ammarali32/Where-s-Whale-do_Competition/blob/main/submissions/score_4887_final/main.py

#######################################
#           import libs               #
#######################################
import sys 
from tqdm import tqdm
import math
import random
from pathlib import Path
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
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
from sklearn.metrics.pairwise import cosine_similarity
import gc
#ROOT_DIRECTORY = Path("/code_execution")
ROOT_DIRECTORY = Path("./")
PREDICTION_FILE = ROOT_DIRECTORY / "submission" / "submission.csv"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"

#######################################
#           configuration             #
#######################################

class CFG:
    DIM = (512,512)
    NUM_WORKERS = 4
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 8
    EPOCHS = 20
    SEED = 42
    device = torch.device('cuda')
    model_name = 'tf_efficientnet_b2_ns'
    loss_module = 'arcface' #'cosface' #'adacos'
    s = 30.0
    m = 0.5 
    ls_eps = 0.0
    easy_margin = False
    scheduler_params = {
          "lr_start": 1e-5,
          "lr_max": 5e-4 ,
          "lr_min": 5e-6,
          "lr_ramp_ep": 15,
          "lr_sus_ep": 0,
          "lr_decay": 0.8,
      }
    model_params = {
      'n_classes':788,
      'model_name':'tf_efficientnet_b5_ns',
      'use_fc':False,
      'fc_dim':2048,
      'dropout':0.0,
      'loss_module':loss_module,
      's':30.0,
      'margin':0.50,
      'ls_eps':0.0,
      'theta_zero':0.785,
      'pretrained':False
    }
    model_params2 = {
      'n_classes':788,
      'model_name':'efficientnetv2_rw_m',
      'use_fc':False,
      'fc_dim':2048,
      'dropout':0.0,
      'loss_module':loss_module,
      's':30.0,
      'margin':0.50,
      'ls_eps':0.0,
      'theta_zero':0.785,
      'pretrained':False
    }

#######################################
#           Transforms                #
#######################################
def get_valid_transforms():

    return albumentations.Compose(
        [
            albumentations.Resize(CFG.DIM[0],CFG.DIM[1],always_apply=True),
            albumentations.Normalize(),
        ToTensorV2(p=1.0)
        ]
    )

#######################################
#             Models                  #
#######################################
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'

class WhaleDataset_testing(Dataset):
    def __init__(self, csv, transforms=None):

        self.csv = csv
        self.augmentations = transforms

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        
        image = cv2.imread('data/' + row.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[0] < image.shape[1]:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image2 = np.fliplr(image)
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']       
            augmented = self.augmentations(image=image2)
            image2 = augmented['image']
        
        return {"image_id": self.csv.index[index], "image": image, "image2": image2}
    
def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output
class ElasticArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50,std=0.0125,plus=False):
        super(ElasticArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)
        self.std=std
        self.plus=plus
    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        margin = torch.normal(mean=self.m, std=self.std, size=label[index, None].size(), device=cos_theta.device) # Fast converge .clamp(self.m-self.std, self.m+self.std)
        if self.plus:
            with torch.no_grad():
                distmat = cos_theta[index, label.view(-1)].detach().clone()
                _, idicate_cosie = torch.sort(distmat, dim=0, descending=True)
                margin, _ = torch.sort(margin, dim=0)
            m_hot.scatter_(1, label[index, None], margin[idicate_cosie])
        else:
            m_hot.scatter_(1, label[index, None], margin)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        return cos_theta
    

class WhaleNet_testing(nn.Module):

    def __init__(self,
                 n_classes,
                 model_name='efficientnet_b0',
                 use_fc=False,
                 fc_dim=512,
                 dropout=0.0,
                 loss_module='softmax',
                 s=30.0,
                 margin=0.50,
                 ls_eps=0.0,
                 theta_zero=0.785,
                 pretrained=False):
        """
        :param n_classes:
        :param model_name: name of model from pretrainedmodels
            e.g. resnet50, resnext101_32x4d, pnasnet5large
        :param pooling: One of ('SPoC', 'MAC', 'RMAC', 'GeM', 'Rpool', 'Flatten', 'CompactBilinearPooling')
        :param loss_module: One of ('arcface', 'cosface', 'softmax')
        """
        super(WhaleNet_testing, self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        final_in_features = self.backbone.classifier.in_features
        
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        
        self.pooling =  GeM()#nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(6144)
        self.ln = nn.Linear(6144, final_in_features)
        self.use_fc = use_fc
        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim

        self.loss_module = loss_module
        if loss_module == 'arcface':
            self.final = ElasticArcFace(final_in_features, n_classes,
                                          s=s, m=margin)#, easy_margin=False, ls_eps=ls_eps)
        elif loss_module == 'cosface':
            self.final = AddMarginProduct(final_in_features, n_classes, s=s, m=margin)
        elif loss_module == 'adacos':
            self.final = AdaCos(final_in_features, n_classes, m=margin, theta_zero=theta_zero)
        else:
            self.final = nn.Linear(final_in_features, n_classes)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        feature = self.extract_feat(x)
        return feature

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x1 = self.pooling(x).view(batch_size, -1)
        x2 = self.pooling(x).view(batch_size, -1)
        x3 = self.pooling(x).view(batch_size, -1)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.bn(x)
        x = self.ln(x)
        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)
        return x

class WhaleNet_Eff_v2(nn.Module):

    def __init__(self,
                 n_classes,
                 model_name='efficientnet_b0',
                 use_fc=False,
                 fc_dim=512,
                 dropout=0.0,
                 loss_module='softmax',
                 s=30.0,
                 margin=0.50,
                 ls_eps=0.0,
                 theta_zero=0.785,
                 pretrained=False):
        """
        :param n_classes:
        :param model_name: name of model from pretrainedmodels
            e.g. resnet50, resnext101_32x4d, pnasnet5large
        :param pooling: One of ('SPoC', 'MAC', 'RMAC', 'GeM', 'Rpool', 'Flatten', 'CompactBilinearPooling')
        :param loss_module: One of ('arcface', 'cosface', 'softmax')
        """
        super(WhaleNet_Eff_v2, self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        final_in_features = self.backbone.classifier.in_features
        
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        
        self.pooling =  GeM()#nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(final_in_features)
        self.use_fc = use_fc
        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.bn = nn.BatchNorm1d(fc_dim)
            self.bn.bias.requires_grad_(False)
            self.fc = nn.Linear(final_in_features, n_classes, bias = False)            
            self.bn.apply(weights_init_kaiming)
            self.fc.apply(weights_init_classifier)
            #self._init_params()
            final_in_features = fc_dim

        self.loss_module = loss_module
        if loss_module == 'arcface':
            self.final = ElasticArcFace(final_in_features, n_classes,
                                          s=s, m=margin)#, easy_margin=False, ls_eps=ls_eps)
        elif loss_module == 'cosface':
            self.final = AddMarginProduct(final_in_features, n_classes, s=s, m=margin)
        elif loss_module == 'adacos':
            self.final = AdaCos(final_in_features, n_classes, m=margin, theta_zero=theta_zero)
        else:
            self.final = nn.Linear(final_in_features, n_classes)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        feature = self.extract_feat(x)
        return feature

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)
        x = self.bn(x)
        if self.use_fc:
            x1 = self.dropout(x)
            x1 = self.bn(x1)
            x1 = self.fc(x1)
        return x

def clear_cache(models):
    del models
    gc.collect()
    torch.cuda.empty_cache()
    
# load test set data and pretrained model
query_scenarios = pd.read_csv(DATA_DIRECTORY / "query_scenarios.csv", index_col="scenario_id")
metadata = pd.read_csv(DATA_DIRECTORY / "metadata.csv", index_col="image_id")
scenario_imgs = []
for row in query_scenarios.itertuples():
    scenario_imgs.extend(pd.read_csv(DATA_DIRECTORY / row.queries_path).query_image_id.values)
    scenario_imgs.extend(pd.read_csv(DATA_DIRECTORY / row.database_path).database_image_id.values)
scenario_imgs = sorted(set(scenario_imgs))
metadata = metadata.loc[scenario_imgs]

# instantiate dataset/loader and generate embeddings for all images
test_dataset = WhaleDataset_testing(metadata, transforms=get_valid_transforms())
dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=4,
    num_workers=CFG.NUM_WORKERS,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
)

#######################################
#      EFF V2 M models Top Only       #
#######################################

paths = ["model_efficientnetv2_rw_m_IMG_SIZE_512_arcface_f0_7-05.bin", "model_efficientnetv2_rw_m_IMG_SIZE_512_arcface_f2_6-79.bin",
         "model_efficientnetv2_rw_m_IMG_SIZE_512_arcface_f4_6-99.bin"]
models = []
for i in paths:
    model = WhaleNet_Eff_v2(**CFG.model_params2)
    state = torch.load(i, map_location=torch.device(CFG.device))
    model.load_state_dict(state)
    model.to(CFG.device)
    model.eval()
    models.append(model)
    del state
    gc.collect()
    torch.cuda.empty_cache()
        
embeddings2 = []
for batch in tqdm(dataloader, total=len(dataloader),disable=True):
    b_embeddings = []
    for model in models:
        embed = (model(batch["image"].to(CFG.device)).detach().cpu().numpy() + model(batch["image2"].to(CFG.device)).detach().cpu().numpy()) / 2
        b_embeddings.append(embed)
    batch_embeddings = np.concatenate(([i for i in b_embeddings]), 1)
    batch_embeddings_df = pd.DataFrame(batch_embeddings, index=batch["image_id"])
    embeddings2.append(batch_embeddings_df)
clear_cache(models)

#######################################
#    EFF B5 models Top and Literal    #
#######################################

top_paths = ["model_tf_efficientnet_b5_ns_IMG_SIZE_512_arcface_f2_6-76.bin",
            "model_tf_efficientnet_b5_ns_IMG_SIZE_512_arcface_f0_7-16.bin",
            "model_tf_efficientnet_b5_ns_IMG_SIZE_512_arcface_f4_7-07.bin",
            "model_tf_efficientnet_b5_ns_IMG_SIZE_512_arcface_f1_6-77.bin",
            "model_tf_efficientnet_b5_ns_IMG_SIZE_512_arcface_f3_6-9.bin"]
top_models = []
for i in top_paths:
    model = WhaleNet_testing(**CFG.model_params)
    state = torch.load(i, map_location=torch.device(CFG.device))
    model.load_state_dict(state)
    model.to(CFG.device)
    model.eval()
    top_models.append(model)
    del state
    gc.collect()
    torch.cuda.empty_cache()


literal_paths = ["model_tf_efficientnet_b2_ns_IMG_SIZE_512_arcface_literal_f0_4-38_nv.bin",
            "model_tf_efficientnet_b2_ns_IMG_SIZE_512_arcface_literal_f1_4-52_nv.bin",
            "model_tf_efficientnet_b2_ns_IMG_SIZE_512_arcface_literal_f2_4-09_nv.bin",
            "model_tf_efficientnet_b2_ns_IMG_SIZE_512_arcface_literal_f4_4-8_nv.bin"]
literal_models = []

for i in literal_paths:
    model = WhaleNet_testing(**CFG.model_params)
    state = torch.load(i, map_location=torch.device(CFG.device))
    model.load_state_dict(state)
    model.to(CFG.device)
    model.eval()
    literal_models.append(model)
    del state
    gc.collect()
    torch.cuda.empty_cache()

embeddings = []
embeddings_literal = []

for batch in tqdm(dataloader, total=len(dataloader),disable=True):
    b_embeddings = []
    for model in top_models:
        embed = (model(batch["image"].to(CFG.device)).detach().cpu().numpy() + model(batch["image2"].to(CFG.device)).detach().cpu().numpy()) / 2
        b_embeddings.append(embed)
    batch_embeddings = np.concatenate(([i for i in b_embeddings]), 1)
    batch_embeddings_df = pd.DataFrame(batch_embeddings, index=batch["image_id"])
    
    
    b_embeddings = []
    for model in literal_models:
        embed = (model(batch["image"].to(CFG.device)).detach().cpu().numpy() + model(batch["image2"].to(CFG.device)).detach().cpu().numpy()) / 2
        b_embeddings.append(embed)
    batch_embeddings_literal = np.concatenate(([i for i in b_embeddings]), 1)
    batch_embeddings_literal_df = pd.DataFrame(batch_embeddings, index=batch["image_id"])
    embeddings.append(batch_embeddings_df)
    embeddings_literal.append(batch_embeddings_literal_df)

clear_cache(top_models)
clear_cache(literal_models)


embeddings = pd.concat(embeddings)
embeddings_eff2 = pd.concat(embeddings2)
embeddings_literal = pd.concat(embeddings_literal)

emb2 = embeddings.copy()
emb2_eff2 = embeddings_eff2.copy()
results = []

#######################################
#                Queries              #
#######################################

for row in query_scenarios.itertuples():
    # load query df and database images; subset embeddings to this scenario's database
    qry_df = pd.read_csv(DATA_DIRECTORY / row.queries_path)
    db_img_ids = pd.read_csv(DATA_DIRECTORY / row.database_path).database_image_id.values
    qr_img_ids = qry_df.query_image_id.values
    top_literal = False
    literal_top = False
    metd = metadata.loc[db_img_ids]
    qrmetd = metadata.loc[qr_img_ids]
    embedding = emb2.copy()
    embeddings_eff2 = emb2_eff2.copy()
    ## Check if yo use top or literal embeddings
    if all(i == "top" for i in metd["viewpoint"].values) and all(i == "top" for i in qrmetd["viewpoint"].values):
        embeddings = embeddings
    else:
        embeddings = embeddings_literal.copy()
    if all(i == "top" for i in qrmetd["viewpoint"].values) and all(i != "top" for i in metd["viewpoint"].values):
        top_literal = True
    if all(i != "top" for i in qrmetd["viewpoint"].values) and all(i == "top" for i in metd["viewpoint"].values):
        literal_top = True
 
    db_embeddings = embeddings.loc[db_img_ids]
    db_embeddings_eff2 = embeddings_eff2.loc[db_img_ids]
    db_embeddings2 = emb2.loc[db_img_ids]
    db_embeddings2_eff2 = emb2_eff2.loc[db_img_ids]
    
    # predict matches for each query in this scenario
    for qry in qry_df.itertuples():
        # get embeddings; drop query from database, if it exists
        qry_embedding = embeddings.loc[[qry.query_image_id]]
        qry_embedding_eff2 = embeddings_eff2.loc[[qry.query_image_id]]
        _db_embeddings = db_embeddings.drop(qry.query_image_id, errors='ignore')
        _db_embeddings2 = db_embeddings2.drop(qry.query_image_id, errors='ignore')
        _db_embeddings_eff2 = db_embeddings_eff2.drop(qry.query_image_id, errors='ignore')
        _db_embeddings2_eff2 = db_embeddings2_eff2.drop(qry.query_image_id, errors='ignore')
        
        # compute cosine similarities and get top 2
        sims = cosine_similarity(qry_embedding, _db_embeddings)[0]        
        if not top_literal and not literal_top:
            sims2 = cosine_similarity(qry_embedding_eff2, _db_embeddings_eff2)[0]            
            sims = 0.75 * sims + 0.25 * sims2
        sims[sims < 0 ] = 0
        sims[sims > 1 ] = 1
        top1 = pd.Series(sims, index=_db_embeddings.index).sort_values(0, ascending=False).head(2)
        
        if literal_top:
            # use literal models to get the top1 then usetop models to find top2 from the data base
            # Calculate the similarities for between top2 on database with other database images
            qry2_embedding = emb2.loc[[top1.index[0]]]
            qry2_embedding_eff2 = emb2_eff2.loc[[top1.index[0]]]
            sims2 = cosine_similarity(qry2_embedding, _db_embeddings2)[0]
            sims3 = cosine_similarity(qry2_embedding_eff2, _db_embeddings2_eff2)[0]
            sims2 = (0.25 * sims3 + 0.75 * sims2)
            sims2[sims2 < 0 ] = 0
            sims2[sims2 > 1 ] = 1
            top2 = pd.Series(sims2, index=_db_embeddings2.index).sort_values(0, ascending=False).head(2)
            qry2_embedding = emb2.loc[[top2.index[1]]]
            qry2_embedding_eff2 = emb2_eff2.loc[[top2.index[1]]]
            sims3 = cosine_similarity(qry2_embedding, _db_embeddings2)[0]
            sims4 = cosine_similarity(qry2_embedding_eff2, _db_embeddings2_eff2)[0]
            sims3 = (0.25 * sims4 + 0.75 * sims3)
            sims3[sims3 < 0 ] = 0
            sims3[sims3 > 1 ] = 1
            
        elif top_literal:
            # Use Literal models to get top2 literals from other literals on the database and 
            # Calculate the similarites with other database images to these top2
            qry2_embedding = embeddings.loc[[top1.index[0]]]
            sims2 = cosine_similarity(qry2_embedding, _db_embeddings)[0]
            qry2_embedding = embeddings.loc[[top1.index[1]]]
            sims3 = cosine_similarity(qry2_embedding, _db_embeddings)[0]
            sims3[sims3 < 0 ] = 0
            sims3[sims3 > 1 ] = 1
            
        else:
            # Use top models to get top2 from database and calculate similarites between these
            # top2 and other dataset images
            qry2_embedding = embeddings.loc[[top1.index[0]]]
            qry2_embedding_eff2 = embeddings_eff2.loc[[top1.index[0]]]
            sims2 = cosine_similarity(qry2_embedding, _db_embeddings)[0]
            sims3 = cosine_similarity(qry2_embedding_eff2, _db_embeddings_eff2)[0]
            sims2 = 0.75 * sims2 + 0.25 * sims3
            qry2_embedding = embeddings.loc[[top1.index[1]]]
            qry2_embedding_eff2 = embeddings_eff2.loc[[top1.index[1]]]
            sims3 = cosine_similarity(qry2_embedding, _db_embeddings)[0]
            sims4 = cosine_similarity(qry2_embedding_eff2, _db_embeddings_eff2)[0]
            sims3 = 0.75 * sims3 + 0.25 * sims4
            sims3[sims3 < 0 ] = 0
            sims3[sims3 > 1 ] = 1
            
        sims2[sims2 < 0 ] = 0
        sims2[sims2 > 1 ] = 1
        
        ## Add similarites using weighted average after rising them to the power 7
        ## this was inspired by Google image recognition 2020 on Kaggle by BO
        ## But i don't know if it did enhance the results or not the number 7 
        ## is tuned using validation results
        if literal_top:
            sims = 0.2*sims**7 + 0.4 * sims2**7 + 0.4 * sims3**7 
        else:
            sims = 0.4 * sims**7 +  0.4 * sims2**7 + 0.2 * sims3**7 
            
        top20 = pd.Series(sims, index=_db_embeddings.index).sort_values(0, ascending=False).head(20)
        # append result
        qry_result = pd.DataFrame(
            {"query_id": qry.query_id, "database_image_id": top20.index, "score": top20.values}
        )
        results.append(qry_result)
submission = pd.concat(results)
submission.to_csv(PREDICTION_FILE, index=False)



