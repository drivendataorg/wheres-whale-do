import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader, ConcatDataset

from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
import torch.distributed as dist


import pandas as pd
import numpy as np
import gc
import json

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


from tqdm.auto import tqdm
from functools import partial

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


# pytorch lighting
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data import *
from models import *
from utils import *
from loss import *


import warnings 
warnings.filterwarnings('ignore')


class Config:
    data_dir = '../input/'
    # train_dir = data_dir + '/train_images/' # images 51033
    train_dir = data_dir + '/train_images_new/' # images 51033
    test_dir = data_dir + '/test_images/' # images 27956
    crop_train_dir = data_dir + '/train_images_new/' # images 51033
    crop_test_dir = data_dir + '/test_images_yolo_x6_public_ensemble_crop_640/' # images 27956
    
    resample_train_dir = data_dir + '/train_resample/'
#     resample_csv_path = data_dir + '/train_5folds_resample.csv'
#     folds_csv_path = data_dir + 'train_5folds.csv'
    # for new data
    folds_csv_path = data_dir + 'train_5folds_new.csv'
    test_csv_path = data_dir + 'sample_submission.csv'
    
    do_crop = True
    do_mask = False
    do_resample = False
    labels_json_path = f'{data_dir}/labels.json'
    species_json_path = f'{data_dir}/species.json'
    ids_json_path = f'{data_dir}/individual_ids_new.json'
#     ids_number_json_path = f'{data_dir}/individual_ids_number.json'
    

    # with open(labels_json_path,'r') as f:
    #     labels_mapping = json.load(f)
    # with open(species_json_path,'r') as f:
    #     species_mapping = json.load(f)
    with open(ids_json_path,'r') as f:
        individual_ids_mapping = json.load(f)
#     with open(ids_number_json_path,'r') as f:
#         individual_ids_number_mapping = json.load(f)
        
    
    # labels_num_classes = len(labels_mapping)
    # species_num_classes = len(species_mapping)
    individual_ids_num_classes = len(individual_ids_mapping)
    
#     model_name = 'convnext_small'
#     model_name = 'convnext_base'
#     model_name = 'convnext_large'
#     model_name = 'convnext_xlarge_384_in22ft1k'
#     model_name = 'tf_efficientnet_b0_ns'
    model_name = 'convnext_large'
    
    image_size = 320           ### input size in training
    embedding_size = 512
    dropout = 0.4
    
    device = 'cuda'             ### set gpu or cpu mode
    debug = False              ### debug flag for checking your modify code
    
    gpus = 1                 ### gpu numbers
    precision = 16             ### training precision 8, 16,32, etc
    batch_size = 32            ### total batch size
    

    lr = 3e-4     ### learning rate default 1e-4,effnet, 2.5e-5,swin transformer
    min_lr = 1e-6              ### min learning rate
    weight_decay = 1e-6
#     gradient_accumulation_steps = 1
#     max_grad_norm =1000         ### 5
    num_workers = 0            ### number workers
    print_freq = 100            ### print log frequency

    seed = 42
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]
    
    optimizer = 'Adam'
    scheduler = 'GradualWarmupSchedulerV2' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'GradualWarmupSchedulerV2']
#     epochs = 10
    # ReduceLROnPlateau
#     factor=0.2 # ReduceLROnPlateau
#     patience=4 # ReduceLROnPlateau
#     eps=1e-6 # ReduceLROnPlateau
    
    ## CosineAnnealingLR
#     T_max= 10 # CosineAnnealingLR

    ## CosineAnnealingWarmRestarts
#     T_0 = 10
#     T_mult = 1
    
    ## GradualWarmupSchedulerV2
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

    ### cls loss for logits
#     criterion = 'LabelSmoothingBinaryCrossEntropy'  ### BinaryCrossEntropy, LabelSmoothingBinaryCrossEntropy
    # CrossEntropy, SCELoss, LabelSmoothingCrossEntropy, FocalCosineLoss, BiTemperedLogisticLoss, TaylorLabelSmoothingCrossEntropy
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
    
    save_dir = f'../results/{model_name}_folds_aug0_{image_size}_{epochs}e_{optimizer}_{scheduler}\
lr{lr}_{criterion}_ls{label_smoothing}_{mt_type}_margin{m}_dropout{dropout}_3head_yolo_x6_public_ensemble_fl_rp/'

from warmup_scheduler import GradualWarmupScheduler
class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]    
        
        
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
#         self.ce = torch.nn.CrossEntropyLoss()
        self.ce = create_criterion(CFG, LOGGER)


    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()   
    
        
class CustomPLModel(pl.LightningModule):
    def __init__(self):
        super(CustomPLModel,self).__init__()
        self.CFG = CFG
        self.model = MultiHeadNet(CFG, pretrained=True)
        self.train_criterion_p3 = FocalLoss(gamma=CFG.fl_gamma)    
        self.train_criterion_p4 = FocalLoss(gamma=CFG.fl_gamma)
        self.train_criterion_p5 = FocalLoss(gamma=CFG.fl_gamma)
        self.val_criterion = nn.CrossEntropyLoss()

    def forward(self, x,individual_ids=None):
        return self.model(x,individual_ids)
    
    ### copy from https://github.com/PyTorchLightning/pytorch-lightning/issues/4690
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    LOGGER.info(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                LOGGER.info(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    ################ freeze bn 
    def freeze_bn(self,net):
        try:
            for m in net.modules():
                # print('m is ', m)
                ### freeze for EffNets , ResNets , ViT
                if isinstance(m,nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
#                     m.eval()  
                    m.weight.requires_grad = False  
                    m.bias.requires_grad = False
               # for n in m.modules():
               #     if isinstance(n,nn.BatchNorm2d) or isinstance(n, nn.LayerNorm):
               #         n.eval()  
                        # n.weight.requires_grad = False  
                        # n.bias.requires_grad = False
        except ValuError:
            print('error with batchnorm2d or layernorm')
            return
        
    def unfreeze_bn(self,net):
        try:
            for m in net.modules():
                # print('m is ', m)
                ### freeze for EffNets , ResNets , ViT
                if isinstance(m,nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
#                     m.train()  
                    m.weight.requires_grad = True  
                    m.bias.requires_grad = True
               # for n in m.modules():
               #     if isinstance(n,nn.BatchNorm2d) or isinstance(n, nn.LayerNorm):
               #         n.eval()  
                        # n.weight.requires_grad = False  
                        # n.bias.requires_grad = False
        except ValuError:
            print('error with batchnorm2d or layernorm')
            return
    
    def get_scheduler(self,optimizer):
        if self.CFG.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=self.CFG.factor, \
                    patience=self.CFG.patience, verbose=True, eps=self.CFG.eps)
        elif self.CFG.scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.CFG.T_max, eta_min=self.CFG.min_lr, last_epoch=-1)
        elif self.CFG.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.CFG.T_0, T_mult=1, eta_min=self.CFG.min_lr, last_epoch=-1) 
        elif self.CFG.scheduler == 'GradualWarmupSchedulerV2':
            scheduler_cosine = CosineAnnealingLR(optimizer, self.CFG.cosine_epochs, eta_min=1e-7)
            scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=self.CFG.multiplier,total_epoch=self.CFG.warmup_epochs, after_scheduler=scheduler_cosine)
            scheduler = scheduler_warmup
        return scheduler
    
    def configure_optimizers(self):
        if self.CFG.optimizer == 'Adam':
            optimizer = Adam(self.parameters(), lr=self.CFG.lr, weight_decay=self.CFG.weight_decay, amsgrad=False)
        elif self.CFG.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.CFG.lr, weight_decay=self.CFG.weight_decay)

        scheduler = self.get_scheduler(optimizer)  
        
        ###https://bleepcoder.com/pytorch-lightning/679052833/how-to-use-reducelronplateau-methon-in-matster-branch
        if self.CFG.scheduler=='ReduceLROnPlateau':
            scheduler = {
                'scheduler': scheduler,
                'reduce_on_plateau': True,
                # val_checkpoint_on is val_loss passed in as checkpoint_on
                'monitor': 'val_loss'
            } 
        
        return [optimizer], [scheduler]

    def mixup_data(self, x, t, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        t_a, t_b = t, t[index]
        
        return mixed_x, t_a, t_b, lam
    
    def mixup_criterion(self, pred, t_a, t_b, lam):
        return lam * self.train_criterion_p5(pred, t_a) + (1 - lam) * self.train_criterion_p5(pred, t_b)
    
    def training_step(self, batch, batch_idx):
        images, individual_ids = batch
        
        if self.current_epoch < self.CFG.mixup_epochs:
            self.CFG.do_mixup = True
        else:
            self.CFG.do_mixup = False
        
        ### mixup copy from https://www.kaggle.com/ttahara/seti-e-t-resnet18d-baseline#Train
        if CFG.do_mixup:
            if batch_idx == 0:
                LOGGER.info(f'Training with mixup prob={self.CFG.mixup_prob} in epoch {self.current_epoch}')
            images, individual_ids_a, individual_ids_b, mixup_lam = self.mixup_data(images, individual_ids, alpha=CFG.mixup_alpha)
            
        ##### fmix
        if CFG.do_fmix:
            if batch_idx == 0:
                LOGGER.info(f'Training with fmix in epoch {self.current_epoch}')
            images,individual_ids = fmix(
                images, 
                individual_ids, 
                alpha=1.0, 
                decay_power=3.0, 
                shape=(self.CFG.image_size,self.CFG.image_size),
                prob=self.CFG.fmix_prob)
            
        ##### cutmix
        if CFG.do_cutmix:
            if batch_idx == 0:
                LOGGER.info(f'Training with cutmix in epoch {self.current_epoch}')
            images,individual_ids = cutmix(
                images, 
                individual_ids, 
                alpha=1.0,
                prob = CFG.cutmix_prob)
        
        
        
        ### plot images
        if batch_idx < 3 and self.current_epoch == 0:
            if os.path.exists(CFG.save_dir):
                save_img(images, CFG.save_dir + f'/train_{self.current_epoch}_{batch_idx}.png', CFG.save_row_num)
                             
        logits_p5 = self(images,individual_ids)
        # loss_p3 = self.train_criterion_p3(logits_p3, labels)
        # loss_p4 = self.train_criterion_p4(logits_p4, species)
        if CFG.do_mixup:
            loss_p5 = self.mixup_criterion(logits_p5, individual_ids_a,individual_ids_b, mixup_lam)
        else:
            loss_p5 = self.train_criterion_p5(logits_p5, individual_ids)
        # loss = loss_p3 + loss_p4 + loss_p5
        loss = loss_p5
#         loss = self.train_criterion_p5(logits_p5, individual_ids)

        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('loss_p3', loss_p3, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('loss_p4', loss_p4, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss_p5', loss_p5, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        logger_logs = {'loss': loss, 'lr': lr}
        output = {
            'loss':loss,
            'progress_bar': logger_logs,
            'log':logger_logs
        }
        return output

    def validation_step(self, batch, batch_idx):
        images, individual_ids = batch

        ### plot images
        if batch_idx < 3 and self.current_epoch == 0:
            if os.path.exists(CFG.save_dir):
                save_img(images, CFG.save_dir + f'/valid_{self.current_epoch}_{batch_idx}.png', CFG.save_row_num)

        logits_p5 = self(images,individual_ids)
        # loss_p3 = self.val_criterion(logits_p3, labels)
        # loss_p4 = self.val_criterion(logits_p4, species)
        loss_p5 = self.val_criterion(logits_p5, individual_ids)
        # val_loss = loss_p3 + loss_p4 + loss_p5
        val_loss = loss_p5
#         val_loss = self.val_criterion(logits_p5, individual_ids)

        preds = torch.argmax(logits_p5, 1).detach()
        targets = individual_ids.detach()
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
               
#         self.log('top1', res[0], on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         self.log('top5', res[1], on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         print(f'top1 is {res[0]:06f}')
#         print(f'top5 is {res[1]:06f}')
         ### gpu
        output = {
            'val_loss': val_loss,
            'logits':logits_p5.detach(),
            'preds': preds, 
            'targets': targets,
        }

        return output
    
    def get_tensor_and_concat(self,tensor):
        gather_t = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gather_t, tensor)
        # out = torch.cat(gather_t).detach().cpu()
        out = torch.cat(gather_t)
        # del gather_t
        # _ = gc.collect()
        return out
    
    def get_list_and_concat(self,list_of_nums):
        tensor = torch.Tensor(list_of_nums).cuda()
        gather_t = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gather_t, tensor)
        return torch.cat(gather_t)
    
    def np_loss_cross_entropy(self, probability, truth):
        batch_size = len(probability)
        truth = truth.reshape(-1)
        p = probability[np.arange(batch_size),truth]
        loss = -np.log(np.clip(p,1e-6,1))
        loss = loss.mean()
        return loss
    
    def precision_at_k(self,output, target, top_k=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k."""
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)
            
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
        
    def apk(self,actual, predicted, k=10):
        actual = [int(actual)]
        if len(predicted)>k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i,p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i+1.0)

        if not actual:
            return 0.0

        return score / min(len(actual), k)
        
    def mapk(self,actual, predicted, k=10):
        _, predicted = predicted.topk(k, 1, True, True)
        actual = actual.data.cpu().numpy()
        predicted = predicted.data.cpu().numpy()
        return np.mean([self.apk(a,p,k) for a,p in zip(actual, predicted)])
    
    def map_per_image(self,label, predictions):
        """Computes the precision score of one image.

        Parameters
        ----------
        label : string
                The true label of the image
        predictions : list
                A list of predicted elements (order does matter, 5 predictions allowed per image)

        Returns
        -------
        score : double
        """    
        try:
            return 1 / (predictions[:5].index(label) + 1)
        except ValueError:
            return 0.0

    def map_per_set(self, labels, predictions):
        """Computes the average over multiple images.

        Parameters
        ----------
        labels : list
                 A list of the true labels. (Only one true label per images allowed!)
        predictions : list of list
                 A list of predicted elements (order does matter, 5 predictions allowed per image)

        Returns
        -------
        score : double
        """
        return np.mean([self.map_per_image(l, p) for l,p in zip(labels, predictions)])
    
    
    def validation_epoch_end(self, outputs):
        ### gpu
        logits = torch.cat([x['logits'] for x in outputs])
        preds = torch.cat([x['preds'] for x in outputs])
        targets = torch.cat([x['targets'] for x in outputs])
        
        all_logits = self.get_tensor_and_concat(logits)
        all_preds = self.get_tensor_and_concat(preds)
        all_targets = self.get_tensor_and_concat(targets)
#         print(f'all_preds shape is {all_preds.shape}')
#         print(f'all_preds is {all_preds}')
#         print(f'all_targets shape is {all_targets.shape}')
#         print(f'all_targets is {all_targets}')
        avg_val_loss = self.val_criterion(all_logits, all_targets)
        all_logits = F.softmax(all_logits,dim=1)
#         all_logits = torch.cat([torch.sigmoid(all_logits), torch.ones_like(all_logits[:, :1]).float().cuda() * 0.5], 1)
#         res = self.precision_at_k(all_logits,all_targets,top_k=(1,5))
        mAP5 = self.mapk(all_targets,all_logits,k=5) 
        
#         _, all_preds = all_logits.topk(5, 1, True, True)
#         all_targets = all_targets.data.cpu().numpy()
#         all_preds = all_preds.data.cpu().numpy()
#         labels = [target_encodings[target] for target in all_targets]
#         predicts = []
#         for preds in all_preds:
#             predict = []
#             for pred in preds:
#                 predict.append(target_encodings[pred])
#             predicts.append(predict)
#         map5 = self.map_per_set(labels,predicts) 
        
#         all_preds = all_preds.data.cpu().numpy()
#         all_targets = all_targets.data.cpu().numpy()
#         accuracy = (all_preds==all_targets).mean()
 
        self.log('avg_val_loss', avg_val_loss)
        self.log('mAP5', mAP5)
#         self.log('map5', map5)
        LOGGER.info(f'Epoch={self.current_epoch},avg_val_loss={avg_val_loss:06f}, mAP5={mAP5:06f}')
        logger_logs = {'avg_val_loss': avg_val_loss}
        
        output = {
            'avg_val_loss':avg_val_loss,
            'progress_bar': logger_logs,
            'log':logger_logs
        }
        
        return output

    
    
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, CFG, fold = 0):
        super().__init__()
        self.CFG = CFG
        self.fold = fold
        
    def setup(self, stage=None):
        # In multi-GPU training, this method is run on each GPU. 
        # So ideal for each training/valid split

        df_fold = pd.read_csv(self.CFG.folds_csv_path)
        if self.CFG.debug:
            df_fold = df_fold.sample(n=self.CFG.batch_size*10, random_state=self.CFG.seed).reset_index(drop=True)
        print(df_fold.groupby(['fold']).size())
        
        #---
        df = df_fold.copy()
        
#         bad_train_images_list = ['3aa9888f58aefc.jpg',
# #              '54aa56e1620b6b.jpg',
# #              'ee6ff2e7e38882.jpg',
# #              '019cec049e2d63.jpg',
# #              '2bb94ab94e1d20.jpg',
# #              '62991e56843500.jpg',
# #              'b115748388ed33.jpg',
#         ]
            
#         df = df[~df['image'].isin(bad_train_images_list)].reset_index(drop=True)

        #---
        df_train = df[df.fold != self.fold].reset_index(drop=True)
        df_valid = df[df.fold == self.fold].reset_index(drop=True)
        
        if self.CFG.do_resample:
            df_number_large = df_train[df_train.number >= 5]
            ### number == 1
            df_number_1 = df_train[df_train.number == 1]
            df_number_1 = pd.DataFrame(np.repeat(df_number_1.values, 5, axis=0), columns=df_number_1.columns)
            
            ### number == 2
            df_number_2 = df_train[df_train.number == 2]
            df_number_2 = pd.DataFrame(np.repeat(df_number_2.values, 4, axis=0), columns=df_number_2.columns)
            
            ### number == 3
            df_number_3 = df_train[df_train.number == 3]
            df_number_3 = pd.DataFrame(np.repeat(df_number_3.values, 3, axis=0), columns=df_number_3.columns)
            
            ### number == 4
            df_number_4 = df_train[df_train.number == 4]
            df_number_4 = pd.DataFrame(np.repeat(df_number_3.values, 2, axis=0), columns=df_number_4.columns)
            
            df_train = pd.concat([df_number_large,df_number_1,df_number_2,df_number_3,df_number_4])
            print(f'df train shape is {df_train.shape}')

        self.train_dataset = TrainMultiHeadDataset(
            CFG, 
            df_train, 
            transforms=get_train_transforms(self.CFG))
        
        
        self.valid_dataset = TrainMultiHeadDataset(
            CFG, 
            df_valid, 
            transforms=get_val_transforms(self.CFG)) 
        
#     def prepare_data(self):
#         model_path = f'{self.CFG.model_dir}/{self.CFG.model_name}/fold_{fold}/mAP5_best.ckpt'
#         LOGGER.info(f'model_path is {model_path}')
#         self.pl_model.on_load_checkpoint(torch.load(model_path))
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.CFG.batch_size, num_workers=4, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, self.CFG.batch_size*2, num_workers=4, shuffle=False) 
    
def train_loop(CFG, fold, LOGGER):
    LOGGER.info(f'=============== fold: {fold} training =============')
    LOGGER.info(f'Training model {CFG.model_name}, params with batch_size={CFG.batch_size}, image_size={CFG.image_size}, embedding_size={CFG.embedding_size}, scheduler={CFG.scheduler}, init_lr={CFG.lr}, fl_gamma={CFG.fl_gamma}, s={CFG.s},m={CFG.m},s_p4={CFG.s_p4},m_p4={CFG.m_p4},s_p3={CFG.s_p3},m_p3={CFG.m_p3}, warmup_epochs={CFG.warmup_epochs}, cosine_epochs={CFG.cosine_epochs}.')
    
    pl.seed_everything(seed=CFG.seed)

    ### load data module
    dm = CustomDataModule(CFG, fold)
    
    ### init model
#     model = MultiHeadNet(CFG, pretrained=True)
    
    ### init 
#     criterion = create_criterion(CFG, LOGGER)
    
    ### init pl model
    pl_model = CustomPLModel()
    
    
#     model_path = f'{CFG.model_dir}/{CFG.model_name}/fold_{fold}/mAP5_best.ckpt'
#     ### resume
#     pl_model = pl_model.load_from_checkpoint(model_path)

#     ### finetune
#     pl_model.on_load_checkpoint(torch.load(model_path))
#     LOGGER.info(f'model_path is {model_path}')

    # Folder hack
    tb_logger = TensorBoardLogger(save_dir=CFG.save_dir, name=f'{CFG.model_name}', version=f'fold_{fold}')
    os.makedirs(f'{CFG.save_dir}/{CFG.model_name}', exist_ok=True)
#     checkpoint_callback = ModelCheckpoint(
#         dirpath=tb_logger.log_dir, 
#         filename=f'checkpoint_best',
#         monitor='mAP', 
#         mode='max')
    checkpoint_callback1 = ModelCheckpoint(
        dirpath=tb_logger.log_dir,
        filename='avg_val_loss_best',
        save_top_k=1, 
        verbose=True,
        monitor='avg_val_loss', 
        mode='min'
        )
    
    checkpoint_callback2 = ModelCheckpoint(
        dirpath=tb_logger.log_dir,
        filename='mAP5_best',
        save_top_k=1, 
        verbose=True,
        monitor='mAP5', 
        mode='max'
        )

#     early_stop_callback = EarlyStopping(
#         monitor='mAP',
#         min_delta=0.0,
#         patience=5,
#         verbose=False,
#         mode='max',
#     )
    
    trainer = pl.Trainer(
        gpus=CFG.gpus,
        precision=CFG.precision,
        max_epochs=CFG.epochs,
#         num_sanity_val_steps=1,
#         resume_from_checkpoint=model_path,
#         num_sanity_val_steps=1 if CFG.debug else 0,
#         checkpoint_callback=checkpoint_callback,#### pl==1.2.4
#         val_check_interval=5.0, # check validation 1 time per 5 epochs
        callbacks=[checkpoint_callback1,checkpoint_callback2], #### pl==1.3.3
#         check_val_every_n_epoch = 1,
#         accelerator='ddp',
        accumulate_grad_batches=1,
#         gradient_clip_val=1000.0,
        strategy='ddp',
        # amp_backend='native', # or apex
        # amp_level='02', # 01,02, etc...
        # benchmark=True,
        # deterministic=True,
       # amp_backend="native",
        sync_batchnorm=True,
        logger=tb_logger,
    )
    
    trainer.fit(pl_model, dm)
    
    
if __name__ == '__main__':
    CFG = Config
    if not os.path.exists(CFG.save_dir):
        os.makedirs(CFG.save_dir)   
    LOGGER = get_log(file_name=CFG.save_dir + 'train.log')

    # target_encodings = {CFG.individual_ids_mapping[x]:x for x in CFG.individual_ids_mapping}
    
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            train_loop(CFG, fold, LOGGER)
