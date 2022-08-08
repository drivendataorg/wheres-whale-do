import sys
sys.path.append('/home/chen/ai-competition/pytorch-image-models-master-220214')
import timm

import torch.nn.functional as F

import torch
import torch.nn as nn
from models.metric_learning import AdaCos, ArcMarginProduct, AddMarginProduct, ArcMarginProduct_subcenter
from models.layers import GeM

from collections import OrderedDict
from timm.models.layers import *
    
class MultiHeadNet(nn.Module):
    def __init__(self, CFG, pretrained=False):
        super(MultiHeadNet,self).__init__()
        self.CFG = CFG
        self.model = timm.create_model(CFG.model_name, pretrained=pretrained)
#         print(self.model)
        ### effnet
        if CFG.model_name == 'tf_efficientnet_b0_ns' or CFG.model_name == 'tf_efficientnet_b4_ns' or CFG.model_name == 'tf_efficientnet_b5_ns' or CFG.model_name == 'tf_efficientnet_b7_ns':
            num_features = self.model.classifier.in_features 
            self.model.global_pool = nn.Identity()
            self.model.classifier = nn.Identity()
            self.pooling = GeM()
            self.drop = nn.Dropout(CFG.dropout)
#             self.fc = nn.Linear(num_features, CFG.embedding_size)
            self.neck = nn.Sequential(
                            nn.Linear(num_features, CFG.embedding_size, bias=True),
                            nn.BatchNorm1d(CFG.embedding_size),
#                             torch.nn.PReLU()
                        )
            if CFG.mt_type == 'cls':
                self.margin = nn.Linear(num_features, CFG.individual_ids_num_classes)
            if CFG.mt_type == 'arcface':
                self.margin = ArcMarginProduct(CFG.embedding_size, 
                                      CFG.individual_ids_num_classes,
                                      s=CFG.s, 
                                      m=CFG.m, 
                                      easy_margin=CFG.easy_margin, 
                                      ls_eps=CFG.ls_eps)  
                
            if CFG.mt_type == 'arcface_adaptive':
                self.margin = ArcMarginProduct_subcenter(CFG.embedding_size, 
                                  CFG.individual_ids_num_classes,
                                  k=CFG.k)
#                 self.margin = ArcFaceLossAdaptiveMargin(CFG.margins,CFG.individual_ids_num_classes,CFG.s)
            
            
        if CFG.model_name == 'convnext_small' or CFG.model_name == 'convnext_base' or CFG.model_name == 'convnext_large' or CFG.model_name == 'convnext_base_384_in22ft1k' or CFG.model_name == 'convnext_xlarge_in22k' or CFG.model_name == 'convnext_xlarge_384_in22ft1k':
            num_features = self.model.head.fc.in_features 
#             self.model.head.global_pool = nn.Identity()
            self.model.head.fc = nn.Identity()
            self.neck = nn.Sequential(
                            nn.Linear(num_features, CFG.embedding_size, bias=True),
                            nn.BatchNorm1d(CFG.embedding_size))
            
            if CFG.mt_type == 'cls':
                self.margin = nn.Linear(CFG.embedding_size, CFG.individual_ids_num_classes)
            if CFG.mt_type == 'arcface':
                self.margin = ArcMarginProduct(CFG.embedding_size, 
                                  CFG.individual_ids_num_classes,
                                  s=CFG.s, 
                                  m=CFG.m, 
                                  easy_margin=CFG.easy_margin, 
                                  ls_eps=CFG.ls_eps) 
            if CFG.mt_type == 'arcface_adaptive':
                self.margin = ArcMarginProduct_subcenter(CFG.embedding_size, 
                                  CFG.individual_ids_num_classes,
                                  k=CFG.k) 
        
        
        hidden_dims_p3 = 256
        hidden_dims_p4 = 512
        if CFG.model_name == 'tf_efficientnet_b0_ns':
            planes_p3 = 112
            planes_p4 = 192
        if CFG.model_name == 'tf_efficientnet_b5_ns':
            planes_p3 = 176
            planes_p4 = 304
        if CFG.model_name == 'tf_efficientnet_b7_ns':
            planes_p3 = 224
            planes_p4 = 384
            
        if CFG.model_name == 'resnet200d':
            planes_p3 = 512
            planes_p4 = 1024
        
        if CFG.model_name == 'convnext_small':
            planes_p3 = 192
            planes_p4 = 384
        
        if CFG.model_name == 'convnext_base':
            planes_p3 = 256
            planes_p4 = 512
            
        if CFG.model_name == 'convnext_large':
            planes_p3 = 384
            planes_p4 = 768
            
        if CFG.model_name == 'convnext_xlarge_in22k':
            planes_p3 = 384
            planes_p4 = 768
            
        if CFG.model_name == 'convnext_xlarge_384_in22ft1k':
            planes_p3 = 512
            planes_p4 = 1024    
            

        # self.head_p3 = nn.Sequential(
        #             nn.Conv2d(planes_p3, hidden_dims_p3, kernel_size=3, padding=1),
        #             nn.BatchNorm2d(hidden_dims_p3),
        #             nn.ReLU(inplace=True),
        #             nn.Conv2d(hidden_dims_p3, hidden_dims_p3, kernel_size=3, padding=1),
        #             nn.BatchNorm2d(hidden_dims_p3),
        #             nn.ReLU(inplace=True),
        #         )
        # self.pool_p3 = GeM()
        # self.drop_p3 = nn.Dropout(CFG.dropout)
        # self.neck_p3 = nn.Sequential(
        #                     nn.Linear(hidden_dims_p3, hidden_dims_p3 // 2, bias=True),
        #                     nn.BatchNorm1d(hidden_dims_p3//2))
        # if CFG.mt_type == 'arcface':
        #     self.margin_p3 = ArcMarginProduct(hidden_dims_p3//2,
        #                                CFG.labels_num_classes,
        #                                s=CFG.s_p3,
        #                                m=CFG.m_p3,
        #                                easy_margin=CFG.easy_margin,
        #                                ls_eps=CFG.ls_eps)
        #
        # if CFG.mt_type == 'arcface_adaptive':
        #     self.margin_p3 = ArcMarginProduct_subcenter(hidden_dims_p3//2,
        #                           CFG.labels_num_classes,
        #                           k=CFG.k)
        #
        # self.head_p4 = nn.Sequential(
        #         nn.Conv2d(planes_p4, hidden_dims_p4, kernel_size=3, padding=1),
        #         nn.BatchNorm2d(hidden_dims_p4),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(hidden_dims_p4, hidden_dims_p4, kernel_size=3, padding=1),
        #         nn.BatchNorm2d(hidden_dims_p4),
        #         nn.ReLU(inplace=True),
        #     )
        # self.pool_p4 = GeM()
        # self.drop_p4 = nn.Dropout(CFG.dropout)
        # self.neck_p4 = nn.Sequential(
        #                     nn.Linear(hidden_dims_p4, hidden_dims_p4 // 2, bias=True),
        #                     nn.BatchNorm1d(hidden_dims_p4//2))
        # if CFG.mt_type == 'arcface':
        #     self.margin_p4 = ArcMarginProduct(hidden_dims_p4//2,
        #                            CFG.species_num_classes,
        #                            s=CFG.s_p4,
        #                            m=CFG.m_p4,
        #                            easy_margin=CFG.easy_margin,
        #                            ls_eps=CFG.ls_eps)
        #
        # if CFG.mt_type == 'arcface_adaptive':
        #     self.margin_p4 = ArcMarginProduct_subcenter(hidden_dims_p4//2,
        #                           CFG.species_num_classes,
        #                           k=CFG.k)
            
            
        
    def forward(self, x,individual_ids=None):
        if self.CFG.model_name == 'tf_efficientnet_b0_ns' or self.CFG.model_name == 'tf_efficientnet_b4_ns' or self.CFG.model_name == 'tf_efficientnet_b5_ns' or self.CFG.model_name == 'tf_efficientnet_b7_ns':
            x = self.model.act1(self.model.bn1(self.model.conv_stem(x)))
            x = self.model.blocks[0](x)
            x = self.model.blocks[1](x)
            x = self.model.blocks[2](x)   
            x = self.model.blocks[3](x)
            x = self.model.blocks[4](x)
            #
            # ####----
            # x_p3 = self.head_p3(x)
            # feat_p3 = self.pool_p3(x_p3).flatten(1)
            # feat_p3 = self.drop_p3(feat_p3)
            # feat_p3 = self.neck_p3(feat_p3)
            # if self.CFG.mt_type == 'arcface':
            #     logits_p3 = self.margin_p3(feat_p3, labels)
            # else:
            #     logits_p3 = self.margin_p3(feat_p3)
            # ####----
            #
            x = self.model.blocks[5](x)
            #
            # ####----
            # x_p4 = self.head_p4(x)
            # feat_p4 = self.pool_p4(x_p4).flatten(1)
            # feat_p4 = self.drop_p4(feat_p4)
            # feat_p4 = self.neck_p4(feat_p4)
            # if self.CFG.mt_type == 'arcface':
            #     logits_p4 = self.margin_p4(feat_p4, species)
            # else:
            #     logits_p4 = self.margin_p4(feat_p4)
            # ####----
            #
            x = self.model.blocks[6](x)
            conv5 = self.model.act2(self.model.bn2(self.model.conv_head(x)))

            features = self.pooling(conv5).flatten(1)
            features = self.drop(features)
            embedding = self.neck(features)
            if self.CFG.mt_type == 'arcface':
                logits = self.margin(embedding, individual_ids)
            else:
                logits = self.margin(embedding)
            
        if self.CFG.model_name == 'convnext_small' or self.CFG.model_name == 'convnext_base' or self.CFG.model_name == 'convnext_large' or self.CFG.model_name == 'convnext_base_384_in22ft1k' or self.CFG.model_name == 'convnext_xlarge_in22k' or self.CFG.model_name == 'convnext_xlarge_384_in22ft1k':
            x = self.model.stem(x)
            x = self.model.stages[0](x)
            x = self.model.stages[1](x)
            #
            # ####----
            # feat_p3 = self.head_p3(x)
            # feat_p3 = self.pool_p3(feat_p3).flatten(1)
            # feat_p3 = self.drop_p3(feat_p3)
            # feat_p3 = self.neck_p3(feat_p3)
            # logits_p3 = self.margin_p3(feat_p3, labels)
            # ####----
            #
            x=self.model.stages[2](x)
            #
            # ####----
            # feat_p4 = self.head_p4(x)
            # feat_p4 = self.pool_p4(feat_p4).flatten(1)
            # feat_p4 = self.drop_p4(feat_p4)
            # feat_p4 = self.neck_p4(feat_p4)
            # logits_p4 = self.margin_p4(feat_p4, species)
            # ####----
            #
            
            x=self.model.stages[3](x)
            x = self.model.norm_pre(x)
            
            features = self.model.head(x)
            embedding = self.neck(features)
            logits = self.margin(embedding, individual_ids)
            
            
        # return logits_p3, logits_p4, logits
        return logits
    
    def predict(self, x):
        if self.CFG.model_name == 'tf_efficientnet_b0_ns' or self.CFG.model_name == 'tf_efficientnet_b4_ns' or self.CFG.model_name == 'tf_efficientnet_b5_ns' or self.CFG.model_name == 'tf_efficientnet_b7_ns':
            x = self.model.act1(self.model.bn1(self.model.conv_stem(x)))
            x = self.model.blocks(x)
            conv5  = self.model.act2(self.model.bn2(self.model.conv_head(x)))
            features = self.pooling(conv5).flatten(1)
            features = self.drop(features)
            embedding = self.neck(features)
            
        if self.CFG.model_name == 'convnext_small' or self.CFG.model_name == 'convnext_base' or self.CFG.model_name == 'convnext_large' or self.CFG.model_name == 'convnext_base_384_in22ft1k' or self.CFG.model_name == 'convnext_xlarge_in22k' or self.CFG.model_name == 'convnext_xlarge_384_in22ft1k':
            x = self.model.stem(x)
            x = self.model.stages[0](x)
            x = self.model.stages[1](x)
            x = self.model.stages[2](x)
            x = self.model.stages[3](x)
            x = self.model.norm_pre(x)
            features = self.model.head(x)
            embedding = self.neck(features)
            
        return embedding