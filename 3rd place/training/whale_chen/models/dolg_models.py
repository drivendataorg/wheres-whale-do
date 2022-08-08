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

### copy from https://github.com/ChristofHenkel/kaggle-landmark-2021-1st-place/blob/main/models/ch_mdl_dolg_efficientnet.py

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class DLOGGeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(DLOGGeM,self).__init__()
        if p_trainable:
            self.p = nn.Parameter(torch.ones(1)*p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)   
        return ret
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class MultiAtrousModule(nn.Module):
    def __init__(self, in_chans, out_chans, dilations):
        super(MultiAtrousModule, self).__init__()
        
        self.d0 = nn.Conv2d(in_chans, 512, kernel_size=3, dilation=dilations[0],padding='same')
        self.d1 = nn.Conv2d(in_chans, 512, kernel_size=3, dilation=dilations[1],padding='same')
        self.d2 = nn.Conv2d(in_chans, 512, kernel_size=3, dilation=dilations[2],padding='same')
        self.conv1 = nn.Conv2d(512 * 3, out_chans, kernel_size=1)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        
        x0 = self.d0(x)
        x1 = self.d1(x)
        x2 = self.d2(x)
        x = torch.cat([x0,x1,x2],dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        return x

class SpatialAttention2d(nn.Module):
    def __init__(self, in_c):
        super(SpatialAttention2d, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 1024, 1, 1)
        self.bn = nn.BatchNorm2d(1024)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(1024, 1, 1, 1)
        self.softplus = nn.Softplus(beta=1, threshold=20) # use default setting.

    def forward(self, x):
        '''
        x : spatial feature map. (b x c x w x h)
        att : softplus attention score 
        '''
        x = self.conv1(x)
        x = self.bn(x)
        
        feature_map_norm = F.normalize(x, p=2, dim=1)
         
        x = self.act1(x)
        x = self.conv2(x)
        att_score = self.softplus(x)
        att = att_score.expand_as(feature_map_norm)
        
        x = att * feature_map_norm
        return x, att_score   

class OrthogonalFusion(nn.Module):
    def __init__(self):
        super(OrthogonalFusion, self).__init__()

    def forward(self, fl, fg):

        bs, c, w, h = fl.shape
        
        fl_dot_fg = torch.bmm(fg[:,None,:],fl.reshape(bs,c,-1))
        fl_dot_fg = fl_dot_fg.reshape(bs,1,w,h)
        fg_norm = torch.norm(fg, dim=1)
        
        fl_proj = (fl_dot_fg / fg_norm[:,None,None,None]) * fg[:,:,None,None]
        fl_orth = fl - fl_proj
        
        f_fused = torch.cat([fl_orth,fg[:,:,None,None].repeat(1,1,w,h)],dim=1)
        return f_fused  


class DOLGNet(nn.Module):
    def __init__(self, CFG, pretrained=False):
        super(DOLGNet,self).__init__()
        self.CFG = CFG
        self.model = timm.create_model(CFG.model_name, pretrained=pretrained)
#         print(self.model)
        ### effnet
        if CFG.model_name == 'tf_efficientnet_b0_ns' or CFG.model_name == 'tf_efficientnet_b4_ns' or CFG.model_name == 'tf_efficientnet_b5_ns' or CFG.model_name == 'tf_efficientnet_b7_ns':
            num_features = self.model.classifier.in_features 
            self.model.global_pool = nn.Identity()
            self.model.classifier = nn.Identity()
            self.global_pool = GeM()
            self.drop = nn.Dropout(CFG.dropout)

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
            

        self.head_p3 = nn.Sequential(
                    nn.Conv2d(planes_p3, hidden_dims_p3, kernel_size=3, padding=1), 
                    nn.BatchNorm2d(hidden_dims_p3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_dims_p3, hidden_dims_p3, kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_dims_p3),
                    nn.ReLU(inplace=True),
                )
#         self.pool_p3 = DLOGGeM(p_trainable=True)
        self.pool_p3 = GeM()
        self.drop_p3 = nn.Dropout(CFG.dropout)
        self.neck_p3 = nn.Sequential(
                  nn.Linear(hidden_dims_p3, hidden_dims_p3//2),
                  nn.BatchNorm1d(hidden_dims_p3//2)
         )
 
        
        if CFG.mt_type == 'arcface':
            self.margin_p3 = ArcMarginProduct(hidden_dims_p3//2, 
                                       CFG.labels_num_classes,
                                       s=CFG.s_p3, 
                                       m=CFG.m_p3, 
                                       easy_margin=CFG.easy_margin, 
                                       ls_eps=CFG.ls_eps)  
                
        if CFG.mt_type == 'arcface_adaptive':
            self.margin_p3 = ArcMarginProduct_subcenter(hidden_dims_p3//2, 
                                  CFG.labels_num_classes,
                                  k=CFG.k)
        

        self.head_p4 = nn.Sequential(
                nn.Conv2d(planes_p4, hidden_dims_p4, kernel_size=3, padding=1), 
                nn.BatchNorm2d(hidden_dims_p4),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dims_p4, hidden_dims_p4, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dims_p4),
                nn.ReLU(inplace=True),
            )
        self.pool_p4 = GeM()
        self.drop_p4 = nn.Dropout(CFG.dropout)
        self.neck_p4 = nn.Sequential(
                  nn.Linear(hidden_dims_p4, hidden_dims_p4//2),
                  nn.BatchNorm1d(hidden_dims_p4//2)
         )

        
        if CFG.mt_type == 'arcface':
            self.margin_p4 = ArcMarginProduct(hidden_dims_p4//2, 
                                   CFG.species_num_classes,
                                   s=CFG.s_p4, 
                                   m=CFG.m_p4, 
                                   easy_margin=CFG.easy_margin, 
                                   ls_eps=CFG.ls_eps) 
                
        if CFG.mt_type == 'arcface_adaptive':
            self.margin_p4 = ArcMarginProduct_subcenter(hidden_dims_p4//2, 
                                  CFG.species_num_classes,
                                  k=CFG.k)
        
         
        ### DOLG
        backbone_out = self.model.feature_info[-1]['num_chs']
        backbone_out_1 = self.model.feature_info[-2]['num_chs']
        feature_dim_l_g = 1024
        fusion_out = 2 * feature_dim_l_g
        
        self.fusion_pool = nn.AdaptiveAvgPool2d(1)
        self.neck = nn.Sequential(
                    nn.Linear(fusion_out, CFG.embedding_size, bias=True),
                    nn.BatchNorm1d(CFG.embedding_size)
                )
        
        self.mam = MultiAtrousModule(backbone_out_1, feature_dim_l_g, CFG.dilations)
        self.conv_g = nn.Conv2d(backbone_out,feature_dim_l_g,kernel_size=1)
        self.bn_g = nn.BatchNorm2d(feature_dim_l_g, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.act_g =  nn.SiLU(inplace=True)
        self.attention2d = SpatialAttention2d(feature_dim_l_g)
        self.fusion = OrthogonalFusion()
            
            
        
    def forward(self, x,labels=None,species=None,individual_ids=None):
        if self.CFG.model_name == 'tf_efficientnet_b0_ns' or self.CFG.model_name == 'tf_efficientnet_b4_ns' or self.CFG.model_name == 'tf_efficientnet_b5_ns' or self.CFG.model_name == 'tf_efficientnet_b7_ns':
            x = self.model.act1(self.model.bn1(self.model.conv_stem(x)))
            x = self.model.blocks[0](x)
            x = self.model.blocks[1](x)
            x = self.model.blocks[2](x)   
            x = self.model.blocks[3](x)
            x = self.model.blocks[4](x)
 
            ####----
            x_l = x
            x_p3 = self.head_p3(x)
            feat_p3 = self.pool_p3(x_p3).flatten(1)
            feat_p3 = self.drop_p3(feat_p3)
            feat_p3 = self.neck_p3(feat_p3)
            if self.CFG.mt_type == 'arcface':
                logits_p3 = self.margin_p3(feat_p3, labels)
            else:
                logits_p3 = self.margin_p3(feat_p3)

            ####----
            
            x = self.model.blocks[5](x)
            
            ####----
            x_p4 = self.head_p4(x)
            feat_p4 = self.pool_p4(x_p4).flatten(1)
            feat_p4 = self.drop_p4(feat_p4)
            feat_p4 = self.neck_p4(feat_p4)
#             logits_p4 = self.margin_p4(feat_p4, species) 
            if self.CFG.mt_type == 'arcface':
                logits_p4 = self.margin_p4(feat_p4, species)
            else:
                logits_p4 = self.margin_p4(feat_p4)
            ####----
            
            x = self.model.blocks[6](x)
#             conv5 = self.model.act2(self.model.bn2(self.model.conv_head(x)))
            x_g = x

            
            x_l = self.mam(x_l)
            x_l, att_score = self.attention2d(x_l)
#             print(f'x_l shape is {x_l.size()}')
            
            x_g = self.conv_g(x_g)
            x_g = self.bn_g(x_g)
            x_g = self.act_g(x_g)
            
            x_g = self.global_pool(x_g)
#             print(f'x_g shape is {x_g.size()}')
#             x_g = x_g[:,:,0,0]
            
            x_fused = self.fusion(x_l, x_g)
            x_fused = self.fusion_pool(x_fused)
            x_fused = x_fused[:,:,0,0]  
            embedding = self.neck(x_fused)
            
            if self.CFG.mt_type == 'arcface':
                logits = self.margin(embedding, individual_ids)
            else:
                logits = self.margin(embedding)   
        return logits_p3, logits_p4, logits
    
    def predict(self, x):
        if self.CFG.model_name == 'tf_efficientnet_b0_ns' or self.CFG.model_name == 'tf_efficientnet_b4_ns' or self.CFG.model_name == 'tf_efficientnet_b5_ns' or self.CFG.model_name == 'tf_efficientnet_b7_ns':
            x = self.model.act1(self.model.bn1(self.model.conv_stem(x)))
            x = self.model.blocks[0](x)
            x = self.model.blocks[1](x)
            x = self.model.blocks[2](x)   
            x = self.model.blocks[3](x)
            x = self.model.blocks[4](x)
 
            ####----
            x_l = x
            ####----
            
            x = self.model.blocks[5](x) 
            x = self.model.blocks[6](x)
            x_g = x

            x_l = self.mam(x_l)
            x_l, att_score = self.attention2d(x_l)
            
            x_g = self.conv_g(x_g)
            x_g = self.bn_g(x_g)
            x_g = self.act_g(x_g)
            x_g = self.global_pool(x_g)
#             x_g = x_g[:,:,0,0]
            
            x_fused = self.fusion(x_l, x_g)
            x_fused = self.fusion_pool(x_fused)
            x_fused = x_fused[:,:,0,0]  
            embedding = self.neck(x_fused)
        
        return embedding