### reference from https://www.kaggle.com/debarshichanda/pytorch-hybrid-swin-transformer-cnn
import sys
sys.path.append('/home/chen/ai-competition/pytorch-image-models-master-210903')
import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.metric_learning import AdaCos, ArcMarginProduct, AddMarginProduct
from models.layers import GeM

class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, patch_size=1, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # NOTE Most reliable way of determining output dims is to run forward pass
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = (feature_size, feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0
        self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class HybridClassifier(nn.Module):
    def __init__(self, CFG, pretrained=True, in_channels=3):
        super().__init__()
        self.CFG = CFG
        self.embedder = timm.create_model(CFG.embedder_name, pretrained=pretrained, features_only=True,out_indices=[2], in_chans=in_channels)
        self.backbone = timm.create_model(CFG.backbone_name, pretrained=pretrained, in_chans=in_channels)
        self.backbone.patch_embed = HybridEmbed(self.embedder, img_size=CFG.image_size, embed_dim=CFG.embed_dim)

        
        n_features = self.backbone.head.in_features
        self.backbone.reset_classifier(0)
#         self.backbone.avgpool = nn.Identity()
#         self.pool_p5 = GeM()
        self.drop_p5 = nn.Dropout(CFG.dropout)
        self.fc_p5 = nn.Linear(n_features, CFG.channel_size)
        self.margin_p5 = ArcMarginProduct(CFG.channel_size, 
                                  CFG.individual_ids_num_classes,
                                  s=CFG.s, 
                                  m=CFG.m, 
                                  easy_margin=CFG.easy_margin, 
                                  ls_eps=CFG.ls_eps)  
        
        
        hidden_dims_p3 = 256
        hidden_dims_p4 = 512
        
        if CFG.backbone_name == 'swin_tiny_patch4_window7_224':
            planes_p3 = 384
            planes_p4 = 768
        
        if CFG.backbone_name == 'swin_large_patch4_window12_384_in22k':
            planes_p3 = 768
            planes_p4 = 1536
        
        
        self.head_p3 = nn.Sequential(
                    nn.Conv2d(planes_p3, hidden_dims_p3, kernel_size=3, padding=1), 
                    nn.BatchNorm2d(hidden_dims_p3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_dims_p3, hidden_dims_p3, kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_dims_p3),
                    nn.ReLU(inplace=True),
                )
        self.pool_p3 = GeM()
        self.drop_p3 = nn.Dropout(CFG.dropout)
        self.fc_p3 = nn.Linear(hidden_dims_p3, hidden_dims_p3//2)
        self.margin_p3 = ArcMarginProduct(hidden_dims_p3//2, 
                                   CFG.labels_num_classes,
                                   s=CFG.s_p3, 
                                   m=CFG.m_p3, 
                                   easy_margin=CFG.easy_margin, 
                                   ls_eps=CFG.ls_eps)
        
        
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
        self.fc_p4 = nn.Linear(hidden_dims_p4, hidden_dims_p4//2)
        self.margin_p4 = ArcMarginProduct(hidden_dims_p4//2, 
                                   CFG.species_num_classes,
                                   s=CFG.s_p4, 
                                   m=CFG.m_p4, 
                                   easy_margin=CFG.easy_margin, 
                                   ls_eps=CFG.ls_eps)

    def forward(self, x, labels,species,individual_ids): 
        x = self.backbone.patch_embed(x)
        if self.backbone.absolute_pos_embed is not None:
            x = x + self.backbone.absolute_pos_embed
        x = self.backbone.pos_drop(x)
        x = self.backbone.layers[0](x)
        x = self.backbone.layers[1](x)
        
                
#         x = self.backbone.layers[2](x)
        for blk in self.backbone.layers[2].blocks:
            if not torch.jit.is_scripting() and self.backbone.layers[2].use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        
        ####----head p3 ####
        # B L C to B C L
        x_p3 = x.transpose(1, 2)
        # B C L to B C H W
        x_p3 = x_p3.reshape(x_p3.size(0),x_p3.size(1), self.CFG.image_size // 32, self.CFG.image_size // 32) # 24 = 384 / 16
        x_p3 = self.head_p3(x_p3)
        feat_p3 = self.pool_p3(x_p3).flatten(1)
        feat_p3 = self.drop_p3(feat_p3)
        feat_p3 = self.fc_p3(feat_p3)
        logits_p3 = self.margin_p3(feat_p3, labels)
        ####----head p3 ####  
            
        if self.backbone.layers[2].downsample is not None:
            x = self.backbone.layers[2].downsample(x)
            
#         x = self.backbone.layers[3](x)
        for blk in self.backbone.layers[3].blocks:
                if not torch.jit.is_scripting() and self.backbone.layers[3].use_checkpoint:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)    
    
    
        ####----head p4 ####
        # B L C to B C L
        x_p4 = x.transpose(1, 2)
        # B C L to B C H W
        x_p4 = x_p4.reshape(x_p4.size(0),x_p4.size(1), self.CFG.image_size // 64, self.CFG.image_size // 64) # 24 = 384 / 16
        x_p4 = self.head_p4(x_p4)
        feat_p4 = self.pool_p4(x_p4).flatten(1)
        feat_p4 = self.drop_p4(feat_p4)
        feat_p4 = self.fc_p4(feat_p4)
        logits_p4 = self.margin_p4(feat_p4, species)
        ####----head p4 ####
        
        if self.backbone.layers[3].downsample is not None:
            x = self.backbone.layers[3].downsample(x)
        
        
        conv5 = x
        x = self.backbone.norm(x)  # B L C
        x = self.backbone.avgpool(x.transpose(1, 2))  # B C 1
        features = torch.flatten(x, 1)
        
        features = self.drop_p5(features)
        embedding = self.fc_p5(features)
        logits_p5 = self.margin_p5(embedding, individual_ids)


        return logits_p3, logits_p4, logits_p5
    
    
