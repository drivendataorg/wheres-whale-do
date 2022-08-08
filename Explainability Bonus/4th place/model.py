
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiAtrousModule(nn.Module):
    def __init__(self, in_chans, out_chans, dilations=[5,7,11,17,19,23,25]):
        super(MultiAtrousModule, self).__init__()
        sz=512
        self.dconvs = [nn.Conv2d(in_chans, sz, kernel_size=3, dilation=dilation,padding=dilation) for dilation in dilations]
        self.dconvs = nn.ModuleList(self.dconvs)
        
        self.conv1 = nn.Conv2d(sz * len(dilations), out_chans, kernel_size=1)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        xs = []
        for dconv in self.dconvs:
            xs.append(dconv(x))

        x = torch.cat(xs,dim=1)
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

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(1).expand_as(x)
        return x

  
class MAM(nn.Module):
    def __init__(self, mam,conv_g,bn_g,act_g,attention2d):
        super(MAM, self).__init__()
       
        self.mam = mam
        self.conv_g = conv_g
        self.bn_g = bn_g
        self.act_g =  act_g
        self.attention2d = attention2d
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fusion = OrthogonalFusion()
        self.fusion_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):

        x_l = x[-2]
        x_g = x[-1]
        
        x_l = self.mam(x_l)
        x_l, att_score = self.attention2d(x_l)
        
        x_g = self.conv_g(x_g)
        x_g = self.bn_g(x_g)
        x_g = self.act_g(x_g)
        
        x_g = self.global_pool(x_g)
        x_g = x_g[:,:,0,0]
        
        x_fused = self.fusion(x_l, x_g)
        x_fused = self.fusion_pool(x_fused)
        x_fused = x_fused[:,:,0,0]        
        
        return x_fused

def load_model(path):
  backbone_ = 'tf_efficientnet_b5_ns' if 'b5_ns' in path else 'tf_efficientnet_b7_ns'
  
  backbone = timm.create_model(backbone_, pretrained=False, num_classes=0, global_pool="", in_chans=3,features_only = True)
  backbone_out = backbone.feature_info[-1]['num_chs']
  backbone_out_1 = backbone.feature_info[-2]['num_chs']
  feature_dim_l_g = 1024
  fusion_out = 2 * feature_dim_l_g

  _mam = MultiAtrousModule(backbone_out_1, feature_dim_l_g)
  _conv_g = nn.Conv2d(backbone_out,feature_dim_l_g,kernel_size=1)
  _bn_g = nn.BatchNorm2d(feature_dim_l_g, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
  _act_g =  nn.SiLU(inplace=True)
  _attention2d = SpatialAttention2d(feature_dim_l_g)
  mam = MAM(_mam,_conv_g,_bn_g,_act_g,_attention2d)
  neck = nn.Sequential(
          nn.Linear(fusion_out, 1024, bias=True),
          nn.BatchNorm1d(1024),
          torch.nn.PReLU()
      )
  model = nn.Sequential(backbone,mam,neck)
  weights = torch.load(path,map_location=DEVICE)
  model.load_state_dict(weights,strict=False)
  model.to(DEVICE); model.eval(); print('')
  return model