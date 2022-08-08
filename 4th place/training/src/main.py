
from pathlib import Path
from loguru import logger
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from fastai.vision.all import *
import sklearn.preprocessing  
from timm import create_model
from timm.data.transforms_factory import create_transform
from fastai.vision.learner import _update_first_layer
from fastai.callback.core import Callback
import copy
import sklearn.preprocessing
import multiprocessing as mp

ROOT_DIRECTORY = Path("/code_execution") 
PREDICTION_FILE = ROOT_DIRECTORY / "submission" / "submission.csv"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_CLASSES = 788
NUM_WORKERS = mp.cpu_count()
logger.info(f'using {NUM_WORKERS} threads')

query_scenarios = pd.read_csv(DATA_DIRECTORY / "query_scenarios.csv", index_col="scenario_id")
metadata = pd.read_csv(DATA_DIRECTORY / "metadata.csv", index_col="image_id")

models_eff7_r888_256 = [f'models/models_eff7_r888_256/tf_efficientnet_b7_ns_f{i}_p1.pth' for i in range(0,5)]
models_eff7_r41_448 = [f'models/models_eff7_r41_448/tf_efficientnet_b7_ns_f{i}_p1.pth' for i in range(0,5)]

models_eff5_r107_416 = [f'models/models_eff5_r107_416/tf_efficientnet_b5_ns_f{i}_p1.pth' for i in range(0,5)]
#models_eff5_r107_416 = [p for p in models_eff5_r107_416 if '_f0_' in p]


tflip = transforms.RandomHorizontalFlip(p=1)

class ImagesDataset(Dataset):
    """Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id and image tensors.
    """

    def __init__(self, metadata, img_size, mean, std):
        self.metadata = metadata
        self.rsz = Resize(img_size, method='squish',pad_mode='zeros')
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=mean, std=std
                ),
            ]
        )
    
    def __getitem__(self, idx):
        image = Image.open(DATA_DIRECTORY / self.metadata.path.iloc[idx]).convert("RGB")
        image = self.rsz(image)
        image = self.transform(image)
        image1 = tflip(image)
        #sample = {"image_id": self.metadata.index[idx], "image": image, "image1":image1}
        sample = {"image_id": self.metadata.index[idx], "image": image}
        return sample

    def __len__(self):
        return len(self.metadata)


from timm.models.layers import SelectAdaptivePool2d


def get_timm_model(arch,pretrained=False):
    model = timm.create_model(arch, num_classes=NUM_CLASSES,pretrained=pretrained)
    return model

def create_timm_body(arch:str, pretrained=True, drop_rate=0.0, cut=None, n_in=3):
    "Creates a body from any model in the `timm` library."
    model = create_model(arch, pretrained=pretrained, drop_rate=drop_rate, 
                         num_classes=0, global_pool='')
    _update_first_layer(model, n_in, pretrained)  #model,in_channels,pretrained 
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int): return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): return cut(model)
    else: raise NameError("cut must be either integer or function")

          
class MultiAtrousModule(nn.Module):
    def __init__(self, in_chans, out_chans, dilations):
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
  
def get_model(cfg):
    backbone = timm.create_model(cfg.backbone, pretrained=False, num_classes=0, global_pool="", in_chans=3,features_only = True)
    backbone_out = backbone.feature_info[-1]['num_chs']
    backbone_out_1 = backbone.feature_info[-2]['num_chs']
    feature_dim_l_g = 1024
    fusion_out = 2 * feature_dim_l_g

    _mam = MultiAtrousModule(backbone_out_1, feature_dim_l_g, cfg.dilations)
    _conv_g = nn.Conv2d(backbone_out,feature_dim_l_g,kernel_size=1)
    _bn_g = nn.BatchNorm2d(feature_dim_l_g, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    _act_g =  nn.SiLU(inplace=True)
    _attention2d = SpatialAttention2d(feature_dim_l_g)
    mam = MAM(_mam,_conv_g,_bn_g,_act_g,_attention2d)
    neck = nn.Sequential(
            nn.Linear(fusion_out, cfg.embedding_size, bias=True),
            nn.BatchNorm1d(cfg.embedding_size),
            torch.nn.PReLU()
        )
    m = nn.Sequential(backbone,mam,neck)
    return m


def get_embeddings(dataloader,models):
    embeddings = [[] for i in range(len(models))]
    for batch in dataloader:
          x = batch["image"].to(DEVICE)
          B0 = []
          for i in range(len(models)):
            model = models[i]
            b0 = model(x).detach().cpu().numpy()
            b0 = sklearn.preprocessing.normalize(b0,norm='l2')
            B0.append(b0)
          
          #x = torch.flip(x, [3])
          x = batch["image1"].to(DEVICE)
          for i in range(len(models)):
            model = models[i]
            b1 = model(x).detach().cpu().numpy()
            b1 = sklearn.preprocessing.normalize(b1,norm='l2')
            b0 = B0[i]
            b = (b0+b1)/2
            b = pd.DataFrame(b, index=batch["image_id"])
            #logger.info(batch["image_id"])
            embeddings[i].append(b)
    
    for i in range(len(embeddings)):
      embeddings[i] = pd.concat(embeddings[i])
    return embeddings

def get_model_embeddings(dataloader,model):
    embeddings = []
    for batch in dataloader:
          x = batch["image"].to(DEVICE)
          b0 = model(x).detach().cpu().numpy()
          # x = torch.flip(x, [1, 0])
          # x = batch["image1"].to(DEVICE)
          x = torch.flip(x, [3])
          b1 = model(x).detach().cpu().numpy()
          b0 = sklearn.preprocessing.normalize(b0,norm='l2')
          b1 = sklearn.preprocessing.normalize(b1,norm='l2')
          b = (b0+b1)/2
          b = pd.DataFrame(b, index=batch["image_id"])
          #logger.info(batch["image_id"])
          embeddings.append(b)
          
    embeddings= pd.concat(embeddings)
    return embeddings
    
EMBEDDINGS = []


def get_matches_for_queries(qry_df_sub,db_img_ids):
    result = []
    for qry in qry_df_sub.itertuples():
        tmp_qry_results = []
          #get all embeddings for each model
        for embeddings in EMBEDDINGS:
          #get database embeddings
          db_embeddings = embeddings.loc[db_img_ids]
          qry_embedding = embeddings.loc[[qry.query_image_id]]
          #drop query from database, if it exists
          _db_embeddings = db_embeddings.drop(qry.query_image_id, errors='ignore')
          # compute cosine similarities
          sims = cosine_similarity(qry_embedding, _db_embeddings)[0]
          #sims = (sims-sims.min()) /(sims.max()-sims.min())

          # res = pd.Series(sims, index=_db_embeddings.index).sort_values(0, ascending=False)
          res = pd.DataFrame(dict(database_image_id=_db_embeddings.index, score=sims)).sort_values(by='score', ascending=False)
          tmp_qry_results.append(res)   
            
        tmp_qry_results = pd.concat(tmp_qry_results)
        # append result 
        qry_result = pd.DataFrame(
            {"query_id": qry.query_id, "database_image_id": tmp_qry_results.database_image_id, "score": tmp_qry_results.score}
        )
        qry_result = qry_result.groupby(['query_id','database_image_id']).mean().reset_index()
        qry_result = qry_result.sort_values(by='score', ascending=False)
        qry_result = qry_result.head(20)

        result.append(qry_result)
    
    result = pd.concat(result)
    return result

def main():
    logger.info("Starting main script")
    # load test set data and pretrained model
    query_scenarios = pd.read_csv(DATA_DIRECTORY / "query_scenarios.csv", index_col="scenario_id")
    metadata = pd.read_csv(DATA_DIRECTORY / "metadata.csv", index_col="image_id",parse_dates=['date'])
    logger.info("Loading pre-trained model")

    cfg = SimpleNamespace()
    #cfg.backbone = 'tf_efficientnet_b4_ns'
    cfg.pool = 'avg'
    cfg.embedding_size = 1024
    cfg.pretrained = False
    cfg.dilations = [5,7,11,17,19,23,25]

    # print('model in gpu: ', next(model.parameters()).is_cuda)

    scenario_imgs = []
    for row in query_scenarios.itertuples():
        scenario_imgs.extend(pd.read_csv(DATA_DIRECTORY / row.queries_path).query_image_id.values)
        scenario_imgs.extend(pd.read_csv(DATA_DIRECTORY / row.database_path).database_image_id.values)
    scenario_imgs = sorted(set(scenario_imgs))
    metadata = metadata.loc[scenario_imgs]

    cfg.backbone = 'tf_efficientnet_b7_ns'
    model = get_model(cfg)
    dataset = ImagesDataset(metadata,img_size=256,mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    dataloader = DataLoader(dataset, batch_size=10,num_workers=NUM_WORKERS)
    
    for path in models_eff7_r888_256:
      weights = torch.load(path,map_location=DEVICE)
      model.load_state_dict(weights,strict=False)
      model.to(DEVICE); model.eval()
      logger.info(f"Precomputing embeddings for model {path}")
      embeddings = get_model_embeddings(dataloader,model)
      EMBEDDINGS.append(embeddings)

    dataset = ImagesDataset(metadata,img_size=448,mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    dataloader = DataLoader(dataset, batch_size=3,num_workers=NUM_WORKERS)

    for path in models_eff7_r41_448:
        weights = torch.load(path,map_location=DEVICE)
        model.load_state_dict(weights,strict=False)
        model.to(DEVICE); model.eval()
        logger.info(f"Precomputing embeddings for model {path}")
        embeddings = get_model_embeddings(dataloader,model)
        EMBEDDINGS.append(embeddings)

    cfg.backbone = 'tf_efficientnet_b5_ns'
    model = get_model(cfg)

    dataset = ImagesDataset(metadata,img_size=416,mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    dataloader = DataLoader(dataset, batch_size=8,num_workers=NUM_WORKERS)
    for path in models_eff5_r107_416:
        weights = torch.load(path,map_location=DEVICE)
        model.load_state_dict(weights,strict=False)
        model.to(DEVICE); model.eval()
        logger.info(f"Precomputing embeddings for model {path}")
        embeddings = get_model_embeddings(dataloader,model)
        EMBEDDINGS.append(embeddings)


    logger.info("Generating image rankings")
    all_results = []
    for row in query_scenarios.itertuples():
        logger.info(f'processing scenario {row}')

        # load query df and database images; subset embeddings to this scenario's database
        qry_df = pd.read_csv(DATA_DIRECTORY / row.queries_path)
        db_img_ids = pd.read_csv(DATA_DIRECTORY / row.database_path).database_image_id.values

        scenario_db = metadata.loc[db_img_ids]
        scenario_qry = metadata.loc[qry_df.query_image_id.values]

        # predict matches for each query in this scenario
        pool = mp.Pool(NUM_WORKERS)
        query_ids = qry_df.query_id.values
        qrys_per_process = len(query_ids) / NUM_WORKERS
        tasks = []
        for num_process in range(1, NUM_WORKERS + 1):
            start_index = (num_process - 1) * qrys_per_process + 1
            end_index = num_process * qrys_per_process
            start_index = int(start_index)
            end_index = int(end_index)
            qry_df_sub = qry_df.loc[qry_df.query_id.isin(query_ids[start_index - 1:end_index])]
            tasks.append((qry_df_sub,db_img_ids,))
            logger.info(f"Task # {num_process} processing {len(qry_df_sub)} queries")

        # start tasks
        results = [pool.apply_async(get_matches_for_queries, t) for t in tasks]

        results = [r.get() for r in results]
        all_results = all_results+results
        

    logger.info(f"Writing predictions file to {PREDICTION_FILE}")
    submission = pd.concat(all_results)
    logger.info(f'submission: {submission.shape} {submission.query_id.nunique()}')
    submission['score'] = np.clip(submission.score,0,1)
    submission.to_csv(PREDICTION_FILE, index=False)

if __name__ == "__main__":
    main()