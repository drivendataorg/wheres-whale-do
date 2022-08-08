
import pandas as pd, numpy as np
import sys, os, random, shutil,time,argparse
from tqdm.auto import tqdm                                                                                                                                                    
from fastai.vision.all import *
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
import sklearn.metrics as skm
import timm
from timm import create_model
from timm.data.transforms_factory import create_transform
from fastai.vision.learner import _update_first_layer
from fastai.callback.core import Callback
from sklearn.metrics.pairwise import cosine_similarity
import albumentations as A
import cv2
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from types import SimpleNamespace


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="Path to root data directory", required=True)
parser.add_argument("--model_dir", help="Directory to save models", required=True, default='models')
parser.add_argument("--fold",type=int, help="Data split to use for validation (0-4)" ) 

args = parser.parse_args()
DATA_DIR = args.data_dir 
MODEL_DIR = args.model_dir 
FOLD = args.fold

IMG_SIZE = 256
N_SPLITS = 5
RANDOM_STATE = 888
MIN_VAL_SAMPLES = 2 # no validation for whale_ids lt MIN_VAL_SAMPLES
MIN_TRAIN_SAMPLES = 4 #no train on whale ids with MIN_TRAIN_SAMPLES but can be in val
# MIN_ENC = 2
NUM_CLASSES = 788
batch_size = 36
mean = std = [0.5,0.5,0.5]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_DIR = f'{MODEL_DIR}/models_eff7_r{RANDOM_STATE}_{IMG_SIZE}'
os.makedirs(MODEL_DIR,exist_ok=True)


def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        dls.rng.seed(seed)
    except NameError:
        pass

fix_seed(RANDOM_STATE)

from sklearn.metrics import average_precision_score
PREDICTION_LIMIT = 20
QUERY_ID_COL = "query_id"
DATABASE_ID_COL = "database_image_id"
SCORE_COL = "score"
class MeanAveragePrecision:
    @classmethod
    def score(cls, predicted: pd.DataFrame, actual: pd.DataFrame, prediction_limit: int):
        """Calculates mean average precision for a ranking task.
        :param predicted: The predicted values as a dataframe with specified column names
        :param actual: The ground truth values as a dataframe with specified column names
        """
        if not predicted[SCORE_COL].between(0.0, 1.0).all():
            raise ValueError("Scores must be in range [0, 1].")
        if predicted.index.name != QUERY_ID_COL:
            raise ValueError(
                f"First column of submission must be named '{QUERY_ID_COL}', "
                f"got {predicted.index.name}."
            )
        if predicted.columns.to_list() != [DATABASE_ID_COL, SCORE_COL]:
            raise ValueError(
                f"Columns of submission must be named '{[DATABASE_ID_COL, SCORE_COL]}', "
                f"got {predicted.columns.to_list()}."
            )

        unadjusted_aps, predicted_n_pos, actual_n_pos = cls._score_per_query(
            predicted, actual, prediction_limit
        )
        adjusted_aps = unadjusted_aps.multiply(predicted_n_pos).divide(actual_n_pos)
        return adjusted_aps.mean()

    @classmethod
    def _score_per_query(
        cls, predicted: pd.DataFrame, actual: pd.DataFrame, prediction_limit: int
    ):
        """Calculates per-query mean average precision for a ranking task."""
        merged = predicted.merge(
            right=actual.assign(actual=1.0),
            how="left",
            on=[QUERY_ID_COL, DATABASE_ID_COL],
        ).fillna({"actual": 0.0})
        # Per-query raw average precisions based on predictions
        unadjusted_aps = merged.groupby(QUERY_ID_COL).apply(
            lambda df: average_precision_score(df["actual"].values, df[SCORE_COL].values)
            if df["actual"].sum()
            else 0.0
        )
        # Total ground truth positive counts for rescaling
        predicted_n_pos = merged["actual"].groupby(QUERY_ID_COL).sum().astype("int64").rename()
        actual_n_pos = actual.groupby(QUERY_ID_COL).size().clip(upper=prediction_limit)
        return unadjusted_aps, predicted_n_pos, actual_n_pos

df = pd.read_csv(f'{DATA_DIR}/metadata.csv',parse_dates = ['date','timestamp'])
df['path'] = f'{DATA_DIR}/'+df.path
d = df.groupby('whale_id').count()[['image_id']].rename(columns={'image_id':'N'}).reset_index()
df = pd.merge(df,d)
d = df.drop_duplicates(subset=['encounter_id']).groupby('whale_id').count()[['encounter_id']].rename(columns={'encounter_id':'N_enc'}).reset_index()
df = pd.merge(df,d)

skf  = StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=RANDOM_STATE)
df['fold'] = -1
for fold_id, (train_index, test_index) in enumerate(skf.split(df, df.whale_id)):
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    df.loc[test_index,'fold'] = fold_id

df.loc[df.N<MIN_VAL_SAMPLES,'fold'] = -1
df['is_valid'] = False
df.loc[df.fold==FOLD,'is_valid']=True
print(df.shape[0],df[~df.is_valid].shape[0],df[df.is_valid].shape[0])
print(df.whale_id.nunique(),df[~df.is_valid].whale_id.nunique(),df[df.is_valid].whale_id.nunique())

assert(df.whale_id.nunique()==df[~df.is_valid].whale_id.nunique()==NUM_CLASSES)

def validate_qry_db(qry,db):
#     qry_db = pd.merge(qry,db,on='whale_id')
#     d = qry_db.groupby('whale_id').count()[['database_image_id']].rename(columns={'database_image_id':'N'}).reset_index()
#     qry_db = pd.merge(qry_db,d)
#     invalid_queries = qry_db[qry_db.N==1].query_image_id.values
#     invalid_whale_ids = qry_db[qry_db.N==1].whale_id.unique()
#     db = db[~db.whale_id.isin(invalid_whale_ids)].copy()
#     qry = qry[~qry.whale_id.isin(invalid_whale_ids)].copy()
    return qry,db

#create scenarios
DATA_DIRECTORY = Path('./')
OUT_DIR = f'{DATA_DIRECTORY}/tmp/val'

os.makedirs(OUT_DIR,exist_ok=True)

df_val = df[df.is_valid].copy()
cols = ['image_id','whale_id']
query_scenarios = [] ##scenario_id,queries_path,database_path

sid = 's0'
db  = df_val[(df_val.viewpoint=='top')][cols].copy().rename(columns={'image_id':'database_image_id'})
qry = df_val[(df_val.viewpoint=='top')][cols].copy().rename(columns={'image_id':'query_image_id'})
qry = qry[qry.whale_id.isin(db.whale_id.unique())]
qry['query_id'] = sid +'-'+qry.query_image_id
qry,db = validate_qry_db(qry,db)
db.to_csv(f'{OUT_DIR}/{sid}_db.csv',index=False)
qry.to_csv(f'{OUT_DIR}/{sid}_qry.csv',index=False)
query_scenarios.append([sid,f'{OUT_DIR}/{sid}_qry.csv',f'{OUT_DIR}/{sid}_db.csv',len(qry),len(db)])

sid = 's1'
db  = df_val[(df_val.viewpoint=='top')&(df_val.date.dt.year==2017)][cols].copy().rename(columns={'image_id':'database_image_id'})
qry = df_val[(df_val.viewpoint=='top')&(df_val.date.dt.year==2017)][cols].copy().rename(columns={'image_id':'query_image_id'})
qry = qry[qry.whale_id.isin(db.whale_id.unique())]
qry['query_id'] = sid +'-'+qry.query_image_id
qry,db = validate_qry_db(qry,db)
db.to_csv(f'{OUT_DIR}/{sid}_db.csv',index=False)
qry.to_csv(f'{OUT_DIR}/{sid}_qry.csv',index=False)
query_scenarios.append([sid,f'{OUT_DIR}/{sid}_qry.csv',f'{OUT_DIR}/{sid}_db.csv',len(qry),len(db)])

sid = 's2'
db  = df_val[(df_val.viewpoint=='top')&(df_val.date.dt.year==2018)][cols].copy().rename(columns={'image_id':'database_image_id'})
qry = df_val[(df_val.viewpoint=='top')&(df_val.date.dt.year==2018)][cols].copy().rename(columns={'image_id':'query_image_id'})
qry = qry[qry.whale_id.isin(db.whale_id.unique())]
qry['query_id'] = sid +'-'+qry.query_image_id
qry,db = validate_qry_db(qry,db)
db.to_csv(f'{OUT_DIR}/{sid}_db.csv',index=False)
qry.to_csv(f'{OUT_DIR}/{sid}_qry.csv',index=False)
query_scenarios.append([sid,f'{OUT_DIR}/{sid}_qry.csv',f'{OUT_DIR}/{sid}_db.csv',len(qry),len(db)])


sid = 's3'
db  = df_val[(df_val.viewpoint=='top')&(df_val.date.dt.year==2019)][cols].copy().rename(columns={'image_id':'database_image_id'})
qry = df_val[(df_val.viewpoint=='top')&(df_val.date.dt.year==2019)][cols].copy().rename(columns={'image_id':'query_image_id'})
qry = qry[qry.whale_id.isin(db.whale_id.unique())]
qry['query_id'] = sid +'-'+qry.query_image_id
qry,db = validate_qry_db(qry,db)
db.to_csv(f'{OUT_DIR}/{sid}_db.csv',index=False)
qry.to_csv(f'{OUT_DIR}/{sid}_qry.csv',index=False)
query_scenarios.append([sid,f'{OUT_DIR}/{sid}_qry.csv',f'{OUT_DIR}/{sid}_db.csv',len(qry),len(db)])


sid = 's4'
db  = df_val[(df_val.viewpoint=='top')&(df_val.date.dt.year==2017)][cols].copy().rename(columns={'image_id':'database_image_id'})
qry = df_val[(df_val.viewpoint=='top')&(df_val.date.dt.year==2018)][cols].copy().rename(columns={'image_id':'query_image_id'})
qry = qry[qry.whale_id.isin(db.whale_id.unique())]
qry['query_id'] = sid +'-'+qry.query_image_id
qry,db = validate_qry_db(qry,db)
db.to_csv(f'{OUT_DIR}/{sid}_db.csv',index=False)
qry.to_csv(f'{OUT_DIR}/{sid}_qry.csv',index=False)
query_scenarios.append([sid,f'{OUT_DIR}/{sid}_qry.csv',f'{OUT_DIR}/{sid}_db.csv',len(qry),len(db)])

sid = 's5'
db  = df_val[(df_val.viewpoint=='top')&(df_val.date.dt.year==2017)][cols].copy().rename(columns={'image_id':'database_image_id'})
qry = df_val[(df_val.viewpoint=='top')&(df_val.date.dt.year==2019)][cols].copy().rename(columns={'image_id':'query_image_id'})
qry = qry[qry.whale_id.isin(db.whale_id.unique())]
qry['query_id'] = sid +'-'+qry.query_image_id
qry,db = validate_qry_db(qry,db)
db.to_csv(f'{OUT_DIR}/{sid}_db.csv',index=False)
qry.to_csv(f'{OUT_DIR}/{sid}_qry.csv',index=False)
query_scenarios.append([sid,f'{OUT_DIR}/{sid}_qry.csv',f'{OUT_DIR}/{sid}_db.csv',len(qry),len(db)])

sid = 's6'
db  = df_val[(df_val.viewpoint=='top')&(df_val.date.dt.year==2018)][cols].copy().rename(columns={'image_id':'database_image_id'})
qry = df_val[(df_val.viewpoint=='top')&(df_val.date.dt.year==2019)][cols].copy().rename(columns={'image_id':'query_image_id'})
qry = qry[qry.whale_id.isin(db.whale_id.unique())]
qry['query_id'] = sid +'-'+qry.query_image_id
qry,db = validate_qry_db(qry,db)
db.to_csv(f'{OUT_DIR}/{sid}_db.csv',index=False)
qry.to_csv(f'{OUT_DIR}/{sid}_qry.csv',index=False)
query_scenarios.append([sid,f'{OUT_DIR}/{sid}_qry.csv',f'{OUT_DIR}/{sid}_db.csv',len(qry),len(db)])


sid = 's7'
db  = df_val[(df_val.viewpoint.isin(['left', 'right']))][cols].copy().rename(columns={'image_id':'database_image_id'})
qry = df_val[(df_val.viewpoint=='top')][cols].copy().rename(columns={'image_id':'query_image_id'})
# db  = df[(df.viewpoint.isin(['left', 'right']))][cols].copy().rename(columns={'image_id':'database_image_id'})
# qry = df_val[(df_val.viewpoint=='top')][cols].copy().rename(columns={'image_id':'query_image_id'})

qry = qry[qry.whale_id.isin(db.whale_id.unique())]
qry['query_id'] = sid +'-'+qry.query_image_id
qry,db = validate_qry_db(qry,db)
db.to_csv(f'{OUT_DIR}/{sid}_db.csv',index=False)
qry.to_csv(f'{OUT_DIR}/{sid}_qry.csv',index=False)
query_scenarios.append([sid,f'{OUT_DIR}/{sid}_qry.csv',f'{OUT_DIR}/{sid}_db.csv',len(qry),len(db)])

sid = 's8'
db  = df_val[(df_val.viewpoint=='top')][cols].copy().rename(columns={'image_id':'database_image_id'})
qry = df_val[(df_val.viewpoint.isin(['left', 'right']))][cols].copy().rename(columns={'image_id':'query_image_id'})

# db  = df_val[(df_val.viewpoint=='top')][cols].copy().rename(columns={'image_id':'database_image_id'})
# qry = df[(df.viewpoint.isin(['left', 'right']))][cols].copy().rename(columns={'image_id':'query_image_id'})

qry = qry[qry.whale_id.isin(db.whale_id.unique())]
qry['query_id'] = sid +'-'+qry.query_image_id
qry,db = validate_qry_db(qry,db)
db.to_csv(f'{OUT_DIR}/{sid}_db.csv',index=False)
qry.to_csv(f'{OUT_DIR}/{sid}_qry.csv',index=False)
query_scenarios.append([sid,f'{OUT_DIR}/{sid}_qry.csv',f'{OUT_DIR}/{sid}_db.csv',len(qry),len(db)])


query_scenarios = pd.DataFrame(query_scenarios,columns=['scenario_id','queries_path','database_path','qry','db'])
query_scenarios.to_csv(f'{DATA_DIRECTORY}/query_scenarios.csv',index=False)

print(query_scenarios)

gt_df = []
for row in query_scenarios.itertuples():
    # load query df and database images; subset embeddings to this scenario's database
    df_qry = pd.read_csv(DATA_DIRECTORY / row.queries_path)
    df_db = pd.read_csv(DATA_DIRECTORY / row.database_path)
    d = pd.merge(df_db,df_qry,on='whale_id')
    gt_df.append(d)
gt_df = pd.concat(gt_df)
gt_df.to_csv(f'{OUT_DIR}/gt.csv',index=False)

scenario_imgs = []
for row in query_scenarios.itertuples():
    scenario_imgs.extend(pd.read_csv(DATA_DIRECTORY / row.queries_path).query_image_id.values)
    scenario_imgs.extend(pd.read_csv(DATA_DIRECTORY / row.database_path).database_image_id.values)
scenario_imgs = sorted(set(scenario_imgs))
df_scenarios_metadata = df.set_index('image_id').loc[scenario_imgs].copy()

n=df[df.N<MIN_TRAIN_SAMPLES].shape[0]
df = df[df.N>=MIN_TRAIN_SAMPLES]
NUM_CLASSES = df.whale_id.nunique()
print(f'removing {n} ids with lt {MIN_TRAIN_SAMPLES} samples; remaining {df.shape[0]} samples; new NUM_CLASSES: {NUM_CLASSES}')

df_trn = df[~df.is_valid].copy()
#df_trn.whale_id.nunique()

def get_dls(df,bs,size,max_size):
    dls = ImageDataLoaders.from_df(df[['path','whale_id','is_valid']],path='/',
                               valid_col='is_valid',
                                item_tfms = [
                                    Resize(max_size,method='squish', pad_mode='zeros'),
                                ],
                               batch_tfms=[
                                          *aug_transforms(size=size,min_scale=1,
                                        do_flip=True,
                                        flip_vert=False,
                                        max_rotate=10.0,
                                        min_zoom=1,
                                        max_zoom=1.1,
                                        max_lighting=0.2,
                                        max_warp=0.2,
                                        p_affine=0.75,
                                        p_lighting=0.75,
                                        mult=1.0,xtra_tfms=None,mode='bilinear',pad_mode='zeros'
                                        ),
                                           
                                          Normalize.from_stats(mean,std)
                                          #Normalize.from_stats(*imagenet_stats)
                                          ],
                              bs=bs,
                              seed = RANDOM_STATE
                              )
    return dls


def map_per_image(label, predictions):
    try:
        return 1 / (predictions[:20].index(label) + 1)
    except ValueError:
        return 0.0

def map20(pred, targ):
    pred = torch.argsort(-pred,axis=1).cpu().numpy()
    pred = pred[:,:20]
    targ = targ.cpu().numpy()
    scores = []
    for i in range(len(pred)):
        p = pred[i].tolist()
        t = targ[i]
        score = map_per_image(t, p)
        scores.append(score)
    score = np.mean(scores)
    return score


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
rsz = Resize(IMG_SIZE, method='squish',pad_mode='zeros')
class ImagesDataset(Dataset):
    """Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id and image tensors.
    """

    def __init__(self, metadata):
        self.metadata = metadata
        self.transform = transforms.Compose(
            [
                #transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
#                     mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    mean = mean, std =std,
                ),
            ]
        )

    def __getitem__(self, idx):
        image = Image.open(self.metadata.path.iloc[idx]).convert("RGB")
        image = rsz(image)
        image = self.transform(image)
        sample = {"image_id": self.metadata.index[idx], "image": image}
        return sample

    def __len__(self):
        return len(self.metadata)

dataset = ImagesDataset(df_scenarios_metadata)
dataloader = DataLoader(dataset, batch_size=batch_size)

class Metr(Callback):
#     def before_epoch(self,**kwargs):
#         "Reset predictions"
#         self.embeddings = Tensor([])
    
    def before_validate(self, **kwargs):
        self.learn.model.eval()
        mdl = nn.Sequential(*list(learn.model.children())[:-1])
        mdl.eval();
        embeddings = []
#         for batch in tqdm(dataloader, total=len(dataloader)):
        with torch.no_grad():
          for batch in dataloader:
  #             batch_embeddings,batch_preds = self.learn.model(batch["image"].to(DEVICE))
              batch_embeddings = mdl(batch["image"].to(DEVICE))
              batch_embeddings = batch_embeddings.detach().cpu().numpy()
              #batch_embeddings = sklearn.preprocessing.normalize(batch_embeddings, norm='l2', axis=1)
              batch_embeddings_df = pd.DataFrame(batch_embeddings, index=batch["image_id"])
              #logger.info(batch["image_id"])
              embeddings.append(batch_embeddings_df)

        embeddings = pd.concat(embeddings)
        results = []
        for row in query_scenarios.itertuples():
            # load query df and database images; subset embeddings to this scenario's database
            qry_df = pd.read_csv(DATA_DIRECTORY / row.queries_path)
            db_img_ids = pd.read_csv(DATA_DIRECTORY / row.database_path).database_image_id.values
            db_embeddings = embeddings.loc[db_img_ids]

            scenario_db = df_scenarios_metadata.loc[db_img_ids]
            scenario_qry = df_scenarios_metadata.loc[qry_df.query_image_id.values]
            # predict matches for each query in this scenario
            for qry in qry_df.itertuples():
                # get embeddings; drop query from database, if it exists
                qry_embedding = embeddings.loc[[qry.query_image_id]]
                _db_embeddings = db_embeddings.drop(qry.query_image_id, errors='ignore')

                # compute cosine similarities and get top 20
                sims = cosine_similarity(qry_embedding, _db_embeddings)[0]
                top20 = pd.Series(sims, index=_db_embeddings.index).sort_values(0, ascending=False).head(20)

                # append result
                qry_result = pd.DataFrame(
                    {"query_id": qry.query_id, "database_image_id": top20.index, "score": top20.values}
                )
                results.append(qry_result)


        predicted = pd.concat(results).set_index('query_id')

        predicted['score'] = np.clip(predicted['score'],0,1)
        actual = gt_df.set_index('query_id')
        mean_avg_prec = MeanAveragePrecision.score(
            predicted=predicted, actual=actual, prediction_limit=PREDICTION_LIMIT
        )
        self.learn.maplb = mean_avg_prec
        



def maplb(x,y):
    return learn.maplb

class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine   


class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    def __init__(self, margins, n_classes, s=30.0):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.margins = margins
        self.out_dim =n_classes
            
    def forward(self, logits, labels):
        ms = []
        ms = self.margins[labels.cpu().numpy()]
        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, self.out_dim).float()
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1,1) - sine * sin_m.view(-1,1)
        phi = torch.where(cosine > th.view(-1,1), phi, cosine - mm.view(-1,1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss     



class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s=16.0, m=0.1, crit="bce", weight=None, reduction="mean",class_weights_norm=None ):
        super().__init__()

        self.weight = weight
        self.reduction = reduction
        self.class_weights_norm = class_weights_norm
        
        self.crit = nn.CrossEntropyLoss(reduction="none")   
        #self.crit = DenseCrossEntropy()
        
        if s is None:
            self.s = torch.nn.Parameter(torch.tensor([16.], requires_grad=True, device='cuda'))
        else:
            self.s = s

        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, logits, labels):

        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        labels2 = torch.zeros_like(cosine)
        labels2.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (labels2 * phi) + ((1.0 - labels2) * cosine)

        s = self.s

        output = output * s
        loss = self.crit(output, labels)

        if self.weight is not None:
            w = self.weight[labels].to(logits.device)

            loss = loss * w
            if self.class_weights_norm == "batch":
                loss = loss.sum() / w.sum()
            if self.class_weights_norm == "global":
                loss = loss.mean()
            else:
                loss = loss.mean()
            
            return loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss    



class MultiAtrousModule(nn.Module):
    def __init__(self, in_chans, out_chans, dilations):
        super(MultiAtrousModule, self).__init__()
        sz=512
        self.dconvs = [nn.Conv2d(in_chans, sz, kernel_size=3, dilation=dilation,padding='same') for dilation in dilations]
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
        att_score : softplus attention score 
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
  


class CustLoss(nn.Module):
    def __init__(self,cfg):
        super(CustLoss, self).__init__()

        self.lfn = ArcFaceLossAdaptiveMargin(cfg.arc_margins,NUM_CLASSES,cfg.arc_s)
        self.lfn = self.lfn.to(DEVICE)

    def forward(self, logits, labels):
        #logits = self.head(embeddings)
        logits,labels = TensorBase(logits), TensorBase(labels)
        loss = self.lfn(logits, labels)
        return loss


cfg = SimpleNamespace()
cfg.backbone = 'tf_efficientnet_b7_ns'
cfg.backbonep0 = f'{MODEL_DIR}/tmp/{cfg.backbone}_f{FOLD}_p0.pth'

cfg.pool = 'avg'
cfg.embedding_size = 1024
cfg.dilations = [5,7,11,17,19,23,25]
cfg.arc_m_ub = 0.3 
cfg.arc_m_lb = 0.05
cfg.arc_s = 0.16
print(cfg)

tmp = np.sqrt(1 / np.sqrt(df_trn['whale_id'].value_counts().sort_index().values))
# tmp = np.sqrt(1 / np.sqrt(df['whale_id'].value_counts().sort_index().values))
arcf_margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * cfg.arc_m_ub + cfg.arc_m_lb
cfg.arc_margins = arcf_margins
    
fix_seed(RANDOM_STATE)
size=max_size=IMG_SIZE
bs = batch_size #64#32

dls = get_dls(df,bs=bs,size=size,max_size=max_size)


backbone = timm.create_model(cfg.backbone, pretrained=True, num_classes=0, global_pool="", in_chans=3,features_only = True)
backbone.load_state_dict(torch.load(cfg.backbonep0),strict=False)
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
head = ArcMarginProduct(cfg.embedding_size, NUM_CLASSES)

model = nn.Sequential(backbone,mam,neck,head)

print(f'training model {cfg.backbone} fold {FOLD}')

def model_split(m):
    return L(m[0], m[1], m[2], m[3]).map(params)
learn = Learner(dls, model, splitter=model_split,    
                loss_func = CustLoss(cfg),
                cbs=[Metr,
                    SaveModelCallback(monitor='maplb',comp=np.greater)
                    ],metrics=[maplb],
               )
learn.to_fp16()

with learn.no_bar():
    learn.fit_one_cycle(10, lr_max=[5e-4,8e-4,1e-3,5e-5]); torch.save(learn.model.state_dict(), f'{MODEL_DIR}/{cfg.backbone}_f{FOLD}_p1.pth')
