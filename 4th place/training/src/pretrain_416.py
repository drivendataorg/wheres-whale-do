
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

IMG_SIZE = 416
N_SPLITS = 5
RANDOM_STATE = 107
MIN_VAL_SAMPLES = 2 # no validation for whale_ids lt MIN_VAL_SAMPLES
MIN_TRAIN_SAMPLES = 4 #no train on whale ids with MIN_TRAIN_SAMPLES but can be in val
NUM_CLASSES = 788
batch_size = 24
mean=(0.485, 0.456, 0.406); std=(0.229, 0.224, 0.225)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_DIR = f'{MODEL_DIR}/models_eff5_r{RANDOM_STATE}_{IMG_SIZE}/tmp'
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
df_scenarios_metadata.shape

n=df[df.N<MIN_TRAIN_SAMPLES].shape[0]
df = df[df.N>=MIN_TRAIN_SAMPLES]
NUM_CLASSES = df.whale_id.nunique()
print(f'removing {n} ids with lt {MIN_TRAIN_SAMPLES} samples; remaining {df.shape[0]} samples; new NUM_CLASSES: {NUM_CLASSES}')

df[df.is_valid].whale_id.nunique(),df[~df.is_valid].whale_id.nunique()

# df1 = df[~df.is_valid].copy()
# df_val_dummy = df[df.is_valid].head(10).copy()
# df_val_dummy['whale_id'] = df1.whale_id.values[0]
# df1 = df1.append(df_val_dummy)
# df = df1.copy()

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


def get_model(arch,pretrained=True):
    model = timm.create_model(arch, num_classes=NUM_CLASSES,pretrained=pretrained)
    return model

def get_dls(df,bs,size,max_size):
    dls = ImageDataLoaders.from_df(df[['path','whale_id','is_valid']],path='/',
                               valid_col='is_valid',
                                item_tfms = [

                                    RandomResizedCrop(max_size,min_scale=0.5, ratio=(0.1, 10),val_xtra=0.,)
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


def saliency_bbox(img, lam):
    size = img.size()
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # initialize OpenCV's static fine grained saliency detector and compute the saliency map
    temp_img = img.cpu().numpy().transpose(1, 2, 0)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(temp_img)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
    x = maximum_indices[0]
    y = maximum_indices[1]

    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class SalMix(Callback):
    def __init__(self,cfg):
        super(SalMix, self).__init__()
        self.lfn = LabelSmoothingCrossEntropyFlat()
        self.lfn.to(DEVICE)
        self.salmix_prob = cfg.salmix_prob
        self.beta = cfg.beta
        self.mixed = False
    
    def before_batch(self,**kwargs):
        if not self.learn.training:
            return
          
        # print(self.learn.xb[0].shape)
        # print(len(self.learn.xb))
        input = self.learn.xb[0]
        r = np.random.rand(1)
        if r < self.salmix_prob:
            self.mixed=True
            target = cast(self.y,Tensor) 
            lam = np.random.beta(self.beta, self.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = saliency_bbox(input[rand_index[0]], lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            self.lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            self.learn.xb = (input,)
          # # compute output
          # input_var = torch.autograd.Variable(input, requires_grad=True)

            self.target_a_var = torch.autograd.Variable(target_a)
            self.target_b_var = torch.autograd.Variable(target_b)

    def after_loss(self, **kwargs):
        if self.mixed:
            output = self.learn.pred
            loss = self.lfn(output, self.target_a_var) * self.lam + self.lfn(output, self.target_b_var) * (1. - self.lam)
            self.learn.loss = loss 
            self.learn.loss_grad = loss 
            self.mixed=False
            self.lam = None
            self.target_a_var = None
            self.target_b_var = None
    


fix_seed(RANDOM_STATE)
size=max_size=IMG_SIZE
dls = get_dls(df,bs=batch_size,size=size,max_size=max_size)
dls.show_batch()

cfg = SimpleNamespace()
cfg.beta = 1 
cfg.salmix_prob = 0.5

arch = 'tf_efficientnet_b5_ns'
print(f'training model {arch} fold {FOLD}')
model = timm.create_model(arch, num_classes=NUM_CLASSES,pretrained=True)

learn = Learner(dls, model, #splitter=default_split,    
                loss_func = LabelSmoothingCrossEntropyFlat(),
                cbs=[
                    Metr,
                    SalMix(cfg),
                    SaveModelCallback(monitor='maplb',comp=np.greater),
                    ],metrics=[accuracy,map20,maplb],
                
               )

learn.to_fp16()
with learn.no_bar():
    learn.fit_one_cycle(40, 5e-4); torch.save(learn.model.state_dict(), f'{MODEL_DIR}/{arch}_f{FOLD}_p0.pth')
