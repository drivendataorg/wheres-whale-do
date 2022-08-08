from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger
import pandas as pd
import numpy as np
import math
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import albumentations as A
from albumentations.pytorch import ToTensorV2

cfg = { "lr_backbone": 1.6e-3,  
        "lr_head": 1.6e-2,      
        "lr_decay_scale": 1.0e-2,
        "batch_size": 32,
        "image_size": (260,260),  # b2-260, b4-380, b5-456
        "max_epochs": 20,
        "optimizer": "AdamW",
        "model_name": "tf_efficientnet_b2_ns",  #
        "out_indices": (3,4),
        "n_splits": -1,  # -1, 5,
        "num_classes": 788,
        "warmup_steps_ratio": 0.2,
        "n_data": -1,

        "s_id": 21.0,              
        "margin_coef_id": 0.5,     
        "margin_power_id": -0.125, 
        "margin_cons_id": 0.05,
        "n_center_id": 2,
       
        "n_nearest_neighbors": 200,
        "num_workers" : 2,
        "wand" : False,
      }

ROOT_DIRECTORY = Path("/code_execution")   
PREDICTION_FILE = ROOT_DIRECTORY / "submission" / "submission.csv"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"

embeddings = []
flip_embeddings = []

class WhaleDataset(Dataset):
    def __init__(self, metadata):
        super().__init__()
        self.metadata = metadata
        augments = []
        augments.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        augments.append(ToTensorV2())  # HWC to CHW
        self.transform = A.Compose(augments)

    def __len__(self):
        return len(self.metadata)

    def get_original_image(self, i: int):
        rgb = Image.open(DATA_DIRECTORY / self.metadata.path.iloc[i]).convert("RGB")
        return rgb

    def __getitem__(self, i: int):
        image = self.get_original_image(i)
        image = np.array(image.resize((cfg["image_size"][0], cfg["image_size"][1]), Image.BICUBIC))
        augmented = self.transform(image=image)["image"]

        return {
            "image_id": self.metadata.index[i],  
            "image": augmented,
        }

class WarmupCosineLambda:
    def __init__(self, warmup_steps: int, cycle_steps: int, decay_scale: float, exponential_warmup: bool = False):
        self.warmup_steps = warmup_steps
        self.cycle_steps = cycle_steps
        self.decay_scale = decay_scale
        self.exponential_warmup = exponential_warmup

    def __call__(self, epoch: int):
        if epoch < self.warmup_steps:
            if self.exponential_warmup:
                return self.decay_scale * pow(self.decay_scale, -epoch / self.warmup_steps)
            ratio = epoch / self.warmup_steps
        else:
            ratio = (1 + math.cos(math.pi * (epoch - self.warmup_steps) / self.cycle_steps)) / 2
        return self.decay_scale + (1 - self.decay_scale) * ratio
    
def topk_average_precision(output: torch.Tensor, y: torch.Tensor, k: int):
    score_array = torch.tensor([1.0 / i for i in range(1, k + 1)], device=output.device)
    topk = output.topk(k)[1]
    match_mat = topk == y[:, None].expand(topk.shape)
    return (match_mat * score_array).sum(dim=1)    

def calc_map5(output: torch.Tensor, y: torch.Tensor, threshold: Optional[float]):
    if threshold is not None:
        output = torch.cat([output, torch.full((output.shape[0], 1), threshold, device=output.device)], dim=1)
    return topk_average_precision(output, y, 5).mean().detach()

def map_dict(output: torch.Tensor, y: torch.Tensor, prefix: str):
    d = {f"{prefix}/acc": topk_average_precision(output, y, 1).mean().detach()}
    for threshold in [None, 0.3, 0.4, 0.5, 0.6, 0.7]:
        d[f"{prefix}/map{threshold}"] = calc_map5(output, y, threshold)
    return d


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, requires_grad=False):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p, requires_grad=requires_grad)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return x.clamp(min=self.eps).pow(self.p).mean((-2, -1)).pow(1.0 / self.p)

class ArcMarginProductSubcenter(nn.Module):
    def __init__(self, in_features: int, out_features: int, k: int = 3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features * k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine

class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    def __init__(self, margins: np.ndarray, n_classes: int, s: float = 30.0):
        super().__init__()
        self.s = s
        self.margins = margins
        self.out_dim = n_classes

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ms = self.margins[labels.cpu().numpy()]
        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, self.out_dim).float()
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1, 1) - sine * sin_m.view(-1, 1)
        phi = torch.where(cosine > th.view(-1, 1), phi, cosine - mm.view(-1, 1))
        return ((labels * phi) + ((1.0 - labels) * cosine)) * self.s

class WhaleDataModule(LightningDataModule):
    def __init__(
        self,
        df: pd.DataFrame,
    ):
        super().__init__()
        self.all_df = df

    def get_dataset(self, df):
        return WhaleDataset(df)

    def all_dataloader(self):
        return DataLoader(
            self.get_dataset(self.all_df),
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers = 4,
            pin_memory=True,
        )


class SphereClassifier(LightningModule):
    def __init__(self, id_class_nums=None):
        super().__init__()
        self.test_results_fp = None

        self.backbone = timm.create_model(
            cfg["model_name"],
            in_chans=3,
            pretrained=False,
            num_classes=0,
            features_only=True,
            out_indices=cfg["out_indices"],
        )
        feature_dims = self.backbone.feature_info.channels()
        # print(f"feature dims: {feature_dims}")
        self.global_pools = torch.nn.ModuleList(
            [GeM(p=3, requires_grad=False) for _ in cfg["out_indices"]]
        )
        self.mid_features = np.sum(feature_dims)
        self.neck = torch.nn.BatchNorm1d(self.mid_features)
        self.head_id = ArcMarginProductSubcenter(self.mid_features, cfg["num_classes"], cfg["n_center_id"])
        if id_class_nums is not None:
            margins_id = np.power(id_class_nums, cfg["margin_power_id"]) * cfg["margin_coef_id"] + cfg["margin_cons_id"]
            self.margin_fn_id = ArcFaceLossAdaptiveMargin(margins_id, cfg["num_classes"], cfg["s_id"])
            self.loss_fn_id = torch.nn.CrossEntropyLoss()

    def get_feat(self, x: torch.Tensor) -> torch.Tensor:
        ms = self.backbone(x)
        h = torch.cat([global_pool(m) for m, global_pool in zip(ms, self.global_pools)], dim=1)
        return self.neck(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.get_feat(x)
        return self.head_id(feat)

    def configure_optimizers(self):
        backbone_params = list(self.backbone.parameters()) + list(self.global_pools.parameters())
        head_params = (list(self.neck.parameters()) + list(self.head_id.parameters()))
        params = [
            {"params": backbone_params, "lr": cfg["lr_backbone"]},
            {"params": head_params, "lr": cfg["lr_head"]},
        ]
        if cfg["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(params)
        elif cfg["optimizer"] == "AdamW":
            optimizer = torch.optim.AdamW(params)

        warmup_steps = cfg["max_epochs"] * cfg["warmup_steps_ratio"]
        cycle_steps = cfg["max_epochs"] - warmup_steps
        lr_lambda = WarmupCosineLambda(warmup_steps, cycle_steps, cfg["lr_decay_scale"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [scheduler] 

    def test_step(self, batch, batch_idx):
        x = batch["image"]
        feat1 = self.get_feat(x)
        feat2 = self.get_feat(x.flip(3))
        batch_embeddings_df = pd.DataFrame(feat1.detach().cpu().numpy(), index=batch["image_id"]) 
        embeddings.append(batch_embeddings_df)

        batch_embeddings_df = pd.DataFrame(feat2.detach().cpu().numpy(), index=batch["image_id"]) 
        flip_embeddings.append(batch_embeddings_df)


def main():
    logger.info("Starting main script")
    # load test set data and pretrained model
    query_scenarios = pd.read_csv(DATA_DIRECTORY / "query_scenarios.csv", index_col="scenario_id")   
    metadata = pd.read_csv(DATA_DIRECTORY / "metadata.csv", index_col="image_id")               
    logger.info("Loading pre-trained model")
    
    # we'll only precompute embeddings for the images in the scenario files (rather than all images), so that the
    # benchmark example can run quickly when doing local testing. this subsetting step is not necessary for an actual
    # code submission since all the images in the test environment metadata also belong to a query or database.
    scenario_imgs = []
    for row in query_scenarios.itertuples():
        scenario_imgs.extend(pd.read_csv(DATA_DIRECTORY / row.queries_path).query_image_id.values)     
        scenario_imgs.extend(pd.read_csv(DATA_DIRECTORY / row.database_path).database_image_id.values) 
    scenario_imgs = sorted(set(scenario_imgs))
    metadata = metadata.loc[scenario_imgs]

    models = []
    data_module = WhaleDataModule(metadata)
    trainer = Trainer(gpus=1, precision=16)

    cfg["image_size"] = (380,380)
    cfg["model_name"] = "tf_efficientnet_b2_ns"
    models.append(SphereClassifier.load_from_checkpoint(checkpoint_path="b2_380.ckpt"))

    cfg["image_size"] = (380,380)
    cfg["model_name"] = "tf_efficientnet_b3_ns"
    models.append(SphereClassifier.load_from_checkpoint(checkpoint_path="b3_380.ckpt"))

    cfg["image_size"] = (380,380)
    cfg["model_name"] = "tf_efficientnet_b4_ns"
    models.append(SphereClassifier.load_from_checkpoint(checkpoint_path="b4_380.ckpt"))

    cfg["image_size"] = (456,456)
    cfg["model_name"] = "tf_efficientnet_b4_ns"
    models.append(SphereClassifier.load_from_checkpoint(checkpoint_path="b4_456.ckpt"))

    cfg["image_size"] = (456,456)
    cfg["model_name"] = "tf_efficientnet_b5_ns"
    models.append(SphereClassifier.load_from_checkpoint(checkpoint_path="b5_456.ckpt"))
 
    cfg["image_size"] = (528,528)
    cfg["model_name"] = "tf_efficientnet_b5_ns"
    models.append(SphereClassifier.load_from_checkpoint(checkpoint_path="b5_528.ckpt"))

    cfg["image_size"] = (380,380)
    cfg["model_name"] = "tf_efficientnetv2_m_in21ft1k"
    models.append(SphereClassifier.load_from_checkpoint(checkpoint_path="v2_380.ckpt"))
    
    all_embeddings = []
    for model in models:
        logger.info("Precomputing embeddings")
        trainer.test(model, data_module.all_dataloader())
        all_embeddings.append(pd.concat(embeddings))
        all_embeddings.append(pd.concat(flip_embeddings))
        embeddings.clear()
        flip_embeddings.clear()

    all_embeddings = pd.concat(all_embeddings, axis=1, ignore_index=False)
    
    logger.info("Generating image rankings")
    # process all scenarios
    results = []
    for row in query_scenarios.itertuples():
        # load query df and database images; subset embeddings to this scenario's database
        qry_df = pd.read_csv(DATA_DIRECTORY / row.queries_path)                                 
        db_img_ids = pd.read_csv(DATA_DIRECTORY / row.database_path).database_image_id.values   
        db_embeddings = all_embeddings.loc[db_img_ids]

        # predict matches for each query in this scenario
        for qry in qry_df.itertuples():
            # get embeddings; drop query from database, if it exists
            qry_embedding = all_embeddings.loc[[qry.query_image_id]]
            _db_embeddings = db_embeddings.drop(qry.query_image_id, errors='ignore')

            # compute cosine similarities and get top 20
            sims = cosine_similarity(qry_embedding, _db_embeddings)[0]
            top20 = pd.Series(sims, index=_db_embeddings.index).sort_values(0, ascending=False).head(20)

            # append result
            qry_result = pd.DataFrame(
                {"query_id": qry.query_id, "database_image_id": top20.index, "score": top20.values}
            )
            results.append(qry_result)

    logger.info(f"Writing predictions file to {PREDICTION_FILE}")
    submission = pd.concat(results)
    submission.to_csv(PREDICTION_FILE, index=False)

if __name__ == "__main__":
    main()

