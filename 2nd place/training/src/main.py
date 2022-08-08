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

cfg = { 
    "model_name": "tf_efficientnet_b2_ns",
    "batch_size": 32,
    "image_size": (380,380),
    "max_epochs": 20,
    "out_indices": (3,4),
    "n_splits": -1,
    "num_classes": 788,
    "warmup_steps_ratio": 0.2,
    "n_data": -1,
    "lr_backbone": 1.6e-3,  
    "lr_head": 1.6e-2,      
    "lr_decay_scale": 1.0e-2,
    "s_id": 21.0,              
    "margin_coef_id": 0.5,     
    "margin_power_id": -0.125, 
    "margin_cons_id": 0.05,
    "n_center_id": 2,
    
    "num_workers" : 4,
    "wandb" : False,
}

ROOT_DIRECTORY = Path("/code_execution")   
PREDICTION_FILE = ROOT_DIRECTORY / "submission" / "submission.csv"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"

embeddings = []
flip_embeddings = []

class BelugaDataset(Dataset):
    def __init__(self, metadata):
        super().__init__()
        self.metadata = metadata
        augments = []
        augments.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        augments.append(ToTensorV2()) 
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

class BelugaDataModule(LightningDataModule):
    def __init__(self,df):
        super().__init__()
        self.all_df = df
    
    def get_dataset(self, df):
        return BelugaDataset(df)

    def all_dataloader(self):
        return DataLoader(
            self.get_dataset(self.all_df),
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers = 4,
            pin_memory=True,
        )

class BelugaClassifier(LightningModule):
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
        self.global_pools = torch.nn.ModuleList([GeM(p=3, requires_grad=False) for _ in cfg["out_indices"]])
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
    
    scenario_imgs = []
    for row in query_scenarios.itertuples():
        scenario_imgs.extend(pd.read_csv(DATA_DIRECTORY / row.queries_path).query_image_id.values)     
        scenario_imgs.extend(pd.read_csv(DATA_DIRECTORY / row.database_path).database_image_id.values) 
    scenario_imgs = sorted(set(scenario_imgs))
    metadata = metadata.loc[scenario_imgs]

    models = []
    data_module = BelugaDataModule(metadata)
    trainer = Trainer(gpus=1, precision=16)

    cfg["image_size"] = (380,380)
    cfg["model_name"] = "tf_efficientnet_b2_ns"
    models.append(BelugaClassifier.load_from_checkpoint(checkpoint_path="tf_efficientnet_b2_ns_380.ckpt"))

    cfg["image_size"] = (380,380)
    cfg["model_name"] = "tf_efficientnet_b3_ns"
    models.append(BelugaClassifier.load_from_checkpoint(checkpoint_path="tf_efficientnet_b3_ns_380.ckpt"))

    cfg["image_size"] = (380,380)
    cfg["model_name"] = "tf_efficientnet_b4_ns"
    models.append(BelugaClassifier.load_from_checkpoint(checkpoint_path="tf_efficientnet_b4_ns_380.ckpt"))

    cfg["image_size"] = (456,456)
    cfg["model_name"] = "tf_efficientnet_b4_ns"
    models.append(BelugaClassifier.load_from_checkpoint(checkpoint_path="tf_efficientnet_b4_ns_456.ckpt"))

    cfg["image_size"] = (456,456)
    cfg["model_name"] = "tf_efficientnet_b5_ns"
    models.append(BelugaClassifier.load_from_checkpoint(checkpoint_path="tf_efficientnet_b5_ns_456.ckpt"))
 
    cfg["image_size"] = (528,528)
    cfg["model_name"] = "tf_efficientnet_b5_ns"
    models.append(BelugaClassifier.load_from_checkpoint(checkpoint_path="tf_efficientnet_b5_ns_528.ckpt"))

    cfg["image_size"] = (380,380)
    cfg["model_name"] = "tf_efficientnetv2_m_in21ft1k"
    models.append(BelugaClassifier.load_from_checkpoint(checkpoint_path="tf_efficientnetv2_m_in21ft1k_380.ckpt"))
    
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

