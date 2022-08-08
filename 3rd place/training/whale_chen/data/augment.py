import numpy as np
import cv2
import torch
import torch.nn.functional as F
from .fmix import *

def load_image(image_path):
    image = np.load(image_path)
    image = image.astype(np.float32)
    image = np.vstack(image).transpose((1, 0))
    return image

### cutout
def cutout(image):
    h, w = image.shape[:2]
    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction

#     min_value = int(np.min(image))
#     max_value = int(np.max(image))
    # scales = [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(1)] ## for 1 channels
#         image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)] ## for 3 channels
        
    return image

### mosaic
def load_mosaic(data_dir, df, label):
    tmp_df = df[df['target'] == label]
    mosaic_df = tmp_df.sample(4)
    mosaic_image_ids = mosaic_df["id"].values
    mosaic_labels = mosaic_df["target"].values

    mosaic_border = [256, 1638]

    # yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in mosaic_border]  # mosaic center x, y
    yc, xc = [int(random.uniform(x // 2 - 64 , x // 2 + 64)) for x in mosaic_border]  # mosaic center x, y
    for i, image_id in enumerate(mosaic_image_ids):
        # load image
#         image_path = data_dir + image_id
        image_path = f'{data_dir}{image_id[0]}/{image_id}.npy'
        img = load_image(image_path)
#         print(f'{i} image shape is {img.shape}')
        h, w = img.shape
        if i == 0:  # top left
            img4 = np.full((h, w), 114, dtype=np.float32)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, w), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(h, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, w), min(h, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            
#         print(f'{i} x shape is x1a={x1a},x2a={x2a}, y1a={y1a}, y2a={y2a}') 
#         print(f'{i} x shape is x1b={x1b},x2b={x2b}, y1b={y1b}, y2b={y2b}')
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

    return img4

def rand_bbox(size, lam):
    H = size[2]
    W = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix(images, targets, alpha=1.0, prob=0.5):
    # images = torch.stack(images).cuda()
    if random.random() < prob:
        shuffle_indices = torch.randperm(images.size(0))
        indices = torch.arange(images.size(0))
        lam = np.clip(np.random.beta(alpha, alpha), 0.35, 0.65)
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)

        images[:, :, bby1:bby2, bbx1:bbx2] = images[shuffle_indices, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))

        for i, si in zip(indices,shuffle_indices):
            targets[i.item()] = lam * targets[i.item()] + (1. - lam)* targets[si.item()]

    return images, targets

def fmix(images, targets, alpha, decay_power, shape, max_soft=0.0, reformulate=False, prob=0.5):
    if random.random() < prob:
        lam, mask = sample_mask(alpha, decay_power, shape, max_soft, reformulate)

        mask_tensor = torch.from_numpy(mask).type(torch.FloatTensor).cuda()
        shuffle_indices = torch.randperm(images.size(0))
        indices = torch.arange(images.size(0))

        shuffle_images = images[shuffle_indices]
        shuffle_targets = targets[shuffle_indices]

        images = images*mask_tensor + shuffle_images * (1 - mask_tensor)
        for i, si in zip(indices,shuffle_indices):
            targets[i.item()] = lam * targets[i.item()] + (1. - lam)* targets[si.item()]
        
    return images, targets

def get_spm(input,target,cfg,model):
    imgsize = (cfg.image_size,cfg.image_size)
    bs = input.size(0)
    with torch.no_grad():
        output,fms = model(input)
        if cfg.model_name == 'resnext50d_32x4d' or cfg.model_name == 'resnet50' or cfg.model_name == 'resnet200d_320': ## resnet
#             clsw = model.model.fc   ## single gpu
            clsw = model.module.model.fc   ## multi gpus
        if cfg.model_name == 'tf_efficientnet_b0_ns' or cfg.model_name == 'tf_efficientnet_b4_ns' or cfg.model_name == 'tf_efficientnet_b5_ns': ## effnet
            clsw = model.model.classifier   ## single gpu
#             clsw = model.module.model.classifier ## multi gpus
        if cfg.model_name == 'vit_base_patch16_384' or cfg.model_name == 'vit_large_patch16_384': ## vit
            clsw = model.model.head ## single gpu
        if cfg.model_name == 'deit_base_distilled_patch16_384': ## deit
#             clsw = model.model.head ## single gpu
            clsw = model.module.model.head ## multi gpus
    
        if cfg.model_name == 'nfnet_l0': 
            clsw = model.model.head ## single gpu
#             clsw = model.module.model.head ## multi gpus
        ## 'https://discuss.pytorch.org/t/attributeerror-sequential-object-has-no-attribute-weight/53855/6'
#         print(clsw)
        
        if cfg.model_name == 'nfnet_l0': 
            weight = clsw.fc.weight.data
            bias = clsw.fc.bias.data
        else:
            weight = clsw.weight.data
            bias = clsw.bias.data

        # weight = clsw[0].weight.data
        # bias = clsw[0].bias.data

        #### dropout
        # weight = clsw[1].weight.data
        # bias = clsw[1].bias.data

        weight = weight.view(weight.size(0),weight.size(1),1,1)
        fms = F.relu(fms)

        if cfg.model_name == 'vit_base_patch16_384' or cfg.model_name == 'vit_large_patch16_384' or cfg.model_name == 'deit_base_distilled_patch16_384':
            normalized_shape = fms.size()[1:]
            poolfea = F.layer_norm(fms,normalized_shape)[:,0]
            fms = fms[:,0].unsqueeze(-1).unsqueeze(-1)
        elif cfg.model_name == 'nfnet_l0':
            poolfea = fms
        else:
            poolfea = F.adaptive_avg_pool2d(fms,(1,1)).squeeze()

        clslogit = F.softmax(clsw.forward(poolfea),dim=-1)
        logitlist = []
        for i in range(bs):
#             logitlist.append(clslogit[i,target[i]])
            logitlist.append(clslogit[i,0])

        clslogit = torch.stack(logitlist)

        out = F.conv2d(fms, weight, bias=bias)
        outmaps = []
        # print(target)
        for i in range(bs):
#             evimap = out[i,target[i]] ### multi label
            evimap = out[i,0] ### multi label
            outmaps.append(evimap)

        outmaps = torch.stack(outmaps)
        if imgsize is not None:
#             outmaps = outmaps.view(outmaps.size(0),1,outmaps.size(1),outmaps.size(2)) ## only for single label
#             outmaps = outmaps.view(outmaps.size(0),outmaps.size(2),outmaps.size(3),outmaps.size(1)) ## only for multi label

            outmaps = outmaps.view(outmaps.size(0),1,outmaps.size(1),outmaps.size(2)) ## only for single label
            outmaps = F.interpolate(outmaps,imgsize,mode='bilinear',align_corners=False)

        outmaps = outmaps.squeeze(-1)
        for i in range(bs):
            outmaps[i] -= outmaps[i].min()
            outmaps[i] /= outmaps[i].sum()

    return outmaps,clslogit

def rand_bbox(size, lam,center=False,attcen=None):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    elif len(size) == 2:
        W = size[0]
        H = size[1]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    if attcen is None:
        # uniform
        cx = 0
        cy = 0
        if W>0 and H>0:
            cx = np.random.randint(W)
            cy = np.random.randint(H)
        if center:
            cx = int(W/2)
            cy = int(H/2)
    else:
        cx = attcen[0]
        cy = attcen[1]

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def snapmix(input,target,cfg,model=None):

    target = target.long()
    r = np.random.rand(1)
    
    lam_a = torch.ones(input.size(0))
#     lam_a = torch.ones((input.size(0),target.size(1)))
    lam_b = 1 - lam_a
    target_b = target.clone()

    if r < cfg.snapmix_prob:
        wfmaps,_ = get_spm(input,target,cfg,model)
        bs = input.size(0)
        lam = np.random.beta(cfg.snapmix_beta, cfg.snapmix_beta)
        lam1 = np.random.beta(cfg.snapmix_beta, cfg.snapmix_beta)
        rand_index = torch.randperm(bs).cuda()
        wfmaps_b = wfmaps[rand_index,:,:]
        target_b = target[rand_index]

        same_label = target == target_b
        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        bbx1_1, bby1_1, bbx2_1, bby2_1 = rand_bbox(input.size(), lam1)

        area = (bby2-bby1)*(bbx2-bbx1)
        area1 = (bby2_1-bby1_1)*(bbx2_1-bbx1_1)

        if  area1 > 0 and  area>0:
            ncont = input[rand_index, :, bbx1_1:bbx2_1, bby1_1:bby2_1].clone()
            ncont = F.interpolate(ncont, size=(bbx2-bbx1,bby2-bby1), mode='bilinear', align_corners=True)
            input[:, :, bbx1:bbx2, bby1:bby2] = ncont
            lam_a = 1 - wfmaps[:,bbx1:bbx2,bby1:bby2].sum(2).sum(1)/(wfmaps.sum(2).sum(1)+1e-8)
            lam_b = wfmaps_b[:,bbx1_1:bbx2_1,bby1_1:bby2_1].sum(2).sum(1)/(wfmaps_b.sum(2).sum(1)+1e-8)
            tmp = lam_a.clone()
            lam_a[same_label] += lam_b[same_label]
            lam_b[same_label] += tmp[same_label]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            lam_a[torch.isnan(lam_a)] = lam
            lam_b[torch.isnan(lam_b)] = 1-lam

    return input,target,target_b,lam_a.cuda(),lam_b.cuda()
