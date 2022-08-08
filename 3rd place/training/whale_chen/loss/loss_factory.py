import torch.nn as nn
from .cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, LabelSmoothingBinaryCrossEntropy
from .sce_loss import SCELoss
from .cosine_loss import CosineLoss, FocalCosineLoss
from .bi_tempered_loss import BiTemperedLogisticLoss
#from .triplet_loss import TripletLoss
from .cb_loss import CBLoss, ClassBalancedLabelSmoothingCrossEntropy
from .taylor_loss import TaylorCrossEntropyLoss, TaylorLabelSmoothingCrossEntropy


def create_criterion(CFG,logger=None):
    criterion = None
    if CFG.criterion == 'LabelSmoothingCrossEntropy':  #### label smoothing cross entropy
        criterion = LabelSmoothingCrossEntropy(smoothing=CFG.label_smoothing)
        logger.info(f'train with LabelSmoothingCrossEntropy, params label_smoothing={CFG.label_smoothing}')
    elif CFG.criterion == 'FocalCosineLoss': ### focal cosine loss
        criterion = FocalCosineLoss(
            alpha=CFG.fcl_alpha,
            gamma=CFG.fcl_gamma,
            xent=CFG.fcl_xent)
        logger.info(f'train with FocalCosineLoss, params alpha={CFG.fcl_alpha}, gamma={CFG.fcl_gamma}, xent={CFG.fcl_xent}')
    elif CFG.criterion == 'BiTemperedLogisticLoss': ### bi-tempered logistic loss
        criterion = BiTemperedLogisticLoss(
            t1=CFG.t1, 
            t2=CFG.t2, 
            label_smoothing=CFG.label_smoothing)
        logger.info(f'train with BiTemperedLogisticLoss, params t1={CFG.t1}, t2={CFG.t2}, label_smoothing={CFG.label_smoothing}')
    elif CFG.criterion == 'CBLoss':
        criterion = CBLoss(
            CFG.cb_samples_per_cls, 
            CFG.num_classes, 
            beta=CFG.cb_beta,
            gamma=CFG.cb_gamma,
            loss_type = CFG.cb_loss_type)
        logger.info(f'train with CBLoss,params beta={CFG.cb_beta}, gamma={CFG.cb_gamma}, loss_type={CFG.cb_loss_type}') 
    elif CFG.criterion == 'ClassBalancedLabelSmoothingCrossEntropy':
        criterion = ClassBalancedLabelSmoothingCrossEntropy(
                    CFG.cb_samples_per_cls, 
                    CFG.num_classes, 
                    beta=CFG.cb_beta,
                    gamma=CFG.cb_gamma,
                    loss_type = CFG.cb_loss_type,
                    smoothing=CFG.label_smoothing)
        logger.info(f'train with ClassBalancedLabelSmoothingCrossEntropy,params beta={CFG.cb_beta}, gamma={CFG.cb_gamma}, loss_type={CFG.cb_loss_type}, smoothing={CFG.label_smoothing}') 
    elif CFG.criterion == 'TaylorLabelSmoothingCrossEntropy':
        criterion = TaylorLabelSmoothingCrossEntropy(
            n=CFG.taylor_n, 
            smoothing=CFG.label_smoothing)
        logger.info(f'train with TaylorLabelSmoothingCrossEntropy, params n={CFG.taylor_n}, label_smoothing={CFG.label_smoothing}')
    elif CFG.criterion == 'LabelSmoothingBinaryCrossEntropy':
        criterion = LabelSmoothingBinaryCrossEntropy(label_smoothing=CFG.label_smoothing)
        logger.info(f'train with LabelSmoothingBinaryCrossEntropy, params label_smoothing={CFG.label_smoothing}.')
    elif CFG.criterion == 'BinaryCrossEntropy':
        criterion = nn.BCEWithLogitsLoss()
        logger.info(f'train with BCEWithLogitsLoss.')
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info(f'train with CrossEntropy.')
    
    return criterion