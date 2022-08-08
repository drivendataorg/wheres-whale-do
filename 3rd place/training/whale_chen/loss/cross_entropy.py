import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1, prob=1.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.prob = prob

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        if random.random() < self.prob:
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        else:
            loss = nll_loss
        return loss.mean()

    
### copy from https://www.kaggle.com/hmendonca/melanoma-neat-pytorch-lightning-native-amp#Model
class LabelSmoothingBinaryCrossEntropy(nn.Module):
    def __init__(self, label_smoothing=0.1,pos_weight=1.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingBinaryCrossEntropy, self).__init__()
        assert label_smoothing < 1.0
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight

    def forward(self, x, y):
         # return batch loss
#         x, y  = batch
#         y_hat = self(x).flatten()
#         y_smo = y.float() * (1 - label_smoothing) + 0.5 * label_smoothing
#         loss  = F.binary_cross_entropy_with_logits(y_hat, y_smo.type_as(y_hat),
#                                                    pos_weight=torch.tensor(pos_weight))
        y_hat = x
        y_smo = y.float() * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
#         loss  = F.binary_cross_entropy_with_logits(y_hat, y_smo.type_as(y_hat),
#                                                    pos_weight=torch.tensor(self.pos_weight))
        loss  = F.binary_cross_entropy_with_logits(y_hat, y_smo.type_as(y_hat))
        
        return loss
    

class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
