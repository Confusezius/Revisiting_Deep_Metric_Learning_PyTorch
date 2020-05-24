import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer
from tqdm import tqdm


"""================================================================================================="""
ALLOWED_MINING_OPS  = list(batchminer.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM       = True
REQUIRES_EMA_NETWORK = True

### MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
class Criterion(torch.nn.Module):
    def __init__(self, opt):
        super(Criterion, self).__init__()
        self.classifier    = torch.nn.Linear(opt.network_feature_dim, 4, bias=False).to(opt.device)
        self.lr            = opt.lr * 10
        self.name          = 'imrot'

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM

    def forward(self, feature_batch, imrot_labels, **kwargs):
        pred_batch    = self.classifier(feature_batch)
        loss          = torch.nn.CrossEntropyLoss()(pred_batch, imrot_labels.to(pred_batch.device))
        return loss
