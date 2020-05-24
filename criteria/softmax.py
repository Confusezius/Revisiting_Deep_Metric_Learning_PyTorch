import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer

"""================================================================================================="""
ALLOWED_MINING_OPS  = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = True

### This Implementation follows: https://github.com/azgo14/classification_metric_learning

class Criterion(torch.nn.Module):
    def __init__(self, opt):
        super(Criterion, self).__init__()
        self.par         = opt

        self.temperature = opt.loss_softmax_temperature

        self.class_map = torch.nn.Parameter(torch.Tensor(opt.n_classes, opt.embed_dim))
        stdv = 1. / np.sqrt(self.class_map.size(1))
        self.class_map.data.uniform_(-stdv, stdv)

        self.name           = 'softmax'

        self.lr = opt.loss_softmax_lr

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM


    def forward(self, batch, labels, **kwargs):
        class_mapped_batch = torch.nn.functional.linear(batch, torch.nn.functional.normalize(self.class_map, dim=1))

        loss = torch.nn.CrossEntropyLoss()(class_mapped_batch/self.temperature, labels.to(torch.long).to(self.par.device))

        return loss
