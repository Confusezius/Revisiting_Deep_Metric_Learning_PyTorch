import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


"""================================================================================================="""
ALLOWED_MINING_OPS = ['barlow']
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = False

class Criterion(torch.nn.Module):
    def __init__(self, opt):
        """
        Args:
        """
        super(Criterion, self).__init__()
        self.pars = opt

        self.name           = 'barlow'

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM


    def forward(self, batch, labels, **kwargs):
        assert torch.all(labels[0::2] == labels[1::2]).item()
        batch_norm = batch.sub(batch.mean(dim=0)).div(batch.std(dim=0))
        cc = batch_norm[0::2].T @ batch_norm[1::2]
        inv_term = torch.square(torch.diag(cc) - 1.0).sum()
        cc.fill_diagonal_(0.0)
        red_term = cc.square().sum()
        loss = inv_term + self.pars.loss_barlow_lambda * red_term

        return loss
