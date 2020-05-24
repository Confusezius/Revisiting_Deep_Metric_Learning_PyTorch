import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer
from tqdm import tqdm


"""================================================================================================="""
ALLOWED_MINING_OPS  = list(batchminer.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = True
REQUIRES_EMA_NETWORK = False


class Criterion(torch.nn.Module):
    def __init__(self, opt):
        super(Criterion, self).__init__()

        self.temperature   = opt.diva_instdiscr_temperature
        self.name          = 'invariantspread'
        self.lr            = opt.lr
        self.reference_labels = torch.zeros(opt.bs//2).to(torch.long).to(opt.device)
        self.diag_mat         = 1 - torch.eye(opt.bs).to(opt.device)

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM
        
    def forward(self, head_1, head_2, **kwargs):
        bs = len(head_1)

        x = torch.cat([head_1, head_2], dim=0)
        reordered_x = torch.cat((x.narrow(0,bs,bs),x.narrow(0,0,bs)), 0)
        #reordered_x = reordered_x.data
        pos = (x*reordered_x.detach()).sum(1).div_(self.temperature).exp_()

        #get all innerproduct, remove diag
        all_prob = torch.mm(x,x.t().detach()).div_(self.temperature).exp_()*self.diag_mat
        all_div  = all_prob.sum(1)

        lnPmt = torch.div(pos, all_div)

        # negative probability
        Pon_div = all_div.repeat(x.shape[0],1)
        lnPon = torch.div(all_prob, Pon_div.t())
        lnPon = -lnPon.add(-1)

        # equation 7 in ref. A (NCE paper)
        lnPon.log_()
        # also remove the pos term
        lnPon = lnPon.sum(1) - (-lnPmt.add(-1)).log_()
        lnPmt.log_()
        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.sum(0)

        # negative multiply m
        loss = - (lnPmtsum + lnPonsum)/x.shape[0]



        return loss
