import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer


"""================================================================================================="""
ALLOWED_MINING_OPS  = list(batchminer.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = True

### MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
class Criterion(torch.nn.Module):
    def __init__(self, opt):
        """
        Args:
            margin:             Triplet Margin.
            nu:                 Regularisation Parameter for beta values if they are learned.
            beta:               Class-Margin values.
            n_classes:          Number of different classes during training.
        """
        super().__init__()

        ####
        self.embed_dim  = opt.embed_dim
        self.proj_dim   = opt.diva_decorrnet_dim

        self.directions = opt.diva_decorrelations
        self.weights    = opt.diva_rho_decorrelation

        self.name       = 'adversarial_separation'

        #Projection network
        self.regressors = nn.ModuleDict()
        for direction in self.directions:
            self.regressors[direction] = torch.nn.Sequential(torch.nn.Linear(self.embed_dim, self.proj_dim), torch.nn.ReLU(), torch.nn.Linear(self.proj_dim, self.embed_dim)).to(torch.float).to(opt.device)

        #Learning Rate for Projection Network
        self.lr        = opt.diva_decorrnet_lr


        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM


        

    def forward(self, feature_dict):
        #Apply gradient reversal on input embeddings.
        adj_feature_dict = {key:torch.nn.functional.normalize(grad_reverse(features),dim=-1) for key, features in feature_dict.items()}
        #Project one embedding to the space of the other (with normalization), then compute the correlation.
        sim_loss = 0
        for weight, direction in zip(self.weights, self.directions):
            source, target = direction.split('-')
            sim_loss += -1.*weight*torch.mean(torch.mean((adj_feature_dict[target]*torch.nn.functional.normalize(self.regressors[direction](adj_feature_dict[source]),dim=-1))**2,dim=-1))
        return sim_loss



### Gradient Reversal Layer
class GradRev(torch.autograd.Function):
    """
    Implements an autograd class to flip gradients during backward pass.
    """
    def forward(self, x):
        """
        Container which applies a simple identity function.

        Input:
            x: any torch tensor input.
        """
        return x.view_as(x)

    def backward(self, grad_output):
        """
        Container to reverse gradient signal during backward pass.

        Input:
            grad_output: any computed gradient.
        """
        return (grad_output * -1.)

### Gradient reverse function
def grad_reverse(x):
    """
    Applies gradient reversal on input.

    Input:
        x: any torch tensor input.
    """
    return GradRev()(x)
