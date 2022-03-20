import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer

"""================================================================================================="""
ALLOWED_MINING_OPS  = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = True

### This implementation follows https://github.com/idstcv/SoftTriple
class Criterion(torch.nn.Module):
    def __init__(self, opt):
        super(Criterion, self).__init__()

        ####
        self.par         = opt
        self.n_classes   = opt.n_classes

        ####
        self.n_centroids  = opt.loss_softtriplet_n_centroids
        self.margin_delta = opt.loss_softtriplet_margin_delta
        self.gamma        = opt.loss_softtriplet_gamma
        self.lam          = opt.loss_softtriplet_lambda
        self.reg_weight   = opt.loss_softtriplet_reg_weight


        ####
        self.reg_norm    = self.n_classes*self.n_centroids*(self.n_centroids-1)
        self.reg_indices = torch.zeros((self.n_classes*self.n_centroids, self.n_classes*self.n_centroids), dtype=torch.bool).to(opt.device)
        for i in range(0, self.n_classes):
            for j in range(0, self.n_centroids):
                self.reg_indices[i*self.n_centroids+j, i*self.n_centroids+j+1:(i+1)*self.n_centroids] = 1


        ####
        self.intra_class_centroids = torch.nn.Parameter(torch.Tensor(opt.embed_dim, self.n_classes*self.n_centroids))
        stdv = 1. / np.sqrt(self.intra_class_centroids.size(1))
        self.intra_class_centroids.data.uniform_(-stdv, stdv)

        self.name = 'softtriplet'

        self.lr   = opt.lr*opt.loss_softtriplet_lr

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM


    def forward(self, batch, labels, **kwargs):
        bs = batch.size(0)

        intra_class_centroids     = torch.nn.functional.normalize(self.intra_class_centroids, dim=1)
        similarities_to_centroids = batch.mm(intra_class_centroids).reshape(-1, self.n_classes, self.n_centroids)

        soft_weight_over_centroids = torch.nn.Softmax(dim=1)(self.gamma*similarities_to_centroids)
        per_class_embed            = torch.sum(soft_weight_over_centroids * similarities_to_centroids, dim=2)

        margin_delta = torch.zeros(per_class_embed.shape).to(self.par.device)
        margin_delta[torch.arange(0, bs), labels] = self.margin_delta

        centroid_classification_loss = torch.nn.CrossEntropyLoss()(self.lam*(per_class_embed-margin_delta), labels.to(torch.long).to(self.par.device))

        inter_centroid_similarity = intra_class_centroids.T.mm(intra_class_centroids)
        regularisation_loss = torch.sum(torch.sqrt(2.00001-2*inter_centroid_similarity[self.reg_indices]))/self.reg_norm

        return centroid_classification_loss + self.reg_weight * regularisation_loss
