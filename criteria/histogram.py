import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer

"""================================================================================================="""
ALLOWED_MINING_OPS  = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = False


#NOTE: This implementation follows: https://github.com/valerystrizh/pytorch-histogram-loss
class Criterion(torch.nn.Module):
    def __init__(self, opt):
        """
        Args:
            margin:             Triplet Margin.
        """
        super(Criterion, self).__init__()
        self.par       = opt

        self.nbins     = opt.loss_histogram_nbins
        self.bin_width = 2/(self.nbins - 1)

        # We require a numpy and torch support as parts of the computation require numpy.
        self.support        = np.linspace(-1,1,self.nbins).reshape(-1,1)
        self.support_torch  = torch.linspace(-1,1,self.nbins).reshape(-1,1).to(opt.device)

        self.name           = 'histogram'

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM


    def forward(self, batch, labels, **kwargs):
        #The original paper utilizes similarities instead of distances.
        similarity = batch.mm(batch.T)

        bs         = labels.size()[0]

        ### We create a equality matrix for labels occuring in the batch
        label_eqs = (labels.repeat(bs, 1)  == labels.view(-1, 1).repeat(1, bs))

        ### Because the similarity matrix is symmetric, we will only utilise the upper triangular.
        ### These values are indexed by sim_inds
        sim_inds = torch.triu(torch.ones(similarity.size()), 1).bool().to(self.par.device)

        ### For the upper triangular similarity matrix, we want to know where our positives/anchors and negatives are:
        pos_inds = label_eqs[sim_inds].repeat(self.nbins, 1)
        neg_inds = ~label_eqs[sim_inds].repeat(self.nbins, 1)

        ###
        n_pos = pos_inds[0].sum()
        n_neg = neg_inds[0].sum()

        ### Extract upper triangular from the similarity matrix. (produces a one-dim vector)
        unique_sim = similarity[sim_inds].view(1, -1)

        ### We broadcast this vector to each histogram bin. Each bin entry requires a different summation in self.histogram()
        unique_sim_rep = unique_sim.repeat(self.nbins, 1)

        ### This assigns bin-values for float-similarities. The conversion to numpy is important to avoid rounding errors in torch.
        assigned_bin_values = ((unique_sim_rep.detach().cpu().numpy() + 1) / self.bin_width).astype(int) * self.bin_width - 1

        ### We now compute the histogram over distances
        hist_pos_sim = self.histogram(unique_sim_rep, assigned_bin_values, pos_inds, n_pos)
        hist_neg_sim = self.histogram(unique_sim_rep, assigned_bin_values, neg_inds, n_neg)

        ### Compute the CDF for the positive similarity histogram
        hist_pos_rep  = hist_pos_sim.view(-1, 1).repeat(1, hist_pos_sim.size()[0])
        hist_pos_inds = torch.tril(torch.ones(hist_pos_rep.size()), -1).bool()
        hist_pos_rep[hist_pos_inds] = 0
        hist_pos_cdf  = hist_pos_rep.sum(0)

        loss = torch.sum(hist_neg_sim * hist_pos_cdf)

        return loss


    def histogram(self, unique_sim_rep, assigned_bin_values, idxs, n_elem):
        """
        Compute the histogram over similarities.
        Args:
            unique_sim_rep:      torch tensor of shape nbins x n_unique_neg_similarities.
            assigned_bin_values: Bin value for each similarity value in unique_sim_rep.
            idxs:                positive/negative entry indices in unique_sim_rep
            n_elem:              number of elements in unique_sim_rep.
        """
        # Cloning is required because we change the similarity matrix in-place, but need it for the
        # positive AND negative histogram. Note that clone() allows for backprop.
        usr = unique_sim_rep.clone()
        # For each bin (and its lower neighbour bin) we find the distance values that belong.
        indsa = torch.tensor((assigned_bin_values==(self.support-self.bin_width) ) & idxs.detach().cpu().numpy())
        indsb = torch.tensor((assigned_bin_values==self.support) & idxs.detach().cpu().numpy())
        # Set all irrelevant similarities to 0
        usr[~(indsb|indsa)]=0
        #
        usr[indsa] = (usr  - self.support_torch + self.bin_width)[indsa] / self.bin_width
        usr[indsb] = (-usr + self.support_torch + self.bin_width)[indsb] / self.bin_width

        return usr.sum(1)/n_elem
