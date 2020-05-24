import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer


class BatchMiner():
    def __init__(self, opt):
        self.par          = opt
        self.lower_cutoff = opt.miner_distance_lower_cutoff
        self.upper_cutoff = opt.miner_distance_upper_cutoff
        self.name         = 'distance'

    def __call__(self, batch, labels, tar_labels=None, return_distances=False, distances=None):
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
        bs, dim = batch.shape

        if distances is None:
            distances = self.pdist(batch.detach()).clamp(min=self.lower_cutoff)
        sel_d = distances.shape[-1]

        positives, negatives = [],[]
        labels_visited       = []
        anchors              = []

        tar_labels = labels if tar_labels is None else tar_labels

        for i in range(bs):
            neg = tar_labels!=labels[i]; pos = tar_labels==labels[i]

            anchors.append(i)
            q_d_inv = self.inverse_sphere_distances(dim, bs, distances[i], tar_labels, labels[i])
            negatives.append(np.random.choice(sel_d,p=q_d_inv))

            if np.sum(pos)>0:
                #Sample positives randomly
                if np.sum(pos)>1: pos[i] = 0
                positives.append(np.random.choice(np.where(pos)[0]))
                #Sample negatives by distance

        sampled_triplets = [[a,p,n] for a,p,n in zip(anchors, positives, negatives)]

        if return_distances:
            return sampled_triplets, distances
        else:
            return sampled_triplets


    def inverse_sphere_distances(self, dim, bs, anchor_to_all_dists, labels, anchor_label):
            dists  = anchor_to_all_dists

            #negated log-distribution of distances of unit sphere in dimension <dim>
            log_q_d_inv = ((2.0 - float(dim)) * torch.log(dists) - (float(dim-3) / 2) * torch.log(1.0 - 0.25 * (dists.pow(2))))
            log_q_d_inv[np.where(labels==anchor_label)[0]] = 0

            q_d_inv     = torch.exp(log_q_d_inv - torch.max(log_q_d_inv)) # - max(log) for stability
            q_d_inv[np.where(labels==anchor_label)[0]] = 0

            ### NOTE: Cutting of values with high distances made the results slightly worse. It can also lead to
            # errors where there are no available negatives (for high samples_per_class cases).
            # q_d_inv[np.where(dists.detach().cpu().numpy()>self.upper_cutoff)[0]]    = 0

            q_d_inv = q_d_inv/q_d_inv.sum()
            return q_d_inv.detach().cpu().numpy()


    def pdist(self, A):
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min = 0)
        return res.sqrt()
