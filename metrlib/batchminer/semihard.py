import numpy as np, torch


class BatchMiner():
    def __init__(self, opt):
        self.par          = opt
        self.name         = 'semihard'
        self.margin       = vars(opt)['loss_'+opt.loss+'_margin']

    def __call__(self, batch, labels, return_distances=False):
        if isinstance(labels, torch.Tensor): labels = labels.detach().numpy()
        bs = batch.size(0)
        #Return distance matrix for all elements in batch (BSxBS)
        distances = self.pdist(batch.detach()).detach().cpu().numpy()

        positives, negatives = [], []
        anchors = []
        for i in range(bs):
            l, d = labels[i], distances[i]
            neg = labels!=l; pos = labels==l

            anchors.append(i)
            pos[i] = 0
            p      = np.random.choice(np.where(pos)[0])
            positives.append(p)

            #Find negatives that violate tripet constraint semi-negatives
            neg_mask = np.logical_and(neg,d>d[p])
            neg_mask = np.logical_and(neg_mask,d<self.margin+d[p])
            if neg_mask.sum()>0:
                negatives.append(np.random.choice(np.where(neg_mask)[0]))
            else:
                negatives.append(np.random.choice(np.where(neg)[0]))

        sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]

        if return_distances:
            return sampled_triplets, distances
        else:
            return sampled_triplets


    def pdist(self, A):
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min = 0)
        return res.clamp(min = 0).sqrt()
