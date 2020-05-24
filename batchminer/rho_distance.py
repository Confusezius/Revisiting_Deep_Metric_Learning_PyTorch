import numpy as np, torch


class BatchMiner():
    def __init__(self, opt):
        self.par          = opt
        self.lower_cutoff  = opt.miner_rho_distance_lower_cutoff
        self.upper_cutoff  = opt.miner_rho_distance_upper_cutoff
        self.contrastive_p = opt.miner_rho_distance_cp

        self.name         = 'rho_distance'

    def __call__(self, batch, labels, return_distances=False):
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
        bs = batch.shape[0]
        distances    = self.pdist(batch.detach()).clamp(min=self.lower_cutoff)

        positives, negatives = [],[]
        labels_visited = []
        anchors = []

        for i in range(bs):
            neg = labels!=labels[i]; pos = labels==labels[i]

            use_contr = np.random.choice(2, p=[1-self.contrastive_p, self.contrastive_p])
            if np.sum(pos)>1:
                anchors.append(i)
                if use_contr:
                    positives.append(i)
                    #Sample negatives by distance
                    pos[i] = 0
                    negatives.append(np.random.choice(np.where(pos)[0]))
                else:
                    q_d_inv = self.inverse_sphere_distances(batch, distances[i], labels, labels[i])
                    #Sample positives randomly
                    pos[i] = 0
                    positives.append(np.random.choice(np.where(pos)[0]))
                    #Sample negatives by distance
                    negatives.append(np.random.choice(bs,p=q_d_inv))

        sampled_triplets   = [[a,p,n] for a,p,n in zip(anchors, positives, negatives)]
        self.push_triplets = np.sum([m[1]==m[2] for m in labels[sampled_triplets]])

        if return_distances:
            return sampled_triplets, distances
        else:
            return sampled_triplets


    def inverse_sphere_distances(self, batch, anchor_to_all_dists, labels, anchor_label):
            dists        = anchor_to_all_dists
            bs,dim       = len(dists),batch.shape[-1]

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


    def pdist(self, A, eps=1e-4):
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min = 0)
        return res.clamp(min = eps).sqrt()
