import numpy as np, torch


class BatchMiner():
    def __init__(self, opt):
        self.par          = opt
        self.mode         = opt.miner_parametric_mode
        self.n_support    = opt.miner_parametric_n_support
        self.support_lim  = opt.miner_parametric_support_lim
        self.name         = 'parametric'

        ###
        self.set_sample_distr()



    def __call__(self, batch, labels):
        bs           = batch.shape[0]
        sample_distr = self.sample_distr

        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()

        ###
        distances = self.pdist(batch.detach())

        p_assigns           = np.sum((distances.cpu().numpy().reshape(-1)>self.support[1:-1].reshape(-1,1)).T,axis=1).reshape(distances.shape)
        outside_support_lim = (distances.cpu().numpy().reshape(-1)<self.support_lim[0]) * (distances.cpu().numpy().reshape(-1)>self.support_lim[1])
        outside_support_lim = outside_support_lim.reshape(distances.shape)

        sample_ps                      = sample_distr[p_assigns]
        sample_ps[outside_support_lim] = 0

        ###
        anchors, labels_visited = [], []
        positives, negatives = [],[]

        ###
        for i in range(bs):
            neg = labels!=labels[i]; pos = labels==labels[i]

            if np.sum(pos)>1:
                anchors.append(i)

                #Sample positives randomly
                pos[i] = 0
                positives.append(np.random.choice(np.where(pos)[0]))

                #Sample negatives by distance
                sample_p = sample_ps[i][neg]
                sample_p = sample_p/sample_p.sum()
                negatives.append(np.random.choice(np.arange(bs)[neg],p=sample_p))

        sampled_triplets = [[a,p,n] for a,p,n in zip(anchors, positives, negatives)]
        return sampled_triplets



    def pdist(self, A, eps=1e-4):
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min = 0)
        return res.clamp(min = eps).sqrt()


    def set_sample_distr(self):
        self.support = np.linspace(self.support_lim[0], self.support_lim[1], self.n_support)

        if self.mode == 'uniform':
            self.sample_distr = np.array([1.] * (self.n_support-1))

        if self.mode == 'hards':
            self.sample_distr = self.support.copy()
            self.sample_distr[self.support<=0.5] = 1
            self.sample_distr[self.support>0.5]  = 0

        if self.mode == 'semihards':
            self.sample_distr = self.support.copy()
            from IPython import embed; embed()
            self.sample_distr[(self.support<=0.7) * (self.support>=0.3)] = 1
            self.sample_distr[(self.support<0.3)  * (self.support>0.7)]  = 0

        if self.mode == 'veryhards':
            self.sample_distr = self.support.copy()
            self.sample_distr[self.support<=0.3] = 1
            self.sample_distr[self.support>0.3]  = 0

        self.sample_distr = np.clip(self.sample_distr, 1e-15, 1)
        self.sample_distr = self.sample_distr/self.sample_distr.sum()
