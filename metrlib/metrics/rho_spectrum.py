from scipy.spatial import distance
from sklearn.preprocessing import normalize
import numpy as np


class Metric():
    def __init__(self, embed_dim, mode,  **kwargs):
        self.mode      = mode
        self.embed_dim = embed_dim
        self.requires = ['features']
        self.name     = 'rho_spectrum@'+str(mode)

    def __call__(self, features):
        from sklearn.decomposition import TruncatedSVD
        from scipy.stats import entropy
        import torch

        if isinstance(features, torch.Tensor):
            _,s,_ = torch.svd(features)
            s     = s.cpu().numpy()
        else:
            svd = TruncatedSVD(n_components=self.embed_dim-1, n_iter=7, random_state=42)
            svd.fit(features)
            s = svd.singular_values_

        if self.mode!=0:
            s = s[np.abs(self.mode)-1:]
        s_norm  = s/np.sum(s)
        uniform = np.ones(len(s))/(len(s))

        if self.mode<0:
            kl = entropy(s_norm, uniform)
        if self.mode>0:
            kl = entropy(uniform, s_norm)
        if self.mode==0:
            kl = s_norm

        return kl
