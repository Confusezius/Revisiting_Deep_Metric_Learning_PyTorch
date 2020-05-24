import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
import random
from scipy import linalg
from scipy.stats import multivariate_normal

"""======================================================"""
REQUIRES_STORAGE = True

###
class Sampler(torch.utils.data.sampler.Sampler):
    """
    Plugs into PyTorch Batchsampler Package.
    """
    def __init__(self, opt, image_dict, image_list):
        self.image_dict = image_dict
        self.image_list = image_list

        self.batch_size         = opt.bs
        self.samples_per_class  = opt.samples_per_class
        self.sampler_length     = len(image_list)//opt.bs
        assert self.batch_size%self.samples_per_class==0, '#Samples per class must divide batchsize!'

        self.name             = 'greedy_coreset_sampler'
        self.requires_storage = True

        self.bigbs           = opt.data_batchmatch_bigbs
        self.update_storage  = not opt.data_storage_no_update
        self.num_batch_comps = opt.data_batchmatch_ncomps

        self.low_proj_dim    = opt.data_sampler_lowproj_dim

        self.lam             = opt.data_d2_coreset_lambda

        self.n_jobs = 16

    def __iter__(self):
        for i in range(self.sampler_length):
            yield self.epoch_indices[i]


    def precompute_indices(self):
        from joblib import Parallel, delayed
        import time

        ### Random Subset from Random classes
        bigb_idxs = np.random.choice(len(self.storage), self.bigbs, replace=True)
        bigbatch  = self.storage[bigb_idxs]

        print('Precomputing Indices... ', end='')
        start = time.time()
        def batchfinder(n_calls, pos):
            idx_sets         = self.d2_coreset(n_calls, pos)
            structured_batches = [list(bigb_idxs[idx_set]) for idx_set in idx_sets]
            # structured_batch = list(bigb_idxs[self.fid_match(bigbatch, batch_size=self.batch_size//self.samples_per_class)])
            #Add random per-class fillers to ensure that the batch is build up correctly.
            for i in range(len(structured_batches)):
                class_idxs = [self.image_list[idx][-1] for idx in structured_batches[i]]
                for class_idx in class_idxs:
                    structured_batches[i].extend([random.choice(self.image_dict[class_idx])[-1] for _ in range(self.samples_per_class-1)])

            return structured_batches

        n_calls            = int(np.ceil(self.sampler_length/self.n_jobs))
        # self.epoch_indices = batchfinder(n_calls, 0)
        self.epoch_indices = Parallel(n_jobs = self.n_jobs)(delayed(batchfinder)(n_calls, i) for i in range(self.n_jobs))
        self.epoch_indices = [x for y in self.epoch_indices for x in y]
        # self.epoch_indices = Parallel(n_jobs = self.n_jobs)(delayed(batchfinder)(self.storage[np.random.choice(len(self.storage), self.bigbs, replace=True)]) for _ in tqdm(range(self.sampler_length), desc='Precomputing Indices...'))

        print('Done in {0:3.4f}s.'.format(time.time()-start))
    def replace_storage_entries(self, embeddings, indices):
        self.storage[indices] = embeddings

    def create_storage(self, dataloader, model, device):
        with torch.no_grad():
            _ = model.eval()
            _ = model.to(device)

            embed_collect = []
            for i,input_tuple in enumerate(tqdm(dataloader, 'Creating data storage...')):
                embed = model(input_tuple[1].type(torch.FloatTensor).to(device))
                if isinstance(embed, tuple): embed = embed[0]
                embed = embed.cpu()
                embed_collect.append(embed)
            embed_collect = torch.cat(embed_collect, dim=0)
            self.storage = embed_collect


    def d2_coreset(self, calls, pos):
        """
        """
        coll = []

        for _ in range(calls):
            bigbatch   = self.storage[np.random.choice(len(self.storage), self.bigbs, replace=False)]
            batch_size = self.batch_size//self.samples_per_class

            if self.low_proj_dim>0:
                low_dim_proj = nn.Linear(bigbatch.shape[-1],self.low_proj_dim,bias=False)
                with torch.no_grad(): bigbatch = low_dim_proj(bigbatch)

            bigbatch = bigbatch.numpy()
            # emp_mean, emp_std = np.mean(bigbatch, axis=0), np.std(bigbatch, axis=0)
            emp_mean, emp_cov = np.mean(bigbatch, axis=0), np.cov(bigbatch.T)

            prod        = np.matmul(bigbatch, bigbatch.T)
            sq          = prod.diagonal().reshape(bigbatch.shape[0], 1)
            dist_matrix = np.clip(-2*prod + sq + sq.T, 0, None)

            start_anchor = np.random.multivariate_normal(emp_mean, emp_cov, 1).reshape(-1)
            start_dists  = np.linalg.norm(bigbatch-start_anchor,axis=1)
            start_point  = np.argmin(start_dists, axis=0)

            idxs = list(range(len(bigbatch)))
            del idxs[start_point]

            k, sampled_indices = 1, [start_point]
            dist_weights = dist_matrix[:,start_point]

            normal_weights = multivariate_normal.pdf(bigbatch,emp_mean,emp_cov)
            while k<batch_size:
                normal_weights_to_use = normal_weights[idxs]/normal_weights[idxs].sum()
                dim = bigbatch.shape[-1]

                sampling_p = normal_weights_to_use*dist_weights[idxs]**self.lam
                sampling_p/= np.sum(sampling_p)

                dm_idx = np.random.choice(range(len(dist_matrix)-k),p=sampling_p.reshape(-1))
                sample = idxs[dm_idx]

                del idxs[dm_idx]

                sampled_indices.append(sample)

                dist_weights = dist_weights + dist_matrix[:,sample]
                k += 1

            coll.append(sampled_indices)

        return coll


    def __len__(self):
        return self.sampler_length
