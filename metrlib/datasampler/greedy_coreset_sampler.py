import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
import random
from scipy import linalg


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
        self.dist_lim        = opt.data_gc_coreset_lim

        self.low_proj_dim    = opt.data_sampler_lowproj_dim

        self.softened = opt.data_gc_softened

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
            idx_sets         = self.greedy_coreset(n_calls, pos)
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


    def full_storage_update(self, dataloader, model, device):
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
            if self.mb_mom>0:
                self.delta_storage = self.mb_mom*self.delta_storage + (1-self.mb_mom)*(embed_collect-self.storage)
                self.storage       = embed_collect + self.mb_lr*self.delta_storage
            else:
                self.storage = embed_collect
                
    def greedy_coreset(self, calls, pos):
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

            prod        = np.matmul(bigbatch, bigbatch.T)
            sq          = prod.diagonal().reshape(bigbatch.shape[0], 1)
            dist_matrix = np.clip(-2*prod + sq + sq.T, 0, None)
            coreset_anchor_dists = np.linalg.norm(dist_matrix, axis=1)

            k, sampled_indices = 0, []

            while k<batch_size:
                if k==0:
                    no    = np.random.randint(len(coreset_anchor_dists))
                else:
                    if self.softened:
                        no = np.random.choice(np.where(coreset_anchor_dists>=np.percentile(coreset_anchor_dists,97))[0])
                    else:
                        no    = np.argmax(coreset_anchor_dists)

                sampled_indices.append(no)
                add_d  = dist_matrix[:, no:no+1]
                #If its closer to the remaining points than the new addition/additions, sample it.
                new_dj = np.concatenate([np.expand_dims(coreset_anchor_dists,-1), add_d], axis=1)
                coreset_anchor_dists = np.min(new_dj, axis=1)
                k += 1

            coll.append(sampled_indices)

        return coll


    def __len__(self):
        return self.sampler_length
