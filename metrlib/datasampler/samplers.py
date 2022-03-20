import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
import random


"""======================================================"""
def sampler_parse_args(parser):
    parser.add_argument('--batch_selection',     default='class_random', type=str,   help='Selection of the data batch: Modes of Selection: random, greedy_coreset')
    parser.add_argument('--primary_subset_perc', default=0.1,            type=float, help='Size of the randomly selected subset before application of coreset selection.')
    return parser



"""======================================================"""
###
# Methods: Full random, Per-Class-Random, CoreSet
class AdvancedSampler(torch.utils.data.sampler.Sampler):
    """
    Plugs into PyTorch Batchsampler Package.
    """
    def __init__(self, method='class_random', random_subset_perc=0.1, batch_size=128, samples_per_class=4):
        self.random_subset_perc = random_subset_perc
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class

        self.method         = method

        self.storage = None
        self.sampler_length = None

        self.methods_requiring_storage = ['greedy_class_coreset', 'greedy_semi_class_coreset', 'presampled_infobatch']

    def create_storage(self, dataloader, model, device):
        self.image_dict = dataloader.dataset.image_dict
        self.image_list = dataloader.dataset.image_list

        self.sampler_length = len(dataloader.dataset)//self.batch_size

        if self.method in self.methods_requiring_storage:
            with torch.no_grad():
                _ = model.eval()
                _ = model.to(device)

                embed_collect = []
                for i,input_tuple in enumerate(tqdm(dataloader, 'Creating data storage...')):
                    embed = model(input_tuple[1].type(torch.FloatTensor).to(device)).cpu()
                    embed_collect.append(embed)
                embed_collect = torch.cat(embed_collect, dim=0)
                self.storage = embed_collect

            self.random_subset_len = int(self.random_subset_perc*len(self.storage))

    def update_storage(self, embeddings, indices):
        if 'coreset' in self.method:
            self.storage[indices] = embeddings

    def __iter__(self):
        for _ in range(self.sampler_length):
            subset = []
            if self.method=='greedy_class_coreset':
                for _ in range(self.batch_size//self.samples_per_class):
                    class_key     = random.choice(list(self.image_dict.keys()))
                    class_indices = np.array([x[1] for x in self.image_dict[class_key]])
                    # print(class_indices)
                    ### Coreset subset of subset
                    subset.extend(class_indices[self.greedy_coreset(self.storage[class_indices], self.samples_per_class)])
                # print([self.image_list[x][1] for x in subset])
            elif self.method=='greedy_semi_class_coreset':
                ### Big random subset
                subset = np.random.randint(0,len(self.storage),self.random_subset_len)
                ### Coreset subset of subset of half the batch size
                subset  = subset[self.greedy_coreset(self.storage[subset], self.batch_size//2)]
                ### Fill the rest of the batch with random samples from each coreset member class
                subset = list(subset)+[random.choice(self.image_dict[self.image_list[idx][-1]])[-1] for idx in subset]
            elif self.method=='presampled_infobatch':
                ### Big random subset
                subset  = np.random.randint(0,len(self.storage),self.random_subset_len)
                classes = torch.tensor([self.image_list[idx][-1] for idx in subset])
                ### Presampled Infobatch for subset of data.
                subset = subset[self.presample_infobatch(classes, self.storage[subset], self.batch_size//2)]
                ### Fill the rest of the batch with random samples from each member class
                subset = list(subset)+[random.choice(self.image_dict[self.image_list[idx][-1]])[-1] for idx in subset]
            elif self.method=='class_random':
                ### Random Subset from Random classes
                for _ in range(self.batch_size//self.samples_per_class):
                    class_key = random.choice(list(self.image_dict.keys()))
                    subset.extend([random.choice(self.image_dict[class_key])[-1] for _ in range(self.samples_per_class)])
            elif self.method=='semi_class_random':
                ### Select half of the indices completely at random, and the other half corresponding to the classes.
                for _ in range(self.batch_size//2):
                    rand_idx       = np.random.randint(len(self.image_list))
                    class_idx      = self.image_list[rand_idx][-1]
                    rand_class_idx = random.choice(self.image_dict[class_idx])[-1]
                    subset.extend([rand_idx, rand_class_idx])
            else:
                raise NotImplementedError('Batch selection method {} not available!'.format(self.method))
            yield subset

    def __len__(self):
        return self.sampler_length

    def pdistsq(self, A):
        prod = torch.mm(A, A.t())
        diag = prod.diag().unsqueeze(1).expand_as(prod)
        return (-2*prod + diag + diag.T)

    def greedy_coreset(self, A, samples):
        dist_matrix          = self.pdistsq(A)
        coreset_anchor_dists = torch.norm(dist_matrix, dim=1)

        sampled_indices, i = [], 0

        while i<samples:
            if i==0:
                sample_idx = np.random.randint(len(coreset_anchor_dists))
            else:
                sample_idx = torch.argmax(coreset_anchor_dists).item()
            sampled_indices.append(sample_idx)
            sample_anchor_dists  = dist_matrix[:, sample_idx:sample_idx+1]
            new_search_dists     = torch.cat([coreset_anchor_dists.unsqueeze(-1), sample_anchor_dists], dim=1)
            coreset_anchor_dists = torch.min(new_search_dists, dim=1)[0]
            i += 1

        return sampled_indices

    def presample_infobatch(self, classes, A, samples):
        equiv_classes = ((classes.reshape(-1,1)-classes.reshape(-1,1).T)==0).type(torch.BoolTensor)

        dim  = A.shape[-1]

        dist = self.pdistsq(A).clamp(min=0.5)
        dist = ((2.0 - float(dim)) * torch.log(dist) - (float(dim-3) / 2) * torch.log(1.0 - 0.25 * (dist.pow(2))))
        dist[equiv_classes] = 0
        dist = torch.exp(dist - torch.max(dist))
        dist[equiv_classes] = 0

        dist = dist/torch.sum(dist)
        dist = dist.flatten().detach().cpu().numpy()

        sampled_idxs = set()
        while len(sampled_idxs)<samples:
            index = np.random.choice(len(dist),p=dist)
            ### Ensure that we do not continously sample the same one in the case of high prob. imbalances!
            dist[index] = 0
            dist = dist/np.sum(dist)
            sample_a, sample_b = index//equiv_classes.shape[0], index%equiv_classes.shape[1]
            sampled_idxs = sampled_idxs.union(set([sample_a, sample_b]))

        sampled_idxs = list(sampled_idxs)
        sampled_idxs = sampled_idxs[:samples]

        return sampled_idxs
        # dist = dist/torch.sum(dist, dim=1).view(-1,1)

        # Divergence of dist from uniform:, normalize with number if NON-ZERO ELEMENTS!
        # non_equiv_classes = (1-equiv_classes.type(torch.LongTensor)).type(torch.BoolTensor)
        # uniform_reference = torch.ones(dist.shape)/torch.sum(non_equiv_classes, dim=1).view(-1,1)
        #
        # kl_divs = []
        # for i in range(len(dist)):
        #     nec_idxs = non_equiv_classes[i,:].type(torch.BoolTensor)
        #     u        = uniform_reference[i,nec_idxs]
        #     d        = dist[i,nec_idxs]
        #     kl_div   = -((u*(d/u).log()).sum())/len(u)
        #     kl_divs.append(kl_div)
        # kl_divs      = torch.stack(kl_divs, dim=0)
        # sample_probs = (kl_divs/kl_divs.sum()).detach().cpu().numpy()


        # return np.random.choice(len(sample_probs), samples, p=sample_probs)
