import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
import random



"""======================================================"""
REQUIRES_STORAGE = False

###
class Sampler(torch.utils.data.sampler.Sampler):
    """
    Plugs into PyTorch Batchsampler Package.
    """
    def __init__(self, opt, image_dict, image_list=None):
        self.image_dict         = image_dict
        self.image_list         = image_list

        self.batch_size         = opt.bs
        self.samples_per_class  = opt.samples_per_class
        self.sampler_length     = len(image_list)//opt.bs
        assert self.batch_size%self.samples_per_class==0, '#Samples per class must divide batchsize!'

        self.name             = 'random_sampler'
        self.requires_storage = False

    def __iter__(self):
        for _ in range(self.sampler_length):
            subset = []
            ### Random Subset from Random classes
            for _ in range(self.batch_size-1):
                class_key  = random.choice(list(self.image_dict.keys()))
                sample_idx = np.random.choice(len(self.image_dict[class_key]))
                subset.append(self.image_dict[class_key][sample_idx][-1])
            #
            subset.append(random.choice(self.image_dict[self.image_list[random.choice(subset)][-1]])[-1])
            yield subset

    def __len__(self):
        return self.sampler_length
