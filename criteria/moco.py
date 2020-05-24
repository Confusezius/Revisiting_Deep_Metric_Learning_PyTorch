import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer
from tqdm import tqdm


"""================================================================================================="""
ALLOWED_MINING_OPS  = list(batchminer.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = True
REQUIRES_EMA_NETWORK = True


class Criterion(torch.nn.Module):
    def __init__(self, opt):
        super(Criterion, self).__init__()

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM

        ####
        self.temperature   = opt.diva_moco_temperature
        if opt.diva_moco_trainable_temp:
            self.temperature = torch.nn.Parameter(torch.tensor(self.temperature).to(torch.float))

        self.lr          = opt.diva_moco_temp_lr
        self.momentum      = opt.diva_moco_momentum
        self.n_key_batches = opt.diva_moco_n_key_batches

        self.name          = 'moco'
        self.reference_labels = torch.zeros(opt.bs).to(torch.long).to(opt.device)

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM


    def update_memory_queue(self, embeddings):
        self.memory_queue = self.memory_queue[len(embeddings):,:]
        self.memory_queue = torch.cat([self.memory_queue, embeddings], dim=0)

    def create_memory_queue(self, model, dataloader, device, opt_key=None):
        with torch.no_grad():
            _ = model.eval()
            _ = model.to(device)

            self.memory_queue = []
            counter = 0
            load_count  = 0
            total_count = self.n_key_batches//len(dataloader) + int(self.n_key_batches%len(dataloader)!=0)
            while counter<self.n_key_batches-1:
                load_count += 1
                for i,input_tuple in enumerate(tqdm(dataloader, 'Filling memory queue [{}/{}]...'.format(load_count, total_count), total=len(dataloader))):
                    embed = model(input_tuple[1].type(torch.FloatTensor).to(device))
                    if isinstance(embed, tuple): embed = embed[0]

                    if opt_key is not None:
                        embed = embed[opt_key].cpu()
                    else:
                        embed = embed.cpu()

                    self.memory_queue.append(embed)

                    counter+=1
                    if counter>=self.n_key_batches:
                        break

            self.memory_queue = torch.cat(self.memory_queue, dim=0).to(device)

        self.n_keys = len(self.memory_queue)

    def shuffleBN(self, bs):
        forward_inds  = torch.randperm(bs).long().cuda()
        backward_inds = torch.zeros(bs).long().cuda()
        value = torch.arange(bs).long().cuda()
        backward_inds.index_copy_(0, forward_inds, value)
        return forward_inds, backward_inds


    def forward(self, query_batch, key_batch, **kwargs):
        bs  = len(query_batch)

        l_pos = query_batch.view(bs, 1, -1).bmm(key_batch.view(bs, -1, 1)).squeeze(-1)
        l_neg = query_batch.view(bs, -1).mm(self.memory_queue.T)

        ### INCLUDE SHUFFLE BN
        logits = torch.cat([l_pos, l_neg], dim=1)

        if isinstance(self.temperature, torch.Tensor):
            loss = torch.nn.CrossEntropyLoss()(logits/self.temperature.clamp(min=1e-8, max=1e4), self.reference_labels)
        else:
            loss = torch.nn.CrossEntropyLoss()(logits/self.temperature, self.reference_labels)

        return loss
