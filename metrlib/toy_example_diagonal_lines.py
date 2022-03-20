import os, numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn, torchvision as tv
import numpy as np
import random

"""==================================================="""
seed = 1
torch.backends.cudnn.deterministic=True; np.random.seed(seed); random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)


"""==================================================="""
ppline     = 100
n_lines    = 4
noise_perc = 0.15
intervals  = [(0.1,0.3), (0.35,0.55), (0.6,0.8), (0.85,1.05)]
lines = [np.stack([np.linspace(intv[0],intv[1],ppline), np.linspace(intv[0],intv[1],ppline)])[:,np.random.choice(ppline, int(ppline*noise_perc), replace=False)] for intv in intervals]
cls   = [x*np.ones(int(ppline*noise_perc)) for x in range(n_lines)]
train_lines = np.concatenate(lines, axis=1).T
train_cls   = np.concatenate(cls)

x_test_line1 = np.stack([0.2*np.ones(ppline), np.linspace(0.2,0.4,ppline)])[:,np.random.choice(ppline, int(ppline*noise_perc), replace=False)]
x_test_line2 = np.stack([0.2*np.ones(ppline), np.linspace(0.55,0.85,ppline)])[:,np.random.choice(ppline, int(ppline*noise_perc), replace=False)]
y_test_line1 = np.stack([np.linspace(0.4,0.6,ppline), 0.2*np.ones(ppline)])[:,np.random.choice(ppline, int(ppline*noise_perc), replace=False)]
y_test_line2 = np.stack([np.linspace(0.7,0.9,ppline), 0.2*np.ones(ppline)])[:,np.random.choice(ppline, int(ppline*noise_perc), replace=False)]
# for line in lines:
#     plt.plot(line[0,:], line[1,:], '.', markersize=6)
# plt.plot(x_test_line1[0,:], x_test_line1[1,:])
# plt.plot(x_test_line2[0,:], x_test_line2[1,:])
# plt.plot(y_test_line1[0,:], y_test_line1[1,:])
# plt.plot(y_test_line2[0,:], y_test_line2[1,:])


###############
os.environ["CUDA_DEVICE_ORDER"]   = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'

###############
import itertools as it
from tqdm import tqdm
import torch.nn.functional as F
bs         = 24
lr         = 0.03
neg_margin = 0.1
train_iter = 200
device = torch.device('cpu')


###############
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(2,30), nn.ReLU(), nn.Linear(30,30), nn.ReLU(), nn.Linear(30,2))

    def forward(self, x):
        return torch.nn.functional.normalize(self.backbone(x),dim=1)



###############
base_net     = Backbone()
main_reg_net = Backbone()



###############
def train(net2train, p_switch=0):
    device = torch.device('cpu')
    _ = net2train.train()
    _ = net2train.to(device)
    optim = torch.optim.Adam(net2train.parameters(), lr=lr)

    loss_collect = []
    for i in range(train_iter):
        idxs   = np.random.choice(len(train_lines), bs, replace=False)
        batch  = torch.from_numpy(train_lines[idxs,:]).to(torch.float).to(device)
        train_labels = train_cls[idxs]
        embed        = net2train(batch)

        unique_cls   = np.unique(train_labels)
        indices      = np.arange(len(batch))
        class_dict   = {i:indices[train_labels==i] for i in unique_cls}

        sampled_triplets = [list(it.product([x],[x],[y for y in unique_cls if x!=y])) for x in unique_cls]
        sampled_triplets = [x for y in sampled_triplets for x in y]
        sampled_triplets = [[x for x in list(it.product(*[class_dict[j] for j in i])) if x[0]!=x[1]] for i in sampled_triplets]
        sampled_triplets = [x for y in sampled_triplets for x in y]

        anchors   = [triplet[0] for triplet in sampled_triplets]
        positives = [triplet[1] for triplet in sampled_triplets]
        negatives = [triplet[2] for triplet in sampled_triplets]

        if p_switch>0:
            negatives = [p if np.random.choice(2, p=[1-p_switch, p_switch]) else n for n,p in zip(negatives, positives)]
            neg_dists = torch.mean(F.relu(neg_margin - nn.PairwiseDistance(p=2)(embed[anchors,:], embed[negatives,:])))
            loss      = neg_dists
        else:
            pos_dists = torch.mean(F.relu(nn.PairwiseDistance(p=2)(embed[anchors,:], embed[positives,:])))
            neg_dists = torch.mean(F.relu(neg_margin - nn.PairwiseDistance(p=2)(embed[anchors,:], embed[negatives,:])))
            loss      = pos_dists + neg_dists

        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_collect.append(loss.item())

    return loss_collect



###############
base_loss    = train(base_net)
_ = train(main_reg_net, p_switch=0.001)



###############
def get_embeds(net):
    _ = net.eval()
    with torch.no_grad():
        train_embed      = net(torch.from_numpy(train_lines).to(torch.float).to(device)).cpu().detach().numpy()
        x_embed_test_line1 = net(torch.from_numpy(x_test_line1.T).to(torch.float).to(device)).cpu().detach().numpy()
        x_embed_test_line2 = net(torch.from_numpy(x_test_line2.T).to(torch.float).to(device)).cpu().detach().numpy()
        y_embed_test_line1 = net(torch.from_numpy(y_test_line1.T).to(torch.float).to(device)).cpu().detach().numpy()
        y_embed_test_line2 = net(torch.from_numpy(y_test_line2.T).to(torch.float).to(device)).cpu().detach().numpy()
    _, s, _ = np.linalg.svd(train_embed)
    s = s/np.sum(s)
    return train_embed, x_embed_test_line1, x_embed_test_line2, y_embed_test_line1, y_embed_test_line2, s




###############
base_embed, x_base_t1, x_base_t2, y_base_t1, y_base_t2, base_s = get_embeds(base_net)
sp                                                             = get_embeds(main_reg_net)

###
theta = np.radians(np.linspace(0,360,300))
x_2 = np.cos(theta)
y_2 = np.sin(theta)


###
plt.style.use('default')
import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0, 0.5, n_lines))
colors = np.array([colors[int(sample_cls)] for sample_cls in train_cls][::-1])
f,ax = plt.subplots(1,4)
for i in range(len(lines)):
    loc = np.where(train_cls==i)[0]
    ax[0].scatter(train_lines[loc,0], train_lines[loc,1], color=list(colors[loc,:]), label='Train Cls {}'.format(i), s=40)
ax[0].scatter(x_test_line1[0,:], x_test_line1[1,:], marker='x', label='Test Cls 1', color='r', s=40)
ax[0].scatter(x_test_line2[0,:], x_test_line2[1,:], marker='x', label='Test Cls 2', color='black', s=60)
ax[0].scatter(y_test_line1[0,:], y_test_line1[1,:], marker='^', label='Test Cls 3', color='brown', s=40)
ax[0].scatter(y_test_line2[0,:], y_test_line2[1,:], marker='^', label='Test Cls 4', color='magenta', s=60)
ax[1].plot(x_2, y_2, '--', color='gray', label='Unit Circle')
for i in range(len(lines)):
    loc = np.where(train_cls==i)[0]
    ax[1].scatter(base_embed[loc,0], base_embed[loc,1], color=list(colors[loc,:]), s=60)
ax[1].scatter(x_base_t1[:,0], x_base_t1[:,1], marker='x', color='r', s=60)
ax[1].scatter(x_base_t2[:,0], x_base_t2[:,1], marker='x', color='black', s=60)
ax[1].scatter(y_base_t1[:,0], y_base_t1[:,1], marker='^', color='brown', s=60)
ax[1].scatter(y_base_t2[:,0], y_base_t2[:,1], marker='^', color='magenta', s=60)
ax[1].set_xlim([np.min(base_embed[:,0])*0.85,np.max(base_embed[:,0]*1.15)])
ax[1].set_ylim([np.min(base_embed[:,1])*1.15,np.max(base_embed[:,1]*0.85)])
ax[2].plot(x_2, y_2, '--', color='gray', label='Unit Circle')
for i in range(len(lines)):
    loc = np.where(train_cls==i)[0]
    ax[2].scatter(sp[0][loc,0], sp[0][loc,1], color=list(colors[loc,:]), alpha=0.4, s=60)
ax[2].scatter(sp[1][:,0], sp[1][:,1], marker='x', color='r', s=60)
ax[2].scatter(sp[2][:,0], sp[2][:,1], marker='x', color='black', s=60)
ax[2].scatter(sp[3][:,0], sp[3][:,1], marker='^', color='brown', s=60)
ax[2].scatter(sp[4][:,0], sp[4][:,1], marker='^', color='magenta', s=60)
ax[2].set_xlim([np.min(sp[0][:,0])*1.15,np.max(sp[0][:,0]*1.15)])
ax[2].set_ylim([np.min(sp[0][:,1])*1.15,np.max(sp[0][:,1]*1.15)])
ax[3].bar(np.array([0,0.25]), base_s, width=0.25, alpha=0.5,edgecolor='k', label=r'$Base$')
ax[3].bar(np.array([0.6,0.85]), sp[5],width=0.25, alpha=0.5,edgecolor='k', label=r'$Reg.~Emb.$')
ax[3].set_xticks([0,0.25,0.6,0.85])
ax[3].set_xticklabels([1,2,1,2])
# ax[1].text(0.5, -1.005, r'$Test~Classes$', fontsize=17, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
# ax[2].text(-0.45, -0.05, r'$Test~Classes$', fontsize=17, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
# ax[1].annotate("", xy=(0.5, -0.9),   xytext=(0.55, -0.98), arrowprops=dict(facecolor='k', headwidth=5, width=2, shrink=0, alpha=1))
# ax[1].annotate("", xy=(0.55, -0.85), xytext=(0.55, -0.98), arrowprops=dict(facecolor='k', headwidth=5, width=2, shrink=0, alpha=1))
# ax[1].annotate("", xy=(0.56, -0.86), xytext=(0.55, -0.98), arrowprops=dict(facecolor='k', headwidth=5, width=2, shrink=0, alpha=1))
# ax[1].annotate("", xy=(0.59, -0.82), xytext=(0.55, -0.98), arrowprops=dict(facecolor='k', headwidth=5, width=2, shrink=0, alpha=1))
# ax[2].annotate("", xy=(0.2, 0.8),  xytext=(0.2, -0.02), arrowprops=dict(facecolor='k', headwidth=5, width=2, shrink=0, alpha=1))
# ax[2].annotate("", xy=(0.9, 0.1), xytext=(0.2, -0.02), arrowprops=dict(facecolor='k', headwidth=5, width=2, shrink=0, alpha=1))
# ax[2].annotate("", xy=(0.8, -0.5), xytext=(0.2, -0.02), arrowprops=dict(facecolor='k', headwidth=5, width=2, shrink=0, alpha=1))
# ax[2].annotate("", xy=(0.7, -0.52), xytext=(0.2, -0.02), arrowprops=dict(facecolor='k', headwidth=5, width=2, shrink=0, alpha=1))
ax[0].set_title(r'$Train/Test~Data$', fontsize=22)
ax[1].set_title(r'$Base~Embed.$', fontsize=22)
ax[2].set_title(r'$Regularized~Embed.$', fontsize=22)
ax[3].set_title(r'$SV~Spectrum$', fontsize=22)
ax[0].legend(loc='upper center',fontsize=16)
ax[1].legend(fontsize=16)
ax[2].legend(loc='center left', fontsize=16)
# ax[1].legend(fontsize=16)
# ax[2].legend(loc=4,fontsize=16)
ax[3].legend(loc=1,fontsize=16)
for a in ax.reshape(-1):
    a.tick_params(axis='both', which='major', labelsize=20)
    a.tick_params(axis='both', which='minor', labelsize=20)
f.set_size_inches(22,6)
f.tight_layout()
f.savefig('diag_line_toy_ex_save.png')
f.savefig('diag_line_toy_ex_save.pdf')
plt.close()
