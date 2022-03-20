import numpy as np
from scipy.special import comb, binom
import torch

class Metric():
    def __init__(self, **kwargs):
        self.requires = ['kmeans_cosine', 'kmeans_nearest_cosine', 'features_cosine', 'target_labels']
        self.name     = 'c_f1'

    def __call__(self, target_labels, computed_cluster_labels_cosine, features_cosine, centroids_cosine):
        import time
        start = time.time()
        if isinstance(features_cosine, torch.Tensor):
            features_cosine = features_cosine.detach().cpu().numpy()
        d = np.zeros(len(features_cosine))
        for i in range(len(features_cosine)):
            d[i] = np.linalg.norm(features_cosine[i,:] - centroids_cosine[computed_cluster_labels_cosine[i],:])

        start = time.time()
        labels_pred = np.zeros(len(features_cosine))
        for i in np.unique(computed_cluster_labels_cosine):
            index = np.where(computed_cluster_labels_cosine == i)[0]
            ind = np.argmin(d[index])
            cid = index[ind]
            labels_pred[index] = cid


        start = time.time()
        N = len(target_labels)

        # cluster n_labels
        avail_labels = np.unique(target_labels)
        n_labels     = len(avail_labels)

        # count the number of objects in each cluster
        count_cluster = np.zeros(n_labels)
        for i in range(n_labels):
            count_cluster[i] = len(np.where(target_labels == avail_labels[i])[0])

        # build a mapping from item_id to item index
        keys     = np.unique(labels_pred)
        num_item = len(keys)
        values   = range(num_item)
        item_map = dict()
        for i in range(len(keys)):
            item_map.update([(keys[i], values[i])])


        # count the number of objects of each item
        count_item = np.zeros(num_item)
        for i in range(N):
            index = item_map[labels_pred[i]]
            count_item[index] = count_item[index] + 1

        # compute True Positive (TP) plus False Positive (FP)
        # tp_fp = 0
        tp_fp = comb(count_cluster, 2).sum()
        # for k in range(n_labels):
        #     if count_cluster[k] > 1:
        #         tp_fp = tp_fp + comb(count_cluster[k], 2)

        # compute True Positive (TP)
        tp     = 0
        start = time.time()
        for k in range(n_labels):
            member     = np.where(target_labels == avail_labels[k])[0]
            member_ids = labels_pred[member]
            count = np.zeros(num_item)
            for j in range(len(member)):
                index = item_map[member_ids[j]]
                count[index] = count[index] + 1
            # for i in range(num_item):
            #     if count[i] > 1:
            #         tp = tp + comb(count[i], 2)
            tp += comb(count,2).sum()
        # False Positive (FP)
        fp = tp_fp - tp

        # Compute False Negative (FN)
        count = comb(count_item, 2).sum()
        # count = 0
        # for j in range(num_item):
            # if count_item[j] > 1:
            #     count = count + comb(count_item[j], 2)
        fn = count - tp

        # compute F measure
        P = tp / (tp + fp)
        R = tp / (tp + fn)
        beta = 1
        F = (beta*beta + 1) * P * R / (beta*beta * P + R)
        return F
