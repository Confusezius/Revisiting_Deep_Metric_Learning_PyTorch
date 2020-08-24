import numpy as np

class Metric():
    def __init__(self, k, **kwargs):
        self.k        = k
        self.requires = ['nearest_features_cosine', 'target_labels']
        self.name     = 'c_recall@{}'.format(k)

    def __call__(self, target_labels, k_closest_classes_cosine, **kwargs):
        recall_at_k = np.sum([1 for target, recalled_predictions in zip(target_labels, k_closest_classes_cosine) if target in recalled_predictions[:self.k]])/len(target_labels)
        return recall_at_k
