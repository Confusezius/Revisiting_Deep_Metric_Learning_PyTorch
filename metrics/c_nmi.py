from sklearn import metrics

class Metric():
    def __init__(self, **kwargs):
        self.requires = ['kmeans_nearest_cosine', 'target_labels']
        self.name     = 'c_nmi'

    def __call__(self, target_labels, computed_cluster_labels_cosine):
        NMI = metrics.cluster.normalized_mutual_info_score(computed_cluster_labels_cosine.reshape(-1), target_labels.reshape(-1))
        return NMI
