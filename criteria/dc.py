import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer
from tqdm import tqdm


"""================================================================================================="""
ALLOWED_MINING_OPS  = list(batchminer.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM       = True
REQUIRES_EMA_NETWORK = True

class Criterion(torch.nn.Module):
    def __init__(self, opt):
        """
        Args:
            margin:             Triplet Margin.
            nu:                 Regularisation Parameter for beta values if they are learned.
            beta:               Class-Margin values.
            n_classes:          Number of different classes during training.
        """
        super(Criterion, self).__init__()

        self.update_f      = opt.diva_dc_update_f
        self.ncluster      = opt.diva_dc_ncluster
        self.red_dim       = opt.embed_dim

        self.classifier    = torch.nn.Linear(opt.network_feature_dim, self.ncluster, bias=False).to(opt.device)

        self.lr            = opt.lr * 10

        self.name          = 'Deep Clustering'

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM

    def update_pseudo_labels(self, model, dataloader, device):
        import faiss, time
        #### Reset Classifier Weights
        torch.cuda.empty_cache()
        self.classifier.weight.data.normal_(0,1/model.feature_dim)

        with torch.no_grad():
            _ = model.eval()
            _ = model.to(device)

            memory_queue = []
            for i,input_tuple in enumerate(tqdm(dataloader, 'Getting DC Embeddings...', total=len(dataloader))):
                embed = model(input_tuple[1].type(torch.FloatTensor).to(device))[-1]
                memory_queue.append(embed)
            memory_queue = torch.cat(memory_queue, dim=0).cpu().numpy()

        #PERFORM PCA
        print('Computing PCA... ', end='')
        start = time.time()
        pca_mat      = faiss.PCAMatrix(memory_queue.shape[-1], self.red_dim)
        pca_mat.train(memory_queue)
        memory_queue = pca_mat.apply_py(memory_queue)
        print('Done in {}s.'.format(time.time()-start))
        #
        #
        print('Computing Pseudolabels... ', end='')
        start = time.time()
        cpu_cluster_index = faiss.IndexFlatL2(memory_queue.shape[-1])
        kmeans            = faiss.Clustering(memory_queue.shape[-1], self.ncluster)
        kmeans.niter      = 20
        kmeans.min_points_per_centroid = 1
        kmeans.max_points_per_centroid = 1000000000
        ### Train Kmeans
        kmeans.train(memory_queue, cpu_cluster_index)
        centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(self.ncluster, memory_queue.shape[-1])
        ###
        faiss_search_index = faiss.IndexFlatL2(centroids.shape[-1])
        faiss_search_index.add(centroids)
        _, computed_cluster_labels = faiss_search_index.search(memory_queue, 1)
        print('Done in {}s.'.format(time.time()-start))
        ###
        self.pseudo_labels = computed_cluster_labels
        ###
        torch.cuda.empty_cache()


    def forward(self, feature_batch, indices, **kwargs):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        pseudo_labels = torch.from_numpy(self.pseudo_labels[indices].reshape(-1))
        pred_batch    = self.classifier(feature_batch)
        loss          = torch.nn.CrossEntropyLoss()(pred_batch, pseudo_labels.to(pred_batch.device))
        return loss
