from metrics import e_recall, dists, rho_spectrum
from metrics import nmi, f1, mAP
import numpy as np
import faiss
import torch
from tqdm import tqdm
import copy


def select(metricname, opt):
    if 'e_recall' in metricname:
        k = int(metricname.split('@')[-1])
        return e_recall.Metric(k)
    elif metricname=='nmi':
        return nmi.Metric()
    elif metricname=='mAP':
        return mAP.Metric()
    elif metricname=='f1':
        return f1.Metric()
    elif 'dists' in metricname:
        mode = metricname.split('@')[-1]
        return dists.Metric(mode)
    elif 'rho_spectrum' in metricname:
        mode = int(metricname.split('@')[-1])
        embed_dim = opt.rho_spectrum_embed_dim
        return rho_spectrum.Metric(embed_dim, mode=mode, opt=opt)
    else:
        raise NotImplementedError("Metric {} not available!".format(metricname))




class MetricComputer():
    def __init__(self, metric_names, opt):
        self.pars            = opt
        self.metric_names    = metric_names
        self.list_of_metrics = [select(metricname, opt) for metricname in metric_names]
        self.requires        = [metric.requires for metric in self.list_of_metrics]
        self.requires        = list(set([x for y in self.requires for x in y]))

    def compute_standard(self, opt, model, dataloader, evaltypes, device, **kwargs):
        evaltypes = copy.deepcopy(evaltypes)

        n_classes = opt.n_classes
        image_paths     = np.array([x[0] for x in dataloader.dataset.image_list])
        _ = model.eval()

        ###
        feature_colls  = {key:[] for key in evaltypes}

        ###
        with torch.no_grad():
            target_labels = []
            final_iter = tqdm(dataloader, desc='Embedding Data...'.format(len(evaltypes)))
            image_paths= [x[0] for x in dataloader.dataset.image_list]
            for idx,inp in enumerate(final_iter):
                input_img,target = inp[1], inp[0]
                target_labels.extend(target.numpy().tolist())
                out = model(input_img.to(device))
                if isinstance(out, tuple): out, aux_f = out

                ### Include embeddings of all output features
                for evaltype in evaltypes:
                    if isinstance(out, dict):
                        feature_colls[evaltype].extend(out[evaltype].cpu().detach().numpy().tolist())
                    else:
                        feature_colls[evaltype].extend(out.cpu().detach().numpy().tolist())


            target_labels = np.hstack(target_labels).reshape(-1,1)


        computed_metrics = {evaltype:{} for evaltype in evaltypes}
        extra_infos      = {evaltype:{} for evaltype in evaltypes}


        ###
        faiss.omp_set_num_threads(self.pars.kernels)
        # faiss.omp_set_num_threads(self.pars.kernels)
        res = None
        torch.cuda.empty_cache()
        if self.pars.evaluate_on_gpu:
            res = faiss.StandardGpuResources()


        import time
        for evaltype in evaltypes:
            features = np.vstack(feature_colls[evaltype]).astype('float32')

            start = time.time()
            if 'kmeans' in self.requires:
                ### Set CPU Cluster index
                cluster_idx = faiss.IndexFlatL2(features.shape[-1])
                if res is not None: cluster_idx = faiss.index_cpu_to_gpu(res, 0, cluster_idx)
                kmeans            = faiss.Clustering(features.shape[-1], n_classes)
                kmeans.niter = 20
                kmeans.min_points_per_centroid = 1
                kmeans.max_points_per_centroid = 1000000000
                ### Train Kmeans
                kmeans.train(features, cluster_idx)
                centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(n_classes, features.shape[-1])


            if 'kmeans_nearest' in self.requires:
                faiss_search_index = faiss.IndexFlatL2(centroids.shape[-1])
                if res is not None: faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
                faiss_search_index.add(centroids)
                _, computed_cluster_labels = faiss_search_index.search(features, 1)

            if 'nearest_features' in self.requires:
                faiss_search_index  = faiss.IndexFlatL2(features.shape[-1])
                if res is not None: faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
                faiss_search_index.add(features)

                max_kval            = np.max([int(x.split('@')[-1]) for x in self.metric_names if 'recall' in x])
                _, k_closest_points = faiss_search_index.search(features, int(max_kval+1))
                k_closest_classes   = target_labels.reshape(-1)[k_closest_points[:,1:]]

            ###
            if self.pars.evaluate_on_gpu:
                features = torch.from_numpy(features).to(self.pars.device)

            start = time.time()
            for metric in self.list_of_metrics:
                input_dict = {}
                if 'features' in metric.requires:         input_dict['features'] = features
                if 'target_labels' in metric.requires:    input_dict['target_labels'] = target_labels
                if 'kmeans' in metric.requires:           input_dict['centroids'] = centroids
                if 'kmeans_nearest' in metric.requires:   input_dict['computed_cluster_labels'] = computed_cluster_labels
                if 'nearest_features' in metric.requires: input_dict['k_closest_classes'] = k_closest_classes
                computed_metrics[evaltype][metric.name] = metric(**input_dict)

            extra_infos[evaltype] = {'features':features, 'target_labels':target_labels,
                                     'image_paths': dataloader.dataset.image_paths,
                                     'query_image_paths':None, 'gallery_image_paths':None}

        torch.cuda.empty_cache()
        return computed_metrics, extra_infos
