# Deep Metric Learning Research in PyTorch

---
## What can I find here?

This repository contains all code and implementations used in:

```
Revisiting Training Strategies and Generalization Performance in Deep Metric Learning
```

accepted to **ICML 2020**.

**Link**: https://arxiv.org/abs/2002.08473

The code is meant to serve as a research starting point in Deep Metric Learning.
By implementing key baselines under a consistent setting and logging a vast set of metrics, it should be easier to ensure that method gains are not due to implementational variations, while better understanding driving factors.

It is set up in a modular way to allow for fast and detailed prototyping, but with key elements written in a way that allows the code to be directly copied into other pipelines. In addition, multiple training and test metrics are logged in W&B to allow for easy and large-scale evaluation.

Finally, please find a public W&B repo with key runs performed in the paper here: https://app.wandb.ai/confusezius/RevisitDML.

**Contact**: Karsten Roth, karsten.rh1@gmail.com  

*Suggestions are always welcome!*

---
## Some Notes:

If you use this code in your research, please cite
```
@misc{roth2020revisiting,
    title={Revisiting Training Strategies and Generalization Performance in Deep Metric Learning},
    author={Karsten Roth and Timo Milbich and Samarth Sinha and Prateek Gupta and Björn Ommer and Joseph Paul Cohen},
    year={2020},
    eprint={2002.08473},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

This repository contains (in parts) code that has been adapted from:
* https://github.com/idstcv/SoftTriple
* https://github.com/bnu-wangxun/Deep_Metric
* https://github.com/valerystrizh/pytorch-histogram-loss
* https://github.com/Confusezius/Deep-Metric-Learning-Baselines

Make sure to also check out the following repo with a great plug-and-play implementation of DML methods:
* https://github.com/KevinMusgrave/pytorch-metric-learning

---

**[All implemented methods and metrics are listed at the bottom!]()**

---

## Paper-related Information

#### Reproduce results from our paper **[Revisiting Training Strategies and Generalization Performance in Deep Metric Learning](https://arxiv.org/pdf/2002.08473.pdf)**

* *ALL* standardized Runs that were used are available in `Revisit_Runs.sh`.
* These runs are also logged in this public W&B repo: https://app.wandb.ai/confusezius/RevisitDML.
* All Runs and their respective metrics can be downloaded and evaluated to generate the plots in our paper by following `Result_Evaluations.py`. This also allows for potential introspection of other relations. It also converts results directly into Latex-table format with mean and standard deviations.
* To utilize different batch-creation methods, simply set the flag `--data_sampler` to the method of choice. Allowed flags are listed in `datasampler/__init__.py`.
* To use the proposed spectral regularization for tuple-based methods, set `--batch_mining rho_distance` with flip probability `--miner_rho_distance_cp e.g. 0.2`.
* A script to run the toy experiments in the paper is provided in `toy_experiments`.

**Note**: There may be small deviations in results based on the Hardware (e.g. between P100 and RTX GPUs) and Software (different PyTorch/Cuda versions) used to run these experiments, but they should be covered in the standard deviations reported in the paper.

---

## How to use this Repo

### Requirements:

* PyTorch 1.2.0+ & Faiss-Gpu
* Python 3.6+
* pretrainedmodels, torchvision 0.3.0+

An exemplary setup of a virtual environment containing everything needed:
```
(1) wget  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
(2) bash Miniconda3-latest-Linux-x86_64.sh (say yes to append path to bashrc)
(3) source .bashrc
(4) conda create -n DL python=3.6
(5) conda activate DL
(6) conda install matplotlib scipy scikit-learn scikit-image tqdm pandas pillow
(7) conda install pytorch torchvision faiss-gpu cudatoolkit=10.0 -c pytorch
(8) pip install wandb pretrainedmodels
(9) Run the scripts!
```

### Datasets:
Data for
* CUB200-2011 (http://www.vision.caltech.edu/visipedia/CUB-200.html)
* CARS196 (https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
* Stanford Online Products (http://cvgl.stanford.edu/projects/lifted_struct/)

can be downloaded either from the respective project sites or directly via Dropbox:

* CUB200-2011 (1.08 GB): https://www.dropbox.com/s/tjhf7fbxw5f9u0q/cub200.tar?dl=0
* CARS196 (1.86 GB): https://www.dropbox.com/s/zi2o92hzqekbmef/cars196.tar?dl=0
* SOP (2.84 GB): https://www.dropbox.com/s/fu8dgxulf10hns9/online_products.tar?dl=0

**The latter ensures that the folder structure is already consistent with this pipeline and the dataloaders**.   

Otherwise, please make sure that the datasets have the following internal structure:

* For CUB200-2011/CARS196:
```
cub200/cars196
└───images
|    └───001.Black_footed_Albatross
|           │   Black_Footed_Albatross_0001_796111
|           │   ...
|    ...
```

* For Stanford Online Products:
```
online_products
└───images
|    └───bicycle_final
|           │   111085122871_0.jpg
|    ...
|
└───Info_Files
|    │   bicycle.txt
|    │   ...
```

Assuming your folder is placed in e.g. `<$datapath/cub200>`, pass `$datapath` as input to `--source`.

### Training:
Training is done by using `main.py` and setting the respective flags, all of which are listed and explained in `parameters.py`. A vast set of exemplary runs is provided in `Revisit_Runs.sh`.

**[I.]** **A basic sample run using default parameters would like this**:

```
python main.py --loss margin --batch_mining distance --log_online \
              --project DML_Project --group Margin_with_Distance --seed 0 \
              --gpu 0 --bs 112 --data_sampler class_random --samples_per_class 2 \
              --arch resnet50_frozen_normalize --source $datapath --n_epochs 150 \
              --lr 0.00001 --embed_dim 128 --evaluate_on_gpu
```

The purpose of each flag explained:

* `--loss <loss_name>`: Name of the training objective used. See folder `criteria` for implementations of these methods.
* `--batch_mining <batchminer_name>`: Name of the batch-miner to use (for tuple-based ranking methods). See folder `batch_mining` for implementations of these methods.
* `--log_online`: Log metrics online via either W&B (Default) or CometML. Regardless, plots, weights and parameters are all stored offline as well.
*  `--project`, `--group`: Project name as well as name of the run. Different seeds will be logged into the same `--group` online. The group as well as the used seed also define the local savename.
* `--seed`, `--gpu`, `--source`: Basic Parameters setting the training seed, the used GPU and the path to the parent folder containing the respective Datasets.
* `--arch`: The utilized backbone, e.g. ResNet50. You can append `_frozen` and `_normalize` to the name to ensure that BatchNorm layers are frozen and embeddings are normalized, respectively.
* `--data_sampler`, `--samples_per_class`: How to construct a batch. The default method, `class_random`, selects classes at random and places `<samples_per_class>` samples into the batch until the batch is filled.
* `--lr`, `--n_epochs`, `--bs` ,`--embed_dim`: Learning rate, number of training epochs, the batchsize and the embedding dimensionality.  
* `--evaluate_on_gpu`: If set, all metrics are computed using the gpu - requires Faiss-GPU and may need additional GPU memory.

#### Some Notes:
* During training, metrics listed in `--evaluation_metrics` will be logged for both training and validation/test set. If you do not care about detailed training metric logging, simply set the flag `--no_train_metrics`. A checkpoint is saved for improvements in metrics listed in `--storage_metrics` on training, validation or test sets.
* If one wishes to use a training/validation split, simply set `--use_tv_split` and `--tv_split_perc <train/val split percentage>`.


**[II.]** **Advanced Runs**:

```
python main.py --loss margin --batch_mining distance --loss_margin_beta 0.6 --miner_distance_lower_cutoff 0.5 ... (basic parameters)
```

* To use specific parameters that are loss, batchminer or e.g. datasampler-related, simply set the respective flag.
* For structure and ease of use, parameters relating to a specifc loss function/batchminer etc. are marked as e.g. `--loss_<lossname>_<parameter_name>`, see `parameters.py`.
* However, every parameter can be called from every class, as all parameters are stored in a shared namespace that is passed to all methods. This makes it easy to create novel fusion losses and the likes.


### Evaluating Results with W&B
Here some information on using W&B (highly encouraged!)

* Create an account here (free): https://wandb.ai
* After the account is set, make sure to include your API key in `parameters.py` under `--wandb_key`.
* To make sure that W&B data can be stored, ensure to run `wandb on` in the folder pointed to by `--save_path`.
* When data is logged online to W&B, one can use `Result_Evaluations.py` to download all data, create named metric and correlation plots and output a summary in the form of a latex-ready table with mean and standard deviations of all metrics. **This ensures that there are no errors between computed and reported results.**


### Creating custom methods:

1. **Create custom objectives**: Simply take a look at e.g. `criteria/margin.py`, and ensure that the used methods has the following properties:
  * Inherit from `torch.nn.Module` and define a custom `forward()` function.
  * When using trainable parameters, make sure to either provide a `self.lr` to set the learning rate of the loss-specific parameters, or set `self.optim_dict_list`, which is a list containing optimization dictionaries passed to the optimizer (see e.g `criteria/proxynca.py`). If both are set, `self.optim_dict_list` has priority.
  * Depending on the loss, remember to set the variables `ALLOWED_MINING_OPS  = None or list of allowed mining operations`, `REQUIRES_BATCHMINER = False or True`, `REQUIRES_OPTIM = False or True` to denote if the method needs a batchminer or optimization of internal parameters.


2. **Create custom batchminer**: Simply take a look at e.g. `batch_mining/distance.py` - The miner needs to be a class with a defined `__call__()`-function, taking in a batch and labels and returning e.g. a list of triplets.

3. **Create custom datasamplers**:Simply take a look at e.g. `datasampler/class_random_sampler.py`. The sampler needs to inherit from `torch.utils.data.sampler.Sampler` and has to provide a `__iter__()` and a `__len__()` function. It has to yield a set of indices that are used to create the batch.


---

## Implemented Methods

For a detailed explanation of everything, please refer to the supplementary of our paper!

### DML criteria

* **Angular** [[Deep Metric Learning with Angular Loss](https://arxiv.org/pdf/1708.01682.pdf)]
* **ArcFace** [[ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.07698.pdf)]
* **Contrastive** [[Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)]
* **Generalized Lifted Structure** [[In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737)]
* **Histogram** [[Learning Deep Embeddings with Histogram Loss](https://arxiv.org/pdf/1611.00822.pdf)]
* **Marginloss** [[Sampling Matters in Deep Embeddings Learning](https://arxiv.org/abs/1706.07567)]
* **MultiSimilarity** [[Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning](https://arxiv.org/abs/1904.06627)]
* **N-Pair** [[Improved Deep Metric Learning with Multi-class N-pair Loss Objective](https://papers.nips.cc/paper/6200-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective)]
* **ProxyNCA** [[No Fuss Distance Metric Learning using Proxies](https://arxiv.org/pdf/1703.07464.pdf)]
* **Quadruplet** [[Beyond triplet loss: a deep quadruplet network for person re-identification](https://arxiv.org/abs/1704.01719)]
* **Signal-to-Noise Ratio (SNR)** [[Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning](https://arxiv.org/pdf/1904.02616.pdf)]
* **SoftTriple** [[SoftTriple Loss: Deep Metric Learning Without Triplet Sampling](https://arxiv.org/abs/1909.05235)]
* **Normalized Softmax** [[Classification is a Strong Baseline for Deep Metric Learning](https://arxiv.org/abs/1811.12649)]
* **Triplet** [[Facenet: A unified embedding for face recognition and clustering](https://arxiv.org/abs/1503.03832)]

### DML batchminer

* **Random** [[Facenet: A unified embedding for face recognition and clustering](https://arxiv.org/abs/1503.03832)]
* **Semihard** [[Facenet: A unified embedding for face recognition and clustering](https://arxiv.org/abs/1503.03832)]
* **Softhard** [https://github.com/Confusezius/Deep-Metric-Learning-Baselines]
* **Distance-based** [[Sampling Matters in Deep Embeddings Learning](https://arxiv.org/abs/1706.07567)]
* **Rho-Distance** [[Revisiting Training Strategies and Generalization Performance in Deep Metric Learning](https://arxiv.org/abs/2002.08473)]
* **Parametric** [[PADS: Policy-Adapted Sampling for Visual Similarity Learning](https://arxiv.org/abs/2003.11113)]

### Architectures

* **ResNet50** [[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)]
* **Inception-BN** [[Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)]
* **GoogLeNet** (torchvision variant w/ BN) [[Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)]

### Datasets
* **CUB200-2011** [[Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)]
* **CARS196** [[Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)]
* **Stanford Online Products** [[Deep Metric Learning via Lifted Structured Feature Embedding](https://cvgl.stanford.edu/projects/lifted_struct/)]

### Evaluation Metrics

* **Recall@k**
* **Normalized Mutual Information (NMI)**
* **F1**
* **mAP (class-averaged)**
* **Spectral Variance**
* **Mean Intraclass Distance**
* **Mean Interclass Distance**
* **Ratio Intra- to Interclass Distance**
