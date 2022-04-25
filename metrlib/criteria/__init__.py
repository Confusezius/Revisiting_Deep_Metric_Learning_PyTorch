### Standard DML criteria
from criteria import triplet, margin, proxynca, npair
from criteria import lifted, contrastive, softmax
from criteria import angular, snr, histogram, arcface
from criteria import softtriplet, multisimilarity, quadruplet
### Non-Standard Criteria
from criteria import adversarial_separation
from criteria import barlow
### Basic Libs
import copy


"""================================================================================================="""
def select(loss, opt, to_optim, batchminer=None):
    #####
    losses = {'triplet': triplet,
              'margin':margin,
              'proxynca':proxynca,
              'npair':npair,
              'angular':angular,
              'contrastive':contrastive,
              'lifted':lifted,
              'snr':snr,
              'multisimilarity':multisimilarity,
              'histogram':histogram,
              'softmax':softmax,
              'softtriplet':softtriplet,
              'arcface':arcface,
              'quadruplet':quadruplet,
              'adversarial_separation':adversarial_separation,
              'barlow': barlow}


    if loss not in losses: raise NotImplementedError('Loss {} not implemented!'.format(loss))

    loss_lib = losses[loss]
    if loss_lib.REQUIRES_BATCHMINER:
        if batchminer is None:
            raise Exception('Loss {} requires one of the following batch mining methods: {}'.format(loss, loss_lib.ALLOWED_MINING_OPS))
        else:
            if batchminer.name not in loss_lib.ALLOWED_MINING_OPS:
                raise Exception('{}-mining not allowed for {}-loss!'.format(batchminer.name, loss))


    loss_par_dict  = {'opt':opt}
    if loss_lib.REQUIRES_BATCHMINER:
        loss_par_dict['batchminer'] = batchminer

    criterion = loss_lib.Criterion(**loss_par_dict)

    if loss_lib.REQUIRES_OPTIM:
        if hasattr(criterion,'optim_dict_list') and criterion.optim_dict_list is not None:
            to_optim += criterion.optim_dict_list
        else:
            to_optim    += [{'params':criterion.parameters(), 'lr':criterion.lr}]

    return criterion, to_optim
