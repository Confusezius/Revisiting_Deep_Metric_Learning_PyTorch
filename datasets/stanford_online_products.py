from datasets.basic_dataset_scaffold import BaseDataset
import os, numpy as np
import pandas as pd


def Give(opt, datapath):
    image_sourcepath  = opt.source_path+'/images'
    training_files = pd.read_table(opt.source_path+'/Info_Files/Ebay_train.txt', header=0, delimiter=' ')
    test_files     = pd.read_table(opt.source_path+'/Info_Files/Ebay_test.txt', header=0, delimiter=' ')

    spi   = np.array([(a,b) for a,b in zip(training_files['super_class_id'], training_files['class_id'])])
    super_dict       = {}
    super_conversion = {}
    for i,(super_ix, class_ix, image_path) in enumerate(zip(training_files['super_class_id'],training_files['class_id'],training_files['path'])):
        if super_ix not in super_dict: super_dict[super_ix] = {}
        if class_ix not in super_dict[super_ix]: super_dict[super_ix][class_ix] = []
        super_dict[super_ix][class_ix].append(image_sourcepath+'/'+image_path)

    if opt.use_tv_split:
        if not opt.tv_split_by_samples:
            train_image_dict, val_image_dict = {},{}
            train_count, val_count = 0, 0
            for super_ix in super_dict.keys():
                class_ixs       = sorted(list(super_dict[super_ix].keys()))
                train_val_split = int(len(super_dict[super_ix])*opt.tv_split_perc)
                train_image_dict[super_ix] = {}
                for _,class_ix in enumerate(class_ixs[:train_val_split]):
                    train_image_dict[super_ix][train_count] = super_dict[super_ix][class_ix]
                    train_count += 1
                val_image_dict[super_ix] = {}
                for _,class_ix in enumerate(class_ixs[train_val_split:]):
                    val_image_dict[super_ix][val_count]     = super_dict[super_ix][class_ix]
                    val_count += 1
        else:
            train_image_dict, val_image_dict = {},{}
            for super_ix in super_dict.keys():
                class_ixs       = sorted(list(super_dict[super_ix].keys()))
                train_image_dict[super_ix] = {}
                val_image_dict[super_ix]   = {}
                for class_ix in class_ixs:
                    train_val_split = int(len(super_dict[super_ix][class_ix])*opt.tv_split_perc)
                    train_image_dict[super_ix][class_ix] = super_dict[super_ix][class_ix][:train_val_split]
                    val_image_dict[super_ix][class_ix]   = super_dict[super_ix][class_ix][train_val_split:]
    else:
        train_image_dict = super_dict
        val_image_dict   = None

    ####
    test_image_dict        = {}
    train_image_dict_temp  = {}
    val_image_dict_temp    = {}
    super_train_image_dict = {}
    super_val_image_dict   = {}
    train_conversion       = {}
    super_train_conversion = {}
    val_conversion         = {}
    super_val_conversion   = {}
    test_conversion        = {}
    super_test_conversion  = {}

    ## Create Training Dictionaries
    i = 0
    for super_ix,super_set in train_image_dict.items():
        super_ix -= 1
        counter   = 0
        super_train_image_dict[super_ix] = []
        for class_ix,class_set in super_set.items():
            class_ix -= 1
            super_train_image_dict[super_ix].extend(class_set)
            train_image_dict_temp[class_ix] = class_set
            if class_ix not in train_conversion:
                train_conversion[class_ix] = class_set[0].split('/')[-1].split('_')[0]
                super_conversion[class_ix] = class_set[0].split('/')[-2]
            counter += 1
            i       += 1
    train_image_dict = train_image_dict_temp

    ## Create Validation Dictionaries
    if opt.use_tv_split:
        i = 0
        for super_ix,super_set in val_image_dict.items():
            super_ix -= 1
            counter   = 0
            super_val_image_dict[super_ix] = []
            for class_ix,class_set in super_set.items():
                class_ix -= 1
                super_val_image_dict[super_ix].extend(class_set)
                val_image_dict_temp[class_ix] = class_set
                if class_ix not in val_conversion:
                    val_conversion[class_ix] = class_set[0].split('/')[-1].split('_')[0]
                    super_conversion[class_ix] = class_set[0].split('/')[-2]
                counter += 1
                i       += 1
        val_image_dict = val_image_dict_temp
    else:
        val_image_dict = None

    ## Create Test Dictioniaries
    for class_ix, img_path in zip(test_files['class_id'],test_files['path']):
        class_ix = class_ix-1
        if not class_ix in test_image_dict.keys():
            test_image_dict[class_ix] = []
        test_image_dict[class_ix].append(image_sourcepath+'/'+img_path)
        test_conversion[class_ix]       = img_path.split('/')[-1].split('_')[0]
        super_test_conversion[class_ix] = img_path.split('/')[-2]

    ##
    if val_image_dict:
        val_dataset            = BaseDataset(val_image_dict,   opt, is_validation=True)
        val_dataset.conversion = val_conversion
    else:
        val_dataset = None

    print('\nDataset Setup:\nUsing Train-Val Split: {0}\n#Classes: Train ({1}) | Val ({2}) | Test ({3})\n'.format(opt.use_tv_split, len(train_image_dict), len(val_image_dict) if val_image_dict else 'X', len(test_image_dict)))

    super_train_dataset = BaseDataset(super_train_image_dict, opt, is_validation=True)
    train_dataset       = BaseDataset(train_image_dict, opt)
    test_dataset        = BaseDataset(test_image_dict,  opt, is_validation=True)
    eval_dataset        = BaseDataset(train_image_dict, opt, is_validation=True)
    eval_train_dataset  = BaseDataset(train_image_dict, opt)

    super_train_dataset.conversion = super_train_conversion
    train_dataset.conversion       = train_conversion
    test_dataset.conversion        = test_conversion
    eval_dataset.conversion        = train_conversion

    return {'training':train_dataset, 'validation':val_dataset, 'testing':test_dataset, 'evaluation':eval_dataset, 'evaluation_train':eval_train_dataset, 'super_evaluation':super_train_dataset}
