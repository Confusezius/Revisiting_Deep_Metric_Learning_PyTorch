from datasets.basic_dataset_scaffold import BaseDataset
import os


def Give(opt, datapath):
    image_sourcepath  = datapath+'/images'
    image_classes     = sorted([x for x in os.listdir(image_sourcepath)])
    total_conversion  = {i:x for i,x in enumerate(image_classes)}
    image_list    = {i:sorted([image_sourcepath+'/'+key+'/'+x for x in os.listdir(image_sourcepath+'/'+key)]) for i,key in enumerate(image_classes)}
    image_list    = [[(key,img_path) for img_path in image_list[key]] for key in image_list.keys()]
    image_list    = [x for y in image_list for x in y]

    ### Dictionary of structure class:list_of_samples_with_said_class
    image_dict    = {}
    for key, img_path in image_list:
        if not key in image_dict.keys():
            image_dict[key] = []
        image_dict[key].append(img_path)

    ### Use the first half of the sorted data as training and the second half as test set
    keys = sorted(list(image_dict.keys()))
    train,test      = keys[:len(keys)//2], keys[len(keys)//2:]

    ### If required, split the training data into a train/val setup either by or per class.
    if opt.use_tv_split:
        if not opt.tv_split_by_samples:
            train_val_split = int(len(train)*opt.tv_split_perc)
            train, val      = train[:train_val_split], train[train_val_split:]
            ###
            train_image_dict = {i:image_dict[key] for i,key in enumerate(train)}
            val_image_dict   = {i:image_dict[key] for i,key in enumerate(val)}
            test_image_dict  = {i:image_dict[key] for i,key in enumerate(test)}
        else:
            val = train
            train_image_dict, val_image_dict = {},{}
            for key in train:
                train_ixs = np.random.choice(len(image_dict[key]), int(len(image_dict[key])*opt.tv_split_perc), replace=False)
                val_ixs   = np.array([x for x in range(len(image_dict[key])) if x not in train_ixs])
                train_image_dict[key] = np.array(image_dict[key])[train_ixs]
                val_image_dict[key]   = np.array(image_dict[key])[val_ixs]
        val_dataset   = BaseDataset(val_image_dict,   opt, is_validation=True)
        val_conversion = {i:total_conversion[key] for i,key in enumerate(val)}
        ###
        val_dataset.conversion   = val_conversion
    else:
        train_image_dict = {key:image_dict[key] for key in train}
        val_image_dict   = None
        val_dataset      = None

    ###
    train_conversion = {i:total_conversion[key] for i,key in enumerate(train)}
    test_conversion  = {i:total_conversion[key] for i,key in enumerate(test)}

    ###
    test_image_dict = {key:image_dict[key] for key in test}

    ###
    print('\nDataset Setup:\nUsing Train-Val Split: {0}\n#Classes: Train ({1}) | Val ({2}) | Test ({3})\n'.format(opt.use_tv_split, len(train_image_dict), len(val_image_dict) if val_image_dict else 'X', len(test_image_dict)))

    ###
    train_dataset       = BaseDataset(train_image_dict, opt)
    test_dataset        = BaseDataset(test_image_dict,  opt, is_validation=True)
    eval_dataset        = BaseDataset(train_image_dict, opt, is_validation=True)
    eval_train_dataset  = BaseDataset(train_image_dict, opt, is_validation=False)
    train_dataset.conversion       = train_conversion
    test_dataset.conversion        = test_conversion
    eval_dataset.conversion        = test_conversion
    eval_train_dataset.conversion  = train_conversion

    return {'training':train_dataset, 'validation':val_dataset, 'testing':test_dataset, 'evaluation':eval_dataset, 'evaluation_train':eval_train_dataset}
