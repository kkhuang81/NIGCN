import numpy as np
import torch
import argparse

def random_splits_papers(dataset_name="papers100M", percls_trn=20, val_lb=500, test_lb=1000, splitType=0, seq=0):
    dataset_str = dataset_name+'/'+dataset_name
    data = np.load(dataset_str + '_labels.npz')
    labels=np.concatenate((data['train_labels'],data['val_labels'],data['test_labels']))
    percls_trn=percls_trn
    val_lb=val_lb
    test_lb=test_lb
    num_classes = labels.max().item() + 1
    
    indices = []
    for i in range(num_classes):
        index = np.nonzero(labels == i)[0]
        index = index[torch.randperm(index.size)]
        indices.append(index)

    idx_train = np.concatenate([i[:percls_trn] for i in indices])
    idx_val=[]
    idx_test=[]

    if splitType == 0:
        rest_index = np.concatenate([i[percls_trn:] for i in indices])
        rest_index = rest_index[torch.randperm(rest_index.size)]
        idx_val = rest_index[:val_lb]
        idx_test = rest_index[val_lb:val_lb + test_lb]

    # per-class of validation and test is val_lb and test_lb, respectively //
    if splitType == 1:
        idx_val = np.concatenate([i[percls_trn:percls_trn + val_lb] for i in indices])
        idx_test = np.concatenate([i[percls_trn + val_lb:percls_trn + val_lb + test_lb] for i in indices])

    reversemap=np.load(dataset_str+'_reverseIDmap.npy', allow_pickle=True).item()

    new_train=[]
    for x in idx_train:
        new_train.append(reversemap[x])
    
    new_val=[]
    for x in idx_val:
        new_val.append(reversemap[x])
    
    new_test=[]
    for x in idx_test:
        new_test.append(reversemap[x])

    np.savez(dataset_str + '_'+str(percls_trn)+'_'+str(val_lb)+'_'+str(test_lb)+'_'+str(seq)+'_splits.npz', idx_train=new_train, idx_val=new_val, idx_test=new_test)

def random_splits_multiclass(dataset_name="cora", percls_trn=20, val_lb=500, test_lb=1000, splitType=0, seq=0):

    dataset_str = dataset_name+'/'+dataset_name
    data = np.load(dataset_str + '_labels.npz')
    labels = data['labels']           
    num_classes = labels.max().item() + 1
    indices = []
    for i in range(num_classes):
        index = np.nonzero(labels == i)[0]
        index = index[torch.randperm(index.size)]
        indices.append(index)

    idx_train = np.concatenate([i[:percls_trn] for i in indices])

    # total val and test are val_lb and test_lb (we use this type for most of the case)
    if splitType == 0:
        rest_index = np.concatenate([i[percls_trn:] for i in indices])
        rest_index = rest_index[torch.randperm(rest_index.size)]
        idx_val = rest_index[:val_lb]
        idx_test = rest_index[val_lb:val_lb + test_lb]

    # per-class of validation and test is val_lb and test_lb, respectively //
    if splitType == 1:
        idx_val = np.concatenate([i[percls_trn:percls_trn + val_lb] for i in indices])
        idx_test = np.concatenate([i[percls_trn + val_lb:percls_trn + val_lb + test_lb] for i in indices])
        #idx_test = idx_test[torch.randperm(idx_test.size)]  # do I need this?

    np.savez(dataset_str + '_'+str(percls_trn)+'_'+str(val_lb)+'_'+str(test_lb)+'_'+str(seq)+'_splits.npz', idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--seed', type=int, default=51290, help='Random seed.')
    parser.add_argument('--dataset', type=str, default="cora", help='Dataset to use.')
    parser.add_argument('--percls_trn', type=int, default=20, help='the number of training nodes per class')
    parser.add_argument('--val_lb', type=int, default=500, help='the number of validation nodes')
    parser.add_argument('--test_lb', type=int, default=1000, help='the number of testing nodes')
    parser.add_argument('--splitType', type=int, default=0, help='the type of split')    
    args = parser.parse_args()

    print(args)
    seeds=[28453, 25139, 26382, 40854, 53845, 79886, 3853, 59739, 94475, 999]
    for i in range(10):
        np.random.seed(seeds[i])
        torch.manual_seed(seeds[i])
        if args.dataset == 'papers100M':
            random_splits_papers(args.dataset, args.percls_trn, args.val_lb, args.test_lb, args.splitType, i)
        else:
            random_splits_multiclass(args.dataset, args.percls_trn, args.val_lb, args.test_lb, args.splitType, i)
