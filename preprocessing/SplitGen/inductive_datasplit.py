import numpy as np
import torch
import argparse

def random_splits_multiclass(dataset_str="cora", percls_trn=20, val_lb=500, test_lb=1000, splitType=0, seq=0):

    dataset_str = dataset_str+'/'+dataset_str
    data = np.load(dataset_str + '_labels.npz')

    labels = data['labels']
    train_list=data['idx_train']
    val_list=data['idx_val']
    test_list=data['idx_test']

    val_test=np.concatenate((val_list, test_list), axis=0)
    val_test = val_test[torch.randperm(val_test.size)]
    

    num_classes = labels.max().item() + 1
    
    indices=[]
    
    for i in range(num_classes):
        index=[x for x in train_list if labels[x]==i]
        np.random.shuffle(index)
        indices.append(index)        

    idx_train = np.concatenate([i[:percls_trn] for i in indices])

    #rest_index = val_test[torch.randperm(val_test.size)]
    idx_val = val_test[:val_lb]
    idx_test = val_test[val_lb:val_lb + test_lb]

    np.savez(dataset_str + '_'+str(percls_trn)+'_'+str(val_lb)+'_'+str(test_lb)+'_'+str(seq)+'_splits.npz', idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)


def random_splits_multilabel(dataset_str="cora", percls_trn=20, val_lb=500, test_lb=1000, seq=0):

    dataset_str = dataset_str+'/'+dataset_str
    data = np.load(dataset_str + '_labels.npz')

    files = []
    for x in data.files:
        files.append(data[x])
    labels, idx_train, idx_val, idx_test = tuple(files)
    
    num_classes=labels.shape[1]
    #num_classes = labels.max().item() + 1
    num_train=min(num_classes*percls_trn, idx_train.size)
    idx_train=idx_train[torch.randperm(idx_train.size)]
    idx_val=idx_val[torch.randperm(idx_val.size)]
    idx_test=idx_test[torch.randperm(idx_test.size)]

    idx_train=idx_train[:num_train]
    idx_val=idx_val[:val_lb]
    idx_test=idx_test[:test_lb]

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
        if args.dataset =='amazon' or args.dataset =='yelp':
            random_splits_multilabel(args.dataset, args.percls_trn, args.val_lb, args.test_lb, i)
        else:
            random_splits_multiclass(args.dataset, args.percls_trn, args.val_lb, args.test_lb, args.splitType, i)
