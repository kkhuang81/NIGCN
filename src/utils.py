import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
import torch
import sys
import pickle as pkl
import networkx as nx
#from normalization import fetch_normalization, row_normalize
from time import perf_counter
import random
import PPR
import gc


 
def load_citation(dataset_name="cora", lamb=0, alpha=0.1, epsilon=0.01, level=6, rr=0.5, opt=True, splitfile=""):    
    dataset_str = 'data/' + dataset_name +'/'+dataset_name    
    data = np.load(dataset_str + '_labels.npz')
    if dataset_name=="papers100M":
        labels=np.concatenate((data['train_labels'],data['val_labels'],data['test_labels']))
    else:
        labels = data['labels']
    
    split=np.load(dataset_str+splitfile)
    files = []
    for x in split.files:
        files.append(split[x])
    idx_train, idx_val, idx_test = tuple(files)
    
    nodelist=[x for x in idx_train]+[x for x in idx_val]+ [x for x in idx_test]

    nodestr=""
    for x in nodelist:
        nodestr+=" "+str(x)
    # feature update
    #features=PPR.ppr(dataset_str, level, alpha, epsilon, rr, 2, nodestr[3:6], seed, opt, thread)
    node_num=0
    edge_num=0    
    if dataset_name == "cora":
        node_num=2708
        edge_num=13264        
    if dataset_name == "citeseer":
        node_num=3327
        edge_num=12431
    if dataset_name == "pubmed":
        node_num=19717
        edge_num=108365       
    if dataset_name == "papers100M":
        node_num=111059956
        edge_num=3339184668    
    
    
    features=PPR.ppr(dataset_str, node_num, edge_num, level, lamb, alpha, epsilon, rr, len(nodelist), nodestr[1:], opt)
    features = torch.FloatTensor(np.array(features))

    if dataset_name=="papers100M":
        trainlist=[]
        trainmap=np.load(dataset_str+'_trainIDmap.npy', allow_pickle=True).item()
        for x in nodelist:
            trainlist.append(trainmap[x]) 
        nodelist=trainlist
        del trainmap

    label_trivaltest=labels[nodelist]
    del labels    
    gc.collect()


    labels = torch.LongTensor(label_trivaltest)    
   
    return features, labels, len(idx_train), len(idx_val), len(idx_test)

def load_inductive(dataset_name="cora", lamb=0, alpha=0.1, epsilon=0.01, level=6, rr=0.5, opt=True, splitfile=""):    
    dataset_str = 'data/' + dataset_name +'/'+dataset_name
    data = np.load(dataset_str + '_labels.npz')
    labels=data['labels']

    #split=np.load(dataset_str+'_'+str(TriNum)+'_'+str(ValNum)+'_'+str(TstNum)+'_'+str(seq)+'_splits.npz')
    print(dataset_str+splitfile)
    split=np.load(dataset_str+splitfile)
    files = []
    for x in split.files:
        files.append(split[x])
    idx_train, idx_val, idx_test = tuple(files)

    #idx_train=idx_train[:1]
    # training graph
    trainlist=[]
    trainmap=np.load(dataset_str+'_trainIDmap.npy', allow_pickle=True).item()
  
    for x in idx_train:
        if x in trainmap:
            trainlist.append(trainmap[x])    
        
    trainstr=""
    for x in trainlist:
        trainstr+=" "+str(x)
    
    node_num=0
    edge_num=0
    if dataset_name == "ogbnarxiv":
        node_num=87599
        edge_num=825665       
    if dataset_name == "reddit":
        node_num=151701
        edge_num=10904939        
    if dataset_name == "amazon":
        node_num=1255968
        edge_num=169689928    
    print('train: ', len(trainlist), trainlist[0])     
    feature_train=PPR.ppr(dataset_str+'_train', node_num, edge_num, level, lamb, alpha, epsilon, rr, len(trainlist), trainstr[1:], opt)    
    feature_train = torch.FloatTensor(np.array(feature_train))

    print(feature_train.shape)
    #print('feat: ',feature_train[0][:10])        

    # whole graph
    nodelist=[x for x in idx_val]+ [x for x in idx_test]
    nodestr=""
    for x in nodelist:
        nodestr+=" "+str(x)

    if dataset_name == "ogbnarxiv":
        node_num=169343
        edge_num=2484941  
    if dataset_name == "reddit":
        node_num=232965
        edge_num=23446803        
    if dataset_name == "amazon":
        node_num=1569960
        edge_num=264339468   
    #print('val+test: ', len(nodelist))    
    features_valtest=PPR.ppr(dataset_str, node_num, edge_num, level, lamb, alpha, epsilon, rr, len(nodelist), nodestr[1:], opt)

    features_valtest = torch.FloatTensor(np.array(features_valtest))


    label_train=labels[idx_train]
    label_valtest=labels[nodelist]
    del labels
    gc.collect()
    label_train=torch.LongTensor(label_train)
    label_valtest=torch.LongTensor(label_valtest)


    return feature_train, features_valtest, label_train, label_valtest, len(idx_val)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def muticlass_f1(output, labels):
    preds = output.max(1)[1]  
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    return micro

def mutilabel_f1(y_true, y_pred):
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = 0
    return f1_score(y_true, y_pred, average="micro")
