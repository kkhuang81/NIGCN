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


 
def load_citation(dataset_name="cora", ome=1, tau=1, epsilon=0.01, rho=1, splitfile=""):    
    dataset_str = '/home/huangkk/data/' + dataset_name +'/'+dataset_name
    #data = np.load(dataset_str + '_5shot_0type_500_1000_labels.npz')
    #print(dataset_str)
    data = np.load(dataset_str + '_labels.npz')
    if dataset_name=="papers100M":
        labels=np.concatenate((data['train_labels'],data['val_labels'],data['test_labels']))
    else:
        labels = data['labels']
    
    '''
    idx_train=data['train_idx']
    idx_val=data['val_idx']
    idx_test=data['test_idx']

    '''
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
    
    
    features=PPR.ppr(dataset_str, node_num, edge_num, ome, tau, epsilon, rho, len(nodelist), nodestr[1:])
    #debug
    #features=PPR.ppr(dataset_str, node_num, edge_num, level, lamb, alpha, epsilon, rr, 1, nodestr[1:], opt)
    ##exit(1)
    #debug end
    features = torch.FloatTensor(np.array(features))
    '''
    #debug mode
    feat=np.load(dataset_str + '_feat64.npy')
    features=feat[nodelist]
    features = torch.FloatTensor(np.array(features))
    '''
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
    #idx_train = torch.LongTensor(idx_train)
    #idx_val = torch.LongTensor(idx_val)
    #idx_test = torch.LongTensor(idx_test)    
    return features, labels, len(idx_train), len(idx_val), len(idx_test)

def load_inductive(dataset_name="cora", ome=1, tau=1, epsilon=0.01, rho=1, splitfile=""):    
    dataset_str = '/home/huangkk/data/' + dataset_name +'/'+dataset_name
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
    if dataset_name == "yelp":
        node_num=537635
        edge_num=7949403       
    if dataset_name == "reddit":
        node_num=151701
        edge_num=10904939        
    if dataset_name == "amazon":
        node_num=1255968
        edge_num=169689928    
    print('train: ', len(trainlist), trainlist[0])   

    feature_train=PPR.ppr(dataset_str+'_train', node_num, edge_num, ome, tau, epsilon, rho, len(trainlist), trainstr[1:])    
    feature_train = torch.FloatTensor(np.array(feature_train))

    print(feature_train.shape)
    #print('feat: ',feature_train[0][:10])        

    # whole graph
    nodelist=[x for x in idx_val]+ [x for x in idx_test]
    nodestr=""
    for x in nodelist:
        nodestr+=" "+str(x)
    # feature update
    #features=PPR.ppr(dataset_str, level, alpha, epsilon, rr, 2, nodestr[3:6], seed, opt, thread)
    if dataset_name == "ogbnarxiv":
        node_num=169343
        edge_num=2484941
    if dataset_name == "yelp":
        node_num=716847
        edge_num=13954819    
    if dataset_name == "reddit":
        node_num=232965
        edge_num=23446803        
    if dataset_name == "amazon":
        node_num=1569960
        edge_num=264339468   
    #print('val+test: ', len(nodelist))    
    features_valtest=PPR.ppr(dataset_str, node_num, edge_num, ome, tau, epsilon, rho, len(nodelist), nodestr[1:])

    features_valtest = torch.FloatTensor(np.array(features_valtest))
    #print(features.shape)
    #print(features[0])
    #np.save('corafeatnew',features)

    label_train=labels[idx_train]
    label_valtest=labels[nodelist]
    del labels
    gc.collect()
    label_train=torch.LongTensor(label_train)
    label_valtest=torch.LongTensor(label_valtest)

    #labels = torch.LongTensor(labels)    
    #idx_train = torch.LongTensor(idx_train)
    #idx_val = torch.LongTensor(idx_val)
    #idx_test = torch.LongTensor(idx_test)    
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
