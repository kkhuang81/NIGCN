import pickle as pkl
import sys
import os
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import scipy.sparse as sp
import sklearn.preprocessing
import json
import time
import scipy.sparse
import struct
from sklearn.preprocessing import StandardScaler

file_name=__
feats=np.load('rawfeats.npy')

feats=np.array(feats,dtype=np.float64)
np.save('feats.npy',feats)

data=np.load(file_name+'_labels.npz')

files = []
for x in data.files:
    files.append(data[x])
labels, idx_train, idx_val, idx_test = tuple(files)

train_feats=feats[idx_train]

scaler = StandardScaler()
scaler.fit(train_feats)
feats = scaler.transform(feats)

train_feats=feats[idx_train]

np.save(file_name+'_feat.npy',feats)
np.save(file_name+'_train_feat.npy',train_feats)