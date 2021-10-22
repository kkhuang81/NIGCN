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


def load_data(dataset_path,prefix, normalize=True):
	adj_full = scipy.sparse.load_npz('{}/{}/adj_full.npz'.format(dataset_path,prefix)).astype(np.bool)
	adj_train = scipy.sparse.load_npz('{}/{}/adj_train.npz'.format(dataset_path,prefix)).astype(np.bool)
	role = json.load(open('{}/{}/role.json'.format(dataset_path,prefix)))
	feats = np.load('{}/{}/feats.npy'.format(dataset_path,prefix))
	class_map = json.load(open('{}/{}/class_map.json'.format(dataset_path,prefix)))
	class_map = {int(k):v for k,v in class_map.items()}
	assert len(class_map) == feats.shape[0]
	# ---- normalize feats ----
	adj_train=adj_train+sp.eye(adj_train.shape[0])
	adj_full=adj_full+sp.eye(adj_full.shape[0])

	train_nodes = np.array(list(set(adj_train.nonzero()[0])))
	train_feats = feats[train_nodes]
	scaler = StandardScaler()
	scaler.fit(train_feats)
	feats = scaler.transform(feats)
	# -------------------------
	num_vertices = adj_full.shape[0]
	if isinstance(list(class_map.values())[0],list):
		num_classes = len(list(class_map.values())[0])
		class_arr = np.zeros((num_vertices, num_classes))
		for k,v in class_map.items():
			class_arr[k] = v
	else:
		num_classes = max(class_map.values()) - min(class_map.values()) + 1
		class_arr = np.zeros((num_vertices, num_classes))
		offset = min(class_map.values())
		for k,v in class_map.items():
			class_arr[k][v-offset] = 1

	node_train = np.array(role['tr'])
	node_val = np.array(role['va'])
	node_test = np.array(role['te'])
	train_feats = feats[node_train]
	adj_train = adj_train[node_train,:][:,node_train]
	labels = class_arr
	return adj_full, adj_train, feats, train_feats, labels, node_train, node_val, node_test

def graphsaint(datastr,dataset_name):
	dataset_path='/data/'+dataset_name
	adj_full, adj_train, feats, train_feats, labels, idx_train, idx_val, idx_test = load_data(datastr,dataset_name)				
	feats=np.array(feats,dtype=np.float64)
	train_feats=np.array(train_feats,dtype=np.float64)
	np.save(dataset_path+'_feat.npy',feats)
	np.save(dataset_path+'_train_feat.npy',train_feats)
	np.savez(dataset_path+'_labels.npz',labels=labels,idx_train=idx_train,idx_val=idx_val,idx_test=idx_test)	

if __name__ == "__main__":
	#Your file storage path. For example, this is shown below.
	datastr="/home/XXX/"

	#dataset name, amazon or reddit
	dataset_name='amazon'
	graphsaint(datastr,dataset_name)

