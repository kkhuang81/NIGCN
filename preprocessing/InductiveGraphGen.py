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

def graphsave(adj,dir):
	if(sp.isspmatrix_csr(adj)):
		el=adj.indices
		pl=adj.indptr
		
		EL=np.array(el,dtype=np.uint32)
		PL=np.array(pl,dtype=np.uint32)

		EL_re=[]

		for i in range(1,PL.shape[0]):
			EL_re+=sorted(EL[PL[i-1]:PL[i]],key=lambda x:PL[x+1]-PL[x])

		EL_re=np.asarray(EL_re,dtype=np.uint32)

		print("EL:",EL_re.shape)
		f1=open(dir+'el.txt','wb')
		for i in EL_re:
			m=struct.pack('I',i)
			f1.write(m)
		f1.close()

		print("PL:",PL.shape)
		f2=open(dir+'pl.txt','wb')
		for i in PL:
			m=struct.pack('I',i)
			f2.write(m)
		f2.close()
	else:
		print("Format Error!")


file_name=__

#training graph
adj_train=sp.load_npz('adj_train.npz')

adj=adj_train.tocoo()
assert(set(adj.row)==set(adj.col))

row=np.unique(adj.row)
nodenum=row.shape[0]
mymap={}
for i in range(nodenum):
	mymap[row[i]]=i

for i in range(len(adj.row)):
	adj.row[i]=mymap[adj.row[i]]
	adj.col[i]=mymap[adj.col[i]]


adj_train=sp.coo_matrix((adj.data,(adj.row, adj.col)),shape=(nodenum, nodenum)).tocsr()

adj_train=adj_train+adj_train.T
adj_train=adj_train+sp.eye(adj_train.shape[0])

np.save(file_name+'_trainIDmap.npy', mymap)
graphsave(adj_train,dir=file_name+'_train_adj_')


#full graph
adj_full=sp.load_npz('adj_full.npz')
adj_full=adj_full+adj_full.T
adj_full=adj_full+sp.eye(adj_full.shape[0])
graphsave(adj_full,dir=file_name+'_full_adj_')