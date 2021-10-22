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


adj=sp.load_npz('adj.npz')
adj=adj+adj.T
adj=adj+sp.eye(adj.shape[0])
graphsave(adj,dir=file_name+'_adj_')