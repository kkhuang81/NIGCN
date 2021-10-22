from __future__ import division
from __future__ import print_function
from utils import load_citation, muticlass_f1
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from models import GCN
import uuid
import os

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="cora",help='Dataset to use.')

parser.add_argument('--seed', type=int, default=51290, help='Random seed.')
parser.add_argument('--type', type=int, default=0, help='the type of the split')
parser.add_argument('--idx', type=int, default=0, help='the index of the split')

parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 loss on parameters).')

parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--hid', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--nlayers', type=int, default=2, help='Number of hidden layers.')

parser.add_argument('--bias', default='none', help='bias.')
parser.add_argument('--batch', type=int, default=64, help='batch size')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--lamb', type=float, default=0, help='lambda value')
parser.add_argument('--alpha', type=float, default=0.1, help='stop probability at each step')
parser.add_argument('--epsilon', type=float, default=0.01, help='residual pagerank value')
parser.add_argument('--rr', type=float, default=0.5, help='the power index of D')
parser.add_argument('--level', type=int, default=6, help='the maximum level')
parser.add_argument('--no-opt', dest='opt', action='store_false', help='Lazy update optimization')
parser.set_defaults(opt=True)

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print("--------------------------")
print(args)

def train():
    model.train()
    loss_list = []
    time_epoch = 0

    for step, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.cuda(args.dev)
        batch_y = batch_y.cuda(args.dev)
        t1 = time.time()
        optimizer.zero_grad()
        output = model(batch_x)
        loss_train = loss_fn(output, batch_y)
        loss_train.backward()
        optimizer.step()
        time_epoch += (time.time() - t1)
        loss_list.append(loss_train.item())
        del batch_x
        del batch_y	
        del output
        torch.cuda.empty_cache()
		
    return np.mean(loss_list), time_epoch

def validate(labels):
	model.eval()	
	out_list = []
	labels=labels.to(args.dev)

	with torch.no_grad():
		for step, batch_x in enumerate(val_loader):
			batch_x=batch_x[0]
			batch_x = batch_x.cuda(args.dev)			
			output = model(batch_x)
			out_list.append(output)		
			del batch_x
			torch.cuda.empty_cache()
		final=out_list[0]
		for i in range(1,len(out_list)):
			final=torch.vstack((final, out_list[i]))
		micro_val = muticlass_f1(final, labels)		
		del labels
		del final
		torch.cuda.empty_cache()
		return micro_val.item()

def test(labels):
	model.load_state_dict(torch.load(checkpt_file))
	model.eval()
	out_list = []
	labels=labels.to(args.dev)

	with torch.no_grad():
		for step, (batch_x) in enumerate(test_loader):
			batch_x=batch_x[0]
			batch_x = batch_x.cuda(args.dev)			
			output = model(batch_x)
			out_list.append(output)		
			del batch_x
			torch.cuda.empty_cache()
		final=out_list[0]
		for i in range(1,len(out_list)):
			final=torch.vstack((final, out_list[i]))
		micro_val = muticlass_f1(final, labels)		
		del labels
		del final
		torch.cuda.empty_cache()
		return micro_val.item()

settings=['_5_125_250', '_10_250_500', '_15_375_750', '_20_500_1000']
#for setting in settings:

splitfile = settings[args.type] + '_' + str(args.idx) + '_splits.npz'
#t=time.time()
features, labels, len_train, len_val, len_test = load_citation(args.dataset, args.lamb, args.alpha, args.epsilon, args.level, args.rr, args.opt, splitfile)
print(features.shape, labels.shape, len_train, len_val, len_test)

checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nlayers=args.nlayers,
            nhidden=args.hid,
            nclass=labels.max().item() + 1,
            dropout=args.dropout,
            bias = args.bias).cuda(args.dev)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

loss_fn = nn.CrossEntropyLoss()

torch_dataset = Data.TensorDataset(features[:len_train], labels[:len_train])
train_loader = Data.DataLoader(dataset=torch_dataset,batch_size=args.batch,shuffle=True,num_workers=0)

torch_dataset = Data.TensorDataset(features[len_train:len_train+len_val])
val_loader = Data.DataLoader(dataset=torch_dataset,batch_size=args.batch,shuffle=False,num_workers=0)

torch_dataset = Data.TensorDataset(features[len_train+len_val:])
test_loader = Data.DataLoader(dataset=torch_dataset,batch_size=args.batch,shuffle=False,num_workers=0)



train_time = 0
bad_counter = 0
best = 0
best_epoch = 0

for epoch in range(args.epochs):
    loss_tra,train_ep = train()
    train_time+=train_ep

    f1_val = validate(labels[len_train:len_train+len_val])

    if(epoch+1)%100 == 0:
        print('Epoch:{:04d}'.format(epoch+1),
            'train',
            'loss:{:.3f}'.format(loss_tra),
            '| val',
            'acc:{:.3f}'.format(f1_val),
            '| cost:{:.3f}'.format(train_time))
    if f1_val > best:
        best = f1_val
        best_epoch = epoch
        torch.save(model.state_dict(), checkpt_file)
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break


f1_test = test(labels[len_train+len_val:])
print("Train cost: {:.4f} s".format(train_time))
print('Load {}th epoch'.format(best_epoch))
print("Test f1: {:.3f}".format(f1_test))
print("--------------------------")
os.remove(checkpt_file)
