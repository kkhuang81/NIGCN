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
#parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=51290, help='Random seed.')

'''
parser.add_argument('--trainnum', type=int, default=20, help='number of train number')
parser.add_argument('--valnum', type=int, default=500, help='number of validation number')
parser.add_argument('--testnum', type=int, default=1000, help='number of test number')
parser.add_argument('--seq', type=int, default=0, help='the sequence of splits')
'''
parser.add_argument('--type', type=int, default=3, help='the type of the split')

parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
#parser.add_argument('--l1', type=float, default=0.05, help='Weight decay (L1 loss on parameters).')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--hid', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--nlayers', type=int, default=2, help='Number of hidden layers.')
#parser.add_argument('--smoo', type=float, default=0.5,help='Smooth for Res layer')
parser.add_argument('--bias', default='none', help='bias.')
parser.add_argument('--batch', type=int, default=64, help='batch size')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

parser.add_argument('--omega', type=float, default=1, help='omega value')
parser.add_argument('--tau', type=float, default=1, help='neighbor range tau')
parser.add_argument('--epsilon', type=float, default=0.01, help='size of selected neighbors')
parser.add_argument('--rho', type=float, default=1, help='the power index')

#parser.add_argument('--level', type=int, default=6, help='the maximum level')
#parser.add_argument('--no-opt', dest='opt', action='store_false', help='Lazy update optimization')
#parser.set_defaults(opt=True)

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
    for step, (batch_x, batch_y) in enumerate(loader):
        # batch_x = batch_x.cuda(args.dev)
        # batch_y = batch_y.cuda(args.dev)
        t1 = time.time()
        optimizer.zero_grad()
        output = model(batch_x)
        loss_train = loss_fn(output, batch_y)
        loss_train.backward()
        optimizer.step()
        time_epoch+=(time.time()-t1)
        loss_list.append(loss_train.item())
    return np.mean(loss_list),time_epoch

def validate():
    model.eval()
    with torch.no_grad():
        output = model(features[len_train:len_train+len_val])
        micro_val = muticlass_f1(output, labels[len_train:len_train+len_val])
        return micro_val.item()

def test():
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(features[len_train+len_val:])
        micro_test = muticlass_f1(output, labels[len_train+len_val:])
        return micro_test.item()


# Load data
#adj, features, labels, idx_train, idx_val, idx_test = load_data()
print('Node wise version 1.0')
settings=['_5_125_250', '_10_250_500', '_15_375_750', '_20_500_1000']
if args.dataset=='papers100M':
    settings=['_250_6250_12500', '_500_12500_25000', '_750_18750_37500', '_1000_25000_50000']
#for setting in settings:
training_time=[]
test_f1score=[]
#for idx in range(10):
for idx in range(10):
    splitfile = settings[args.type] + '_' + str(idx) + '_splits.npz'
    #t=time.time()
    features, labels, len_train, len_val, len_test = load_citation(args.dataset, args.omega, args.tau, args.epsilon, args.rho, splitfile)
    #len_train, len_val, len_test=len(idx_train), len(idx_val), len(idx_test)
    #print('Precompute time ', time.time()-t) we have our own precompute time
    #np.save('corafeatnofilter',features)
    #np.save('corafeatopt',features)
    #quit()
    #features=torch.FloatTensor(np.load('../GBP/pubfeat.npz')['arr_0.npy'])
    #print('Train length ', len_train, idx_train[0], ' ', idx_train[-1])
    #print('Val length ', len_val, idx_val[0], ' ', idx_val[-1])
    #print('Test length ', len_test, idx_test[0], ' ', idx_test[-1])

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


    features = features.cuda(args.dev)
    labels = labels.cuda(args.dev)

    torch_dataset = Data.TensorDataset(features[:len_train], labels[:len_train])
    loader = Data.DataLoader(dataset=torch_dataset,
                            batch_size=args.batch,
                            shuffle=True,
                            num_workers=0)



    train_time = 0
    bad_counter = 0
    best = 0
    best_epoch = 0

    # when the trianing data is not sufficient enough, set the patience value small in case of overfitting.
    for epoch in range(args.epochs):
        loss_tra,train_ep = train()
        f1_val = validate()
        train_time+=train_ep
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


    f1_test = test()
    print("Train cost: {:.4f}s".format(train_time))
    print('Load {}th epoch'.format(best_epoch))
    print("Test f1:{:.3f}".format(f1_test))
    print("--------------------------")

    training_time.append(train_time)
    test_f1score.append(f1_test)
    os.remove(checkpt_file)

print("avg_train_time: {:.4f} s".format(np.mean(training_time)))
print("std_train_time: {:.4f} s".format(np.std(training_time)))
print("avg_f1_score: {:.4f}".format(np.mean(test_f1score)))
print("std_f1_score: {:.4f}".format(np.std(test_f1score)))
