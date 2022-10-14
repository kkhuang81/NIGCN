from __future__ import division
from __future__ import print_function
from utils import load_inductive, mutilabel_f1
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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="cora",help='Dataset to use.')
parser.add_argument('--seed', type=int, default=51290, help='Random seed.')
parser.add_argument('--type', type=int, default=3, help='the type of the split')

parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--hid', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--nlayers', type=int, default=2, help='Number of hidden layers.')
parser.add_argument('--bias', default='none', help='bias.')
parser.add_argument('--batch', type=int, default=64, help='batch size')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

parser.add_argument('--omega', type=float, default=1, help='omega value')
parser.add_argument('--tau', type=float, default=1, help='neighbor range tau')
parser.add_argument('--epsilon', type=float, default=0.01, help='size of selected neighbors')
parser.add_argument('--rho', type=float, default=1, help='the power index')

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
        batch_x = batch_x.cuda(args.dev)
        batch_y = batch_y.cuda(args.dev)
        t1 = time.time()
        optimizer.zero_grad()
        output = model(batch_x)
        loss_train = loss_fn(output, batch_y)
        loss_train.backward()
        optimizer.step()
        time_epoch+=(time.time()-t1)
        loss_list.append(loss_train.item())
    return np.mean(loss_list),time_epoch


def evaluate(model,feats,labels):
    model.eval()
    with torch.no_grad():
        logits = model(feats)
        f1_mic = mutilabel_f1(labels.cpu().numpy(),logits.cpu().numpy())
        return f1_mic


def validate():
    return evaluate(model, feature_valtest[:len_val].cuda(args.dev), label_valtest[:len_val])

def test():
    model.load_state_dict(torch.load(checkpt_file))
    return evaluate(model, feature_valtest[len_val:].cuda(args.dev), label_valtest[len_val:])    


print('Node wise version 1.0')
settings=['_5_125_250', '_10_250_500', '_15_375_750', '_20_500_1000']
#settings=['_35_875_1750', '_69_1725_3450', '_86_2150_4300', '_173_4325_8650', '_431_10775_21550']

#settings=['_5_125_250', '_10_250_500', '_15_375_750', '_20_500_1000']
training_time=[]
test_f1score=[]
for idx in range(10):
    splitfile = settings[args.type] + '_' + str(idx) + '_splits.npz'
    feature_train, feature_valtest, label_train, label_valtest, len_val = load_inductive(args.dataset, args.omega, args.tau, args.epsilon, args.rho, splitfile)

    checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'

    print("class no.: ", label_train.shape[1])

    model = GCN(nfeat=feature_train.shape[1],
                nlayers=args.nlayers,
                nhidden=args.hid,
                nclass=label_train.shape[1],
                dropout=args.dropout,
                bias = args.bias).cuda(args.dev)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_fn = nn.BCEWithLogitsLoss()

    label_train=label_train.float()
    label_valtest=label_valtest.float()


    torch_dataset = Data.TensorDataset(feature_train, label_train)
    loader = Data.DataLoader(dataset=torch_dataset,
                            batch_size=args.batch,
                            shuffle=True,
                            num_workers=0)


    train_time = 0
    bad_counter = 0
    best = 0
    best_epoch = 0

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
