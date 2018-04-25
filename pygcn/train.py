from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
from pygcn.models import GCN
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from pygcn.utils import load_data, accuracy
import loadfile
# from loadfile import loadData, MAX_LEN, EMBEDDING_SIZE

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=100,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden2', type=int, default = 25,
                    help='Number of hidden units in the second layer.')

# MAX_LEN = 25
# EMBEDDING_SIZE = 100

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# features, labels, idx_train, idx_val, idx_test = load_data()
adj1, adj2, x1, x2, y = loadfile.loadData('../../stsbenchmark/sts-train.csv')
adj1dev, adj2dev, x1dev, x2dev, ydev = loadfile.loadData('../../stsbenchmark/sts-dev.csv')
adj1test, adj2test, x1test, x2test, ytest = loadfile.loadData('../../stsbenchmark/sts-test.csv')

# Model and optimizer
model = GCN(nfeat=loadfile.EMBEDDING_SIZE,
            nhid=args.hidden,
            nclass=args.hidden2,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                      lr=args.lr, weight_decay=args.weight_decay)
# optimizer = optim.SGD(model.parameters(), lr = args.lr)

print("Model Initialised")
"""
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
"""
if args.cuda:
    model.cuda()
    for i in tqdm(range(len(adj1))):
        adj1[i] = adj1[i].cuda()
        adj2[i] = adj2[i].cuda()
        x1[i] = x1[i].cuda()
        x2[i] = x2[i].cuda()
    y = y.cuda()
    for i in tqdm(range(len(adj1test))):
        adj1test[i] = adj1test[i].cuda()
        adj2test[i] = adj2test[i].cuda()
        x1test[i] = x1test[i].cuda()
        x2test[i] = x2test[i].cuda()
    ytest = ytest.cuda()
print("Model Step CUDa")

# features, labels = Variable(features), Variable(labels)
for i in tqdm(range(len(adj1))):
    adj1[i] = Variable(adj1[i])
    adj2[i] = Variable(adj2[i])
    x1[i] = Variable(x1[i])
    x2[i] = Variable(x2[i])
y = Variable(y)
for i in tqdm(range(len(adj1test))):
    adj1test[i] = Variable(adj1test[i])
    adj2test[i] = Variable(adj2test[i])
    x1test[i] = Variable(x1test[i])
    x2test[i] = Variable(x2test[i])
ytest = Variable(ytest)
print("Model Variables ready")
"""
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data[0]),
          'acc_train: {:.4f}'.format(acc_train.data[0]),
          'loss_val: {:.4f}'.format(loss_val.data[0]),
          'acc_val: {:.4f}'.format(acc_val.data[0]),
          'time: {:.4f}s'.format(time.time() - t))
"""
# assuming (x1, ag1, ag2, x2, y) are the training tuples
# loss_function = F.mse_loss()
f = open("loss.txt","w+")
def myTrain(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    currentLoss = 0
    # print("Hello: ", len(x1))
    for i in tqdm(range(len(x1) - 1)):
        # print("I ",i)
        # print(x1[i].data.size(), adj1[i].data.size())
        # optimizer.zero_grad()
        output = model(x1[i], adj1[i], x2[i], adj2[i])
        loss_train = F.l1_loss(output, y[i])
        loss_train.backward()
        currentLoss += loss_train.data[0]
        # optimizer.step()
        # if (i%500 == 249):
            # print("some loss statistic: ", currentLoss)
            # currentLoss = 0
    optimizer.step()
    print("Epoch:{:04d} over.".format(epoch+1),
          "time: {:.4f}s".format(time.time() - t),
          "Loss this Epoch: {:.4f}.".format(currentLoss))
    f.write(str(epoch)+"\t"+str(currentLoss)+"\n")
"""
def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data[0]),
          "accuracy= {:.4f}".format(acc_test.data[0]))
"""
def myTest():
    f = open("file.txt","w+")
    g = open("file2.txt","w+")
    model.eval()
    for i in tqdm(range(len(x1test) - 1)):
        output = model(x1test[i], adj1test[i], x2test[i], adj2test[i])
        f.write(str(output.data.cpu().numpy()[0]) + "\n")
        g.write(str(ytest.data.cpu().numpy()[i]) + "\n")


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    myTrain(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
myTest()
