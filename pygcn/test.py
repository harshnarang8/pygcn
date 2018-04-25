import torch
from models import GCN
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

def test1():
  model = GCN(nfeat=100,nhid=50,nclass=25,dropout=0.5)
  # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
  optimizer = optim.SGD(model.parameters(), lr=0.05)
  # optimizer.zero_grad()
  i = []
  k = []
  for j in range(0,12):
    i.append([j,j])
    # i.append([0,j,j])
    k.append(1)
    # k.append(1)
  adj1 = [Variable(torch.sparse.FloatTensor(torch.LongTensor(i).t(), torch.FloatTensor(k), torch.Size([12,12])))]
  adj2 = [Variable(torch.sparse.FloatTensor(torch.LongTensor(i).t(), torch.FloatTensor(k), torch.Size([12,12])))]
  input1 = [Variable(torch.randn(12,100))]
  input2 = [Variable(torch.randn(12,100))]
  y = [Variable(torch.FloatTensor([2.5]))]
  # print(adj1.data.to_dense())
  model.train()
  for i in range(1):
    optimizer.zero_grad()
    output = model(input1[0], adj1[0], input2[0], adj2[0])
    # if (i%10 == 0):
      # print(output.data.numpy()[0])
    loss_train = F.l1_loss(output, y[0])
    t = loss_train.cuda()
    z = loss_train.data
    print(z.numpy()[0])
    loss_train.backward()
    optimizer.step()
  # optimizer.zero_grad()
  # output = model(input1, adj1, input2, adj2)
  # print(output.data.numpy()[0])
  model.eval()
  output = model(input1[0], adj1[0], input2[0], adj2[0])
  print(output.data.numpy()[0])

def test2():
  x = torch.sparse.FloatTensor(5,5)
  y = torch.FloatTensor(5,5)
  torch.mm(x,y)
  torch.mm(Variable(x),Variable(y))

if __name__=="__main__":
  test1()
