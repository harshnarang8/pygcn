import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, x1, adj1, x2, adj2):
        x1 = F.relu(self.gc1(x1, adj1))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x1 = self.gc2(x1, adj1)
        x1 = nn.MaxPool1d(x1.size(0))(x1.t().contiguous().view(1,x1.size(1),-1))

        x2 = F.relu(self.gc1(x2, adj2))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x2 = self.gc2(x2, adj2)
        x2 = nn.MaxPool1d(x2.size(0))(x2.t().contiguous().view(1,x2.size(1),-1))

        return torch.abs(5*self.cos(x1,x2))

        # return F.log_softmax(x1)
