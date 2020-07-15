import torch.nn as nn
from dgl.graph import DGLGraph
from dgl.nn.pytorch import GatedGraphConv
import torch
import dgl
import numpy as np


class Devign(nn.Module):
    def __init__(self, vocab_dict, embedding_dim, embedding_tensor, device, n_steps=6, n_etypes=4, class_num=2, dropout=0.5):
        super(Devign, self).__init__()
        vocab_size = len(vocab_dict)
        self.vocab_dict = vocab_dict
        self.device = device
        # use esisting embedding vectors
        if torch.is_tensor(embedding_tensor):
            self.encoder = nn.Embedding(vocab_size, embedding_dim, _weight=embedding_tensor)
            self.encoder.weight.requires_grad = False
        else:
            self.encoder = nn.Embedding(vocab_size, embedding_dim)

        self.embedding_dim = embedding_dim
        self.n_etypes = n_etypes
        self.criterion = nn.CrossEntropyLoss()

        self.ggnn = \
            GatedGraphConv(self.embedding_dim * 2, self.embedding_dim * 2, n_steps,  n_etypes,)

        self.dropout = nn.Dropout(dropout)

        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv1d(1, 1, 3)
        torch.nn.init.xavier_uniform(self.conv_1.weight, gain=1)
        self.pool_1 = nn.MaxPool1d(kernel_size=3, stride=2,)
        self.conv_2 = nn.Conv1d(1, 1, 1)
        torch.nn.init.xavier_uniform(self.conv_2.weight, gain=1)
        self.pool_2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.class_num = class_num
        self.linear_1 = nn.Linear(99, 2)
        self.linear_2 = nn.Linear(49, 2)

        self.softmax = nn.Softmax()
        self.name = 'devign'

    def get_localvec(self, n_list, device):
        local = []
        for node in n_list:
            node_local = []
            for n in node:
                emb = self.embedding_node(n)
                node_local.append(torch.tensor(emb))
            node_local = torch.stack(node_local).to(device)
            local.append(node_local)  # normalize
        return local


    def embedding_node(self, node): #ToDo add label embedding
        codes = node['code'].split()
        _type = '__' + node['type'] + '__'
        c_vec = torch.zeros(self.embedding_dim, device=self.device)
        for tk in codes:
            if tk in self.vocab_dict:
                tk = self.vocab_dict[tk]
            else:
                tk = 0
            c_vec += self.encoder(torch.tensor(tk, device=self.device))
        if len(codes) != 0:
            c_vec = c_vec / len(codes)
        if _type in self.vocab_dict:
            _type = 0
        else:
            _type = self.vocab_dict[_type]
        t_vec = self.encoder(torch.tensor(_type, device=self.device))
        return torch.cat((c_vec, t_vec), dim=0)

    def calRes(self, batch_graph, y, device):
        e_type = batch_graph.edata['type'].to(device)
        local = batch_graph.ndata['tk'].to(device)
        feature = self.ggnn(batch_graph, local, e_type)
        #feature = self.dropout(feature)
        num = len(local)

        global_feature = torch.cat([local, feature], dim=1).view([num, -1, 1])
        global_feature = global_feature.permute(0,2,1)
        local = local.view([num, -1, 1])
        local = local.permute(0,2,1)

        z_vec = self.conv_1(global_feature)
        z_vec = self.relu(z_vec)
        z_vec = self.pool_1(z_vec)
        z_vec = self.conv_2(z_vec)
        z_vec = self.relu(z_vec)
        z_vec = self.dropout(z_vec)
        z_vec = self.pool_2(z_vec).view([num, -1])

        y_vec = self.conv_1(local)
        y_vec = self.relu(y_vec)
        y_vec = self.dropout(y_vec)
        y_vec = self.pool_1(y_vec)
        y_vec = self.conv_2(y_vec)
        y_vec = self.relu(y_vec)
        y_vec = self.dropout(y_vec)
        y_vec = self.pool_2(y_vec).view([num, -1])

        z_vec = self.linear_1(z_vec)
        y_vec = self.linear_2(y_vec)

        res = (z_vec * y_vec)
        batch_graph.ndata['res'] = res
        res = dgl.mean_nodes(batch_graph, 'res')
        loss = self.criterion(res, y)
        res = self.softmax(res)
        value, predicted = torch.max(res, dim = 1)
        return value, predicted, loss

    def forward(self, graph, y, device):
        y = torch.tensor(y, dtype=torch.long, device=device)
        N, A = map(list, zip(*graph))
        local = self.get_localvec(N, device)
        #local = [self.dropout(l) for l in local]
        batch_graph = self.perpare_dgl(A, local)
        return self.calRes(batch_graph, y, device)

    @staticmethod
    def perpare_dgl(A, local):
        dgl_list = []
        for i in range(len(local)):
            dgl_graph = DGLGraph()
            dgl_graph.add_nodes(len(local[i]))
            st, ed, e_type = np.nonzero(A[i])
            dgl_graph.add_edges(st, ed, {'type':torch.tensor(e_type)})
            dgl_graph.ndata['tk'] = local[i]
            dgl_list.append(dgl_graph)
        batched_graph = dgl.batch(dgl_list)
        return batched_graph




def testScript():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Devign(embedding_dim = 100, vocab_size = 10, n_steps = 2, n_etypes = 2)
    N = [
        torch.tensor([2,3,4,5]),   torch.tensor([2, 3, 4, 5]),  torch.tensor([2, 3, ])
    ]
    A = np.zeros([3, 3, 4])
    A[0,1,:] = 1
    A[1,2,:] = 1
    graph = [(N, A), (N, A)]
    model(graph, device)

if __name__ == '__main__':
    testScript()