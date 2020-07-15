import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dgl.graph import DGLGraph
import torch
from torch.utils.data import DataLoader
import dgl
import numpy as np
import copy


class CodeDataSet(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CodeGraphLoader(DataLoader):
    def __init__(self, dataset, device, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None):
        super(CodeGraphLoader, self).__init__(
            dataset, batch_size, shuffle, sampler,
            batch_sampler, num_workers, collate_fn,
            pin_memory, drop_last, timeout,
            worker_init_fn, multiprocessing_context)
        #self.id = [data['id'] for data in dataset]
        #self.func_name = [data['name'] for data in dataset]
        self.collate_fn = self.collate_fcunc
        self.device = device

    def collate_fcunc(self, batch):
        def transfer_edge(edges):
            new_edges = np.array(edges)
            for i, edge in enumerate(new_edges):
                new_edges[i] = np.array(edge)
            return new_edges.reshape([len(edges), len(edges), -1])
        batch_graph = [(i['cfg']['nodes'], transfer_edge(i['cfg']['edges'])) for i in batch]
        batch_y = [i['label'] for i in batch]
        #batch_graph, batch_y = self.augmentation(batch_graph, batch_y)
        return batch_graph, batch_y

    def collect_tokens(self, nodes):
        change_map = {
            'var':{},
            'func':{}
        }
        for node in nodes:
            for tk in node:
                tk = tk.item()
                if tk in self.token_set['var'] and tk not in change_map['var']:
                    change_map['var'][tk] = self.token_set['var'].index(tk)
                if tk in self.token_set['func'] and tk not in change_map['func']:
                    change_map['func'][tk] =self.token_set['func'].index(tk)
        return change_map


    def noise_graph(self, graph, iteration = 10):
        res = []
        nodes, edges = graph
        change_map = self.collect_tokens(nodes)
        for i in range(iteration):
            newnodes = []
            for node in nodes:
                newnode = []
                for tk in node:
                    if tk.item() in change_map['var']:
                        index = (change_map['var'][tk.item()] + 10) % len(self.token_set['var'])
                        newnode.append(self.token_set['var'][index])
                    else:
                        newnode.append(tk.item())
                newnode = torch.tensor(newnode, dtype=torch.long)
                newnodes.append(newnode)
            res.append((newnodes, edges))
        return res


    def augmentation(self, batch_graph, batch_y):
        res_x = []
        res_y = []

        for i, graph in enumerate(batch_graph):
            new_x = self.noise_graph(graph)
            new_y = [batch_y[i]] * len(new_x)
            res_x.extend(new_x)
            res_y.extend(new_y)
        import random
        random.seed(100)
        random.shuffle(res_x)
        random.seed(100)
        random.shuffle(res_y)
        return res_x, res_y



    def perpare_dgl(self, batch_graph):
        nodes, edges = map(list, zip(*batch_graph))
        edges = [np.sum(edge,axis= 2) for edge in edges]
        dgl_list = []
        for i in range(len(nodes)):
            dgl = DGLGraph()
            dgl.add_nodes(len(nodes[i]))
            st, ed = np.nonzero(edges[i])
            dgl.add_edges(st, ed)
            dgl.ndata['tk'] = nodes[i]
            dgl_list.append(dgl)
        batched_graph = dgl.batch(dgl_list)
        return batched_graph




class CodeSquentialLoader(DataLoader):
    def __init__(self, dataset, data_type, token_set, device, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None):
        super(CodeSquentialLoader, self).__init__(
            dataset, batch_size, shuffle, sampler,
            batch_sampler, num_workers, collate_fn,
            pin_memory, drop_last, timeout,
            worker_init_fn, multiprocessing_context)
        self.file_name = [data['file_name'] for data in dataset]
        self.func_name = [data['func_name'] for data in dataset]
        self.collate_fn = self.collate_fcunc
        self.device = device
        self.data_type = data_type
        self.token_set = token_set
        self.max_len = 100

    def get_maxlength(self, batch):
        X_lengths = [[len(path) for path in paths] for paths in batch]
        max_length = max([max(length) for length in X_lengths])
        max_length = max_length if max_length < self.max_len else self.max_len
        return max_length, X_lengths

    def collate_fcunc(self, batch):
        data_batch = [i[self.data_type] for i in batch]
        label_batch = [torch.tensor(i['y'], dtype=torch.long, device=self.device) for i in batch]

        max_len, path_length = self.get_maxlength(data_batch)
        batch_size, path_num = len(data_batch), len(data_batch[0])
        padded_x = torch.zeros((batch_size, path_num,  max_len), dtype=torch.long).to(self.device)

        for i, select_paths in enumerate(path_length):
            for j, x_len in enumerate(select_paths):
                x_len = x_len if x_len < self.max_len else self.max_len
                sequence = torch.Tensor(data_batch[i][j])
                padded_x[i, j, :x_len] = sequence[:x_len]
        return padded_x, label_batch, path_length



def main():
    x = [
        [torch.Tensor([1,2,3,4,5]),torch.Tensor([1,2,3])],
        [torch.Tensor([1,2]),torch.Tensor([1])]
    ]
    device = 'cpu'
    a = CodeSquentialLoader(x, device, batch_size=2)
    for x in a:
        print()

if __name__ == '__main__':
    main()