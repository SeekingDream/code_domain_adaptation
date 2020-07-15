# -- coding: utf-8 --
import os
import json
from scripts.data_perparation.DataLoaderClass import *
from scripts.DL_model.devign_model import Devign
import pickle
import warnings
import torch
import random
warnings.filterwarnings('ignore')


if torch.cuda.is_available():
    IS_DEBUG = False
else:
    IS_DEBUG = True


class CodeReader:
    def __init__(self, path, device, split_ratio=0.1):
        self.dataset = []
        self.train = []
        self.val = []
        self.nodenum = []
        self.edgenum = []
        self.split_ratio = split_ratio
        self.good = 0
        self.bad = 0
        for file in os.listdir(path):
            filepath = os.path.join(path, file)
            with open(filepath, 'r') as f:
                func = json.load(f)
            if len(func['cfg']['nodes']) > 80:
                continue
            if func['label'] is None:
                continue
            if func['label'] == 1:
                self.bad += 1
            else:
                self.good += 1
            self.dataset.append(func)
            if IS_DEBUG and len(self.dataset) >= 1000:
                break
        np.random.shuffle(self.dataset)
        data_size = len(self.dataset)
        num_train, num_val = int(data_size * (1 - self.split_ratio)), int(data_size * self.split_ratio)
        train_indices = random.sample(range(num_train + num_val), num_train)
        val_indices = [i for i in range(num_train + num_val) if i not in train_indices]
        self.train = [self.dataset[i] for i in train_indices]
        self.val = [self.dataset[i] for i in val_indices]

    def old__init__(self, path, device, split_ratio=0.1):
        self.data = []
        self.train = []
        self.val = []
        self.nodenum = []
        self.edgenum = []
        self.split_ratio = split_ratio
        self.good = 0
        self.bad = 0
        for file in os.listdir(path):
            filepath = os.path.join(path, file)
            with open(filepath, 'rb') as f:
                func = pickle.load(f)
            if func['label'] is None:
                continue
            if func['label'] == 1:
                self.bad += 1
            else:
                self.good += 1
            self.data.append(func)
            if IS_DEBUG and len(self.data) >= 1000:
                break
        np.random.shuffle(self.data)
        data_size = len(self.data)
        num_train, num_val = int(data_size * (1 - self.split_ratio)), int(data_size * self.split_ratio)
        train_indices = np.random.sample(range(num_train + num_val), num_train)
        val_indices = [i for i in range(num_train + num_val) if i not in train_indices]
        self.train = [self.data[i] for i in train_indices]
        self.val = [self.data[i] for i in val_indices]

    def count_vuln(self):
        for d in self.data:
            if d['y'] == 1:
                self.bad += 1
            else:
                self.good += 1
        print('good function number is', self.good, 'bad function number is', self.bad)

    def get_training(self):
        return CodeDataSet(self.train)

    def get_validation(self):
        return CodeDataSet(self.val)

    def get_data(self):
        random.shuffle(self.dataset)
        return CodeDataSet(self.dataset)

    def get_datainfo(self):
        return None
        for d in self.data:
            self.nodenum.append(len(d['nodes']))
            self.edgenum.append(torch.sum(d['edges']).item())
        return [int(np.mean(self.nodenum)), int(np.mean(self.edgenum))]


def get_dataloader(dataset, device, batch_size=32):
    data_loader = CodeGraphLoader(dataset, device=device, batch_size=batch_size)
    return data_loader


def common_get_metric(pred, turth):
    if np.sum(pred) == 0:
        p = 0
    else:
        p = np.sum(turth * pred) / np.sum(pred)
    if np.sum(turth) == 0:
        r = 0
    else:
        r = np.sum(turth * pred) / np.sum(turth)
    acc = np.sum(turth == pred) / len(turth)
    if p + r == 0:
        f1 = 0
    else:
        f1 = p*r*2/(p+r)
    return {'accuracy':acc, 'percision':p, 'recall':r, 'F1': f1}


def common_metric(tp, tn, fp, fn):
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    if (recall + precision) != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    acc = (tp + tn) / (tp + tn + fp + fn + 1)
    return precision, recall, acc, f1



def get_commontoken(token_dict):
    res = {
        'var':[],
        'func':[]
    }
    for token in token_dict:
        if 'VAR_' in token:
            res['var'].append(token_dict[token])
        if 'FUNC_' in token:
            res['func'].append(token_dict[token])
    return res


def common_predict(model, data_loader, device):
    predict_score, predict_label, ground_y = [], [], []
    for data in data_loader:
        x, y = data[0], data[1]
        values, predicted, batch_loss = model(x, y, device)
        values = values.view([-1])
        predicted = predicted.view([-1])
        predict_score.append(values)
        predict_label.append(predicted)
        ground_y.extend(y)
    predict_score = torch.cat(predict_score).detach().cpu().numpy()
    predict_label = torch.cat(predict_label).detach().cpu().numpy()
    ground_y = torch.tensor(ground_y).detach().cpu().numpy()
    return predict_score, predict_label, ground_y



def testClass():
    tokenmap = TokenMap()
    tokenmap.load("tokenmap_sard_pdg_simin.pickle")
    data_path = "../data_repo/SARD_Preprocessed/PreprocessedTest"
    datasetname = 'CWE89_'
    with open('sard_norm.json', 'r') as f:
        norm_token = json.load(f)
    dataset = CodeReader(data_path, datasetname, tokenmap, norm_token)
    train_dataset = dataset.get_training()

    random_path = [data['random_path'] for data in train_dataset]
    print()


if __name__ == '__main__':
    testClass()
