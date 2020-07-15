from scripts.uncertainty import Vinalla, MCDropout, Mutation
import os
import numpy as np
import torch
import torch.optim as optim
import argparse
import random
from scripts.utils import IS_DEBUG
from scripts.utils import TokenMap, CodeReader, get_dataloader, get_model, common_metric, get_commontoken
import datetime
import matplotlib.pyplot as plt
from gensim.models import word2vec
from scripts.devign_model import Devign
from scripts.utils import common_predict
import warnings
warnings.filterwarnings('ignore')

class BestThresh:
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

        self.train_ground = UncertaintyMetric.get_ground(model, train_loader, device)
        self.test_ground = UncertaintyMetric.get_ground(model, test_loader, device)

        self.metric = Vinalla(self.model, self.train_loader, self.test_loader,
                    self.train_ground, self.test_ground, self.device)

    def find_optimization(self):
        best_s, best_acc = -1000, 0
        for s in self.metric.train_score:
            is_correct = self.metric.train_score > s
            corr_index = np.where(is_correct > 0)
            err_index = np.where(is_correct == 0)
            predict = self.metric.train_ground
            predict[err_index] = 1 - predict[err_index]
            acc = np.sum(predict) / len(predict)
            if acc > best_acc:
                best_acc = acc
                best_s = s
        is_correct = self.metric.test_score > best_s
        corr_index = np.where(is_correct > 0)
        err_index = np.where(is_correct == 0)
        predict = self.metric.test_ground
        predict[err_index] = 1 - predict[err_index]
        acc = np.sum(predict) / len(predict)
        print(acc)
        return acc


class UncertaintyMetric:
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

        self.train_ground = self.get_ground(model, train_loader, device)
        self.test_ground = self.get_ground(model, test_loader, device)

        self.metric = []

        v = MCDropout(self.model, self.train_loader, self.test_loader,
                    self.train_ground, self.test_ground, self.device)
        self.metric.append(v)

        v = Mutation(self.model, self.train_loader, self.test_loader,
                    self.train_ground, self.test_ground, self.device)
        self.metric.append(v)

        v = Vinalla(self.model, self.train_loader, self.test_loader,
                    self.train_ground, self.test_ground, self.device)
        self.metric.append(v)


    @staticmethod
    def get_ground(model, data_loader, device):
        predict_score, predict_label, ground_y \
            = common_predict(model, data_loader, device)
        return predict_label == ground_y

    def run(self):
        r_list = [0.7, 0.8, 0.9, 0.95, 0.99]
        for requirement in r_list:
            print('requirement is %0.3f' % requirement)
            cover = []
            for v in self.metric:
                acc, cover_index = v.get_coverindex(requirement)
                cover.append(cover_index)
                print(v.__class__.__name__, 'true accuracy is', acc)


def main():
    train_path = "../../data/quem_cfg"
    test_path = "../../data/quem_cfg"
    model_path = "../../models/devign.h5"
    device = torch.device("cuda:" + str(2) if torch.cuda.is_available() else "cpu")

    train_dataset = CodeReader(train_path, device)
    test_dataset = CodeReader(test_path, device)
    trainset = train_dataset.get_data()[:2500]
    train_loader = get_dataloader(trainset, device, batch_size=1000)
    valset = test_dataset.get_data()[2500:]
    val_loader = get_dataloader(valset, device, batch_size=1000)
    w2v = word2vec.Word2Vec.load('../../models/w2v.model')
    model = Devign(100, w2v).to(device)
    model.load_state_dict(torch.load(model_path))
    uncertainty = UncertaintyMetric(model, train_loader, val_loader, device)
    uncertainty.run()
    opt = BestThresh(model, train_loader, val_loader, device)
    opt.find_optimization()


if __name__ == '__main__':
    main()