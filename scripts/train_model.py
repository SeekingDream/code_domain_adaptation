# -- coding: utf-8 --
import os
import numpy as np
import torch
import torch.optim as optim
import argparse
import random
from scripts.utils import CodeReader, get_dataloader, common_metric
import datetime
import matplotlib.pyplot as plt
from gensim.models import word2vec
from scripts.DL_model.devign_model import Devign
import warnings
warnings.filterwarnings('ignore')


def build_vocab(train_dataset):
    vocab_dict = {'__UNKNOW__' : 0, '__PAD__': 1}
    for data in train_dataset.dataset:
        x = data['cfg']
        for node in x['nodes']:
            _code, _type = node['code'], '__' + node['type'] + '__'
            _tks = _code.split()
            if _type not in vocab_dict:
                vocab_dict[_type] = len(vocab_dict)
            for tk in _tks:
                if tk not in vocab_dict:
                    vocab_dict[tk] = len(vocab_dict)
    return vocab_dict


def train_model(train_path, test_path, model_path, w2v):
    st_time = datetime.datetime.now()
    train_dataset = CodeReader(train_path, device, split_ratio=0.3)
    test_dataset = CodeReader(test_path, device)
    ed_time = datetime.datetime.now()
    print('load the dataset, cost time', ed_time - st_time)

    trainset = train_dataset.get_training()
    train_loader = get_dataloader(trainset, device, batch_size=256)
    valset = train_dataset.get_validation()
    val_loader = get_dataloader(valset, device, batch_size= 5000)
    if w2v is None:
        vocab_dict = build_vocab(train_dataset)
    else:
        vocab_dict = w2v['dict']

    print('good method number:', train_dataset.good, 'bad method number:', train_dataset.bad)
    print('taining dataset size:', len(trainset), 'validation dataset size:',len(valset))
    model = Devign(vocab_dict, 100, w2v, device, n_steps=3, n_etypes=1).to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=1e-4)

    val_losses, avg_valid_losses = [],[]
    loss_list, acc_list = [], []
    for epoch in range(args.epoch):
        running_loss = 0
        tp, fp, fn, tn, = 0, 0, 0, 0
        train_tp, train_fp, train_fn, train_tn = 0, 0, 0, 0
        model.train()
        for data in train_loader:
            y = data[1]
            x = data[0]
            optimizer.zero_grad()
            values, predicted, batch_loss = model(x, y, device)
            batch_loss.backward()
            optimizer.step()
            running_loss += batch_loss.item()
            y = torch.tensor(y, device= device)
            predicted = predicted.view([-1])
            train_tp += torch.sum(y * predicted, dtype=torch.float64)
            train_tn += torch.sum((1 - y) * (1 - predicted), dtype=torch.float64)
            train_fn += torch.sum(y * (1 - predicted), dtype=torch.float64)
            train_fp += torch.sum((1 - y) * predicted, dtype=torch.float64)
        train_precision, train_recall, train_acc, train_f1 = common_metric(train_tp, train_tn, train_fp, train_fn)
        running_loss = np.average(running_loss)
        loss_list.append(running_loss)
        acc_list.append(train_acc)
        model.eval()
        for data in val_loader:
            y = data[1]
            x = data[0]
            values, predicted, batch_loss = model(x, y, device)
            val_losses.append(batch_loss.item())
            y = torch.tensor(y, device= device).view([-1])
            predicted = predicted.view([-1])
            tp += torch.sum(y * predicted, dtype=torch.float64)
            tn += torch.sum((1 - y) * (1 - predicted),dtype=torch.float64)
            fn += torch.sum(y * (1 - predicted),dtype=torch.float64)
            fp += torch.sum((1 - y) * predicted,dtype=torch.float64)
        precision, recall, acc, f1 = common_metric(tp, tn, fp, fn)
        print(
            '[%d] loss: %.3f accuracy: %.3f precision: %.3f recall: %.3f f1: %.3f train_precision: %.3f train_recall: %.3f train_f1: %.3f' %
            (epoch + 1, running_loss, acc, precision, recall, f1, train_precision, train_recall, train_f1))

        valid_loss = np.average(val_losses)
        avg_valid_losses.append(valid_loss)
        val_losses = []

    loss_list = np.array(loss_list).reshape([-1, 1])
    acc_list = np.array(acc_list).reshape([-1, 1])
    plt.figure()
    plt.plot(loss_list)
    plt.savefig(model_path + 'loss.png')

    plt.figure()
    plt.plot(acc_list)
    plt.savefig(model_path + 'acc.png')
    res = np.concatenate([loss_list, acc_list], axis=1)
    np.savetxt(model_path + 'res.csv', res, delimiter=',')
    torch.save(model.state_dict(), model_path)


def main():
    model_dir = args.model_dir
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    train_path = args.train_path
    test_path = args.test_path
    try:
        w2v = torch.load(args.vec_path)
    except:
        w2v = None
    model_path = model_dir + 'devign' + ".h5"
    train_model(train_path, test_path, model_path, w2v)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--device', type=int, default=1, help='GPU No.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky relu.')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='epochs.')

    parser.add_argument('--train_path', type=str, default="../data/qemu_cfg", help='the directory store the data')
    parser.add_argument('--test_path', type=str, default="../data/qemu_cfg", help='the directory test the data')
    parser.add_argument('--model_dir', type=str, default="../models/", help='The target to parse')
    parser.add_argument('--vec_path', type=str, default='../models/w2v.model', help='the token map pickle')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print("========== Parameter Settings ==========")
    for arg in vars(args):
        print(arg, "=", getattr(args, arg))
    print("========== ========== ==========")

    random.seed(args.seed)
    np.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")
    main()
