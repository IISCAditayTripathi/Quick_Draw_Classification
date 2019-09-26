import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import itertools
import numpy.random as nprnd
import random
from model import DoodleClassifier
from DataLoaderDraw import DrawLoaderDraw, DrawDataset
parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default='processed_quick_draw_paths.pkl')
parser.add_argument('--label2idx', type=str, default='label2idx_draw.pkl')
#parser.add_argument('--label2key', type=str, default='label2key_draw.pkl')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--num_workers', type=int, default=14)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--clip_grad', type=bool, default=True)
parser.add_argument('--split_ratio', type=float, default=0.2)
parser.add_argument('--data', type=str, default='data_draw_imp.pkl')
#parser.add_argument('--key2array_idx', type=str, default='key2array_idx_draw.pkl')

args = parser.parse_args()

def valid_step(valid_loader, model, device):
    model.eval()
    loader = tqdm(valid_loader, total=len(valid_loader), unit='batches')
    running_loss = 0
    total = 0
    correct = 0
    for i_batch, data in enumerate(loader):
        valid_x = data[0].to(device)
        valid_y = data[1].to(device)

        loss = model(valid_x, valid_y)
        pred = model.pred(valid_x)
        total += valid_y.size(0)
        correct += (pred == valid_y).sum().item()

        running_loss += loss.item()
    print("Accuracy", 100* correct/total)
    return running_loss/(i_batch+1)

def prepare_labels(label2key):
    labels = list(label2key.keys())
    idx2label = {}
    label2idx = {}
    for i in tqdm(range(len(labels))):
        idx2label[i] = labels[i]
        label2idx[labels[i]] = i
    return label2idx, idx2label


def train(train_data, valid_data, test_data, gpu, batch_size, shuffle, num_workers, num_epoch, patience, clip_grad, label2idx):
    device = torch.device(gpu)

    dataset_train = DrawDataset(train_data, label2idx)
    dataset_valid = DrawDataset(valid_data, label2idx)
    #pickle.dump(label2idx, open('label2idx_draw.pkl', 'wb'))
    #pickle.dump(idx2label, open('idx2label_draw.pkl', 'wb'))

    data_loader_train = DrawLoaderDraw(dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers= num_workers)
    data_loader_valid = DrawLoaderDraw(dataset_valid, batch_size=12, num_workers=2)
    model = DoodleClassifier(num_classes=345)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    best_loss = float('inf')
    for epoch in range(num_epoch):
        for phase in ['train', 'valid']:
            if phase =='train':
                model.train()
            if phase == 'valid':
                model.eval()
            if phase == 'train':
                loader = tqdm(data_loader_train, total = len(data_loader_train), unit='batches')
                running_loss = 0
                for i_batch , data in enumerate(loader):
                    model.zero_grad()
                    train_x = data[0].to(device)
                    train_y = data[1].to(device)
                    #train_x = train_x.view(train_x.size(0),1,train_x.size(1), train_x.size(2))
                    loss = model(train_x, train_y)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    optimizer.step()
                    running_loss += loss.item()
                    loader.set_postfix(Loss=running_loss/(i_batch+1), Epoch=epoch)
            if phase == 'valid':
                 valid_loss = valid_step(valid_loader=data_loader_valid, model=model, device=device)
                 if (valid_loss < best_loss-0.001):
                     best_loss = valid_loss
                     no_update = 0
                     best_model = model.state_dict()
                     print('Validation loss decreased from previous epoch', valid_loss)
                     torch.save(best_model, "checkpoints/resnet152/best_model.pt")
                 elif (valid_loss > best_loss +  0.001) and (no_update < patience):
                     no_update += 1
                     print('validation loss increased', valid_loss)
                 elif (no_update == patience):
                     print("Model has exceeded patience. Exiting...")
                     exit(0)
                 if epoch == num_epoch - 1:
                     print("Model has reached final epoch.  Stoping and saving model.")
                     torch.save(model.state_dict(), "checkpoints/resnet152/"+str(epoch)+'pt')
                     exit(0)


def prepare_dataset(key2image, key2label, label2key, split_ratio):
    train_x = []
    train_y = []

    valid_x = []
    valid_y = []
    data_size  = len(key2image)
    random.seed(1)
    key_array = list(key2image.keys())

    array = [i for i in range(data_size)]
    random.shuffle(array)
    train_data_list = array[0:int(1-split_ratio)*data_size]

    valid_data_list = array[int(1 - split_ratio)*data_size]
    for i in train_data_list:
        key = key_array[i]
        train_x.append(key2image[key])
        train_y.append(key2label[key])
    for i in valid_data_list:
        key = key_array[i]
        valid_x.append(key2image[key])
        valid_y.append(key2lable[key])

    data = {'train_x': train_x, 'train_y': train_y, 'valid_x': valid_x, 'valid_y': valid_y}

    pickle.dump(data, open('data_draw.pkl', 'wb'))
    return train_x, train_y, valid_x, valid_y


if args.mode == 'train':
    print("Loading Data")
    data_paths = pickle.load(open(args.data_path, 'rb'))
    train_paths = data_paths['train_x']
    valid_paths = data_paths['valid_x']
    test_paths = data_paths['test_x']
    label2idx = pickle.load(open(args.label2idx, 'rb'))
    #key2label = pickle.load(open(args.key2label, 'rb'))
    #key2image = pickle.load(open(args.key2image, 'rb'))
    #label2key = pickle.load(open(args.label2key, 'rb'))
    print("Preparing training and test split.")
    #data = pickle.load(open(args.data, 'rb'))
    #key2array_idx = pickle.load(open(args.key2array_idx, 'rb'))

    #train_x, train_y, valid_x, valid_y = prepare_dataset(key2image=key2image, key2label=key2label, label2key=label2key, split_ratio=args.split_ratio)
    print("Starting Training!")
    train(train_data=train_paths,
            valid_data=valid_paths,
            test_data=test_paths,
            batch_size=args.batch_size,
            clip_grad=args.clip_grad,
            num_workers=args.num_workers,
            patience=args.patience,
            shuffle=args.shuffle,
            num_epoch=args.num_epochs,
            gpu=args.gpu,
            label2idx=label2idx
            )



