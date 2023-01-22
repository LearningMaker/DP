import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import torchvision.transforms as transforms
import h5py
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Normalization(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        _range = np.max(input, axis=0) - np.min(input, axis=0) + 1e-7
        input = (input - np.min(input, axis=0)) / _range
        return input


class Resize(torch.nn.Module):
    def __init__(self, length):
        super().__init__()
        self.length = length

    def forward(self, inputs):
        out = [inputs[int(i * inputs.shape[0] / self.length)] for i in range(self.length)]
        out = np.array(out).transpose([1, 0])
        return out


class ToTargets(torch.nn.Module):
    def __init__(self, mode, label):
        super().__init__()
        self.mode = mode
        self.label = label

    def forward(self, targets):
        if self.mode == 'kernel_size':
            targets = targets[self.label]
            targets = (targets - 1) / 2
            targets = targets - 1 if targets == 3 else targets
        elif self.mode == 'stride':
            targets = targets[self.label]
            targets = targets - 1
        return targets


class Rapl(torch.utils.data.Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        (self.feature, self.label) = data
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        feat, lab = self.feature[index], self.label[index]
        feat = self.transform(feat) if self.transform is not None else feat
        lab = self.target_transform(lab) if self.target_transform is not None else lab
        return feat, lab

    def __len__(self):
        return len(self.feature)


class RaplLoader(object):
    def __init__(self, batch_size, mode, num_workers=0):
        self.label = {'in_channels': 0, 'out_channels': 1, 'kernel_size': 2,
                      'stride': 3, 'padding': 4, 'dilation': 5,
                      'groups': 6, 'input_size': 7, 'output_size': 8}[mode]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = {'out_channels': 6, 'kernel_size': 3, 'stride': 2}[mode]
        self.data = self.preprocess()

        self.transform = transforms.Compose([
            Normalization(),
            Resize(1024),
        ])
        self.target_transform = transforms.Compose([
            ToTargets(mode, self.label),
        ])

    def preprocess(self):
        x, y = [], []
        data = h5py.File('../datasets/data.h5', 'r')
        hp_ = h5py.File('../datasets/hp.h5', 'r')

        for k in data['data'].keys():
            d = data['data'][k][:]
            pos = data['position'][k][:]
            hp = hp_[k][:]
            hp = hp[hp[:, 0] != -1]
            hp = hp[hp[:, -1] != -1]
            bs = np.ones_like(hp)[:, 0:1]*128
            hp = np.concatenate([hp, bs], axis=1)

            hp_index = 0
            for (i, j) in pos:
                if d[:, -1][i] == 0:
                    x.append(d[i:j + 1, :-1])
                    y.append(hp[hp_index])
                    hp_index += 1
            assert hp_index == len(hp)

        return x, y

    def loader(self, data, shuffle=False, transform=None, target_transform=None):
        dataset = Rapl(data, transform=transform, target_transform=target_transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)
        return dataloader

    def get_loader(self):
        dataloader = self.loader(self.data, shuffle=True, transform=self.transform, target_transform=self.target_transform)
        return dataloader
