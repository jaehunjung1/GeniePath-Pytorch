from dgl.data import PPIDataset
import torch
from torch.utils.data import DataLoader

from dataset import collate

import ipdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def eval(valid_loader):
    # TODO
    pass


def main(_):



    train_dataset = PPIDataset(mode='train')
    valid_dataset = PPIDataset(mode='valid')
    test_dataset = PPIDataset(mode='test')
    train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collate, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=2, collate_fn=collate, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, collate_fn=collate, shuffle=False)


    for batch_idx, batch in enumerate(train_loader):
        ipdb.set_trace()
        pass