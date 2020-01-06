from absl import app
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils import tensorboard
from dgl.data import PPIDataset
from sklearn import metrics
import numpy as np

from flags import FLAGS
from dataset import collate
from model import GeniePath

import ipdb

device = torch.device('cuda' if torch.cuda.is_available() and FLAGS.use_gpu else 'cpu')


def eval(model: nn.Module, data_loader: DataLoader, summary_writer: tensorboard.SummaryWriter, epoch_id: int):
    model.eval()

    correct = 0
    micro_f1 = 0.0
    num_entry = 0
    num_batch = 0
    for batch in data_loader:
        graph, feats, labels = batch
        feats = feats.to(device)

        out = model(graph, feats)
        pred = (out > 0).float().cpu()  # out > 0 means p > 0.5 in BCEWithLogitsLoss

        correct += (pred == labels).sum()
        micro_f1 += metrics.f1_score(labels, pred, average='micro')

        num_entry += np.prod(pred.size())
        num_batch += 1

    summary_writer.add_scalar("Metrics/eval_acc", correct / num_entry, global_step=epoch_id)
    summary_writer.add_scalar("Metrics/eval_micro_f1", micro_f1 / num_batch, global_step=epoch_id)
    pass


def main(_):
    lr = FLAGS.lr
    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    val_batch_size = FLAGS.val_batch_size
    hidden_dim = FLAGS.hidden_dim
    num_layers = FLAGS.num_layers

    train_dataset = PPIDataset(mode='train')
    valid_dataset = PPIDataset(mode='valid')
    # test_dataset = PPIDataset(mode='test')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=val_batch_size, collate_fn=collate, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=val_batch_size, collate_fn=collate, shuffle=False)

    model = GeniePath(50, 121, gat_dim=hidden_dim, lstm_hidden_dim=hidden_dim,
                      num_layers=num_layers, device=device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    summary_writer = tensorboard.SummaryWriter(log_dir=FLAGS.tensorboard_path)

    start_epoch_id = 1
    step = 0
    # TODO best score tracking, saving checkpoints, test code

    # Training Phase
    for epoch_id in tqdm(range(start_epoch_id, epochs + 1)):
        model.train()
        for batch in train_loader:
            graph, feats, labels = batch
            feats = feats.to(device)

            out = model(graph, feats)
            optimizer.zero_grad()
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            step += 1

            summary_writer.add_scalar("Loss/train", loss.detach().cpu().numpy(), global_step=step)

        if epoch_id % 5 == 0:
            eval(model, valid_loader, summary_writer, epoch_id)


if __name__ == "__main__":
    app.run(main)
