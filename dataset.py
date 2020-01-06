import torch


def collate(samples):
    """
    Each sample contains (DGLGraph, ndarray) where ndarray denotes labels for each node
    The node features can be accessed through DGLGraph.ndata['feat'] => sized (num_nodes, 50)
    The labels are ndarray => sized (num_nodes, 121)
    """
    graphs, labels = map(list, zip(*samples))
    feats = torch.cat(list(map(lambda x: x.ndata['feat'], graphs)), dim=0)  # (sum of num_nodes in batch, 50)
    labels = torch.cat(list(map(torch.from_numpy, labels)), dim=0)  # (sum of num_nodes in batch, 121)
    return graphs, feats, labels

