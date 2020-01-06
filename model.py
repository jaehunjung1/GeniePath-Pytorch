import torch
import torch.nn as nn
from dgl.nn.pytorch.conv.gatconv import GATConv


class Breadth(nn.Module):
    def __init__(self, in_dim, out_dim, residual=False):
        super(Breadth, self).__init__()
        self.GATConv = GATConv(in_dim, out_dim, num_heads=1, residual=residual)

    def forward(self, graph, x):
        """
        :param graph: batched DGLGraph
        :param x: node feature tensor of size (num_nodes, feature_dim)
        :return: updated node feature tensor (same size with x) / Graph nData/eData does not matter here.
        """
        x = torch.tanh(self.GATConv(graph, x))
        return x.squeeze(1)


class Depth(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Depth, self).__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=1, bias=False)

    def forward(self, x, h, c):
        """This is where a discrepancy from the paper exists. The equations in the paper illustrates that
        no hidden state h (which originally exists in LSTM) is employed in computing for each time step,
        but in their code, the h for LSTM is continuously fed into LSTM for each time step.
        In order to follow the paper exactly, h should be washed out (to zero vector) at every time step."""
        x, (h, c) = self.lstm(x, (h, c))
        return x, (h, c)


class GeniePathLayer(nn.Module):
    def __init__(self, gat_dim, lstm_hidden_dim):
        super(GeniePathLayer, self).__init__()
        self.breadth = Breadth(gat_dim, gat_dim)
        self.depth = Depth(gat_dim, lstm_hidden_dim)

    def forward(self, graph, x, h, c):
        x = self.breadth(graph, x).unsqueeze(0)
        x, (h, c) = self.depth(x, h, c)
        return x.squeeze(0), (h, c)


class GeniePath(nn.Module):
    def __init__(self, in_dim, gat_dim, lstm_hidden_dim, out_dim, num_layers, device):
        super(GeniePath, self).__init__()
        self.device = device
        self.lstm_hidden_dim = lstm_hidden_dim

        self.linear1 = nn.Linear(in_dim, gat_dim)
        self.path_layers = nn.ModuleList([GeniePathLayer(gat_dim, lstm_hidden_dim) for _ in range(num_layers)])
        self.linear2 = nn.Linear(gat_dim, out_dim)

    def forward(self, graph, node_feat):
        x = self.linear1(node_feat)
        h = torch.zeros((1, x.size(0), self.lstm_hidden_dim)).to(self.device)
        c = torch.zeros((1, x.size(0), self.lstm_hidden_dim)).to(self.device)
        for genie_path_layer in self.path_layers:
            x, (h, c) = genie_path_layer(graph, x, h, c)

        return self.linear2(x)


if __name__ == '__main__':
    import dgl
    g = dgl.DGLGraph()
    g.add_nodes(3)
    g.add_edges([0, 2], [1])

    # NOTE: gat_dim === lstm_hidden_dim 이어야함 모델의 구조 자체가!!!
    model = GeniePath(1, gat_dim=10, lstm_hidden_dim=10, out_dim=20, num_layers=4, device=torch.device('cpu'))
    out = model(g, torch.tensor([[1], [2], [3]], dtype=torch.float))
    assert out.size() == torch.Size([3, 20])
