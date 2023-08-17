from typing import List
import torch
import torch_geometric


class MLP(torch.nn.Module):
    r"""A multi layer perceptron factory.
    Args:
        structure (list(int)): Implicitly defines the MLP. len(structure)-1 is the number of linear layers and each pair
        (structure[i], structure[i+1]) of entries defines the number of (inputs, outputs) of a linear layer.
        activation (nn.Module, optional): Defines the activation function used after each linear layer.
            (default: nn.LeakyReLU)
    """
    def __init__(self, structure: List[int], activation):
        super(MLP, self).__init__()
        self.activation = activation
        self.linear_layers = torch.nn.ModuleList()
        self.batch_norm_layers = torch.nn.ModuleList()
        for i in range(len(structure)-1):
            self.linear_layers.append(torch.nn.Linear(structure[i], structure[i+1]))
        for i in range(len(structure)-2):
            self.batch_norm_layers.append(torch.nn.BatchNorm1d(structure[i+1]))

    def forward(self, x):
        for layer in range(len(self.batch_norm_layers)):
            x = self.activation(self.batch_norm_layers[layer](self.linear_layers[layer](x)))
        return self.linear_layers[len(self.batch_norm_layers)](x)

    def __getitem__(self, key):
        return self.linear_layers[key]


class Base(torch_geometric.nn.MessagePassing):
    r"""A Base model to unify initialization."""

    def __init__(self, d_in=1, features=64, layers=5, activation=torch.nn.functional.relu, edge_dim=0, **kwargs):
        super(Base, self).__init__(aggr='add')
        self.d_in = d_in
        self.const = features
        self.num_layers = layers
        self.activation = activation
        self.edge_dim = edge_dim


class GIN(Base):
    r"""A GIN model using GIN layers, where each GIN layer uses an MLP."""
    def __init__(self, *args, **kwargs):
        super(GIN, self).__init__(aggr='add', *args, **kwargs)
        self.mlp_in = [self.d_in, self.const, self.const]
        self.mlp_out = [self.const, self.const, self.const]
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch_geometric.nn.conv.gin_conv.GINEConv(MLP(self.mlp_in, self.activation),
                                                                         train_eps=True, edge_dim=self.edge_dim))
        for _ in range(self.num_layers-1):
            self.layers.append(torch_geometric.nn.conv.gin_conv.GINEConv(MLP(self.mlp_out, self.activation),
                                                                             train_eps=True, edge_dim=self.edge_dim))

    def forward(self, x, edge_index, edge_attr):
        x, edge_index, edge_attr = x, edge_index, edge_attr
        representations = [x]
        for layer in self.layers:
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
            representations += [x]
        return representations

    def reset(self):
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch_geometric.nn.conv.gin_conv.GINEConv(MLP(self.mlp_in, self.activation), train_eps=True,
                                                                     edge_dim=self.edge_dim))
        for _ in range(self.num_layers - 1):
            self.layers.append(torch_geometric.nn.conv.gin_conv.GINEConv(MLP(self.mlp_out, self.activation),
                                                                         train_eps=True, edge_dim=self.edge_dim))


class GAT(Base):
    r"""A GAT model using GATv2 layers."""
    def __init__(self):
        super(GAT, self).__init__(aggr='add')
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch_geometric.nn.conv.gatv2_conv.GATv2Conv(self.d_in, int(self.const/4), heads=4,
                                                                        edge_dim=1))
        for _ in range(self.num_layers-1):
            self.layers.append(torch_geometric.nn.conv.gatv2_conv.GATv2Conv(self.const, int(self.const/4), heads=4,
                                                                            edge_dim=1))

    def forward(self, x, edge_index, edge_attr):
        x, edge_index, edge_attr = x, edge_index, edge_attr
        representations = [x]
        for layer in self.layers:
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
            representations += [x]
        return representations

    def reset(self):
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch_geometric.nn.conv.gatv2_conv.GATv2Conv(self.d_in, int(self.const / 4), heads=4,
                                                                        edge_dim=self.edge_dim))
        for _ in range(self.num_layers - 1):
            self.layers.append(torch_geometric.nn.conv.gatv2_conv.GATv2Conv(self.const, int(self.const / 4), heads=4,
                                                                            edge_dim=self.edge_dim))


class GCN(Base):
    r"""A GCN model using GCN layers, where each GCN layer uses an MLP for edge attributes."""
    def __init__(self):
        super(GCN, self).__init__(aggr='add')
        self.mlp_in = [self.d_in, self.const, self.const]
        self.mlp_out = [self.const, self.const, self.const]
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch_geometric.nn.conv.nn_conv.NNConv(self.d_in, self.const,
                                                                  MLP([self.edge_dim, self.const],
                                                                      self.activation)))
        for _ in range(self.num_layers-1):
            self.layers.append(torch_geometric.nn.conv.nn_conv.NNConv(self.const, self.const,
                                                                      MLP([self.edge_dim, self.const],
                                                                          self.activation)))

    def forward(self, x, edge_index, edge_attr):
        x, edge_index, edge_attr = x, edge_index, edge_attr
        representations = [x]
        for layer in self.layers:
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
            representations += [x]
        return representations

    def reset(self):
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch_geometric.nn.conv.nn_conv.NNConv(self.d_in, self.const,
                                                                  MLP([self.edge_dim, self.const],
                                                                      self.activation)))
        for _ in range(self.num_layers - 1):
            self.layers.append(torch_geometric.nn.conv.nn_conv.NNConv(self.const, self.const,
                                                                      MLP([self.edge_dim, self.const],
                                                                          self.activation)))


out = dict()
out['GIN'] = GIN
out['GAT'] = GAT
out['GCN'] = GCN
choices = out
