import torch
import torch_geometric


class Base(torch.nn.Module):
    def __init__(self, net, **kwargs):
        super(Base, self).__init__()
        self.net = net(**kwargs)

    def forward(self, data):
        out = self.net(data.x, data.edge_index, data.edge_attr)
        return out

    def loss(self, scores, y):
        pass

    def reset(self):
        self.net.reset()


class BCE(Base):
    def __init__(self, net, node_classification=False, num_classes=2, **kwargs):
        super(BCE, self).__init__(net, **kwargs)
        self.linears = torch.nn.ModuleList()
        self.num_classes = num_classes
        self.linears.append(torch.nn.Linear(self.net.d_in, self.num_classes))
        for _ in range(self.net.num_layers):
            self.linears.append(torch.nn.Linear(self.net.const, self.num_classes))
        self.node_classification = node_classification


    def forward(self, data):
        out = self.net(data.x, data.edge_index, data.edge_attr)
        sum_pool = 0
        for i in range(len(out)):
            pool = out[i]
            if not self.node_classification:
                pool = torch_geometric.nn.global_mean_pool(out[i], data.batch)
            sum_pool += torch.nn.functional.dropout(self.linears[i](pool), p=0.5, training=self.training)
        logits = sum_pool
        return logits

    def loss(self, scores, y):
        loss = torch.nn.CrossEntropyLoss()
        #loss = torch.nn.functional.binary_cross_entropy(scores[:, 0], y.to(torch.float))
        return loss(scores, y.to(torch.long))#loss

    def reset(self):
        super(BCE, self).reset()
        self.linears = torch.nn.ModuleList()
        self.linears.append(torch.nn.Linear(self.net.d_in, self.num_classes))
        for _ in range(self.net.num_layers):
            self.linears.append(torch.nn.Linear(self.net.const, self.num_classes))