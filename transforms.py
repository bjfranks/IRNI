import torch
import random
import numpy as np
from torch_geometric.transforms import BaseTransform
import dejavu_gi
np.seterr(over='ignore')


class EdgeConstant(BaseTransform):
    r"""Adds a constant value to each edge attribute.

    Args:
        value (float, optional): The value to add. (default: :obj:`1.0`)
    """
    def __init__(self, value: float = 1.0):
        self.value = value

    def __call__(self, data):
        c = torch.full((data.edge_index.shape[1], 1), self.value, dtype=torch.float)
        data.edge_attr = c
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(value={self.value})'


class Nothing(BaseTransform):
    r"""Does nothing, acts as a counterpart to the other node-initialization schemes.
    """
    def __init__(self, dims=1, edge_labels=False):
        pass

    def __call__(self, data):
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class RNI(BaseTransform):
    r"""Adds a randomized value to each node feature.

    Args:
        cat (bool, optional): If set to :obj:`False`, all existing node
            features will be replaced. (default: :obj:`True`)
    """
    def __init__(self, prob=1, dims=1, edge_labels=False, cat=True):
        self.prob = prob
        self.dims = dims
        self.cat = cat

    def __call__(self, data):
        if self.dims != 0:
            x = data.x

            c = torch.rand((data.num_nodes, self.dims), dtype=torch.float)
            n = torch.full((data.num_nodes, self.dims), 0,  dtype=torch.float)
            r = torch.rand((data.num_nodes, 1), dtype=torch.float)
            c = torch.where(r < self.prob, c, n)

            if x is not None and self.cat:
                x = x.view(-1, 1) if x.dim() == 1 else x
                data.x = torch.cat([x, c.to(x.dtype).to(x.device)], dim=-1)
            else:
                data.x = c

        return data

    def __repr__(self):
        return '{}(prob={}, dims={}, cat={})'.format(self.__class__.__name__, self.prob, self.dims, self.cat)


class IRNI(BaseTransform):
    r"""Adds a one-hot encoding of its individualization to each node according to a random Individualization Refinement
    Path.
    """
    def __init__(self, dims=1, edge_labels=False):
        self.depth = int(dims)
        self.edge_labels = edge_labels

    def __call__(self, data):
        if self.depth != 0:
            colors = mash1(data.x)  # [mash0(x) for x in data.x]
            if self.edge_labels:
                edge_colors = mash1(data.edge_attr)  # [mash0(x) for x in data.edge_attr]
            else:
                edge_colors = []
            try:
                test = dejavu_gi.random_ir_paths(data.num_nodes, data.edge_index.T.tolist(), self.depth,
                                                 vertex_labels=colors, edge_labels=edge_colors,
                                                 fill_paths=True, directed_dimacs=True)
            except OSError as e:
                import traceback
                traceback.print_exc()
                print(data.num_nodes, data.edge_index.T.tolist(), self.depth, colors, edge_colors)
            nodes = test[0]['base_points']
            o = torch.full((data.num_nodes,self.depth), 0, dtype=data.x.dtype)
            k = 0
            for node in nodes:
                o[node, k] = 1
                k += 1
            data.x = torch.cat([data.x, o.to(data.x.dtype).to(data.x.device)], dim=-1)
        return data

    def __repr__(self):
        return '{}(depth={}, fill={})'.format(self.__class__.__name__, self.depth, self.fill)


class ORNI(object):
    r"""Randomly individualizes num_nodes nodes.
    """
    def __init__(self, dims=1, edge_labels=False):
        self.num_nodes = dims

    def __call__(self, data):
        if self.num_nodes != 0:
            nodes = list(range(data.num_nodes))
            random.shuffle(nodes)
            nodes = nodes[:self.num_nodes]

            o = torch.full((data.num_nodes, self.num_nodes), 0, dtype=data.x.dtype)
            k = 0
            for node in nodes:
                o[node, k] = 1
                k += 1
            data.x = torch.cat([data.x, o.to(data.x.dtype).to(data.x.device)], dim=-1)
        return data

    def __repr__(self):
        return '{}(num_nodes={})'.format(self.__class__.__name__, self.num_nodes)


class CLIP(object):
    r"""Randomly individualizes num_nodes nodes.
    """
    def __init__(self, dims=1, edge_labels=False):
        self.dims = dims
        self.edge_labels = edge_labels

    def __call__(self, data):
        if self.dims != 0:
            colors = mash1(data.x)  # [mash0(x) for x in data.x]
            if self.edge_labels:
                edge_colors = mash1(data.edge_attr)  # [mash0(x) for x in data.edge_attr]
            else:
                edge_colors = []
            try:
                node2color = dejavu_gi.color_refinement(data.num_nodes, data.edge_index.T.tolist(),
                                                        vertex_labels=colors, edge_labels=edge_colors,
                                                        directed_dimacs=True)
            except OSError:
                import traceback
                traceback.print_exc()
                print(data.num_nodes, data.edge_index.T.tolist(), colors, edge_colors)
            color_classes = dict()
            for i in range(len(node2color)):
                if node2color[i] in color_classes:
                    color_classes[node2color[i]] += [i]
                else:
                    color_classes[node2color[i]] = [i]
            o = torch.full((data.num_nodes, self.dims), 0, dtype=data.x.dtype)
            for i in color_classes.values():
                if len(i) > 1:
                    random.shuffle(i)
                    i = i[:self.dims]
                    k = 0
                    for node in i:
                        o[node, k] = 1
                        k += 1
            data.x = torch.cat([data.x, o.to(data.x.device)], dim=-1)
        return data

    def __repr__(self):
        return '{}(dims={})'.format(self.__class__.__name__, self.dims)


# this is better in general, however quite slow and adds considerable cost. PyTorch needs a good hash
def mash0(string):
    dtype = np.uint16
    large = dtype(35235237)
    small = dtype(5)
    string = string.numpy().astype(dtype)
    mash0 = dtype(0)
    for i in range(len(string)):
        mash0 += string[i]
        mash0 = mash0 * (large - mash0 * small)
    return int(mash0)


# this works for datasets considered here, but in general might not hash colors properly.
# This is about 1000 times faster than using mash0
def mash1(input):
    output = torch.sum(input*torch.tensor([2 ** (input.shape[1]-1-i) for i in range(input.shape[1])]), dim=1,
                       dtype=torch.int64) % 12345
    return output.tolist()


out = dict()
out['None'] = Nothing
out['RNI'] = RNI
out['IRNI'] = IRNI
out['ORNI'] = ORNI
out['CLIP'] = CLIP
choices = out
