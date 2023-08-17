import torch
import torch_geometric
import pickle
import os
import transforms
import torchvision
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
import models

#['EXP', 'CEXP', 'IMDB-BINARY', 'NCI1', 'MUTAG', 'PROTEINS', 'TRI', 'TRIX', 'DD']
#data_set_types = ['CV', 'TrTe']


class TRI(torch_geometric.data.InMemoryDataset):
    def __init__(self, root, train=True, transform=None, pre_transform=None, extrapolate=False):
        super(TRI, self).__init__(root, transform, pre_transform)
        if extrapolate:
            path = self.processed_paths[0] if train else self.processed_paths[2]
        else:
            path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt', 'X_test.pt']

    def process(self):
        data_list = generate_triangle_graphs()
        torch.save(self.collate(data_list), self.processed_paths[0])
        data_list = generate_triangle_graphs()
        torch.save(self.collate(data_list), self.processed_paths[1])
        data_list = generate_triangle_graphs(num_nodes=100)
        torch.save(self.collate(data_list), self.processed_paths[2])


class LCC(torch_geometric.data.InMemoryDataset):
    def __init__(self, root, train=True, transform=None, pre_transform=None, extrapolate=False):
        super(LCC, self).__init__(root, transform, pre_transform)
        if extrapolate:
            path = self.processed_paths[0] if train else self.processed_paths[2]
        else:
            path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt', 'X_test.pt']

    def process(self):
        data_list = generate_lcc_graphs()
        torch.save(self.collate(data_list), self.processed_paths[0])
        data_list = generate_lcc_graphs()
        torch.save(self.collate(data_list), self.processed_paths[1])
        data_list = generate_lcc_graphs(num_nodes=100)
        torch.save(self.collate(data_list), self.processed_paths[2])


class PlanarSATPairsDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(PlanarSATPairsDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["GraphSAT.pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        with open(self.root+'/raw/GraphSAT.pkl', 'rb') as f:
            dataset = pickle.load(f)
        data_list = [torch_geometric.data.Data(x=torch.tensor(x), edge_index=torch.tensor(edge_list), y=torch.tensor(y))
                     for edge_list, x, y in dataset]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def get_data_set(args, train=True):
    args.EXP = False
    if args.data_set not in choices:
        raise RuntimeError(args.data_set+' is not a valid data set choice!')
    enum = choices.index(args.data_set)
    base_transforms = [transforms.IR(depth=args.ir_depth, paths=args.ir_paths, fill=not args.ir_no_fill,
                                     sum=args.ir_sum, rni=args.ir_rni, one_hot=False)]
    if args.rni_one_hot:
        base_transforms += [transforms.PartialRNIOneHot(num_nodes=args.rni_dims)]
    else:
        base_transforms += [transforms.RNI(prob=args.rni_prob, dims=args.rni_dims)]

    if enum == 0:
        args.EXP = True
        transform = torchvision.transforms.Compose(base_transforms)
        data_set = PlanarSATPairsDataset("./data/EXP", transform=transform)
        data_set[0].to
        model_type = models.model_types[0]
        data_set_type = data_set_types[0]
    if enum == 1:
        args.EXP = True
        transform = torchvision.transforms.Compose(base_transforms)
        data_set = PlanarSATPairsDataset("./data/CEXP", transform=transform)
        model_type = models.model_types[0]
        data_set_type = data_set_types[0]
    if enum == 2:
        transform = torchvision.transforms.Compose([torch_geometric.transforms.Constant()]+base_transforms)
        data_set = TUDataset(root='./data', name='IMDB-BINARY', transform=transform)
        model_type = models.model_types[0]
        data_set_type = data_set_types[0]
    if enum == 3:
        transform = torchvision.transforms.Compose([torch_geometric.transforms.Constant()]+base_transforms)
        data_set = TUDataset(root='./data', name='NCI1', transform=transform)
        model_type = models.model_types[0]
        data_set_type = data_set_types[0]
    if enum == 4:
        transform = torchvision.transforms.Compose(base_transforms)
        data_set = TUDataset(root='./data', name='MUTAG', transform=transform)
        model_type = models.model_types[0]
        data_set_type = data_set_types[0]
    if enum == 5:
        transform = torchvision.transforms.Compose(base_transforms)
        data_set = TUDataset(root='./data', name='PROTEINS', use_node_attr=True, transform=transform)
        model_type = models.model_types[0]
        data_set_type = data_set_types[0]
    if enum == 6:
        transform = torchvision.transforms.Compose([torch_geometric.transforms.Constant()]+base_transforms)
        data_set = TRI(root='./data/TRI', train=train, transform=transform)
        model_type = models.model_types[1]
        data_set_type = data_set_types[1]
    if enum == 7:
        transform = torchvision.transforms.Compose([torch_geometric.transforms.Constant()]+base_transforms)
        data_set = TRI(root='./data/TRI', train=train, transform=transform, extrapolate=True)
        model_type = models.model_types[1]
        data_set_type = data_set_types[1]
    if enum == 8:
        transform = torchvision.transforms.Compose(base_transforms)
        data_set = TUDataset(root='./data', name='DD', use_node_attr=True, transform=transform)
        model_type = models.model_types[0]
        data_set_type = data_set_types[0]

    if args.hsc_flip:
        for datum in data_set:
            datum.y = 1-datum.y
    return data_set, model_type, data_set_type


def generate_triangle_graphs(num_graphs=1000, num_nodes=20):
    import networkx as nx
    data_set = []
    for _ in range(num_graphs):
        G = nx.random_regular_graph(3, num_nodes)
        dict = nx.triangles(G)
        edge_index = [[],[]]
        labels = [0]*num_nodes
        for u, v, _ in G.edges(data=True):
            edge_index[0] += [u, v]
            edge_index[1] += [v, u]
        for key, value in dict.items():
            labels[key] = int(value>0)
        data = torch_geometric.data.Data(edge_index=torch.tensor(edge_index), y=torch.tensor(labels))
        data.num_nodes = num_nodes
        data_set += [data]
    return data_set


def generate_lcc_graphs(num_graphs=1000, num_nodes=20):
    import networkx as nx
    data_set = []
    for _ in range(num_graphs):
        G = nx.random_regular_graph(3, num_nodes)
        dict = nx.triangles(G)
        labels = [0] * num_nodes
        for key, value in dict.items():
            labels[key] = int(value > 0)
        data = torch_geometric.utils.convert.from_networkx(G)
        data.y = torch.tensor(labels)
        data_set += [data]
    return data_set


from utils import LazyFunction
out = dict()
# [node_classifictation, statified, num_classes, edge_labels, trainset, evalset if exists]
out['PROTEINS'] = [False, True, 2, False, LazyFunction(TUDataset, root='./data', name='PROTEINS', use_node_attr=True,
                                                       transform=torch_geometric.transforms.Compose(
                                                           [transforms.EdgeConstant()]))
                   ]
out['MUTAG'] = [False, True, 2, True, LazyFunction(TUDataset, root='./data', name='MUTAG', use_node_attr=True,
                                                   use_edge_attr=True,
                                                   transform=torch_geometric.transforms.Compose(
                                                           [transforms.Nothing()]))
                ]
out['NCI1'] = [False, True, 2, False, LazyFunction(TUDataset, root='./data', name='NCI1',
                                                   transform=torch_geometric.transforms.Compose(
                                                           [transforms.EdgeConstant()]))
               ]
out['TRI'] = [True, False, 2, False, LazyFunction(TRI, root='./data/TRI', train=True,
                                                  transform=torch_geometric.transforms.Compose(
                                                      [torch_geometric.transforms.Constant(),
                                                       transforms.EdgeConstant()])),
              LazyFunction(TRI, root='./data/TRI', train=False,
                           transform=torch_geometric.transforms.Compose(
                               [torch_geometric.transforms.Constant(), transforms.EdgeConstant()]))
              ]
out['TRIX'] = [True, False, 2, False, LazyFunction(TRI, root='./data/TRI', train=True, extrapolate=True,
                                                  transform=torch_geometric.transforms.Compose(
                                                      [torch_geometric.transforms.Constant(),
                                                       transforms.EdgeConstant()])),
              LazyFunction(TRI, root='./data/TRI', train=False, extrapolate=True,
                           transform=torch_geometric.transforms.Compose(
                               [torch_geometric.transforms.Constant(), transforms.EdgeConstant()]))
              ]
out['EXP'] = [False, True, 2, False, LazyFunction(PlanarSATPairsDataset, root='./data/EXP',
                                                  transform=torch_geometric.transforms.Compose(
                                                      [transforms.EdgeConstant()]))
              ]
out['CEXP'] = [False, True, 2, False, LazyFunction(PlanarSATPairsDataset, root='./data/CEXP',
                                                   transform=torch_geometric.transforms.Compose(
                                                       [transforms.EdgeConstant()]))
              ]
out['CSL'] = [False, True, 10, False, LazyFunction(GNNBenchmarkDataset, root='./data', name='CSL',
                                                   transform=torch_geometric.transforms.Compose(
                                                       [torch_geometric.transforms.Constant(),
                                                        transforms.EdgeConstant()]))
              ]
choices = out
