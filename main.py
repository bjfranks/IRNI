import argparse
from optimization import optimize, Objective
import datasets
import models
import transforms
from skopt.space import Real, Integer, Categorical
from model_heads import BCE
from utils import LazyFunction
import torch
import torch_geometric
import copy
import time
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
from skopt.utils import use_named_args
import random


def setup_parser():
    out = argparse.ArgumentParser()
    out.add_argument('-dataset', default='TRI', type=str, choices=datasets.choices.keys(),
                     help='The dataset used for training and evaluation.')
    out.add_argument('-model', default='GIN', type=str, choices=models.choices.keys(),
                     help='The model used for training and evaluation.')
    out.add_argument('-node_initialization', default='None', type=str, choices=transforms.choices.keys(),
                     help='The model used for training and evaluation.')
    out.add_argument('-num_workers', default=0, type=int,
                     help='The number of workers used for training and evaluation.')
    out.add_argument('-num_bayes_samples', default=50, type=int,
                     help='The number of samples used to estimate the optimal hyperparameters.')
    out.add_argument('-seeds', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], type=int, nargs='+',
                     help='The seeds used, one after the other, to determine the splits as well as the baysian'
                          'sampling.')
    out.add_argument('-cuda', default=0, type=int, help='The cuda device used.')
    out.add_argument('-num_train_eval_samples', default=3, type=int,
                     help='The number samples used during the train/eval cycle. This influences the variance of the'
                          'bayes samples')
    out.add_argument('-batch_size', default=[8, 256], type=int, nargs=2, help='The minimum and maximum batch size.')
    out.add_argument('-epochs', default=[32, 512], type=int, nargs=2, help='The minimum and maximum amount of epochs.')
    out.add_argument('-lr', default=[-6.0, -2.0], type=float, nargs=2, help='The minimum and maximum logarithmic learning '
                                                                        'rate, i.e. 10**lr is used for training.')
    out.add_argument('-weight_decay', default=[-10.0, -0.3], type=float, nargs=2,
                     help='The minimum and maximum logarithmic weight decay.')
    out.add_argument('-features', default=[16, 128], type=int, nargs=2,
                     help='The minimum and maximum number of features used in the GNN.')
    out.add_argument('-layers', default=[2, 10], type=int, nargs=2,
                     help='The minimum and maximum number of layers used in the GNN.')
    out.add_argument('-dims', default=[1, 5], type=int, nargs=2,
                     help='The minimum and maximum number of dimensions used for the node initialization.')
    out.add_argument('-step_size', default=[0.01, 1.0], type=float, nargs=2,
                     help='The minimum and maximum step_size used for the learning rate to drop relative to the number '
                          'of epochs.')
    out.add_argument('-ensembling', default=[1, 1], type=int, nargs=2,
                     help='The minimum and maximum number of ensemblings over randomness used for the evaluation.')
    return out


def get_space(name, tuple):
    if len(tuple) == 2:
        if tuple[0] == tuple[1]:
            return Categorical(name=name, categories=[tuple[0]])
        else:
            if isinstance(tuple[0], int):
                return Integer(name=name, low=tuple[0], high=tuple[1])
            elif isinstance(tuple[0], float):
                return Real(name=name, low=tuple[0], high=tuple[1])
    return Categorical(name=name, categories=tuple)


def main(args, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    #from torch_geometric.datasets import GNNBenchmarkDataset
    #print(GNNBenchmarkDataset(root='./data', name='CSL').data.y)
    search_space = [get_space(name='batch_size', tuple=args.batch_size),
                    get_space(name='epochs', tuple=args.epochs),
                    get_space(name='lr', tuple=args.lr),
                    get_space(name='weight_decay', tuple=args.weight_decay),
                    get_space(name='features', tuple=args.features),
                    get_space(name='layers', tuple=args.layers),
                    get_space(name='dims', tuple=args.dims),
                    get_space(name='step_size', tuple=args.step_size),
                    get_space(name='ensembling', tuple=args.ensembling),
                    get_space(name='node_initialization', tuple=[args.node_initialization]),
                    get_space(name='edge_labels', tuple=[datasets.choices[args.dataset][3]])
                    ]
    model = LazyFunction(BCE, net=models.choices[args.model], node_classification=datasets.choices[args.dataset][0],
                         num_classes=datasets.choices[args.dataset][2])
    start = time.time()
    if len(datasets.choices[args.dataset]) == 5:
        if datasets.choices[args.dataset][1]:
            skf1 = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
            y = datasets.choices[args.dataset][4]().data.y
            gen = skf1.split(list(range(len(y))), y)
            train_pre, test = next(gen)
            skf2 = StratifiedKFold(n_splits=9, random_state=seed, shuffle=True)
            gen = skf2.split(train_pre, y[train_pre])
            train_eval = []
            for _ in range(args.num_train_eval_samples):
                train_indices, val_indices = next(gen)
                train = train_pre[train_indices]
                train = train.astype(np.int64)
                val = train_pre[val_indices]
                val = val.astype(np.int64)
                train_eval += [(train, val)]
            test = test.astype(np.int64)
            trainset = datasets.choices[args.dataset][4]
            evalset = copy.deepcopy(datasets.choices[args.dataset][4])
            testset = copy.deepcopy(datasets.choices[args.dataset][4])
        else:
            raise NotImplementedError('Still has to be done')
    elif len(datasets.choices[args.dataset]) == 6:
        if datasets.choices[args.dataset][1]:
            skf = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
            y = datasets.choices[args.dataset][4]().data.y
            gen = skf.split(list(range(len(y))), y)
            train_eval = []
            for _ in range(args.num_train_eval_samples):
                train, val = next(gen)
                train = train.astype(np.int64)
                val = val.astype(np.int64)
                train_eval += [(train, val)]
            test = np.array(range(len(datasets.choices[args.dataset][5]())), dtype=np.int64)
            trainset = datasets.choices[args.dataset][4]
            evalset = copy.deepcopy(datasets.choices[args.dataset][4])
            testset = datasets.choices[args.dataset][4]
        else:
            kf = KFold(n_splits=10, random_state=seed, shuffle=True)
            gen = kf.split(list(range(len(datasets.choices[args.dataset][4]()))))
            train_eval = []
            for _ in range(args.num_train_eval_samples):
                train, val = next(gen)
                train = train.astype(np.int64)
                val = val.astype(np.int64)
                train_eval += [(train, val)]
            test = np.array(range(len(datasets.choices[args.dataset][5]())), dtype=np.int64)
            trainset = datasets.choices[args.dataset][4]
            evalset = copy.deepcopy(datasets.choices[args.dataset][4])
            testset = datasets.choices[args.dataset][5]
    result = optimize(model=model, trainset=trainset, indices=train_eval, evalset=evalset, search_space=search_space,
                      num_workers=args.num_workers, num_bayes_samples=args.num_bayes_samples, seed=seed)

    obj = Objective(model, trainset, testset, [(np.concatenate((train, val)), test)], {'num_workers': args.num_workers})

    @use_named_args(dimensions=search_space)
    def objective(**kwargs):
        return obj.train_eval(**kwargs)
    test_auroc = objective(result.x)

    end = time.time()
    print(end-start, 's')

    opt_x, opt_fun = result.x, result.fun
    print(opt_x, opt_fun, test_auroc)

    bayes_samples = []
    for x, fun in zip(result.x_iters, result.func_vals):
        bayes_samples += [(x, fun)]
    print(bayes_samples)

    import pickle
    import os
    try:
        os.mkdir('results_EoR')
    except FileExistsError:
        pass
    with open('results_EoR/{}_{}.pkl'.format(args.node_initialization, args.dataset), 'ab') as f:
        pickle.dump([seed, test_auroc, (opt_x, opt_fun), bayes_samples, end-start], f)



if __name__ == "__main__":
    parser = setup_parser()

    _args = parser.parse_args()

    for seed in _args.seeds:
        with torch.cuda.device(_args.cuda):
            main(_args, seed)
