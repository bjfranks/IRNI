import torch
import torch_geometric
import skopt
from skopt.utils import use_named_args
from sklearn.metrics import roc_auc_score
import transforms
import time
import numpy as np


class Objective:
    def __init__(self, model, trainset, evalset, indices, kwargs):
        self.model = model
        self.trainset = trainset
        self.evalset = evalset
        self.indices = indices
        self.kwargs = kwargs

    def train(self, model, trainset, lr, weight_decay, batch_size, epochs, step_size, **kwargs):
        model.train()
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=10**lr, weight_decay=10**weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5,
                                                    step_size=int(step_size*epochs)
                                                    if int(step_size*epochs) >= 1 else 1)
        if trainset[0].y.shape[0] == 1:
            target = torch.tensor([datum.y for datum in trainset])
            class_sample_count = np.array(
                [len(np.where(target == t)[0]) for t in np.unique(target)])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in target])

            samples_weight = torch.from_numpy(samples_weight)
            sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
        else:
            sampler = torch.utils.data.RandomSampler(trainset)
        loader = torch_geometric.loader.DataLoader(trainset, batch_size=int(batch_size), sampler=sampler,
                                                   num_workers=self.kwargs['num_workers'])
        for epoch in range(epochs):
            for data in loader:
                data = data.cuda()
                optimizer.zero_grad()
                loss = model.loss(model(data), data.y)
                loss.backward()
                optimizer.step()
            scheduler.step()
            print(epoch, end='\r')
        print(epochs, end=' ')

    def eval(self, model, evalset, batch_size, ensembling, **kwargs):
        model.eval()
        model = model.cuda()
        loader = torch_geometric.loader.DataLoader(evalset, batch_size=int(batch_size),
                                                   num_workers=self.kwargs['num_workers'])
        with torch.no_grad():
            avg_logits = None
            for _ in range(ensembling):
                if model.num_classes > 2:
                    logits = torch.empty([0, model.linears[0].out_features])
                else:
                    logits = torch.empty(0)
                labels = []
                for data in loader:
                    data = data.cuda()
                    out = model(data)
                    if model.num_classes > 2:
                        logit = torch.softmax(out, dim=1).cpu()
                    else:
                        logit = torch.argmax(out, dim=1).cpu()
                    if model.node_classification:
                        logits = torch.cat((logits, logit.view(-1)), dim=0)
                    else:
                        logits = torch.cat((logits, logit), dim=0)
                    labels += data.y.cpu().tolist()
                logits = logits.unsqueeze(0)
                if _ == 0:
                    if model.num_classes > 2:
                        avg_logits = torch.empty([0, logits.shape[1], logits.shape[2]])
                    else:
                        avg_logits = torch.empty([0, logits.shape[1]])
                avg_logits = torch.cat((avg_logits, logits), dim=0)
            avg_logits = torch.mean(avg_logits, dim=0)
            return roc_auc_score(labels, avg_logits.tolist(), multi_class='ovo')

    def train_eval(self, node_initialization, dims, edge_labels, **kwargs):
        avg_objective = 0
        for (train_indices, eval_indices) in self.indices:
            trainset = self.trainset(transform=torch_geometric.transforms.Compose(
                [self.trainset.kwargs['transform'],
                 transforms.choices[node_initialization](dims=dims, edge_labels=edge_labels)]))[train_indices]
            evalset = self.evalset(transform=torch_geometric.transforms.Compose(
                [self.evalset.kwargs['transform'],
                 transforms.choices[node_initialization](dims=dims, edge_labels=edge_labels)]))[eval_indices]
            model = self.model(d_in=trainset[0].x.shape[1], edge_dim=trainset[0].edge_attr.shape[1], **kwargs)
            self.train(model=model, trainset=trainset, **kwargs, **self.kwargs)
            avg_objective += self.eval(model=model, evalset=evalset, **kwargs, **self.kwargs)
        return avg_objective/len(self.indices)

    def __call__(self, **kwargs):
        start = time.time()
        objective = self.train_eval(**kwargs)
        end = time.time()
        calc_factor = 0
        calc_factor += 1 - (kwargs['batch_size'] / 256)
        calc_factor += kwargs['epochs'] / 512
        calc_factor += kwargs['features'] / 128
        calc_factor += kwargs['layers'] / 10
        calc_factor += kwargs['ensembling'] / 64
        calc_factor /= 100
        print(-objective+calc_factor, end=' ')
        print(end - start, 's')
        return -objective+calc_factor


def optimize(model, trainset, indices, evalset, search_space, num_bayes_samples, seed, **kwargs):
    obj = Objective(model, trainset, evalset, indices, kwargs)

    @use_named_args(dimensions=search_space)
    def objective(**kwargs):
        return obj(**kwargs)

    return skopt.gp_minimize(objective, search_space, n_calls=num_bayes_samples, random_state=seed,
                             n_initial_points=int(num_bayes_samples/5) if num_bayes_samples/5 >= 1 else 1)
