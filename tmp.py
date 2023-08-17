import pickle
import numpy as np

models = ['None', 'RNI', 'CLIP', 'ORNI', 'IRNI']
datasets = ['PROTEINS', 'MUTAG', 'NCI1', 'TRI', 'TRIX', 'EXP', 'CEXP', 'CSL']

print('Models', end=' ')
for dataset in datasets:
    print(dataset, end=' ')
print()

for model in models:
    print(model, end=' ')
    for dataset in datasets:
        try:
            data = dict()
            with open('results/{}_{}.pkl'.format(model, dataset), 'rb') as f:
                while True:
                    try:
                        out = pickle.load(f)
                        data[str(out[0])]=out[1]
                    except EOFError:
                        break
            values = list(data.values())
            mean = np.mean(values)
            std = np.std(values)
            print('{:.2f}+-{:.2f}'.format(mean,std), end=' ')
        except OSError:
            print('0.00+-0.00',end=' ')
    print()
