# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from typing import Dict

import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


from kbc.datasets import Dataset
from kbc.models import CP, ComplEx, MobiusESM, MobiusESMRot, QuatE
from kbc.regularizers import F2, N3
from kbc.optimizers import KBCOptimizer

torch.manual_seed(0)

big_datasets = ['FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10', 'NELL-995-h25', 'NELL-995-h50', 'NELL-995-h75', 'NELL-995-h100', 'Mock-DS']
datasets = big_datasets

parser = argparse.ArgumentParser(
    description="Relational learning contraption"
)

parser.add_argument(
    '--dataset', choices=datasets,
    help="Dataset in {}".format(datasets)
)

models = ['CP', 'ComplEx', 'MobiusESM', 'MobiusESMRot', 'QuatE']
parser.add_argument(
    '--model', choices=models,
    help="Model in {}".format(models)
)

regularizers = ['N3', 'F2']
parser.add_argument(
    '--regularizer', choices=regularizers, default='N3',
    help="Regularizer in {}".format(regularizers)
)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument(
    '--optimizer', choices=optimizers, default='Adagrad',
    help="Optimizer in {}".format(optimizers)
)

parser.add_argument(
    '--max_epochs', default=50, type=int,
    help="Number of epochs."
)

parser.add_argument(
    '--save_model', default=False, action='store_true'
)

parser.add_argument(
    '--load_model', default=False, action='store_true'
)

parser.add_argument(
    '--valid', default=3, type=float,
    help="Number of epochs before valid."
)

parser.add_argument(
    '--plot_epochs', default=1000, type=float,
    help="Number of epochs before plotting."
)

parser.add_argument(
    '--rank', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--reg', default=0, type=float,
    help="Regularization weight"
)
parser.add_argument(
    '--init', default=1e-3, type=float,
    help="Initial scale"
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--decay1', default=0.9, type=float,
    help="decay rate for the first moment estimate in Adam"
)
parser.add_argument(
    '--decay2', default=0.999, type=float,
    help="decay rate for second moment estimate in Adam"
)



args = parser.parse_args()

dataset = Dataset(args.dataset)
examples = torch.from_numpy(dataset.get_train().astype('int64'))

print(dataset.get_shape())
model = {
    'CP': lambda: CP(dataset.get_shape(), args.rank, args.init),
    'ComplEx': lambda: ComplEx(dataset.get_shape(), args.rank, args.init),
    'MobiusESM': lambda: MobiusESM(dataset.get_shape(), args.rank, args.init),
    'MobiusESMRot': lambda: MobiusESMRot(dataset.get_shape(), args.rank, args.init),
    'QuatE': lambda: QuatE(dataset.get_shape(), args.rank, args.init),
}[args.model]()

if args.load_model:
  if args.model == "MobiusESM":
    print("Mobius")
    print(type(model))
    print("shape before:" + str(model.embeddings[0].weight.data.shape))
    model.embeddings[0].weight.data = torch.from_numpy(np.load("/content/drive/My Drive/masters/thesis/trained-embeddings/old-mobius-embeddings/WN18-50epoch/entity_embedding.npy"))
    print("shape after:" + str(model.embeddings[0].weight.data.shape))
    model.embeddings[1].weight.data = torch.from_numpy(np.load("/content/drive/My Drive/masters/thesis/trained-embeddings/old-mobius-embeddings/WN18-50epoch/relation_embedding.npy"))
    print("Trained embeddings are loaded...")
  if args.model == "QuatE":
    print("QuatE")
    print(type(model))
    model.embeddings[0].weight.data = torch.from_numpy(np.load("/content/drive/My Drive/masters/thesis/trained-embeddings/WN18-QuatE-Dim-250/entity_embedding_QuatE_250.npy"))
    model.embeddings[1].weight.data = torch.from_numpy(np.load("/content/drive/My Drive/masters/thesis/trained-embeddings/WN18-QuatE-Dim-250/relation_embedding_QuatE_250.npy"))
    print("Trained embeddings are loaded...")
  if args.model == "ComplEx":
    print("ComplEx")
    print(type(model))
    model.embeddings[0].weight.data = torch.from_numpy(np.load("/content/drive/My Drive/masters/thesis/trained-embeddings/WN18-ComplEx-Dim-500/entity_embedding_ComplEx_40.npy"))
    model.embeddings[1].weight.data = torch.from_numpy(np.load("/content/drive/My Drive/masters/thesis/trained-embeddings/WN18-ComplEx-Dim-500/relation_embedding_ComplEx_40.npy"))
    print("Trained embeddings are loaded...")

regularizer = {
    'F2': F2(args.reg),
    'N3': N3(args.reg),
}[args.regularizer]

device = 'cuda'
model.to(device)

optim_method = {
    'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
    'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2)),
    'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate)
}[args.optimizer]()

optimizer = KBCOptimizer(model, regularizer, optim_method, args.batch_size)


def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': m, 'hits@[1,3,10]': h}

cur_loss = 0
curve = {'train': [], 'valid': [], 'test': []}
ent_id = pd.read_csv("ent_id", sep='\t', header=None)
path = str(args.model)
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)
for e in range(args.max_epochs):
    cur_loss = optimizer.epoch(examples)

    if (e + 1) % args.valid == 0:
        valid, test, train = [
            avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
            for split in ['valid', 'test', 'train']
        ]

        curve['valid'].append(valid)
        curve['test'].append(test)
        curve['train'].append(train)

        print("\t TRAIN: ", train)
        print("\t VALID : ", valid)


    #plotting part...
    if (e + 1) % args.plot_epochs == 0:
      with torch.no_grad():
        ent_embeddings = model.embeddings[0].weight.cpu().numpy()
        f, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]},
                                  figsize=(7, 14))

        ax1.scatter(x=ent_embeddings[:10,0], y=ent_embeddings[:10,1],  s=100)
        for i, txt in enumerate(np.asarray(ent_id)[:10,0]):
            ax1.annotate(txt, (ent_embeddings[i,0], ent_embeddings[i,1]) ,fontsize=25)


        ax1.scatter(x=ent_embeddings[10:,0], y=-ent_embeddings[10:,1],  s=100)
        for i, txt in enumerate(np.asarray(ent_id)[10:, 0]):
            ax1.annotate(txt, (ent_embeddings[i+10, 0], -ent_embeddings[i+10, 1]),fontsize=25)
        model_name = str(args.model)
        plt.savefig(model_name+"/"+model_name + "-" +str(args.max_epochs))
        


results = dataset.eval(model, 'test', -1)
print("\n\nTEST : ", results)

if args.save_model:
  import numpy as np

  entity_embedding = model.embeddings[0].weight.detach().cpu().numpy()
  np.save(
      'entity_embedding_'+args.model+'_'+str(args.max_epochs),
      entity_embedding
  )

  relation_embedding = model.embeddings[1].weight.detach().cpu().numpy()
  np.save(
      'relation_embedding_'+args.model+'_'+str(args.max_epochs),
      relation_embedding
  )



