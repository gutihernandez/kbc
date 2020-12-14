import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '--dataset'
)
parser.add_argument(
    '--k', default=10, type=int,
    help="k-folds"
)
parser.add_argument(
    '--input_path'
)

args = parser.parse_args()
dataset = args.dataset
input_path_dset_folder = args.input_path
k_fold = args.k

input_path_dset = os.path.join(input_path_dset_folder, dataset)

# read the data
train_set = pd.read_csv(os.path.join(input_path_dset, "train"), sep="\t", header=None)
valid_set = pd.read_csv(os.path.join(input_path_dset, "valid"), sep="\t", header=None)
test_set = pd.read_csv(os.path.join(input_path_dset, "test"), sep="\t", header=None)
# concat train+test sets
concat_dataset = pd.concat([train_set, test_set])
size = concat_dataset.shape[0]

# shuffle the total data
shuffled_dataset = concat_dataset.sample(frac=1).reset_index()

# calculate the new size of test set
new_test_size = shuffled_dataset.shape[0] // k_fold

for k in range(k_fold):
    print(k)
    # create a folder space for output datasets
    try:
        output_folder = os.path.join(input_path_dset_folder, dataset + str(k))
        os.mkdir(os.path.join(input_path_dset_folder, dataset + str(k)))
    except:
        print("it exists! so moving on...")
    # split train+test data
    new_test = shuffled_dataset.iloc[new_test_size * k:new_test_size * (k + 1)]
    print(new_test_size * (k))
    print(new_test_size * (k + 1))
    new_train = shuffled_dataset.drop(np.arange(new_test_size * k, new_test_size * (k + 1)), axis=0)

    # last folding needs special care because of remainders
    if k == 9:
        new_test = shuffled_dataset.iloc[new_test_size * k:]
        new_train = shuffled_dataset.drop(np.arange(new_test_size * k, size), axis=0)

    print("sizes:")
    print(new_test.shape)
    print(new_train.shape)
    # save train+test data
    new_test.to_csv(output_folder + "/test", sep="\t", header=None, index=False)
    valid_set.to_csv(output_folder + "/valid", sep="\t", header=None, index=False)
    new_train.to_csv(output_folder + "/train", sep="\t", header=None, index=False)

    # below lines of codes ensures that train - test sets do not have any common elements in them
    print(new_train.columns.tolist())
    result = pd.merge(new_train, new_test, on=new_test.columns.tolist()[1:])
    print(result.shape)
