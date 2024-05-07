# data/dataloader.py
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset, ConcatDataset
import random

meta_path = '/data/ephemeral/home/data/meta.csv'
train_path = '/data/ephemeral/home/data/train.csv'
submission_path = '/data/ephemeral/home/data/sample_submission.csv'

meta_data = pd.read_csv(meta_path)
df_train = pd.read_csv(train_path)
df_submission = pd.read_csv(submission_path)

merge = pd.merge(df_train, meta_data, how='inner')

oversampling_factors = [1.0] * 17
oversampling_factors[1] = 2.0  # (100/50)
oversampling_factors[13] = 1.35  # (100/74)
oversampling_factors[14] = 2.0  # (100/50)

def oversample_subset_per_class(dataset, oversampling_factors):
    oversampled_datasets = []
    class_to_indices = {}
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(i)

    for label, indices in class_to_indices.items():
        oversampling_factor = oversampling_factors[label]
        oversampled_indices = random.choices(indices, k=int(len(indices) * oversampling_factor) // 2 * 2)
        oversampled_subset = Subset(dataset, oversampled_indices)
        oversampled_datasets.append(oversampled_subset)
    
    oversampled_dataset = ConcatDataset(oversampled_datasets)
    print(f"Oversampled | {len(dataset)} -> {len(oversampled_dataset)}")
    return oversampled_dataset