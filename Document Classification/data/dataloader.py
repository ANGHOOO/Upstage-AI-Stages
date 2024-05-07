# data/dataloader.py
import pandas as pd

meta_path = '/data/ephemeral/home/data/meta.csv'
train_path = '/data/ephemeral/home/data/train.csv'
submission_path = '/data/ephemeral/home/data/sample_submission.csv'

meta_data = pd.read_csv(meta_path)
df_train = pd.read_csv(train_path)
df_submission = pd.read_csv(submission_path)

merge = pd.merge(df_train, meta_data, how='inner')