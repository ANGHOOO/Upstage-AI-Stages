import os
import torch
import pandas as pd
from config import Config
from dataset import load_datasets
from model import get_model
from train import train_loop
from evaluate import evaluate
from transform import get_transforms

def main():
    config = Config()
    train_loader, valid_loader, test_loader = load_datasets(config)
    model, criterion, mixup_fn = get_model(config)
    optimizer = config.optimizer(model)
    scheduler = config.scheduler(optimizer)
    model_name = 'effb4-add_fc'
    model = train_loop(model, train_loader, valid_loader, config.device, criterion, optimizer, scheduler, mixup_fn, config.num_epochs, config.patience)
    model, metrics = evaluate(model, test_loader, config.device, criterion, 0, 1)
    print(metrics)
    pred_df = pd.DataFrame(test_loader.dataset.data, columns=['ID', 'target'])
    pred_df['target'] = metrics['preds']
    submission_path = '/data/ephemeral/home/data/sample_submission.csv'
    sample_submission_df = pd.read_csv(submission_path)
    assert (sample_submission_df['ID'] == pred_df['ID']).all()
    pred_df.to_csv('/data/ephemeral/home/outputs/effb4-add_fc.csv', index=False)

if __name__ == '__main__':
    main()
