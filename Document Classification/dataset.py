import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from augraphy import *

class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, augraphy_transform=None):
        self.data = pd.read_csv(csv_file).values
        self.img_dir = img_dir
        self.transform = transform
        self.augraphy_transform = augraphy_transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name, target = self.data[idx]
        img_path = os.path.join(self.img_dir, img_name)
        try:
            img = Image.open(img_path)
            img = np.array(img)
            if self.augraphy_transform:
                img = self.augraphy_transform(img)
            if self.transform:
                img = self.transform(image=img)['image']
            return img, target
        except (IOError, OSError):
            print(f"Cannot read image: {img_path}")
            return None, None

def load_datasets(config):
    totensor_transform = A.Compose([A.Resize(380, 380), ToTensorV2()])
    train_csv = '/data/ephemeral/home/filtered_final.csv'
    img_dir = '/data/ephemeral/home/lmj2'
    test_csv = '/data/ephemeral/home/data/sample_submission.csv'
    test_img_dir = '/data/ephemeral/home/data/test/'

    train_dataset = ImageDataset(train_csv, img_dir, transform=totensor_transform)
    test_dataset = ImageDataset(test_csv, test_img_dir, transform=totensor_transform)

    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
