# data/dataset.py
import os
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, csv, path, album_transform=None, augraphy_transform=None):
        self.df = pd.read_csv(csv).values
        self.path = path 
        self.album_transform = album_transform
        self.augraphy_transform = augraphy_transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        name, target = self.df[idx]
        img_path = os.path.join(self.path, name)
        
        try:
            img = Image.open(img_path)
            img = np.array(img)
            
            if self.augraphy_transform:
                img = self.augraphy_transform(img)

            if self.album_transform:
                img = self.album_transform(image=img)['image']
            
            return img, target
        except (IOError, OSError):
            print(f"Cannot read image: {img_path}")
            return None, None