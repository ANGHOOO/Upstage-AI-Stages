# utils.py
import sys
import glob
import cv2
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils, datasets, models
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim import lr_scheduler
from torch.autograd import Variable
from matplotlib import pyplot as plt
from time import time
import os
import time
import random
import timm
import torch
import albumentations as A
import pandas as pd
import numpy as np
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from torch.utils.data.dataloader import DataLoader, default_collate
import wandb
from augraphy import *

# Your function and class definitions here

# main.py
from utils import *

def main():
    # Your main execution code here

if __name__ == "__main__":
    main()