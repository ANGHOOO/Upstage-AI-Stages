import torch
from torch.optim.lr_scheduler import StepLR

class Config:
    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_epochs = 100
    model_path = '/data/ephemeral/home/models'
    lr = 0.001
    step_size = 5
    gamma = 0.5
    patience = 10
    mixup_alpha = 0.3
    cutmix_alpha = 0.0
    prob = 0.8
    switch_prob = 0.5
    label_smoothing = 0.1
    num_classes = 17

    @staticmethod
    def optimizer(model):
        return torch.optim.Adam(model.parameters(), lr=Config.lr)

    @staticmethod
    def scheduler(optimizer):
        return StepLR(optimizer, step_size=Config.step_size, gamma=Config.gamma)
