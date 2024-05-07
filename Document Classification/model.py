import torch.nn as nn
import timm
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
import torch

class EfficientNetB4(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB4, self).__init__()
        self.effb4 = timm.create_model('efficientnet_b4', pretrained=True)
        in_features = self.effb4.classifier.in_features
        self.effb4.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512),
            nn.BatchBatch1d(512),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.BatchBatch1d(256),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.effb4(x)

def get_model(config):
    model = EfficientNetB4(num_classes=config.num_classes)
    model = model.to(config.device)
    mixup_fn = Mixup(
        mixup_alpha=config.mixup_alpha,
        cutmix_alpha=config.cutmix_alpha,
        prob=config.prob,
        switch_prob=config.switch_prob,
        mode='elem',
        label_smoothing=config.label_smoothing,
        num_classes=config.num_classes
    )
    criterion = SoftTargetCrossEntropy() if mixup_fn is not None else nn.CrossEntropyLoss()
    return model, criterion, mixup_fn
