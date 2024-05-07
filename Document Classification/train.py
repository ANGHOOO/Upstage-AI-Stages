from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
import torch.nn as nn

# 여기서 Mixup을 위한 설정을 추가합니다.
mixup_fn = Mixup(
    mixup_alpha=0.3, cutmix_alpha=0.0, prob=0.8, switch_prob=0.5, mode='elem',
    label_smoothing=0.1, num_classes=17
)

# Mixup 사용 시 SoftTargetCrossEntropy 사용, 아니면 기본 CrossEntropyLoss 사용
criterion = SoftTargetCrossEntropy() if mixup_fn is not None else nn.CrossEntropyLoss()