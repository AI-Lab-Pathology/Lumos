
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========= Focal Loss =========
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        ce_loss = F.cross_entropy(
            input, target,
            weight=self.alpha,
            reduction='none',
            ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ========= Label Smoothing CrossEntropy =========
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, input, target):
        log_probs = F.log_softmax(input, dim=-1)
        n_classes = input.size(-1)

        
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(true_dist * log_probs).sum(dim=1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def build_loss(loss_name: str, **kwargs):

    loss_name = loss_name.lower()
    if loss_name == 'ce':
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_name == 'focal':
        return FocalLoss(**kwargs)
    elif loss_name == 'lsce':  
        return LabelSmoothingCrossEntropy(**kwargs)
    elif loss_name == 'bce':
        return nn.BCEWithLogitsLoss(**kwargs)
    elif loss_name == 'mse':
        return nn.MSELoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")
