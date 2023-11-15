import torch
import torch.nn.functional as F
from mmseg.ops import resize

def non_saturating_loss(logits, targets):
    logits = resize(
            input=logits,
            size=targets.shape[2:],
            mode='bilinear')
    targets = targets.squeeze(1)
    probs = logits.softmax(1)
    log_prob = torch.log(1 - probs + 1e-12)
    if targets.ndim == 2:
        return - (targets * log_prob).sum(1).mean()
    else:
        return F.nll_loss(log_prob, targets, ignore_index=255)
        # targets = targets.permute(0,3,1,2)
        # return - (targets * log_prob).sum(1).mean()

class NonSaturatingLoss(torch.nn.Module):
    def __init__(self, epsilon=0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        if self.epsilon > 0: # label smoothing
            targets = targets.clone()
            n_classes = logits.shape[1]
            targets[torch.where(targets==255)] = n_classes
            onehot_targets = F.one_hot(targets, n_classes+1).float()
            onehot_targets = onehot_targets[:,:,:,:,:-1]
            targets = (1 - self.epsilon) * onehot_targets + self.epsilon / n_classes
        return non_saturating_loss(logits, targets)
