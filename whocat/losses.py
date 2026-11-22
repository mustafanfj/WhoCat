import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    """Additive angular margin softmax (Deng et al., CVPR 2019) with margin warm-up."""

    def __init__(self, dim, n_classes, s=32.0, m=0.4):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, dim) * 0.01)
        self.s, self.m_max, self.m = s, m, m

    def forward(self, emb, y):
        cos = F.linear(F.normalize(emb), F.normalize(self.W)).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos.gather(1, y[:, None]))
        target = torch.cos(theta + self.m)
        # keep the target monotonic when theta + m > pi
        target = torch.where(theta + self.m < math.pi, target, cos.gather(1, y[:, None]) - self.m * math.sin(self.m))
        logits = cos.scatter(1, y[:, None], target) * self.s
        return F.cross_entropy(logits, y)


def batch_hard_triplet(emb, y, margin=0.3):
    """Hardest positive / hardest negative in the batch, cosine distance (Hermans et al., 2017).

    With coat-aware batches the hardest negatives are same-coat cats, so this term is
    where look-alike-aware batching changes what the model learns.
    """
    e = F.normalize(emb.float())
    d = 1 - e @ e.T
    same = y[:, None] == y[None, :]
    eye = torch.eye(len(y), dtype=torch.bool, device=y.device)
    pos = d.masked_fill(~same | eye, -1).max(1).values
    neg = d.masked_fill(same, 9).min(1).values
    valid = (same & ~eye).any(1)
    return F.relu(pos - neg + margin)[valid].mean(), (neg < pos)[valid].float().mean()
