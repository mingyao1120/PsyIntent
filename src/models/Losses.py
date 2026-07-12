import torch
import torch.nn as nn
import math

class DUDCLoss(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, eps=1e-5):
        super(DUDCLoss, self).__init__()
        self.eps = eps
        self.alpha = None
        self.out1=self.out2=self.loss_multi=self.loss_sample=self.new1=self.new2=self.loss_single=None

    def forward(self, out1, out2, target, alpha):
        """"
        Parameters
        ----------
        out1: input probability from one stream
        out2: input probability from other stream
        """
        # alpha balances the UDC (multi) and DDC (single) terms; default 0.4,
        # inherited from the LabCR baseline (Shi et al., 2024).
        self.target = target
        self.alpha = alpha

        # Calculating Probabilities on single — vectorized multi_sharp
        self.out1 = out1
        self.out2 = out2
        loss_list = []
        for i in range(out1.size(0)):
            self.new1 = multi_sharp(out1[i], target[i])
            self.new2 = multi_sharp(out2[i], target[i])
            loss_list.append(cross(self.new1, self.new2, self.eps) + cross(self.new2, self.new1, self.eps))
        self.loss_single = torch.stack(loss_list).mean()

        # Calculating Probabilities on multi
        self.out1 = nn.functional.sigmoid(out1)
        self.out2 = nn.functional.sigmoid(out2)
        self.loss_multi = cross(self.out1, self.out2, self.eps) + cross(self.out2, self.out1, self.eps)


        loss = self.loss_multi * self.alpha + (1-self.alpha)*self.loss_single

        return loss

class ISDLoss(nn.Module):
    '''Instance Similarity Distribution (IRP) loss.

    The temperature ``tau`` (tau_feat, default 0.4) is a fixed hyperparameter
    inherited from the LabCR baseline (Shi et al., TIP 2024), kept for a fair
    comparison with the baseline.
    '''

    def __init__(self, tau=0.4, eps=1e-5):
        super(ISDLoss, self).__init__()
        self.tau = tau
        self.eps = eps
        self.f1=self.f2=self.length_b=self.mean_f1=self.mean_f2=self.length=self.diag=self.new1=self.new2=self.out1=self.out2=None

    def forward(self, feat1, feat2):
        """"
        Parameters
        ----------
        feat1: input features from one stream
        feat2: input features from other stream
        """
        self.f1 = feat1
        self.f2 = feat2

        self.length_b = feat1.size(0)  # batch_size

        self.mean_f1 = torch.reshape(self.f1, (self.length_b, -1))
        self.mean_f2 = torch.reshape(self.f2, (self.length_b, -1))
        self.mean_f1 = nn.functional.normalize(self.mean_f1, 2, dim=1)
        self.mean_f2 = nn.functional.normalize(self.mean_f2, 2, dim=1)

        self.length = self.mean_f1.size(0)
        # Zero out diagonal without allocating a full eye matrix on CPU then moving to GPU
        self.new1 = torch.mm(self.mean_f1, self.mean_f1.t()) / self.tau
        self.new2 = torch.mm(self.mean_f2, self.mean_f2.t()) / self.tau

        # Fill diagonal with 0 (in-place, avoids CPU↔GPU transfer)
        self.new1.fill_diagonal_(0)
        self.new2.fill_diagonal_(0)

        self.out1 = self.new1.flatten()[:-1].view(self.length - 1, self.length + 1)[:, 1:].flatten().view(self.length, self.length - 1)  # B*(B-1)
        self.out2 = self.new2.flatten()[:-1].view(self.length - 1, self.length + 1)[:, 1:].flatten().view(self.length, self.length - 1)  # B*(B-1)

        self.out1 = nn.functional.softmax(self.out1, dim=-1)
        self.out2 = nn.functional.softmax(self.out2, dim=-1)

        loss = KL(self.out1, self.out2, self.eps) + KL(self.out2, self.out1, self.eps)
        return loss

def EH(probs, eps):

    ent = - (probs * (probs + eps).log()).sum(dim=1)
    mean = ent.mean()
    torch.distributed.all_reduce(mean)
    return mean

def KL(out1,out2,eps):
    kl = (out1 * (out1 + eps).log() - out1 * (out2 + eps).log()).sum(dim=1)
    kl = kl.mean()
    torch.distributed.all_reduce(kl)
    return kl

def cross(out1,out2,eps):
    loss = -(out1 * (out2 + eps).log()).sum(dim=1)
    loss = loss.mean()
    torch.distributed.all_reduce(loss)
    return loss

def multi_sharp(out, target):
    non = torch.nonzero(target)  # index of positive classes
    count = len(non)
    neg_logits = out[target == 0]  # (n_neg,)
    pos_logits = out[non]          # (count,)

    # Build sharp matrix: (count, n_neg + 1), last column is the positive logit
    neg_expanded = neg_logits.unsqueeze(0).expand(count, -1)  # (count, n_neg) — no copy
    sharp_mar = torch.cat([neg_expanded, pos_logits.view(-1, 1)], dim=1)  # (count, n_neg + 1)
    sharp_mar = nn.functional.softmax(sharp_mar, dim=-1)

    return sharp_mar
