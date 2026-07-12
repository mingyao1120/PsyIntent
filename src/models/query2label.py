# --------------------------------------------------------
# Query2Label
# Written by Shilong Liu
# --------------------------------------------------------

import os, sys
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import math

from models.backbone import build_backbone
from models.transformer import build_transformer
from utils.misc import clean_state_dict

class GroupWiseLinear(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

from models.potential_module.EmotionNet import EmoClassifier


def _emotion_weights_path():
    """Path to the pre-trained EAMB-Net emotion weights.

    Resolves to ``<release_root>/pretrained/emotion_model.pth`` so the model file
    can locate the weights regardless of the current working directory. Override
    the location with the ``PSYINTENT_EMOTION_WEIGHTS`` environment variable.
    """
    env = os.environ.get('PSYINTENT_EMOTION_WEIGHTS', '')
    if env:
        return env
    release_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(release_root, 'pretrained', 'emotion_model.pth')


def emotion_model():
    emotion_model = EmoClassifier()
    emotion_model_path = _emotion_weights_path()
    checkpoint_emotion_model = torch.load(emotion_model_path, map_location='cpu', weights_only=False)
    emotion_model.load_state_dict(checkpoint_emotion_model['model'])
    return emotion_model


class Qeruy2Label(nn.Module):
    def __init__(self, backbone, transfomer, num_class):
        """PsyIntent model for image-based datasets (Intentonomy, MET-MeMe).

        Args:
            backbone: visual backbone (ResNet-101).
            transfomer: PsyIntent transformer (VPI + PQG + decoder).
            num_class: number of intent categories.
        """
        super().__init__()

        # EAMB-Net emotion encoder (frozen), used as the emotion feature ℰ.
        self.emotion_model = emotion_model()
        for p in self.emotion_model.parameters():
            p.requires_grad = False
        emo_out_channels = 256  # EAMB-Net emotion module output channels

        self.backbone = backbone
        self.transformer = transfomer
        self.num_class = num_class

        hidden_dim = transfomer.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_class, hidden_dim)  # learnable queries
        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)

        # Project BERT (768-d) psychological-text features to the hidden dim.
        self.senten_proj = nn.Linear(768, hidden_dim)

        # Project the EAMB-Net emotion feature map to the hidden dim.
        self.emo_proj = nn.Conv2d(in_channels=emo_out_channels, out_channels=hidden_dim, kernel_size=1)


    def forward(self, input, cap_feats):

        src, pos = self.backbone(input)  # torch.Size([B, 3, 224, 224]) -> torch.Size([B, 2048, 7, 7])
        src, pos = src[-1], pos[-1]

        # --- Psychological-text features 𝒯 (BERT last hidden state -> hidden dim) ---
        cap_feats_proj = self.senten_proj(cap_feats.last_hidden_state)
        if cap_feats_proj.size(0) != src.size(0):
            cap_feats_proj = cap_feats_proj.repeat(2, 1, 1)

        # --- Emotion features ℰ (EAMB-Net -> 7x7 feature map) ---
        logits, cam, eam, conf, gap, x = self.emotion_model(input)
        x = F.interpolate(x, size=(7, 7), mode='bilinear', align_corners=False)
        emo_feats = self.emo_proj(x)

        query_input = self.query_embed.weight  # (num_class, hidden_dim)
        hs, _, text_embd, visual_embd = self.transformer(
            self.input_proj(src), query_input, pos, cap_feats_proj, emo_feats,
            use_peqg=True,
        )
        features = hs[-1]  # (B, num_class, hidden_dim)
        out = self.fc(hs[-1])  # (B, num_class)

        return out, features, text_embd, visual_embd

    def load_backbone(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=torch.device(dist.get_rank()), weights_only=False)
        self.backbone[0].body.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(path, checkpoint['epoch']))


def build_q2l(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = Qeruy2Label(
        backbone=backbone,
        transfomer=transformer,
        num_class=args.num_class,
    )

    if not args.keep_input_proj:
        model.input_proj = nn.Identity()
        print("set model.input_proj to Identity!")

    return model
