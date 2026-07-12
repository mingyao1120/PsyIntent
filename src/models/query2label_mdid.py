# --------------------------------------------------------
# Query2Label — MDID variant (feature-based, no image backbone)
# Written by Shilong Liu
# --------------------------------------------------------

import os, sys
import os.path as osp

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import math

from models.backbone_mdid import build_backbone
from models.transformer_mdid import build_transformer
from utils.misc import clean_state_dict

class GroupWiseLinear(nn.Module):
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


class Qeruy2Label(nn.Module):
    def __init__(self, backbone, transfomer, num_class):
        """PsyIntent model for MDID.

        MDID images are not publicly available, so PsyIntent runs on the
        user-provided image caption ("Raw") together with pre-extracted visual
        features; there is no image backbone and no emotion encoder here.

        Args:
            backbone: dummy backbone exposing ``num_channels``.
            transfomer: PsyIntent transformer (VPI + PQG + decoder).
            num_class: number of intent categories.
        """
        super().__init__()

        self.transformer = transfomer
        self.num_class = num_class

        hidden_dim = transfomer.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_class, hidden_dim)  # learnable queries
        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)

        # Project BERT (768-d) psychological-text features to the hidden dim.
        self.senten_proj = nn.Linear(768, hidden_dim)


    def forward(self, input, cap_feats):
        # MDID: ``input`` is a pre-extracted visual feature vector.
        src = input

        cap_feats_proj = self.senten_proj(cap_feats.last_hidden_state)  # (B, len, 768) -> (B, len, hidden_dim)
        if cap_feats_proj.size(0) != src.size(0):
            cap_feats_proj = cap_feats_proj.repeat(2, 1, 1)

        query_input = self.query_embed.weight  # (num_class, hidden_dim)
        # For MDID the visual and emotion pathways coincide (pre-extracted features).
        hs, _, text_embd, visual_embd = self.transformer(
            self.input_proj(src), query_input, None, cap_feats_proj, self.input_proj(src)
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
