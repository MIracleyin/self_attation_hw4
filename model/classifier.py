#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/4/15 下午9:08
# !@Author  : miracleyin @email: miracleyin@live.com
# !@File    : classifier.py

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/4/14 下午3:51
# !@Author  : miracleyin @email: miracleyin@live.com
# !@File    : model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.conformer import ConformerBlock


class Classifier(nn.Module):
    def __init__(self, d_model=240, n_spks=600, dropout=0.1):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=256, nhead=3
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        # Project the the dimension of features from d_model into speaker nums.
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_spks)
        )

    def forward(self, mels):
        """
        args:
          mels: (batch size, length, 40)
        return:
          out: (batch size, n_spks)
        """
        # out: (batch size, length, d_model)
        out = self.prenet(mels)
        # out: (length, batch size, d_model)
        out = out.permute(1, 0, 2)
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        out = self.encoder_layer(out)
        # out: (batch size, length, d_model)
        out = out.transpose(0, 1)
        # mean pooling
        stats = out.mean(dim=1)

        # out: (batch, n_spks)
        out = self.pred_layer(stats)
        return out


class ConClassifier(nn.Module):
    def __init__(self, d_model=240, n_spks=600, dropout=0.1):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)

        # 修改以下代码 conformer
        # self.encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=d_model, dim_feedforward=256, nhead=3
        # )
        # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.encoder_layer = ConformerBlock(
            dim=d_model, dim_head=256, heads=3
        )
        self.encoder = nn.Sequential(
            self.encoder_layer
        )
        # Project the the dimension of features from d_model into speaker nums.
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_spks)
        )

    def forward(self, mels):
        """
        args:
          mels: (batch size, length, 40)
        return:
          out: (batch size, n_spks)
        """
        # out: (batch size, length, d_model)
        out = self.prenet(mels)
        # out: (length, batch size, d_model)
        out = out.permute(1, 0, 2)
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        out = self.encoder_layer(out)
        # out: (batch size, length, d_model)
        out = out.transpose(0, 1)
        # mean pooling
        stats = out.mean(dim=1)

        # out: (batch, n_spks)
        out = self.pred_layer(stats)
        return out


if __name__ == '__main__':
    pass
