import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict

from detectron2.layers import Conv2d, Linear, ShapeSpec, get_norm
from detectron2.modeling.roi_heads import ROI_BOX_HEAD_REGISTRY
from ..attention import SelfAttentionBlock, CrossAttentionBlock
from mvdnet.layers import Conv3d

@ROI_BOX_HEAD_REGISTRY.register()
class MVDNetBoxHead(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim     = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm       = cfg.MODEL.ROI_BOX_HEAD.NORM
        self.history_on = cfg.INPUT.HISTORY_ON
        self.num_history = cfg.INPUT.NUM_HISTORY+1
        self.pooler_size = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        assert num_fc > 0

        for f in input_shape.keys():
            if f.startswith("radar"):
                self.radar_key = f
                self.radar_output_size = input_shape[f].channels * input_shape[f].height * input_shape[f].width
                self.radar_input_channels = input_shape[f].channels
            elif f.startswith("lidar"):
                self.lidar_key = f
                self.lidar_output_size = input_shape[f].channels * input_shape[f].height * input_shape[f].width
                self.lidar_input_channels = input_shape[f].channels

        assert(self.lidar_output_size >= self.radar_output_size)
        if self.lidar_output_size != self.radar_output_size:
            self.match_conv = Conv2d(
                in_channels = self.lidar_input_channels,
                out_channels = self.radar_input_channels,
                kernel_size = 3,
                padding = 1,
                bias = False,
                norm = nn.BatchNorm2d(self.radar_input_channels),
                activation = F.leaky_relu_
            )
        else:
            self.match_conv = None
        self.radar_self_attention = SelfAttentionBlock(self.radar_output_size)
        self.lidar_self_attention = SelfAttentionBlock(self.radar_output_size)
        self.radar_cross_attention = CrossAttentionBlock(self.radar_output_size)
        self.lidar_cross_attention = CrossAttentionBlock(self.radar_output_size)

        if self.history_on:
            self.tnn1 = Conv3d(
                in_channels = self.radar_input_channels*2,
                out_channels = self.radar_input_channels,
                kernel_size = [3, 3, 3],
                padding = [1, 1, 1],
                bias=False,
                norm=nn.BatchNorm3d(self.radar_input_channels),
                activation=F.leaky_relu_
            )
            self.tnn2 = Conv3d(
                in_channels = self.radar_input_channels,
                out_channels = self.radar_input_channels,
                kernel_size = [3, 3, 3],
                padding = [1, 1, 1],
                bias=False,
                norm=nn.BatchNorm3d(self.radar_input_channels),
                activation=F.leaky_relu_
            )
            self.tnn3 = Conv3d(
                in_channels = self.radar_input_channels,
                out_channels = self.radar_input_channels,
                kernel_size = [self.num_history, 3, 3],
                padding = [0, 1, 1],
                bias=False,
                norm=nn.BatchNorm3d(self.radar_input_channels),
                activation=F.leaky_relu_
            )
            self.tnns = [self.tnn1, self.tnn2, self.tnn3]
        else:
            self.tnn = Conv2d(
                in_channels = self.radar_input_channels*2,
                out_channels = self.radar_input_channels,
                kernel_size = 3,
                padding = 1,
                bias=False,
                norm=nn.BatchNorm2d(self.radar_input_channels),
                activation=F.leaky_relu_
            )
        self._output_size = self.radar_output_size
        
        self.fcs = []
        for k in range(num_fc):
            fc = Linear(self._output_size, fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)
        if self.match_conv is not None:
            weight_init.c2_msra_fill(self.match_conv)
        if self.history_on:
            for layer in self.tnns:
                weight_init.c2_msra_fill(layer)
        else:
            weight_init.c2_msra_fill(self.tnn)

    def forward(self, x):
        radar_features = x[self.radar_key]
        lidar_features = x[self.lidar_key]

        if self.history_on:
            fusion_feature = []
            for radar_x, lidar_x in zip(radar_features, lidar_features):
                if self.match_conv is not None:
                    lidar_x = self.match_conv(lidar_x)
                radar_x = torch.flatten(radar_x, start_dim=1)
                lidar_x = torch.flatten(lidar_x, start_dim=1)
                radar_x = self.radar_self_attention(radar_x)
                lidar_x = self.lidar_self_attention(lidar_x)
                radar_y = self.radar_cross_attention([radar_x, lidar_x])
                lidar_y = self.lidar_cross_attention([lidar_x, radar_x])
                radar_y = radar_y.reshape(-1, self.radar_input_channels,
                    self.pooler_size, self.pooler_size)
                lidar_y = lidar_y.reshape(-1, self.radar_input_channels,
                    self.pooler_size, self.pooler_size)
                feature_x = torch.cat([radar_y, lidar_y], dim=1)
                fusion_feature.append(feature_x)
            fusion_feature = torch.stack(fusion_feature).permute(1,2,0,3,4).contiguous()
            for layer in self.tnns:
                fusion_feature = layer(fusion_feature)
            fusion_feature = torch.flatten(fusion_feature, start_dim=1)
        else:
            if self.match_conv is not None:
                lidar_features = self.match_conv(lidar_features)
            radar_x = torch.flatten(radar_features, start_dim=1)
            lidar_x = torch.flatten(lidar_features, start_dim=1)
            radar_x = self.radar_self_attention(radar_x)
            lidar_x = self.lidar_self_attention(lidar_x)
            radar_y = self.radar_cross_attention([radar_x, lidar_x])
            lidar_y = self.lidar_cross_attention([lidar_x, radar_x])
            radar_y = radar_y.reshape(-1, self.radar_input_channels,
                self.pooler_size, self.pooler_size)
            lidar_y = lidar_y.reshape(-1, self.radar_input_channels,
                self.pooler_size, self.pooler_size)
            feature_x = torch.cat([radar_y, lidar_y], dim=1)
            feature_x = self.tnn(feature_x)
            fusion_feature = torch.flatten(feature_x, start_dim=1)
        
        for layer in self.fcs:
            fusion_feature = F.leaky_relu_(layer(fusion_feature))
        return fusion_feature

    @property
    def output_size(self):
        return self._output_size