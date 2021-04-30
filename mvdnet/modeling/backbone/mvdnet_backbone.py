import math
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
from torch import nn
import torch
import numpy as np

from detectron2.layers import (
    Conv2d, ShapeSpec, get_norm, cat
)
from mvdnet.layers import MaxPool2d, ConvTranspose2d

from detectron2.modeling import Backbone
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.modeling.backbone.resnet import BasicStem

class MVDNetStem(BasicStem):
    def __init__(self, in_channels=3, out_channels=64, norm="BN"):
        super().__init__()
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        weight_init.c2_msra_fill(self.conv1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu_(x)
        return x

    @property
    def out_channels(self):
        return self.conv1.out_channels

    @property
    def stride(self):
        return 1

class VGG(Backbone):
    def __init__(self, in_channels=3, stem_out_channels=16, norm="FrozenBN",
        num_layers_per_stage=[2,2,3,3], out_channels_per_stage=[32,64,128,256],
        history_on=True, num_history=5):

        super(VGG, self).__init__()
        self.num_layers_per_stage = num_layers_per_stage
        self.out_channels_per_stage = out_channels_per_stage
        self.history_on = history_on        
        self.num_history = num_history

        self.stem = MVDNetStem(
            in_channels=in_channels,
            out_channels=stem_out_channels,
            norm="FrozenBN"
        )

        self.downsampler = []
        num_channels = [stem_out_channels] + self.out_channels_per_stage
        for i, num_layers in enumerate(num_layers_per_stage):
            in_channels = num_channels[i]
            out_channels = num_channels[i+1]
            layers = []
            if i > 0:
                pool = MaxPool2d(2)
                layers.append(pool)
            for j in range(num_layers):
                conv = Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    norm=get_norm(norm, out_channels),
                    # activation=F.relu_
                    activation=F.leaky_relu_
                )
                weight_init.c2_msra_fill(conv)
                layers.append(conv)
                in_channels = out_channels
            stage = nn.Sequential(*layers)
            self.downsampler.append(stage)
            name = "down" + str(i+1)
            self.add_module(name, stage)

        out_channels_per_stage_flip = self.out_channels_per_stage[::-1]
        num_stage = len(out_channels_per_stage_flip)
        self.upsampler = []
        self.pyramid_fusion = []
        for i in range(num_stage-1):
            in_channels = out_channels_per_stage_flip[i]
            out_channels = out_channels_per_stage_flip[i+1]
            upconv = ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,              
                output_padding=1,
                bias=False,
                norm=get_norm(norm, out_channels),
                # activation=F.relu_
                activation=F.leaky_relu_
            )
            weight_init.c2_msra_fill(upconv)
            self.upsampler.append(upconv)
            name = "up" + str(num_stage-1-i)
            self.add_module(name, upconv)

            in_channels = 2 * out_channels
            out_channels = out_channels
            fusion = Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                norm=get_norm(norm, out_channels),
                # activation=F.relu_
                activation=F.leaky_relu_
            )
            weight_init.c2_msra_fill(fusion)
            self.pyramid_fusion.append(fusion)
            name = "fusion" + str(num_stage-1-i)
            self.add_module(name, fusion)

        if self.history_on:
            self.final_fusion = Conv2d(
                out_channels * self.num_history,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False,
                norm=get_norm(norm, out_channels),
                # activation=F.relu_
                activation=F.leaky_relu_
            )

    def forward(self, x):
        if self.history_on:
            feature_maps = []
            for j in range(self.num_history):
                x_j = self.stem(x[:,j,:,:,:])
                vgg_output = []
                for i, downsampler in enumerate(self.downsampler):
                    x_j = downsampler(x_j)
                    vgg_output.append(x_j)

                for i, (upsampler, shortcut, fusion) in enumerate(zip(
                    self.upsampler, vgg_output[-2::-1], self.pyramid_fusion
                )):
                    x_j = upsampler(x_j)
                    x_j = cat([x_j, shortcut], dim=1)
                    x_j = fusion(x_j)
                feature_maps.append(x_j)
            rpn_feature_map = self.final_fusion(cat(feature_maps, dim=1))
            features = {"rpn":rpn_feature_map, "rcnn":feature_maps}
        else:
            x = self.stem(x)
            vgg_output = []
            for i, downsampler in enumerate(self.downsampler):
                x = downsampler(x)
                vgg_output.append(x)

            for i, (upsampler, shortcut, fusion) in enumerate(zip(
                self.upsampler, vgg_output[-2::-1], self.pyramid_fusion
            )):
                x = upsampler(x)
                x = cat([x, shortcut], dim=1)
                x = fusion(x)

            features = {"rpn":x, "rcnn":x}
        return features

    def output_shape(self):
        output_shapes = {
            "rpn": ShapeSpec(
                channels=self.out_channels_per_stage[0], stride=self.stem.stride
            ),
            "rcnn": ShapeSpec(
                channels=self.out_channels_per_stage[0], stride=self.stem.stride
            )
        }
        return output_shapes

class MVDNetBackbone(Backbone):
    def __init__(self, radar_backbone, lidar_backbone):
        super(MVDNetBackbone, self).__init__()
        self.radar_backbone = radar_backbone
        self.lidar_backbone = lidar_backbone

    def forward(self, radar_data, lidar_data):
        radar_features = self.radar_backbone(radar_data)
        lidar_features = self.lidar_backbone(lidar_data)

        features = {"radar_rpn":radar_features["rpn"], 
            "radar_rcnn":radar_features["rcnn"],
            "lidar_rpn":lidar_features["rpn"],
            "lidar_rcnn":lidar_features["rcnn"]}

        return features

    def output_shape(self):
        output_shapes = {
            "radar_rpn": ShapeSpec(
                channels=self.radar_backbone.out_channels_per_stage[0], stride=self.radar_backbone.stem.stride
            ),
            "radar_rcnn": ShapeSpec(
                channels=self.radar_backbone.out_channels_per_stage[0], stride=self.radar_backbone.stem.stride
            ),
            "lidar_rpn": ShapeSpec(
                channels=self.lidar_backbone.out_channels_per_stage[0], stride=self.lidar_backbone.stem.stride
            ),
            "lidar_rcnn": ShapeSpec(
                channels=self.lidar_backbone.out_channels_per_stage[0], stride=self.lidar_backbone.stem.stride
            )
        }
        return output_shapes

@BACKBONE_REGISTRY.register()
def build_mvdnet_backbone(cfg, input_shape: ShapeSpec):
    stem_out_channels = cfg.MODEL.MVDNET.STEM_OUT_CHANNELS
    stage_layers = cfg.MODEL.MVDNET.STAGE_LAYERS
    radar_stage_out_channels = cfg.MODEL.MVDNET.RADAR_STAGE_OUT_CHANNELS
    lidar_stage_out_channels = cfg.MODEL.MVDNET.LIDAR_STAGE_OUT_CHANNELS
    radar_norm = cfg.MODEL.MVDNET.RADAR_NORM
    lidar_norm = cfg.MODEL.MVDNET.LIDAR_NORM
    num_history = cfg.INPUT.NUM_HISTORY+1
    history_on = cfg.INPUT.HISTORY_ON

    lidar_channels = np.int(np.round(
        (cfg.INPUT.LIDAR.PROJECTION.HEIGHT_UB 
        - cfg.INPUT.LIDAR.PROJECTION.HEIGHT_LB)
        / cfg.INPUT.LIDAR.PROJECTION.DELTA_H))+1

    radar_backbone = VGG(
        in_channels=1,
        stem_out_channels=stem_out_channels,
        norm=radar_norm,
        num_layers_per_stage=stage_layers,
        out_channels_per_stage=radar_stage_out_channels,
        history_on=history_on,
        num_history=num_history
    )

    lidar_backbone = VGG(
        in_channels=lidar_channels,
        stem_out_channels=stem_out_channels,
        norm=lidar_norm,
        num_layers_per_stage=stage_layers,
        out_channels_per_stage=lidar_stage_out_channels,
        history_on=history_on,
        num_history=num_history
    )

    backbone = MVDNetBackbone(
        radar_backbone,
        lidar_backbone
    )
    return backbone