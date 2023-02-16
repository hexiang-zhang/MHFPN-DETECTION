# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16
import torch

from ..builder import NECKS
from .fpn import FPN


@NECKS.register_module()
class MHFPN(FPN):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(PAHRFPN, self).__init__(
            in_channels,
            out_channels,
            num_outs,
            start_level,
            end_level,
            add_extra_convs,
            relu_before_extra_convs,
            no_norm_on_lateral,
            conv_cfg,
            norm_cfg,
            act_cfg,
            init_cfg=init_cfg)
        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        for i in range(self.start_level + 1, self.backbone_end_level):
            d_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            pafpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)
        self.reduction_conv = ConvModule(
            512,
            out_channels,
            kernel_size=1,
            act_cfg=None)

        self.pooling = F.max_pool2d


    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')

        # build outputs
        # part 1: from original levels
        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            inter_outs[i + 1] += self.downsample_convs[i](inter_outs[i])
        
        # part 3:Build two aggregate headers
        hr_in12 = []
        hr_in34 = []
        hr_in12.append(inter_outs[2])
        hr_in12.extend([F.interpolate(inter_outs[3], scale_factor=2, mode='nearest')])
        hr_in34.append(inter_outs[0])
        hr_in34.extend([F.interpolate(inter_outs[1], scale_factor=2, mode='nearest')])
        #Add convolution module
        hr_out12 = self.reduction_conv(torch.cat(hr_in12, dim=1))
        hr_out34 = self.reduction_conv(torch.cat(hr_in34, dim=1))
        #Use pool (maximum pooling) to expand
        outputs = []
        outputs.append(self.pafpn_convs[0](hr_out34))
        outputs.extend([self.pafpn_convs[0](self.pooling(hr_out34, kernel_size=2, stride=2))])
        outputs.extend([self.pafpn_convs[1](hr_out12)])
        outputs.extend([self.pafpn_convs[2](self.pooling(hr_out12, kernel_size=2, stride=2))])
      
        # part 4: add extra levels
        if self.num_outs > len(outputs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outputs.append(F.max_pool2d(outputs[-1], 1, stride=2))
        
        return tuple(outputs)
