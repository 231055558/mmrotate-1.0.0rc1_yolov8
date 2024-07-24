# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union
import torch
import torch.nn as nn
from mmdet.utils import ConfigType, OptMultiConfig
import mmdet.models
from mmrotate.registry import MODELS
from .. import CSPLayerWithTwoConv
from ..utils import make_divisible, make_round
from .yolov5_pafpn import YOLOv5PAFPN


@MODELS.register_module()
class YOLOv8PAFPN_EX(YOLOv5PAFPN):
    """Path Aggregation Network used in YOLOv8.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Defaults to 1.
        freeze_all(bool): Whether to freeze the model
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 real_out_channels: List[int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None,
                 ):
        self.real_out_channels = real_out_channels
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            num_csp_blocks=num_csp_blocks,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)
        self.real_out_layers = nn.ModuleList()
        self.ez_down_layer = nn.ModuleList()
        self.extra_layer = nn.ModuleList()
        for idx in range(len(self.real_out_channels)):
            self.real_out_layers.append(self.build_real_out_layer(idx))
        for idx in range(len(self.real_out_layers)):
            if idx >= len(self.in_channels):
                self.ez_down_layer.append(self.build_ez_down_layer(idx))
        # self.extra_layer.append(nn.Sequential(nn.Conv2d(int(self.in_channels[-1]*self.widen_factor),
        #                                                 self.out_channels[-1],
        #                                                 kernel_size=3,
        #                                                 stride=(2,2),
        #                                                 padding=1,
        #                                                 bias=False),
        #                                       nn.BatchNorm2d(self.out_channels[-1]),
        #                                       nn.ReLU(inplace=True)))

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        return nn.Identity()

    def build_extra_layer(self, idx: int) -> nn.Module:
        return

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        return CSPLayerWithTwoConv(
            make_divisible((self.in_channels[idx - 1] + self.in_channels[idx]),
                           self.widen_factor),
            make_divisible(self.out_channels[idx - 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        return CSPLayerWithTwoConv(
            make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]),
                self.widen_factor),
            make_divisible(self.out_channels[idx + 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_out_layer(self, idx: int) -> nn.Module:
        return nn.Identity()

    def build_real_out_layer(self, idx: int) -> nn.Module:
            return nn.Sequential(nn.Conv2d(int(self.out_channels[idx] * self.widen_factor),
                                   self.real_out_channels[idx],
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias=False),
                         nn.BatchNorm2d(self.real_out_channels[idx]),
                         nn.ReLU(inplace=True))

    def build_fpn_layer(self, idx: int) -> nn.Module:
        return nn.Conv2d(int(self.out_channels[idx-1] * self.widen_factor),
                                           int(self.out_channels[idx] * self.widen_factor),
                                           kernel_size=3,
                                           stride=(2,2),
                                           padding=1,
                                           bias=False)

    def build_ez_down_layer(self, idx: int) -> nn.Module:
        return nn.Conv2d(int(self.out_channels[idx-1] * self.widen_factor),
                         int(self.out_channels[idx] * self.widen_factor),
                         kernel_size=1,
                         stride=(2,2),
                         padding=0,
                         bias=False)

    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](inputs[idx]))

        # reduce_outs.append(self.extra_layer[0](reduce_outs[-1]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](
                                                     feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs)
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))


        x = results
        for idx in range(len(self.real_out_layers)-len(self.in_channels)):
            x.append(self.ez_down_layer[idx](x[idx + len(self.in_channels)-1]))
        results = []
        for idx in range(len(self.real_out_channels)):
            results.append(self.real_out_layers[idx](x[idx]))

        return tuple(results)

@MODELS.register_module()
class YOLOv8PAFPN_newcon(YOLOv5PAFPN):
    """Path Aggregation Network used in YOLOv8.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Defaults to 1.
        freeze_all(bool): Whether to freeze the model
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 real_out_channels: List[int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None,
                 ):
        self.real_out_channels = real_out_channels
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            num_csp_blocks=num_csp_blocks,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)
        self.real_out_layers = nn.ModuleList()
        self.ez_down_layer = nn.ModuleList()
        self.extra_layer = nn.ModuleList()
        for idx in range(len(self.real_out_channels)):
            self.real_out_layers.append(self.build_real_out_layer(idx))
        for idx in range(len(self.real_out_layers)):
            if idx >= len(self.in_channels):
                self.ez_down_layer.append(self.build_ez_down_layer(idx))


    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        return nn.Identity()

    def build_extra_layer(self, idx: int) -> nn.Module:
        return

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        return CSPLayerWithTwoConv(
            make_divisible((self.in_channels[idx - 1] + self.in_channels[idx]),
                           self.widen_factor),
            make_divisible(self.out_channels[idx - 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        return CSPLayerWithTwoConv(
            make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]),
                self.widen_factor),
            make_divisible(self.out_channels[idx + 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_out_layer(self, idx: int) -> nn.Module:
        return nn.Identity()

    def build_real_out_layer(self, idx: int) -> nn.Module:
            return nn.Sequential(nn.Conv2d(int(self.out_channels[idx] * self.widen_factor),
                                   self.real_out_channels[idx],
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias=False),
                         nn.BatchNorm2d(self.real_out_channels[idx]),
                         nn.ReLU(inplace=True))

    def build_fpn_layer(self, idx: int) -> nn.Module:
        return nn.Conv2d(int(self.out_channels[idx-1] * self.widen_factor),
                                           int(self.out_channels[idx] * self.widen_factor),
                                           kernel_size=3,
                                           stride=(2,2),
                                           padding=1,
                                           bias=False)

    def build_ez_down_layer(self, idx: int) -> nn.Module:
        return nn.Conv2d(int(self.out_channels[idx-1] * self.widen_factor),
                         int(self.out_channels[idx] * self.widen_factor),
                         kernel_size=1,
                         stride=(2,2),
                         padding=0,
                         bias=False)

    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](inputs[idx]))


        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](
                                                     feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs)
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))


        x = results
        for idx in range(len(self.real_out_layers)-len(self.in_channels)):
            x.append(self.ez_down_layer[idx](x[idx + len(self.in_channels)-1]))
        results = []
        for idx in range(len(self.real_out_channels)):
            results.append(self.real_out_layers[idx](x[idx]))




        return tuple(results)

@MODELS.register_module()
class YOLOv8PAFPN_NBUP(YOLOv5PAFPN):
    """Path Aggregation Network used in YOLOv8.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Defaults to 1.
        freeze_all(bool): Whether to freeze the model
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 real_out_channels: List[int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None,
                 ):
        self.real_out_channels = real_out_channels
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            num_csp_blocks=num_csp_blocks,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)
        self.real_out_layers = nn.ModuleList()
        self.ez_down_layer = nn.ModuleList()
        self.nb_upsample_layers = nn.ModuleList()
        for idx in range(len(self.real_out_channels)):
            self.real_out_layers.append(self.build_real_out_layer(idx))
        for idx in range(len(self.real_out_layers)):
            if idx >= len(self.in_channels):
                self.ez_down_layer.append(self.build_ez_down_layer(idx))
        for idx in range(len(self.in_channels) - 1):
            self.nb_upsample_layers.append(self.build_nb_upsample_layers(idx))
        print("ok")
    def build_nb_upsample_layers(self, idx: int) -> nn.Module:
        ln = len(self.in_channels)
        return nn.Sequential(nn.ConvTranspose2d(int(self.in_channels[ln-idx-1] * self.widen_factor),
                                                int(self.in_channels[ln-idx-1] * self.widen_factor),
                                                kernel_size=4,
                                                stride=2,
                                                padding=1,
                                                bias=False),
                             nn.BatchNorm2d(int(self.in_channels[ln-idx-1] * self.widen_factor)),
                             nn.ReLU(inplace=True))



    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        return nn.Identity()

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        return CSPLayerWithTwoConv(
            make_divisible((self.in_channels[idx - 1] + self.in_channels[idx]),
                           self.widen_factor),
            make_divisible(self.out_channels[idx - 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        return CSPLayerWithTwoConv(
            make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]),
                self.widen_factor),
            make_divisible(self.out_channels[idx + 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_out_layer(self, idx: int) -> nn.Module:
        return nn.Identity()

    def build_real_out_layer(self, idx: int) -> nn.Module:
            return nn.Sequential(nn.Conv2d(int(self.out_channels[idx] * self.widen_factor),
                                   self.real_out_channels[idx],
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias=False),
                         nn.BatchNorm2d(self.real_out_channels[idx]),
                         nn.ReLU(inplace=True))

    def build_fpn_layer(self, idx: int) -> nn.Module:
        return nn.Conv2d(int(self.out_channels[idx-1] * self.widen_factor),
                                           int(self.out_channels[idx] * self.widen_factor),
                                           kernel_size=3,
                                           stride=(2,2),
                                           padding=1,
                                           bias=False)

    def build_ez_down_layer(self, idx: int) -> nn.Module:
        return nn.Conv2d(int(self.out_channels[idx-1] * self.widen_factor),
                         int(self.out_channels[idx] * self.widen_factor),
                         kernel_size=1,
                         stride=(2,2),
                         padding=0,
                         bias=False)

    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](inputs[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]

            upsample_feat = self.nb_upsample_layers[len(self.in_channels) - 1 -
                                                 idx](
                                                     feat_high)

            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs)
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))


        x = results
        for idx in range(len(self.real_out_layers)-len(self.in_channels)):
            x.append(self.ez_down_layer[idx](x[idx + len(self.in_channels)-1]))
        results = []
        for idx in range(len(self.real_out_channels)):
            results.append(self.real_out_layers[idx](x[idx]))

        return tuple(results)


@MODELS.register_module()
class YOLOv8PAFPN_EXTRACSP(YOLOv5PAFPN):
    """Path Aggregation Network used in YOLOv8.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Defaults to 1.
        freeze_all(bool): Whether to freeze the model
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 real_out_channels: List[int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None,
                 ):
        self.real_out_channels = real_out_channels
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            num_csp_blocks=num_csp_blocks,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)
        self.real_out_layers = nn.ModuleList()
        self.fpn_layers = nn.ModuleList()
        self.before_top = nn.ModuleList()
        self.before_down = nn.ModuleList()
        for idx in range(len(self.real_out_channels)):
            self.real_out_layers.append(self.build_real_out_layer(idx))
        for idx in range(len(self.real_out_layers)):
            if idx >= len(self.in_channels):
                self.fpn_layers.append(self.build_fpn_layer(idx))
        for idx in range(len(self.in_channels) - 1):
            self.before_top.append(self.build_before_top(idx))
        for idx in range(len(self.in_channels) - 1):
            self.before_down.append(self.build_before_down(idx))

    def build_before_down(self, idx: int) -> nn.Module:
        return CSPLayerWithTwoConv(
            make_divisible(self.in_channels[idx + 1],
                           self.widen_factor),
            make_divisible(self.in_channels[idx + 1],
                           self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_before_top(self, idx: int) -> nn.Module:
        return CSPLayerWithTwoConv(
            make_divisible(self.in_channels[idx],
                           self.widen_factor),
            make_divisible(self.in_channels[idx],
                           self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        return nn.Identity()

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        return CSPLayerWithTwoConv(
            make_divisible((self.in_channels[idx - 1] + self.in_channels[idx]),
                           self.widen_factor),
            make_divisible(self.out_channels[idx - 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        return CSPLayerWithTwoConv(
            make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]),
                self.widen_factor),
            make_divisible(self.out_channels[idx + 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_out_layer(self, idx: int) -> nn.Module:
        return nn.Identity()

    def build_real_out_layer(self, idx: int) -> nn.Module:
        return nn.Sequential(nn.Conv2d(int(self.out_channels[idx] * self.widen_factor),
                                       self.real_out_channels[idx],
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=False),
                             nn.BatchNorm2d(self.real_out_channels[idx]),
                             nn.ReLU(inplace=True))

    def build_fpn_layer(self, idx: int) -> nn.Module:
        return nn.Conv2d(int(self.out_channels[idx - 1] * self.widen_factor),
                         int(self.out_channels[idx] * self.widen_factor),
                         kernel_size=3,
                         stride=(2, 2),
                         padding=1,
                         bias=False)

    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](inputs[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            feat_low = self.before_top[idx-1](feat_low)
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](
                feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs)
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            feat_high = self.before_down[idx](feat_high)
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        x = results
        for idx in range(len(self.real_out_layers) - len(self.in_channels)):
            x.append(self.fpn_layers[idx](x[idx + len(self.in_channels) - 1]))
        results = []
        for idx in range(len(self.real_out_channels)):
            results.append(self.real_out_layers[idx](x[idx]))

        return tuple(results)

@MODELS.register_module()
class YOLOv8PAFPN_SPLIT(YOLOv5PAFPN):
    """Path Aggregation Network used in YOLOv8.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Defaults to 1.
        freeze_all(bool): Whether to freeze the model
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 real_out_channels: List[int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None,
                 ):
        self.real_out_channels = real_out_channels
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            num_csp_blocks=num_csp_blocks,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)
        self.real_out_layers = nn.ModuleList()
        for idx in range(len(self.real_out_channels)):
            self.real_out_layers.append(self.build_real_out_layer(idx))

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        return nn.Identity()

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        return CSPLayerWithTwoConv(
            make_divisible((self.in_channels[idx - 1] + self.in_channels[idx]),
                           self.widen_factor),
            make_divisible(self.out_channels[idx - 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        return CSPLayerWithTwoConv(
            make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]),
                self.widen_factor),
            make_divisible(self.out_channels[idx + 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_out_layer(self, idx: int) -> nn.Module:
        return nn.Identity()

    def build_real_out_layer(self, idx: int) -> nn.Module:
        if idx < 2:
            return nn.Sequential(nn.Conv2d(int(self.out_channels[idx] * self.widen_factor),
                                   self.real_out_channels[idx],
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias=False),
                         nn.BatchNorm2d(self.real_out_channels[idx]),
                         nn.ReLU(inplace=True))
        else:
            return nn.Sequential(nn.Conv2d(int(self.out_channels[-1] *0.5 * self.widen_factor),
                                           self.real_out_channels[idx],
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           bias=False),
                                 nn.BatchNorm2d(self.real_out_channels[idx]),
                                 nn.ReLU(inplace=True))



    def split_channels(self, inputs) -> list:
        channels = inputs.size(1)
        split_1 = inputs[:,:channels//2,:,:]
        split_2 = inputs[:,channels//2:,:,:]
        return [split_1, split_2]

    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](inputs[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](
                                                     feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs)
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)-1):
            results.append(self.out_layers[idx](outs[idx]))
        for i in self.split_channels(outs[-1]):
            results.append(i)
        outputs = []
        for idx in range(len(self.real_out_channels)):
            outputs.append(self.real_out_layers[idx](results[idx]))

        return tuple(outputs)


@MODELS.register_module()
class YOLOv8PAFPN_FPN(YOLOv5PAFPN):
    """Path Aggregation Network used in YOLOv8.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Defaults to 1.
        freeze_all(bool): Whether to freeze the model
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 real_out_channels: List[int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        self.real_out_channels = real_out_channels
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            num_csp_blocks=num_csp_blocks,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)
        self.fpn_layers = nn.ModuleList()
        self.fpn_layers.append(mmdet.models.FPN(in_channels=[x // 2 for x in self.in_channels], out_channels=self.real_out_channels[0], num_outs=5))

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        return nn.Identity()

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        return CSPLayerWithTwoConv(
            make_divisible((self.in_channels[idx - 1] + self.in_channels[idx]),
                           self.widen_factor),
            make_divisible(self.out_channels[idx - 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        return CSPLayerWithTwoConv(
            make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]),
                self.widen_factor),
            make_divisible(self.out_channels[idx + 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](inputs[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](
                feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs)
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        # out_to_fpn
        outputs = self.fpn_layers[0](tuple(results))

        return tuple(outputs)