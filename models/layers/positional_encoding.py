# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from torch import Tensor
from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING
from mmdet.models.utils.positional_encoding import SinePositionalEncoding


@POSITIONAL_ENCODING.register_module()
class CustomLearnedPositionalEncoding3D(BaseModule):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 row_num_embed=256,
                 col_num_embed=256,
                 tub_num_embed=32,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        
        super(CustomLearnedPositionalEncoding3D, self).__init__(init_cfg)
        self.row_embed = nn.Embedding(row_num_embed, num_feats[0])
        self.col_embed = nn.Embedding(col_num_embed, num_feats[1])
        self.tub_embed = nn.Embedding(tub_num_embed, num_feats[2])
        
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed
        self.tub_num_embed = tub_num_embed

    def forward(self, mask, stride):
        """Forward function for `LearnedPositionalEncoding3D`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        X, Y, Z = mask.shape[-3:]
        
        x = torch.arange(0, X, step=stride, device=mask.device)
        y = torch.arange(0, Y, step=stride, device=mask.device)
        z = torch.arange(0, Z, step=stride, device=mask.device)
        
        x_embed = self.row_embed(x).view(X, 1, 1, -1).expand(X, Y, Z, self.num_feats[0])
        y_embed = self.col_embed(y).view(1, Y, 1, -1).expand(X, Y, Z, self.num_feats[1])
        z_embed = self.tub_embed(z).view(1, 1, Z, -1).expand(X, Y, Z, self.num_feats[2])

        # [X, Y, Z, num_feat * 3] ==> [C, X, Y, Z] ==> [1, C, X, Y, Z] ==> [B, C, X, Y, Z]
        pos = torch.cat((x_embed, y_embed, z_embed), dim=-1).permute(3, 0, 1, 2).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1, 1)
        
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'
        
        return repr_str

@POSITIONAL_ENCODING.register_module()
class SinePositionalEncoding3D(SinePositionalEncoding):
    """Position encoding with sine and cosine functions.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        NOTE:num_feats (int): Different with SinePositionalEncoding,
        the feature dimension is for the sum of x, y, z axis.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def forward(self, B, mask: Tensor) -> Tensor:
        """Forward function for `SinePositionalEncoding3D`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, x, y, z].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [num_feats, h, w].
        """
        assert mask.dim() == 3,\
            f'{mask.shape} should be a 3-dimensional Tensor,' \
            f' got {mask.dim()}-dimensional Tensor instead '
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        x_embed = not_mask.cumsum(0, dtype=torch.float32)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        z_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            x_embed = (x_embed + self.offset) / \
                      (x_embed[-1:, :, :] + self.eps) * self.scale
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, -1:, :] + self.eps) * self.scale
            z_embed = (z_embed + self.offset) / \
                      (z_embed[:, :, -1:] + self.eps) * self.scale
        
        dim_xy = self.num_feats // 3
        if dim_xy % 2 != 0:
            dim_xy += 1 # make sure dim_xy is even
        dim_z = self.num_feats - 2 * dim_xy
        t_xy = torch.arange(
            dim_xy, dtype=torch.float32, device=mask.device)
        t_xy = self.temperature**(2 * (t_xy // 2) / dim_xy)

        t_z = torch.arange(
            dim_z, dtype=torch.float32, device=mask.device)
        t_z = self.temperature**(2 * (t_z // 2) / dim_z)

        pos_x = x_embed[:, :, :, None] / t_xy
        pos_y = y_embed[:, :, :, None] / t_xy
        pos_z = z_embed[:, :, :, None] / t_z
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        X, Y, Z = mask.size()
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(X, Y, Z, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).view(X, Y, Z, -1)
        pos_z = torch.stack(
            (pos_z[:, :, :, 0::2].sin(), pos_z[:, :, :, 1::2].cos()),
            dim=4).view(X, Y, Z, -1)
        pos = torch.cat([pos_x, pos_y, pos_z], dim=-1).permute(3, 0, 1, 2) # (C, X，Y, Z)
        pos = pos.unsqueeze(0).expand(B, -1, -1, -1, -1) # (B, C, X，Y, Z)
        return pos


@POSITIONAL_ENCODING.register_module()
class CustomSinePositionalEncoding3D(BaseModule):
    """Position encoding with sine and cosine functions.
    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.,
                 init_cfg=None):
        super(CustomSinePositionalEncoding3D, self).__init__(init_cfg)
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask):
        """Forward function for `SinePositionalEncoding`.
        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].
        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        ndim = 3
        pos_length = (self.num_feats // (ndim * 2)) * 2
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        n_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            n_embed = (n_embed + self.offset) / \
                      (n_embed[:, -1:, :, :] + self.eps) * self.scale
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, :, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            pos_length, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / pos_length)
        pos_n = n_embed[:, :, :, :, None] / dim_t
        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, N, H, W = mask.size()
        pos_n = torch.stack(
            (pos_n[:, :, :, :, 0::2].sin(), pos_n[:, :, :, :, 1::2].cos()),
            dim=4).view(B, N, H, W, -1)
        pos_x = torch.stack(
            (pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()),
            dim=4).view(B, N, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()),
            dim=4).view(B, N, H, W, -1)
        pos = torch.cat((pos_n, pos_y, pos_x), dim=4)#.permute(0, 1, 4, 2, 3)

        gap = self.num_feats - pos.size(4)
        if gap > 0:
            pos_p = torch.zeros((*pos.shape[:-1], gap), dtype=torch.float32, device=mask.device)
            pos = torch.cat([pos, pos_p], dim=-1)
        pos = pos.permute(0, 1, 4, 2, 3)
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'temperature={self.temperature}, '
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'eps={self.eps})'
        return repr_str


@POSITIONAL_ENCODING.register_module()
class LearnedPositionalEncoding3D(BaseModule):
    """Position embedding with learnable embedding weights.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super(LearnedPositionalEncoding3D, self).__init__(init_cfg)
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.
        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].
        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat(
            (x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(
                1, w, 1)),
            dim=-1).permute(2, 0,
                            1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'
        return repr_str