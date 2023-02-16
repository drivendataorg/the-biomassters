import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from copy import deepcopy
from einops import rearrange
from torch.nn import LayerNorm
from typing import Optional, Sequence, Type

from monai.networks.blocks import PatchEmbed
from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.layers import DropPath, trunc_normal_


def window_partition(x, window_size):
    '''window partition operation based on: 'Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>'
    https://github.com/microsoft/Swin-Transformer

    Args:
        x:           input tensor.
        window_size: local window size.
    '''
    x_shape = x.size()
    if len(x_shape) == 5:
        b, d, h, w, c = x_shape
        x = x.view(
            b,
            d // window_size[0],
            window_size[0],
            h // window_size[1],
            window_size[1],
            w // window_size[2],
            window_size[2],
            c,
        )
        windows = (
            x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
        )
    elif len(x_shape) == 4:
        b, h, w, c = x.shape
        x = x.view(b, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], c)
    return windows


def window_reverse(windows, window_size, dims):
    '''window reverse operation based on: 'Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>'
    https://github.com/microsoft/Swin-Transformer

    Args:
        windows:     windows tensor.
        window_size: local window size.
        dims:        dimension values.
    '''
    if len(dims) == 4:
        b, d, h, w = dims
        x = windows.view(
            b,
            d // window_size[0],
            h // window_size[1],
            w // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    elif len(dims) == 3:
        b, h, w = dims
        x = windows.view(b, h // window_size[0], w // window_size[1], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    '''Computing window size based on: 'Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>'
    https://github.com/microsoft/Swin-Transformer

    Args:
        x_size:      input size.
        window_size: local window size.
        shift_size:  window shifting size.
    '''

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttentionV1(nn.Module):
    '''
    Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>
    https://github.com/microsoft/Swin-Transformer
    '''

    def __init__(
        self,
        dim:         int,
        num_heads:   int,
        window_size: Sequence[int],
        qkv_bias:    bool = False,
        attn_drop:   float = 0.0,
        proj_drop:   float = 0.0,
    ) -> None:
        '''
        Args:
            dim: number  of feature channels.
            num_heads:   number of attention heads.
            window_size: local window size.
            qkv_bias:    add a learnable bias to query, key, value.
            attn_drop:   attention dropout rate.
            proj_drop:   dropout rate of output.
        '''

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        if len(self.window_size) == 3:

            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1

        elif len(self.window_size) == 2:

            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
            else:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowAttentionV2(nn.Module):
    '''
    Window based multi-head self attention module with relative position bias based on: 'Liu et al.,
    Swin Transformer V2: Scaling Up Capacity and Resolution
    <https://arxiv.org/abs/2111.09883>'
    https://github.com/microsoft/Swin-Transformer
    '''

    def __init__(
        self,
        dim:         int,
        num_heads:   int,
        window_size: Sequence[int],
        qkv_bias:    bool = False,
        attn_drop:   float = 0.0,
        proj_drop:   float = 0.0,
    ) -> None:
        '''
        Args:
            dim: number  of feature channels.
            num_heads:   number of attention heads.
            window_size: local window size.
            qkv_bias:    add a learnable bias to query, key, value.
            attn_drop:   attention dropout rate.
            proj_drop:   dropout rate of output.
        '''

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        mesh_args = torch.meshgrid.__kwdefaults__

        scale_params = torch.log(10 * torch.ones((num_heads, 1, 1)))
        self.logit_scale = nn.Parameter(scale_params, requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(3, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )

        if len(self.window_size) == 3:

            # get relative_coords_table
            relative_coords_d = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
            relative_coords_h = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
            relative_coords_w = torch.arange(-(self.window_size[2] - 1), self.window_size[2], dtype=torch.float32)
            if mesh_args is not None:
                relative_coords_table = torch.stack(torch.meshgrid(relative_coords_d, relative_coords_h, relative_coords_w, indexing='ij'))
            else:
                relative_coords_table = torch.stack(torch.meshgrid(relative_coords_d, relative_coords_h, relative_coords_w))
            relative_coords_table = relative_coords_table.permute(1, 2, 3, 0).contiguous().unsqueeze(0)
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
            relative_coords_table[:, :, :, 2] /= (self.window_size[2] - 1)

            # get pair-wise relative position index for each token inside the window
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1

        elif len(self.window_size) == 2:

            # get relative_coords_table
            relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
            relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
            if mesh_args is not None:
                relative_coords_table = torch.stack(torch.meshgrid(relative_coords_h, relative_coords_w, indexing='ij'))
            else:
                relative_coords_table = torch.stack(torch.meshgrid(relative_coords_h, relative_coords_w))
            relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous().unsqueeze(0)
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
            else:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)
        self.register_buffer('relative_coords_table', relative_coords_table)

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        b, n, c = x.shape
        device = x.device

        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        max_scale = torch.log(torch.tensor(1.0 / 0.01, device=device))
        logit_scale = torch.clamp(self.logit_scale, max=max_scale).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    '''
    Swin Transformer block based on: 'Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>'
    https://github.com/microsoft/Swin-Transformer
    '''

    def __init__(
        self,
        dim:            int,
        num_heads:      int,
        window_size:    Sequence[int],
        shift_size:     Sequence[int],
        mlp_ratio:      float = 4.0,
        qkv_bias:       bool = True,
        drop:           float = 0.0,
        attn_drop:      float = 0.0,
        attn_version:   str = 'v2',
        drop_path:      float = 0.0,
        act_layer:      str = 'GELU',
        norm_layer:     Type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
    ) -> None:
        '''
        Args:
            dim: number     of feature channels.
            num_heads:      number of attention heads.
            window_size:    local window size.
            shift_size:     window shift size.
            mlp_ratio:      ratio of mlp hidden dim to embedding dim.
            qkv_bias:       add a learnable bias to query, key, value.
            drop:           dropout rate.
            attn_drop:      attention dropout rate.
            drop_path:      stochastic depth rate.
            act_layer:      activation layer.
            norm_layer:     normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        '''

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(dim)

        self.attn_version = attn_version
        if attn_version == 'v1':
            self.attn = WindowAttentionV1(
                dim,
                window_size = self.window_size,
                num_heads   = num_heads,
                qkv_bias    = qkv_bias,
                attn_drop   = attn_drop,
                proj_drop   = drop,
            )
        elif attn_version == 'v2':
            self.attn = WindowAttentionV2(
                dim,
                window_size = self.window_size,
                num_heads   = num_heads,
                qkv_bias    = qkv_bias,
                attn_drop   = attn_drop,
                proj_drop   = drop,
            )
        else:
            raise ValueError('unknown attn_version')

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            hidden_size  = dim,
            mlp_dim      = mlp_hidden_dim,
            act          = act_layer,
            dropout_rate = drop,
            dropout_mode = 'swin'
        )

    def forward_part1(self, x, mask_matrix):
        x_shape = x.size()
        x = self.norm1(x)
        if len(x_shape) == 5:
            b, d, h, w, c = x.shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            _, dp, hp, wp, _ = x.shape
            dims = [b, dp, hp, wp]

        elif len(x_shape) == 4:
            b, h, w, c = x.shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            pad_l = pad_t = 0
            pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
            pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, hp, wp, _ = x.shape
            dims = [b, hp, wp]

        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if len(x_shape) == 5:
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :d, :h, :w, :].contiguous()
        elif len(x_shape) == 4:
            if pad_r > 0 or pad_b > 0:
                x = x[:, :h, :w, :].contiguous()

        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
        return x


class PatchMerging(nn.Module):
    '''The `PatchMerging` module previously defined in v0.9.0.'''

    def __init__(
        self,
        input_resolution,
        dim:          int,
        norm_layer:   Type[LayerNorm] = nn.LayerNorm,
        spatial_dims: int = 3
    ) -> None:
        '''
        Args:
            input_resolution: resolution of input feature maps.
            dim:              number of feature channels.
            norm_layer:       normalization layer.
            spatial_dims:     number of spatial dims.
        '''

        super().__init__()
        assert spatial_dims == 3, 'PatchMerging supports spatial_dims 3 only'

        self.dim = dim
        self.input_resolution = input_resolution
        D, H, W = self.input_resolution
        self.merge_D = D % 2 == 0
        self.merge_H = H % 2 == 0
        self.merge_W = W % 2 == 0

        dims_tmp = dim
        self.resample_scale = [1, 1, 1]
        if self.merge_D:
            dims_tmp *= 2
            self.resample_scale[0] = 2
        if self.merge_H:
            dims_tmp *= 2
            self.resample_scale[1] = 2
        if self.merge_W:
            dims_tmp *= 2
            self.resample_scale[2] = 2

        self.output_resolution = [
            i // s for i, s in
            zip(input_resolution, self.resample_scale)
        ]

        self.norm = norm_layer(dims_tmp)
        self.output_dims = 2 * dim
        self.reduction = nn.Linear(dims_tmp, self.output_dims, bias=False)

    def forward(self, x):
        # x: (B, D, H, W, C)

        if self.merge_D and self.merge_H and self.merge_W:
            x = torch.cat([
                x[:, 0::2, 0::2, 0::2, :],
                x[:, 1::2, 0::2, 0::2, :],
                x[:, 0::2, 1::2, 0::2, :],
                x[:, 0::2, 0::2, 1::2, :],
                x[:, 1::2, 1::2, 0::2, :],
                x[:, 1::2, 0::2, 1::2, :],
                x[:, 0::2, 1::2, 1::2, :],
                x[:, 1::2, 1::2, 1::2, :],
            ], dim=-1)  # x: (B, D//2, H//2, W//2, 8C)
        elif (not self.merge_D) and self.merge_H and self.merge_W:
            x = torch.cat([
                x[:, :, 0::2, 0::2, :],
                x[:, :, 1::2, 0::2, :],
                x[:, :, 0::2, 1::2, :],
                x[:, :, 1::2, 1::2, :],
            ], dim=-1)  # x: (B, D, H//2, W//2, 4C)
        elif self.merge_D and (not self.merge_H) and self.merge_W:
            x = torch.cat([
                x[:, 0::2, :, 0::2, :],
                x[:, 1::2, :, 0::2, :],
                x[:, 0::2, :, 1::2, :],
                x[:, 1::2, :, 1::2, :],
            ], dim=-1)  # x: (B, D//2, H, W//2, 4C)
        elif self.merge_D and self.merge_H and (not self.merge_W):
            x = torch.cat([
                x[:, 0::2, 0::2, :, :],
                x[:, 1::2, 0::2, :, :],
                x[:, 0::2, 1::2, :, :],
                x[:, 1::2, 1::2, :, :],
            ], dim=-1)  # x: (B, D//2, H//2, W, 4C)
        elif (not self.merge_D) and (not self.merge_H) and self.merge_W:
            x = torch.cat([
                x[:, :, :, 0::2, :],
                x[:, :, :, 1::2, :],
            ], dim=-1)  # x: (B, D, H, W//2, 2C)
        elif (not self.merge_D) and self.merge_H and (not self.merge_W):
            x = torch.cat([
                x[:, :, 0::2, :, :],
                x[:, :, 1::2, :, :],
            ], dim=-1)  # x: (B, D, H//2, W, 2C)
        elif self.merge_D and (not self.merge_H) and (not self.merge_W):
            x = torch.cat([
                x[:, 0::2, :, :, :],
                x[:, 1::2, :, :, :],
            ], dim=-1)  # x: (B, D//2, H, W, 2C)
        else:
            pass

        x = self.norm(x)
        x = self.reduction(x)
        # out: (B, D*, H*, W*, 2C)

        return x


def compute_mask(dims, window_size, shift_size, device):
    '''Computing region masks based on: 'Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>'
    https://github.com/microsoft/Swin-Transformer

    Args:
        dims:        dimension values.
        window_size: local window size.
        shift_size:  shift size.
        device:      device.
    '''

    cnt = 0

    if len(dims) == 3:
        d, h, w = dims
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

    elif len(dims) == 2:
        h, w = dims
        img_mask = torch.zeros((1, h, w, 1), device=device)
        for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                img_mask[:, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


class BasicLayer(nn.Module):
    '''
    Basic Swin Transformer layer in one stage based on: 'Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>'
    https://github.com/microsoft/Swin-Transformer
    '''

    def __init__(
        self,
        input_resolution: Sequence[int],
        dim:              int,
        depth:            int,
        num_heads:        int,
        window_size:      Sequence[int],
        drop_path:        list,
        mlp_ratio:        float = 4.0,
        qkv_bias:         bool = False,
        drop:             float = 0.0,
        attn_drop:        float = 0.0,
        attn_version:     str = 'v2',
        norm_layer:       Type[LayerNorm] = nn.LayerNorm,
        downsample:       Optional[nn.Module] = None,
        use_checkpoint:   bool = False,
    ) -> None:
        '''
        Args:
            input_resolution: resolution of input feature maps.
            dim:              number of feature channels.
            depth:            number of layers in each stage.
            num_heads:        number of attention heads.
            window_size:      local window size.
            drop_path:        stochastic depth rate.
            mlp_ratio:        ratio of mlp hidden dim to embedding dim.
            qkv_bias:         add a learnable bias to query, key, value.
            drop:             dropout rate.
            attn_drop:        attention dropout rate.
            norm_layer:       normalization layer.
            downsample:       an optional downsampling layer at the end of the layer.
            use_checkpoint:   use gradient checkpointing for reduced memory usage.
        '''

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim            = dim,
                    num_heads      = num_heads,
                    window_size    = self.window_size,
                    shift_size     = self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio      = mlp_ratio,
                    qkv_bias       = qkv_bias,
                    drop           = drop,
                    attn_drop      = attn_drop,
                    attn_version   = attn_version,
                    drop_path      = drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer     = norm_layer,
                    use_checkpoint = use_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(
                input_resolution = input_resolution,
                dim              = dim,
                norm_layer       = norm_layer,
                spatial_dims     = len(self.window_size)
            )

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, c, d, h, w = x_shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            x = rearrange(x, 'b c d h w -> b d h w c')
            dp = int(np.ceil(d / window_size[0])) * window_size[0]
            hp = int(np.ceil(h / window_size[1])) * window_size[1]
            wp = int(np.ceil(w / window_size[2])) * window_size[2]
            attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask)
            x = x.view(b, d, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, 'b d h w c -> b c d h w')

        elif len(x_shape) == 4:
            b, c, h, w = x_shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            x = rearrange(x, 'b c h w -> b h w c')
            hp = int(np.ceil(h / window_size[0])) * window_size[0]
            wp = int(np.ceil(w / window_size[1])) * window_size[1]
            attn_mask = compute_mask([hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask)
            x = x.view(b, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, 'b h w c -> b c h w')
        return x


class SwinTransformer(nn.Module):
    '''
    Swin Transformer based on: 'Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>'
    https://github.com/microsoft/Swin-Transformer
    '''

    def __init__(
        self,
        image_size:     Sequence[int],
        in_chans:       int,
        embed_dim:      int,
        window_size:    Sequence[int],
        patch_size:     Sequence[int],
        depths:         Sequence[int],
        num_heads:      Sequence[int],
        mlp_ratio:      float = 4.0,
        qkv_bias:       bool = True,
        drop_rate:      float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        attn_version:   str = 'v2',
        norm_layer:     Type[LayerNorm] = nn.LayerNorm,
        patch_norm:     bool = False,
        use_checkpoint: bool = False,
        spatial_dims:   int = 3,
    ) -> None:
        '''
        Args:
            image_size:     dimension of input image.
            in_chans:       dimension of input channels.
            embed_dim:      number of linear projection output channels.
            window_size:    local window size.
            patch_size:     patch size.
            depths:         number of layers in each stage.
            num_heads:      number of attention heads.
            mlp_ratio:      ratio of mlp hidden dim to embedding dim.
            qkv_bias:       add a learnable bias to query, key, value.
            drop_rate:      dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer:     normalization layer.
            patch_norm:     add normalization after patch embedding.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims:   spatial dimension.
        '''

        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            patch_size   = self.patch_size,
            in_chans     = in_chans,
            embed_dim    = embed_dim,
            norm_layer   = norm_layer if self.patch_norm else None,
            spatial_dims = spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        self.resamples = []

        input_size = deepcopy(image_size)
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                input_resolution = input_size,
                dim              = int(embed_dim * 2**i_layer),
                depth            = depths[i_layer],
                num_heads        = num_heads[i_layer],
                window_size      = self.window_size,
                drop_path        = dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio        = mlp_ratio,
                qkv_bias         = qkv_bias,
                drop             = drop_rate,
                attn_drop        = attn_drop_rate,
                attn_version     = attn_version,
                norm_layer       = norm_layer,
                downsample       = PatchMerging,
                use_checkpoint   = use_checkpoint,
            )
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)

            input_size = layer.downsample.output_resolution
            self.resamples.insert(0, layer.downsample.resample_scale)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, 'n c d h w -> n d h w c')
                x = F.layer_norm(x, [ch])
                x = rearrange(x, 'n d h w c -> n c d h w')
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, 'n c h w -> n h w c')
                x = F.layer_norm(x, [ch])
                x = rearrange(x, 'n h w c -> n c h w')
        return x

    def forward(self, x, normalize=True):
        x0 = self.patch_embed(x)
        x0 = self.pos_drop(x0)
        x0_out = self.proj_out(x0, normalize)
        x1 = self.layers1[0](x0.contiguous())
        x1_out = self.proj_out(x1, normalize)
        x2 = self.layers2[0](x1.contiguous())
        x2_out = self.proj_out(x2, normalize)
        x3 = self.layers3[0](x2.contiguous())
        x3_out = self.proj_out(x3, normalize)
        x4 = self.layers4[0](x3.contiguous())
        x4_out = self.proj_out(x4, normalize)
        return [x0_out, x1_out, x2_out, x3_out, x4_out]
