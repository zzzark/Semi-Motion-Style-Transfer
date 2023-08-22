import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from motion_tensor.kinematics import forward_kinematics
from motion_tensor.rotations import rotation_6d_to_matrix


def get_conv_pad(pad, kernel_size, stride) -> List[nn.Module]:
    pad_l = (kernel_size - stride) // 2
    pad_r = (kernel_size - stride) - pad_l

    if pad == 'reflect':
        pad_nn = nn.ReflectionPad1d((pad_l, pad_r))
    elif pad == 'replicate':
        pad_nn = nn.ReplicationPad1d((pad_l, pad_r))
    elif pad == 'zero':
        pad_nn = nn.ConstantPad1d((pad_l, pad_r), 0.0)
    else:
        raise ValueError(f"Unsupported Operator: {pad}")

    return [pad_nn]


def get_acti_layer(acti='relu', inplace=False) -> List[nn.Module]:
    if acti == 'relu':
        return [nn.ReLU(inplace=inplace)]
    elif acti == 'lrelu':
        return [nn.LeakyReLU(0.2, inplace=inplace)]
    elif acti == 'tanh':
        return [nn.Tanh()]
    elif acti == 'none':
        return []
    else:
        raise ValueError(f"Unsupported Operator: {acti}")


def get_norm_layer(norm, norm_dim) -> List[nn.Module]:

    if norm == 'bn':
        return [nn.BatchNorm1d(norm_dim)]
    elif norm == 'in':
        return [nn.InstanceNorm1d(norm_dim, affine=True, track_running_stats=False)]
    elif norm == 'adain':
        raise NotImplementedError("This module is not implemented in the `get_norm_layer` function.")
    elif norm == 'none':
        return []
    else:
        raise ValueError(f"Unsupported Operator: {norm}")


def get_dropout_layer(dropout=None) -> List[nn.Module]:
    if dropout is not None:
        return [nn.Dropout(p=dropout)]
    else:
        return []


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, m):
        return F.interpolate(m, scale_factor=self.scale_factor, mode=self.mode)


def get_upsample_layer(scale_factor, mode='nearest'):
    return [Upsample(scale_factor, mode)]


class AdaIN(nn.Module):
    def __init__(self, in_feat, mpl_layers=3):
        super(AdaIN, self).__init__()

        mlp = [LinearBlock(in_feat, in_feat, norm='none', acti='lrelu') for _ in range(mpl_layers - 1)]
        mlp += [LinearBlock(in_feat, in_feat * 2, norm='none', acti='none')]
        self.mlp = nn.Sequential(*mlp)
        self.norm = nn.InstanceNorm1d(in_feat, affine=False)

    def forward(self, x_c, x_s):
        x_s = F.max_pool1d(x_s, x_s.shape[-1])
        x_s = x_s.view(x_s.shape[0], -1)
        ms = self.mlp(x_s)
        sz = ms.shape[-1] // 2
        m = ms[..., :sz, None]  # mean
        v = ms[..., sz:, None]  # var

        x = self.norm(x_c)
        return (v + 1.0) * x + m


class SANet(nn.Module):
    def __init__(self, in_feat):
        super(SANet, self).__init__()
        self.in_feat = in_feat
        self.net_n = nn.Conv1d(in_feat, in_feat, kernel_size=(1,), stride=(1,))
        self.net_m = nn.Conv1d(in_feat, in_feat, kernel_size=(1,), stride=(1,))
        self.net_l = nn.Conv1d(in_feat, in_feat, kernel_size=(1,), stride=(1,))
        self.net_o = nn.Conv1d(in_feat, in_feat, kernel_size=(1,), stride=(1,))
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.InstanceNorm1d(in_feat)

    def forward(self, x_c, x_s):
        Mn = self.net_n(self.norm(x_c))
        Mm = self.net_m(self.norm(x_s))

        MA = torch.bmm(Mn.permute(0, 2, 1), Mm)  # [B, Tc, Ts]
        MA = self.softmax(MA)  # [B, Tc, Ts]

        Ml = self.net_l(x_s)  # [B, C, Ts]
        Mo = torch.bmm(Ml, MA.permute(0, 2, 1))  # [B, C, Tc]

        Mo = self.net_o(Mo)  # [B, C, Tc]
        Mo += x_c

        return Mo


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding='reflect', dropout=None,
                 norm='none', acti='lrelu', bias=True, inplace=False):

        super(ConvBlock, self).__init__()

        layers: List[nn.Module] = []
        layers += get_conv_pad(padding, kernel_size, stride)
        layers += [nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=(stride,), bias=bias)]
        layers += get_norm_layer(norm, norm_dim=out_channels)
        layers += get_acti_layer(acti, inplace=inplace)
        layers += get_dropout_layer(dropout)

        self.net = nn.Sequential(*layers)

    def forward(self, m):
        return self.net(m)


class ResConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride, padding='reflect', dropout=None,
                 norm='none', acti='lrelu', bias=True, inplace=False):
        super(ResConvBlock, self).__init__()
        self.cov = nn.Sequential(
            ConvBlock(in_channels,  mid_channels, kernel_size, stride, padding, dropout, norm, acti, bias, inplace),
            ConvBlock(mid_channels, out_channels, kernel_size, stride, padding, dropout, norm, acti, bias, inplace),
        )

        if in_channels != out_channels:  # shortcut
            self.sht = ConvBlock(in_channels, out_channels, 1, 1, padding,
                                 dropout='none', norm='none', acti='none', bias=False, inplace=False)
        else:
            self.sht = None

    def forward(self, x):
        s = x if self.sht is None else self.sht(x)
        x = self.cov(x)
        return s + x


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=None, norm='none', acti='relu', bias=True):
        super(LinearBlock, self).__init__()
        layers = [nn.Linear(in_dim, out_dim, bias=bias)]
        layers += get_norm_layer(norm, norm_dim=out_dim)
        layers += get_acti_layer(acti)
        layers += get_dropout_layer(dropout)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AdditionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, reduce=0.01,
                 padding='reflect', dropout=None, norm='none', acti='lrelu', bias=True, inplace=False):
        super(AdditionConv, self).__init__()

        layers: List[nn.Module] = []
        layers += get_conv_pad(padding, kernel_size, stride)
        layers += [nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=(stride,), bias=bias)]
        self.conv = nn.Sequential(*layers)

        layers: List[nn.Module] = []
        layers += get_norm_layer(norm, norm_dim=out_channels)
        layers += get_acti_layer(acti, inplace=inplace)
        layers += get_dropout_layer(dropout)
        self.out = nn.Sequential(*layers)

        self.reduce = reduce  # reduce the input offset
        self.linear = LinearBlock(in_channels, out_channels, dropout, norm='none', acti='none', bias=True)

    def forward(self, m, o):
        m = self.conv(m)  # [B, C, T]
        o = self.linear(o)  # [B, C]
        o = o[..., None] * self.reduce  # [B, C, 1]

        return self.out(m + o)


class ForwardKinematics(nn.Module):
    def __init__(self, p_index: list, offset: torch.Tensor, world=True, is_edge=False):
        super(ForwardKinematics, self).__init__()
        self.p_index = p_index

        offset = offset.detach().cpu()
        if len(offset.shape) == 3:
            offset = offset[None, ...]
        assert len(offset.shape) == 4, f"offset should be [1, J, 3, 1], but got {offset.shape}"
        assert offset.shape[0] == 1, f"offset should be [1, J, 3, 1], but got {offset.shape}"
        assert offset.shape[2] == 3, f"offset should be [1, J, 3, 1], but got {offset.shape}"
        assert offset.shape[3] == 1, f"offset should be [1, J, 3, 1], but got {offset.shape}"

        # self.offset = offset
        self.register_buffer('offset', offset, persistent=False)  # .to(device)

        self.world = world
        self.is_edge = is_edge

    def forward(self, r6d, trs=None):
        """
        [B, J, 6, T] or [B, 6, J, T]
        """
        if r6d.shape[-2] != 6:
            assert r6d.shape[-3] == 6
            r6d = r6d.permute(0, 2, 1, 3)  # [N, V, C, T]
        offset = self.offset.expand(r6d.shape[0], -1, -1, -1)
        mat = rotation_6d_to_matrix(r6d)
        pos = forward_kinematics(self.p_index, mat, trs, offset, self.world, self.is_edge)
        return pos
