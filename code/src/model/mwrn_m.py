import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


def make_model(args, parent=False):
    return MODEL(args)


class CALayer(nn.Module):
    def __init__(self, wn, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                wn(nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True)),
                nn.ReLU(inplace=True),
                wn(nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True)),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class MWRCA(nn.Module):
    def __init__(self,  wn, n_feats, act=nn.ReLU(True)):
        super(MWRCA, self).__init__()
        self.expand = wn(nn.Conv2d(n_feats, 48, 1, 1, 0))
        self.conv_k3 = nn.Sequential(
            wn(nn.Conv2d(24, 96, 3, 1, 1)),
            act,
            wn(nn.Conv2d(96, 24, 3, 1, 1))
        )
        self.conv_k5 = nn.Sequential(
            wn(nn.Conv2d(16, 64, 3, 1, 1)),
            act,
            wn(nn.Conv2d(64, 16, 3, 1, 2, dilation=2))
        )
        self.conv_k7 = nn.Sequential(
            wn(nn.Conv2d(8, 32, 3, 1, 1)),
            act,
            wn(nn.Conv2d(32, 8, 3, 1, 3, dilation=3))
        )
        self.reduction = wn(nn.Conv2d(48, n_feats, 1, 1, 0))
        self.ca = CALayer(wn, n_feats, reduction=8)

    def forward(self, x):
        x_exp = self.expand(x)
        x_slice_k3 = x_exp[:, :24, :, :]
        x_slice_k5 = x_exp[:, 24:40, :, :]
        x_slice_k7 = x_exp[:, 40:, :, :]
        res_k3 = self.conv_k3(x_slice_k3)
        res_k3 += x_slice_k3
        res_k5 = self.conv_k5(x_slice_k5)
        res_k5 += x_slice_k5
        res_k7 = self.conv_k7(x_slice_k7)
        res_k7 += x_slice_k7
        res = torch.cat([res_k3, res_k5, res_k7], dim=1)
        res = self.reduction(res)
        res = self.ca(res)
        res += x
        return res


class FCA(nn.Module):
    def __init__(self, wn, in_channels, out_channels):
        super(FCA, self).__init__()
        self.fusion = wn(nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        self.body = MWRCA(wn, out_channels, act=nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.fusion(x)
        res = self.body(x)
        return res


class MWRB(nn.Module):
    def __init__(self,  wn, n_feats, act=nn.ReLU(True)):
        super(MWRB, self).__init__()
        self.expand = wn(nn.Conv2d(n_feats, 48, 1, 1, 0))
        self.conv_k3 = nn.Sequential(
            wn(nn.Conv2d(24, 96, 3, 1, 1)),
            act,
            wn(nn.Conv2d(96, 24, 3, 1, 1))
        )
        self.conv_k5 = nn.Sequential(
            wn(nn.Conv2d(16, 64, 3, 1, 1)),
            act,
            wn(nn.Conv2d(64, 16, 3, 1, 2, dilation=2))
        )
        self.conv_k7 = nn.Sequential(
            wn(nn.Conv2d(8, 32, 3, 1, 1)),
            act,
            wn(nn.Conv2d(32, 8, 3, 1, 3, dilation=3))
        )
        self.reduction = wn(nn.Conv2d(48, n_feats, 1, 1, 0))

    def forward(self, x):
        x_exp = self.expand(x)
        x_slice_k3 = x_exp[:, :24, :, :]
        x_slice_k5 = x_exp[:, 24:40, :, :]
        x_slice_k7 = x_exp[:, 40:, :, :]
        res_k3 = self.conv_k3(x_slice_k3)
        res_k3 += x_slice_k3
        res_k5 = self.conv_k5(x_slice_k5)
        res_k5 += x_slice_k5
        res_k7 = self.conv_k7(x_slice_k7)
        res_k7 += x_slice_k7
        res = torch.cat([res_k3, res_k5, res_k7], dim=1)
        res = self.reduction(res)
        res += x
        return res


class CB(nn.Module):
    def __init__(self, wn, n_feats, act=nn.ReLU(True)):
        super(CB, self).__init__()
        self.b0 = MWRB(wn, n_feats, act=act)
        self.b1 = MWRB(wn, n_feats, act=act)
        self.b2 = MWRB(wn, n_feats, act=act)
        self.b3 = MWRB(wn, n_feats, act=act)
        self.b4 = MWRB(wn, n_feats, act=act)
        self.b5 = MWRB(wn, n_feats, act=act)
        self.b6 = MWRB(wn, n_feats, act=act)
        self.b7 = MWRB(wn, n_feats, act=act)
        self.b8 = MWRB(wn, n_feats, act=act)
        self.b9 = MWRB(wn, n_feats, act=act)

        # self.body = nn.ModuleList()
        # for i in range(8):
        #     self.body.append(MWRB(wn, n_feats, act=act))

    def forward(self, x):
        # for i in range(8):
        #     x = self.body[i](x)

        x0 = self.b0(x)
        x1 = self.b1(x0)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        x4 = self.b4(x3)
        x5 = self.b5(x4)
        x6 = self.b6(x5)
        x7 = self.b7(x6)
        x8 = self.b8(x7)
        out = self.b9(x8)

        return out


class MODEL(nn.Module):
    def __init__(self, args):
        super(MODEL, self).__init__()
        # hyper-params
        self.args = args
        scale = args.scale[0]
        n_feats = args.n_feats
        act = nn.ReLU(True)
        # wn = lambda x: x
        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(
            [0.4488, 0.4371, 0.4040])).view([1, 3, 1, 1])

        # define head module
        head = []
        head.append(
            wn(nn.Conv2d(args.n_colors, n_feats, 3, padding=3//2)))

        # define tail module
        tail = []
        out_feats = scale*scale*args.n_colors
        tail.append(
            wn(nn.Conv2d(n_feats, out_feats, 3, padding=3//2)))
        tail.append(nn.PixelShuffle(scale))
        skip = []
        skip.append(
            wn(nn.Conv2d(args.n_colors, out_feats, 5, padding=5//2))
        )
        skip.append(nn.PixelShuffle(scale))

        self.head = nn.Sequential(*head)
        self.b1 = CB(wn, n_feats, act)
        self.b2 = CB(wn, n_feats, act)
        self.b3 = CB(wn, n_feats, act)
        self.b4 = CB(wn, n_feats, act)
        self.b5 = CB(wn, n_feats, act)
        self.b6 = CB(wn, n_feats, act)
        self.b7 = CB(wn, n_feats, act)
        self.b8 = CB(wn, n_feats, act)
        self.c1 = FCA(wn, n_feats*2, n_feats)
        self.c2 = FCA(wn, n_feats*3, n_feats)
        self.c3 = FCA(wn, n_feats*4, n_feats)
        self.c4 = FCA(wn, n_feats*5, n_feats)
        self.c5 = FCA(wn, n_feats*6, n_feats)
        self.c6 = FCA(wn, n_feats*7, n_feats)
        self.c7 = FCA(wn, n_feats * 8, n_feats)
        self.c8 = FCA(wn, n_feats * 9, n_feats)

        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

    def forward(self, x):
        x = (x - self.rgb_mean.cuda()*255)/127.5
        s = self.skip(x)
        x = self.head(x)
        o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([o0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([o0, o1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([o0, o1, o2, b3], dim=1)
        o3 = self.c3(c3)

        b4 = self.b4(o3)
        c4 = torch.cat([o0, o1, o2, o3, b4], dim=1)
        o4 = self.c4(c4)

        b5 = self.b5(o4)
        c5 = torch.cat([o0, o1, o2, o3, o4, b5], dim=1)
        o5 = self.c5(c5)

        b6 = self.b6(o5)
        c6 = torch.cat([o0, o1, o2, o3, o4, o5, b6], dim=1)
        o6 = self.c6(c6)

        b7 = self.b7(o6)
        c7 = torch.cat([o0, o1, o2, o3, o4, o5, o6, b7], dim=1)
        o7 = self.c7(c7)

        b8 = self.b8(o7)
        c8 = torch.cat([o0, o1, o2, o3, o4, o5, o6, o7, b8], dim=1)
        o8 = self.c8(c8)

        x = self.tail(o8)
        x += s
        x = x*127.5 + self.rgb_mean.cuda()*255
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0 or name.find('skip') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
