
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def preprocess(x):
    B, C, H, W = x.shape
    total_pixels = H * W
    x_flat = x.reshape(B, -1)

    k1 = max(1, int(0.01 * total_pixels))
    k99 = min(total_pixels, int(0.99 * total_pixels))

    p1 = torch.kthvalue(x_flat, k1, dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    p99 = torch.kthvalue(x_flat, k99, dim=1, keepdim=True)[0].view(B, 1, 1, 1)

    x = torch.clamp(x, p1, p99)
    x = (x - p1) / (p99 - p1 + 1e-8)

    return x, p99, p1


class LayerNorm2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + 1e-6)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class s_f_extraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


class UpsampleBlock(nn.Module):
    # Final version present in the supplied notebook:
    # Conv -> PixelShuffle(2)
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        x = self.conv(x)
        return self.pixel_shuffle(x)


class restormer(nn.Module):
    def __init__(self, C):
        super().__init__()

        self.conv1 = nn.Conv2d(C, C, 1)
        self.conv2 = nn.Conv2d(C, C, 1)
        self.conv3 = nn.Conv2d(C, 2 * C, 1)
        self.conv4 = nn.Conv2d(2 * C, 2 * C, 3, padding=1, groups=2 * C)
        self.conv5 = nn.Conv2d(C, C, 1)

        self.dwconv_q = nn.Conv2d(C, C, 3, padding=1, groups=C)
        self.dwconv_k = nn.Conv2d(C, C, 3, padding=1, groups=C)
        self.dwconv_v = nn.Conv2d(C, C, 3, padding=1, groups=C)
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))

        self.LayerNorm1 = LayerNorm2D(C)
        self.LayerNorm2 = LayerNorm2D(C)
        self.gelu = nn.GELU()

    def forward(self, x):
        x_in = x
        B, C, H, W = x.shape

        x_ln = self.LayerNorm1(x)
        x_edit = self.conv1(x_ln)

        q = self.dwconv_q(x_edit).reshape(B, C, H * W)
        k = self.dwconv_k(x_edit).reshape(B, C, H * W)
        v = self.dwconv_v(x_edit).reshape(B, C, H * W)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = q @ k.transpose(-1, -2) * self.temperature
        attn = torch.softmax(attn, dim=-1)
        score = attn @ v

        z = self.conv2(score.reshape(B, C, H, W))
        y = x_in + z

        y_in = y
        y_ln = self.LayerNorm2(y)
        y_edit = self.conv3(y_ln)

        y1, y2 = self.conv4(y_edit).chunk(2, dim=1)
        gated = self.gelu(y1) * y2

        out = self.conv5(gated)
        return out + y_in


class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.stage1 = nn.ModuleList([restormer(channels) for _ in range(4)])
        self.stage2 = nn.ModuleList([restormer(2 * channels) for _ in range(6)])
        self.stage3 = nn.ModuleList([restormer(4 * channels) for _ in range(6)])

        self.conv1 = nn.Conv2d(channels, 2 * channels, 3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(2 * channels, 4 * channels, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(4 * channels, 8 * channels, 3, padding=1, stride=2)

    def forward(self, x):
        add_decoder = []

        for block in self.stage1:
            x = block(x)
        add_decoder.append(x)
        x = self.conv1(x)

        for block in self.stage2:
            x = block(x)
        add_decoder.append(x)
        x = self.conv2(x)

        for block in self.stage3:
            x = block(x)
        add_decoder.append(x)
        x = self.conv3(x)

        return x, add_decoder


class BottleNeck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.stage = nn.ModuleList([restormer(8 * channels) for _ in range(8)])

    def forward(self, x):
        for block in self.stage:
            x = block(x)
        return x


class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.stage1 = nn.ModuleList([restormer(4 * channels) for _ in range(6)])
        self.stage2 = nn.ModuleList([restormer(2 * channels) for _ in range(6)])
        self.stage3 = nn.ModuleList([restormer(channels) for _ in range(4)])

        self.upsample = nn.ModuleList([
            UpsampleBlock(8 * channels, 4 * channels),
            UpsampleBlock(4 * channels, 2 * channels),
            UpsampleBlock(2 * channels, channels),
        ])

    def forward(self, x, add_decoder):
        x = self.upsample[0](x)
        x = x + add_decoder[2]
        for block in self.stage1:
            x = block(x)

        x = self.upsample[1](x)
        x = x + add_decoder[1]
        for block in self.stage2:
            x = block(x)

        x = self.upsample[2](x)
        x = x + add_decoder[0]
        for block in self.stage3:
            x = block(x)

        return x


class Arch(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.s_f_e = s_f_extraction()
        self.encoder = Encoder(channels)
        self.bottle = BottleNeck(channels)
        self.decoder = Decoder(channels)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(int(channels / 4), 1, 3, padding=1)

    def forward(self, x):
        x, p99, p1 = preprocess(x)
        base = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)

        x = self.s_f_e(x)
        x, add_decoder = self.encoder(x)
        x = self.bottle(x)
        x = self.decoder(x, add_decoder)

        x = self.conv(x)
        x = F.pixel_shuffle(x, upscale_factor=2)
        res = self.conv2(x)

        out = base + res
        out = out * (p99 - p1) + p1
        out = torch.clamp(out, 0.0, 1.0)
        return out


