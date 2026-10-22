"""U-Net family used for B0 (whole image), B1 (patch -> centre) and B2 (B1 + FiLM conditioning).

Blocks follow the PtyRANNOSAUR description: two 3x3 convolutions, batch norm, leaky ReLU,
2x pooling / upsampling. Outputs 8 channels: heat logit, 6 element logits, height.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_CH = 8  # 0 heat, 1..6 class logits, 7 height


class Block(nn.Module):
    def __init__(self, cin, cout, cond_dim=0):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.b2 = nn.BatchNorm2d(cout)
        self.film = nn.Linear(cond_dim, 2 * cout) if cond_dim else None

    def forward(self, x, cond=None):
        x = F.leaky_relu(self.b1(self.c1(x)), 0.1)
        x = self.b2(self.c2(x))
        if self.film is not None:
            g, b = self.film(cond).chunk(2, dim=1)
            x = x * (1 + g[:, :, None, None]) + b[:, :, None, None]
        return F.leaky_relu(x, 0.1)


class UNet(nn.Module):
    def __init__(self, in_ch=10, base=32, depth=4, cond_dim=0):
        super().__init__()
        self.depth = depth
        self.cond_dim = cond_dim
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList([Block(in_ch, ch[0], cond_dim)] +
                                  [Block(ch[i], ch[i + 1], cond_dim) for i in range(depth)])
        self.up = nn.ModuleList([Block(ch[i + 1] + ch[i], ch[i], cond_dim) for i in reversed(range(depth))])
        self.head = nn.Conv2d(ch[0], OUT_CH, 1)

    def forward(self, x, cond=None):
        skips = []
        for i, blk in enumerate(self.down):
            if i > 0:
                x = F.max_pool2d(x, 2)
            x = blk(x, cond)
            skips.append(x)
        skips.pop()
        for blk in self.up:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = blk(torch.cat([x, skips.pop()], 1), cond)
        return self.head(x)


def make_model(kind):
    if kind == "B0":
        return UNet(depth=4, base=32)
    if kind == "B1":
        return UNet(depth=3, base=32)
    if kind == "B2":
        return UNet(depth=3, base=32, cond_dim=2)
    raise ValueError(kind)


PATCH, CENTRE = 48, 16  # B1/B2 window and scored centre (px)


def norm_cond(amp, stiffness):
    """Map (amplitude A, CO stiffness N/m) to roughly [-1, 1]."""
    return torch.stack([(amp - 0.9) / 0.5, (stiffness - 0.7) / 0.3], dim=-1).float()
