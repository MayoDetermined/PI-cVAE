import torch.nn as nn

class ConvBlock(nn.Module):
    """Conv2d + GroupNorm + GELU."""
    def __init__(self, in_ch, out_ch, kernel, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding)
        self.norm = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    
class ResBlock(nn.Module):
    """Pre-activation residual block: GN-GELU-Conv-GN-GELU-Conv + skip."""
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(min(32, ch), ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(min(32, ch), ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

