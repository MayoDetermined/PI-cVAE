import torch
import torch.nn as nn
from ..blocks.custom_layers import ResBlock, ConvBlock


class Decoder(nn.Module):
    """
    Decoder: Maps latent representation back to input space.

    Shared CNN backbone (4x upsampling blocks) produces a 32-channel feature
    map at half resolution.  Four per-quantity heads then do the final 2x
    upsampling and channel projection.  Each head has capacity matched to the
    complexity of the field it must predict:

      te  — smooth log-scalar (1 ch)  : direct ConvTranspose  (no extra block)
      ti  — smooth log-scalar (1 ch)  : direct ConvTranspose
      na  — smooth log-scalar (10 ch) : 1 × ResBlock + ConvTranspose
      ua  — bipolar velocities (10 ch): 2 × ResBlock + ConvTranspose
    """

    def __init__(
        self,
        out_channels: int,
        latent_dim: int,
        cond_dim: int = 8,
        nx: int = 104,
        ny: int = 50,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.nx = nx
        self.ny = ny

        # Calculate intermediate spatial dimensions
        self.flat_h = nx // 16
        self.flat_w = ny // 16
        self.flat_size = 256 * self.flat_h * self.flat_w

        # FC layers to expand latent vector to spatial feature map
        self.fc_expand = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 512),
            nn.GELU(),
            nn.Linear(512, self.flat_size),
            nn.GELU(),
        )

        # ── Shared backbone: 4 upsampling stages ──────────────────────────
        self.res1 = ResBlock(256)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=(1, 0))
        self.norm1 = nn.GroupNorm(min(32, 128), 128)

        self.res2 = ResBlock(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=(0, 0))
        self.norm2 = nn.GroupNorm(min(32, 64), 64)

        self.res3 = ResBlock(64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=(0, 1))
        self.norm3 = nn.GroupNorm(min(32, 32), 32)

        self.res4 = ResBlock(32)
        # After res4: feature map is (B, 32, nx//2, ny//2)

        self.act = nn.GELU()

        # ── Per-quantity heads (final 2x upsample + projection) ───────────
        _kw = dict(kernel_size=4, stride=2, padding=1, output_padding=(0, 0))

        # te: 1 smooth channel — ResBlock + ConvTranspose (consistent with na/ua heads)
        self.head_te = nn.Sequential(
            ResBlock(32),
            nn.ConvTranspose2d(32, 1, **_kw),
        )

        # ti: 1 smooth channel — ResBlock + ConvTranspose
        self.head_ti = nn.Sequential(
            ResBlock(32),
            nn.ConvTranspose2d(32, 1, **_kw),
        )

        # na: 10 density channels — one extra ResBlock for species variation
        self.head_na = nn.Sequential(
            ResBlock(32),
            nn.ConvTranspose2d(32, 10, **_kw),
        )

        # ua: 10 velocity channels — two extra ResBlocks for bipolar structure
        self.head_ua = nn.Sequential(
            ResBlock(32),
            ResBlock(32),
            nn.ConvTranspose2d(32, 10, **_kw),
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            z: Latent representation (batch, latent_dim)
            c: Conditioning information (batch, cond_dim), optional

        Returns:
            Reconstructed output (batch, out_channels, nx, ny)
            Channel order: te(1) | ti(1) | na(10) | ua(10)
        """
        # Concatenate latent vector with conditioning if provided
        if c is not None:
            z_c = torch.cat([z, c], dim=1)
        else:
            z_c = z

        # Expand to spatial feature map
        h = self.fc_expand(z_c)
        h = h.view(-1, 256, self.flat_h, self.flat_w)

        # Shared backbone
        h = self.res1(h)
        h = self.act(self.norm1(self.deconv1(h)))

        h = self.res2(h)
        h = self.act(self.norm2(self.deconv2(h)))

        h = self.res3(h)
        h = self.act(self.norm3(self.deconv3(h)))

        h = self.res4(h)   # (B, 32, nx//2, ny//2)

        # Per-quantity heads
        te = self.head_te(h)          # (B,  1, nx, ny)
        ti = self.head_ti(h)          # (B,  1, nx, ny)
        na = self.head_na(h)          # (B, 10, nx, ny)
        ua = self.head_ua(h)          # (B, 10, nx, ny)

        out = torch.cat([te, ti, na, ua], dim=1)   # (B, 22, nx, ny)

        # Sigmoid: map all outputs to [0, 1] (normalized space)
        return torch.sigmoid(out)
