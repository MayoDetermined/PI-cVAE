import torch
import torch.nn as nn
from ..blocks.custom_layers import ResBlock, ConvBlock


class Decoder(nn.Module):
    """Decoder: Maps latent representation back to input space."""
    
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
            nn.ReLU(),
            nn.Linear(512, self.flat_size),
            nn.ReLU(),
        )
        
        # Transposed convolutional layers with residual blocks
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
        self.deconv4 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1, output_padding=(0, 0))
        
        self.act = nn.GELU()

    def forward(self, z: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            z: Latent representation (batch, latent_dim)
            c: Conditioning information (batch, cond_dim), optional
            
        Returns:
            Reconstructed output (batch, out_channels, nx, ny)
        """
        # Concatenate latent vector with conditioning if provided
        if c is not None:
            z_c = torch.cat([z, c], dim=1)
        else:
            z_c = z
            
        # Expand to spatial feature map
        h = self.fc_expand(z_c)
        h = h.view(-1, 256, self.flat_h, self.flat_w)
        
        # Decode with residual connections and upsampling
        h = self.res1(h)
        h = self.act(self.norm1(self.deconv1(h)))
        
        h = self.res2(h)
        h = self.act(self.norm2(self.deconv2(h)))
        
        h = self.res3(h)
        h = self.act(self.norm3(self.deconv3(h)))
        
        h = self.res4(h)
        h = self.deconv4(h)
        
        # Sigmoid to map output to [0, 1] (normalized space)
        h = torch.sigmoid(h)
        
        return h