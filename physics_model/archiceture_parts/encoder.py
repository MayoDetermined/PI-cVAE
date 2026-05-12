import torch
import torch.nn as nn
from ..blocks.custom_layers import ResBlock, ConvBlock


class Encoder(nn.Module):
    """Encoder: CNN backbone that maps input to latent distribution."""
    
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        cond_dim: int = 8,
        nx: int = 104,
        ny: int = 50,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        
        # CNN backbone with residual blocks after each downsampling
        self.conv1 = ConvBlock(in_channels, 32, kernel=4, stride=2, padding=1)
        self.res1 = ResBlock(32)

        self.conv2 = ConvBlock(32, 64, kernel=4, stride=2, padding=1)
        self.res2 = ResBlock(64)

        self.conv3 = ConvBlock(64, 128, kernel=4, stride=2, padding=1)
        self.res3 = ResBlock(128)

        self.conv4 = ConvBlock(128, 256, kernel=4, stride=2, padding=1)
        self.res4 = ResBlock(256)
        
        # Calculate flattened size after 4 downsampling steps (2^4 = 16x)
        flat_h = nx // 16
        flat_w = ny // 16
        self.flat_size = 256 * flat_h * flat_w
        
        # Fully connected layers for latent distribution
        self.fc_hidden = nn.Linear(self.flat_size, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        self.act = nn.GELU()
        
        # Conditioning projection: adds c into hidden representation
        self.cond_proj = nn.Linear(cond_dim, 512)

    def forward(self, x: torch.Tensor, c: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch, in_channels, nx, ny)
            c: Conditioning tensor (batch, cond_dim), optional
            
        Returns:
            mu: Mean of latent distribution (batch, latent_dim)
            logvar: Log variance of latent distribution (batch, latent_dim)
        """
        # Convolutional backbone
        h = self.conv1(x)
        h = self.res1(h)
        
        h = self.conv2(h)
        h = self.res2(h)
        
        h = self.conv3(h)
        h = self.res3(h)
        
        h = self.conv4(h)
        h = self.res4(h)
        
        # Flatten and pass through FC layers
        h = h.view(h.size(0), -1)
        h = self.fc_hidden(h)
        
        # Add conditioning before activation so c can modulate feature selection
        if c is not None:
            h = h + self.cond_proj(c.float())
        
        h = self.act(h)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar