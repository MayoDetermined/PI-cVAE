import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .archiceture_parts.encoder import Encoder
from .archiceture_parts.decoder import Decoder
from .archiceture_parts.prior_net import PriorNet, softplus_logvar


# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE(nn.Module):
    """Base Variational Autoencoder implementation."""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_dim: int = 16,
        nx: int = 104,
        ny: int = 50,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            nx=nx,
            ny=ny,
        )
        
        # Decoder
        self.decoder = Decoder(
            out_channels=out_channels,
            latent_dim=latent_dim,
            nx=nx,
            ny=ny,
        )

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + eps * std
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, in_channels, nx, ny)
            
        Returns:
            decoded: Reconstructed output
            mu: Mean of posterior
            logvar: Log variance of posterior
            z: Sampled latent code
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        decoded = self.decoder(z)
        
        return decoded, mu, logvar, z

    def sample(
        self,
        num_samples: int,
        device: torch.device = None
    ) -> torch.Tensor:
        """Generate samples from standard normal prior."""
        if device is None:
            device = next(self.parameters()).device
            
        with torch.no_grad():
            # Sample from standard normal
            z = torch.randn(num_samples, self.latent_dim, device=device)
            # Decode
            samples = self.decoder(z)
        return samples


class ConditionalVAE(VAE):
    """Conditional VAE: Condition latent space on class labels or continuous conditions."""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_dim: int = 16,
        cond_dim: int = 8,
        nx: int = 104,
        ny: int = 50,
        use_prior_net: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_dim=latent_dim,
            nx=nx,
            ny=ny,
        )
        self.cond_dim = cond_dim
        self.use_prior_net = use_prior_net
        
        # Override base encoder with conditioning-aware version
        self.encoder = Encoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            nx=nx,
            ny=ny,
        )
        
        # Optional prior network for learning p(z|c)
        if use_prior_net:
            self.prior_net = PriorNet(
                cond_dim=cond_dim,
                latent_dim=latent_dim,
                hidden=256,
            )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, in_channels, nx, ny)
            c: Conditioning information (batch, cond_dim)
            
        Returns:
            decoded: Reconstructed output
            mu: Mean of posterior
            logvar: Log variance of posterior
            z: Sampled latent code
        """
        # Encode with conditioning
        mu, logvar = self.encoder(x, c)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode with conditioning (c is concatenated inside decoder)
        decoded = self.decoder(z, c)
        
        return decoded, mu, logvar, z

    def sample(
        self,
        num_samples: int,
        c: torch.Tensor,
        use_prior: bool = True,
        device: torch.device = None
    ) -> torch.Tensor:
        """Generate samples from conditional prior."""
        if device is None:
            device = next(self.parameters()).device
            
        with torch.no_grad():
            if use_prior and self.use_prior_net:
                # Sample from learned prior p(z|c)
                mu_prior, logvar_prior = self.prior_net(c)
                z = self.reparameterize(mu_prior, logvar_prior)
            else:
                # Sample from standard normal prior
                z = torch.randn(num_samples, self.latent_dim, device=device)
            
            # Decode
            samples = self.decoder(z, c)
        return samples